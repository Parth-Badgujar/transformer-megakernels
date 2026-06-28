import json
import torch
import cutlass
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, T
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class WarpgroupMeta:
    tidx: int
    group_tidx: int
    group_id: int
    lane_id: int
    warp_id: int

@dataclass
class Phases:
    compute_phase: int
    input_phase: int
    output_phase: int

@dataclass
class PipelineMeta:
    current_idx: int
    next_idx: int
    expected_cnt: int

@dsl_user_op
def ld_acquire_u32(ptr, *, loc=None, ip=None) -> cutlass.Int32:
    return cutlass.Int32(
        llvm.inline_asm(
            cutlass.Int32.mlir_type,
            [ptr.ir_value(loc=loc, ip=ip)],
            "ld.acquire.gpu.global.u32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )

@dsl_user_op
def atomic_add_release(ptr: cutlass.Int64, val: cutlass.Int32, *, loc=None, ip=None) -> cutlass.Int32:
    return cutlass.Int32(
        llvm.inline_asm(
            cutlass.Int32.mlir_type,
            [ptr.ir_value(loc=loc, ip=ip), cutlass.Int32(val).ir_value(loc=loc, ip=ip)],
            "atom.release.gpu.global.add.u32 $0, [$1], $2;",
            "=r,l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )

@dsl_user_op
def fence_proxy_async_shared_cta(*, loc=None, ip=None) -> None:
    llvm.inline_asm(
        ir.Type.parse("!llvm.void"),
        [],
        "fence.proxy.async.shared::cta;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )

@dsl_user_op
def fence_proxy_async_global(*, loc=None, ip=None) -> None:
    llvm.inline_asm(
        ir.Type.parse("!llvm.void"),
        [],
        "fence.proxy.async.global;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


## Profiler Utils

PROBE_HEADER = 1
PROBE_ENTRY = 4

# Number of tag slots; must be > the largest tag value in TAG_NAMES.
# probe tensor width = PROBE_HEADER + MAX_TAG_SLOTS * PROBE_ENTRY
MAX_TAG_SLOTS = 16

NUM_PROBE_ROLES = 2

ROLE_NAMES = ["warpgroup0", "warpgroup1"]

TAG_NAMES = {
    0: "LOAD_K",
    1: "LOAD_V",
    2: "COMPUTE_QK",
    3: "ROW_REDUCE_SOFTMAX",
    4: "COMPUTE_PV",
    5: "ATTENTION_STORE",
    7: "LOAD_A",
    8: "LOAD_B",
    9: "COMPUTE_AB",
    10: "STORE_C",
    11: "LOAD_ROW",
    12: "COMPUTE_ROW",
    13: "STORE_ROW",
    14: "LOAD_WEIGHTS",
    15: "LOAD_QK"
}
TAGS = {v:k for k, v in TAG_NAMES.items()}


@dsl_user_op
def globaltimer_u64(*, loc=None, ip=None) -> cutlass.Int64:
    t = llvm.inline_asm(
        T.i64(), [],
        "mov.u64 $0, %globaltimer;",
        "=l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )
    return cutlass.Int64(t)


@dsl_user_op
def smid_u32(*, loc=None, ip=None) -> cutlass.Int32:
    t = llvm.inline_asm(
        T.i32(), [],
        "mov.u32 $0, %smid;",
        "=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )
    return cutlass.Int32(t)


# ---------------------------------------------------------------------------
# Probe layout (tag-indexed, supports parallel / overlapping ranges)
# ---------------------------------------------------------------------------
# Each tag gets a fixed slot determined by its tag value:
#
#   col 0                        : header (bitmask of active tags, set by range_finalize)
#   col PROBE_HEADER + tag*4 + 0 : sm_id
#   col PROBE_HEADER + tag*4 + 1 : tag value (redundant but keeps parity with old format)
#   col PROBE_HEADER + tag*4 + 2 : start timestamp (ns)
#   col PROBE_HEADER + tag*4 + 3 : end timestamp (ns, written by range_stop)
#
# Because the slot is keyed by tag, any number of ranges can be open at the
# same time — range_stop identifies its matching range_start via tag_val.
#
# Required probe tensor width:
#   PROBE_HEADER + MAX_TAG_SLOTS * PROBE_ENTRY  (= 1 + 16*4 = 65 columns)
# ---------------------------------------------------------------------------

def range_start(start_probe, row, sm_val, tag_val, warpgroup):
    """Record the start of the range identified by *tag_val*.

    Multiple ranges with different tags can be open simultaneously.
    """
    start_probe[row, sm_val, warpgroup, 0] = cutlass.Int64(tag_val)
    start_probe[row, sm_val, warpgroup, 1] = cutlass.Int64(warpgroup)
    start_probe[row, sm_val, warpgroup, 2] = globaltimer_u64()


def range_stop(stop_probe, row, sm_val, tag_val, warpgroup):
    stop_probe[row, sm_val, warpgroup, 0] = globaltimer_u64()
    stop_probe[row, sm_val, warpgroup, 1] = cutlass.Int64(tag_val)


def dump_probe(
    start_probe: torch.Tensor,
    stop_probe: torch.Tensor,
    num_sms: int,
    max_events,
    out_path: str = "pipeline_trace.json",
):
    """Convert range_start/range_stop probe tensors into a Perfetto/Chrome
    Trace Event Format JSON file.
 
    Expected shapes:
        start_probe: [max_events, num_sms, 2, 3]   # fields: tag, warpgroup, ts_ns
        stop_probe:  [max_events, num_sms, 2, 3]    # fields: ts_ns, tag, (unused)
 
    Mapping to Perfetto:
        SM index        -> pid (process)
        warpgroup index -> tid (thread)   (always 0 or 1)
 
    Rows with tag_val == 0 are unused/empty slots and are skipped.
 
    Start/stop events sharing the same (sm, warpgroup, tag) are paired in
    row order: the 1st start for that tag pairs with the 1st stop for that
    tag, the 2nd with the 2nd, etc. This is correct both for properly
    nested (LIFO) ranges and for non-nested (FIFO) ranges of the same tag.
    """
    SENTINEL_TAG = 0
    NUM_WARPGROUPS = 2
 
    start_np = start_probe.detach().to("cpu").numpy()
    stop_np = stop_probe.detach().to("cpu").numpy()
 
    max_rows = min(max_events, start_np.shape[0], stop_np.shape[0])
 
    starts = defaultdict(list)  # (sm, wg, tag) -> [(row, ts_ns), ...]
    stops = defaultdict(list)   # (sm, wg, tag) -> [(row, ts_ns), ...]
 
    for row in range(max_rows):
        for sm in range(num_sms):
            for wg in range(NUM_WARPGROUPS):
                s_tag = int(start_np[row, sm, wg, 0])
                if s_tag != SENTINEL_TAG:
                    s_ts = int(start_np[row, sm, wg, 2])
                    starts[(sm, wg, s_tag)].append((row, s_ts))
 
                e_tag = int(stop_np[row, sm, wg, 1])
                if e_tag != SENTINEL_TAG:
                    e_ts = int(stop_np[row, sm, wg, 0])
                    stops[(sm, wg, e_tag)].append((row, e_ts))
 
    trace_events = []
 
    all_keys = set(starts.keys()) | set(stops.keys())
    for (sm, wg, tag) in all_keys:
        s_list = sorted(starts.get((sm, wg, tag), []), key=lambda x: x[0])
        e_list = sorted(stops.get((sm, wg, tag), []), key=lambda x: x[0])
 
        n_pairs = min(len(s_list), len(e_list))
        tag_name = TAG_NAMES.get(tag, f"tag_{tag}")
 
        for i in range(n_pairs):
            _, s_ts_ns = s_list[i]
            _, e_ts_ns = e_list[i]
 
            ts_us = s_ts_ns / 1000.0
            dur_us = (e_ts_ns - s_ts_ns) / 1000.0
 
            if dur_us < 0:
                # Stale/garbage pairing — skip rather than emit a
                # negative-duration event Perfetto can't render sensibly.
                continue
 
            trace_events.append(
                {
                    "name": tag_name,
                    "cat": "range",
                    "ph": "X",  # complete event: start + duration in one entry
                    "ts": ts_us,
                    "dur": dur_us,
                    "pid": sm,
                    "tid": wg,
                    "args": {
                        "tag": tag,
                        "tag_name": tag_name,
                        "sm": sm,
                        "warpgroup": wg,
                    },
                }
            )
 
    # Process/thread name metadata so Perfetto shows "SM N" / "Warpgroup N"
    # instead of bare pid/tid numbers.
    for sm in range(num_sms):
        trace_events.append(
            {
                "name": "process_name",
                "ph": "M",
                "pid": sm,
                "tid": 0,
                "args": {"name": f"SM {sm}"},
            }
        )
        for wg in range(NUM_WARPGROUPS):
            trace_events.append(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": sm,
                    "tid": wg,
                    "args": {"name": f"Warpgroup {wg}"},
                }
            )
 
    trace = {
        "traceEvents": trace_events,
        "displayTimeUnit": "ns",
    }
 
    with open(out_path, "w") as f:
        json.dump(trace, f)
 
    return out_path