import json
import heapq
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
    1: "LOAD_K",
    2: "LOAD_V",
    3: "COMPUTE_QK",
    4: "ROW_REDUCE_SOFTMAX",
    5: "COMPUTE_PV",
    6: "ATTENTION_STORE",
    7: "LOAD_A",
    8: "LOAD_B",
    9: "COMPUTE_AB",
    10: "STORE_C",
    11: "LOAD_ROW",
    12: "COMPUTE_ROW",
    13: "STORE_ROW",
    14: "LOAD_WEIGHTS",
    15: "LOAD_Q"
}
TAGS = {v:k for k, v in TAG_NAMES.items()}


@dsl_user_op
def clock64(*, loc=None, ip=None) -> cutlass.Int64:
    """Read per-SM %clock64 — high resolution but NOT synchronized across SMs."""
    t = llvm.inline_asm(
        T.i64(), [],
        "mov.u64 $0, %clock64;",
        "=l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )
    return cutlass.Int64(t)


@dsl_user_op
def globaltimer(*, loc=None, ip=None) -> cutlass.Int64:
    """Read %globaltimer — synchronized across all SMs (lower resolution)."""
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
    start_probe[row, sm_val, warpgroup, 2] = clock64()


def range_stop(stop_probe, row, sm_val, tag_val, warpgroup):
    stop_probe[row, sm_val, warpgroup, 0] = clock64()
    stop_probe[row, sm_val, warpgroup, 1] = cutlass.Int64(tag_val)



def dump_probe(
    start_probe: torch.Tensor,
    stop_probe: torch.Tensor,
    num_sms: int,
    max_events,
    out_path: str = "pipeline_trace.json",
    clock_offsets: torch.Tensor = None,
):
    """Convert range_start/range_stop probe tensors into a Perfetto JSON trace.

    Shapes:
        start_probe : [max_events, num_sms, 2, 3]  — (tag, warpgroup, timestamp)
        stop_probe  : [max_events, num_sms, 2, 2]  — (timestamp, tag)

    Perfetto mapping:  SM → pid,  warpgroup → tid (with lane sub-rows for overlap).
    Start/stop pairs with the same (sm, wg, tag) are matched in row order.
    """
    NUM_WG = 2

    start_np = start_probe.detach().cpu().numpy()
    stop_np  = stop_probe.detach().cpu().numpy()
    offsets  = clock_offsets.detach().cpu().numpy() if clock_offsets is not None else None
    max_rows = min(max_events, start_np.shape[0], stop_np.shape[0])

    # ── Gather start/stop timestamps keyed by (sm, wg, tag) ──────────────
    starts = defaultdict(list)
    stops  = defaultdict(list)

    for row in range(max_rows):
        for sm in range(num_sms):
            off = int(offsets[sm]) if offsets is not None else 0
            for wg in range(NUM_WG):
                s_tag = int(start_np[row, sm, wg, 0])
                if s_tag:
                    starts[(sm, wg, s_tag)].append(int(start_np[row, sm, wg, 2]) + off)
                e_tag = int(stop_np[row, sm, wg, 1])
                if e_tag:
                    stops[(sm, wg, e_tag)].append(int(stop_np[row, sm, wg, 0]) + off)

    # ── Pair starts/stops → events grouped by (sm, wg) ───────────────────
    events_by_wg = defaultdict(list)  # (sm, wg) → list of (start_us, end_us, tag)

    for key in set(starts) | set(stops):
        sm, wg, tag = key
        s_list = starts.get(key, [])
        e_list = stops.get(key, [])
        name = TAG_NAMES.get(tag, f"tag_{tag}")
        for s_ns, e_ns in zip(s_list, e_list):
            dur = e_ns - s_ns
            if dur >= 0:
                events_by_wg[(sm, wg)].append((s_ns / 1000.0, e_ns / 1000.0, tag, name))

    # ── Lane assignment (interval scheduling) + emit trace events ────────
    trace_events = []
    max_lanes = {}  # (sm, wg) → lane count

    for (sm, wg), evs in events_by_wg.items():
        evs.sort()  # by (start, end)

        free  = []   # min-heap of reusable lane indices
        active = []  # min-heap of (end_time, lane)
        next_lane = 0

        for start, end, tag, name in evs:
            # release finished lanes
            while active and active[0][0] <= start:
                heapq.heappush(free, heapq.heappop(active)[1])

            if free:
                lane = heapq.heappop(free)
            else:
                lane = next_lane
                next_lane += 1

            heapq.heappush(active, (end, lane))
            max_lanes[(sm, wg)] = max(max_lanes.get((sm, wg), 0), lane + 1)

            trace_events.append({
                "name": name, "cat": "range", "ph": "X",
                "ts": start, "dur": end - start,
                "pid": sm, "tid": f"wg{wg}.{lane}",
                "args": {"tag": tag, "tag_name": name,
                         "sm": sm, "warpgroup": wg, "lane": lane},
            })

    # ── Perfetto metadata (process / thread names) ───────────────────────
    for sm in range(num_sms):
        trace_events.append({"name": "process_name",       "ph": "M", "pid": sm, "tid": 0,
                             "args": {"name": f"SM {sm}"}})
        trace_events.append({"name": "process_sort_index", "ph": "M", "pid": sm,
                             "args": {"sort_index": sm}})
        for wg in range(NUM_WG):
            for lane in range(max_lanes.get((sm, wg), 1)):
                label = f"Warpgroup {wg}" if lane == 0 else f"Warpgroup {wg} (lane {lane})"
                trace_events.append({"name": "thread_name", "ph": "M", "pid": sm,
                                     "tid": f"wg{wg}.{lane}", "args": {"name": label}})

    with open(out_path, "w") as f:
        json.dump({"traceEvents": trace_events, "displayTimeUnit": "ns"}, f)

    return out_path