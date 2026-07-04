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
def globaltimer_u64(*, loc=None, ip=None) -> cutlass.Int64:
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
        SM index        -> pid (process)   one swimlane per SM
        warpgroup index -> base tid        one sub-row per warpgroup within that SM

    A single tid/track can only show one slice at a time (Perfetto enforces
    strict LIFO nesting per track and drops the second of any overlapping
    pair — see slice_drop_overlapping_complete_event). So when two tags
    within the same (sm, wg) genuinely overlap in time, the second one is
    assigned an extra "lane" tid (wg, wg+0.001, wg+0.002, ...) so it draws
    on its own sub-row instead of being dropped. Lanes are reused once
    freed, so most of the time you still just see one row per warpgroup,
    and extra rows only appear during actual concurrent overlap, matching
    the LOAD_A/LOAD_B-stacked-rows look.

    Rows with tag_val == 0 are unused/empty slots and are skipped.

    Start/stop events sharing the same (sm, warpgroup, tag) are paired in
    row order: the 1st start for that tag pairs with the 1st stop for that
    tag, the 2nd with the 2nd, etc. This is correct both for properly
    nested (LIFO) ranges and for non-nested (FIFO) ranges of the same tag.
    """
    SENTINEL_TAG = 0
    NUM_WARPGROUPS = 2

    start_np = start_probe.detach().cpu().numpy()
    stop_np = stop_probe.detach().cpu().numpy()

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

    # First pass: build the flat list of (sm, wg, tag, start_us, end_us)
    # events, grouped by (sm, wg) so we can do lane assignment per
    # warpgroup (lanes must be computed across ALL tags sharing that
    # warpgroup, since it's the warpgroup's tid that's contended, not the
    # tag's).
    events_by_sm_wg = defaultdict(list)  # (sm, wg) -> [event dict, ...]

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
            end_us = e_ts_ns / 1000.0
            dur_us = end_us - ts_us

            if dur_us < 0:
                continue

            events_by_sm_wg[(sm, wg)].append(
                {
                    "start": ts_us,
                    "end": end_us,
                    "tag": tag,
                    "tag_name": tag_name,
                    "sm": sm,
                    "wg": wg,
                }
            )

    trace_events = []

    # Second pass: for each (sm, wg), assign lanes via interval scheduling
    # so overlapping events land on different sub-tids, then emit X events
    # (back to complete events — simpler and correct now that each lane
    # only ever has non-overlapping events on it, satisfying Perfetto's
    # strict-nesting requirement).
    max_lanes_per_sm_wg = {}  # (sm, wg) -> lane count, used for thread_name metadata

    for (sm, wg), evs in events_by_sm_wg.items():
        # Sort by start time; ties broken by end time so shorter events
        # don't unnecessarily hold a lane open.
        evs.sort(key=lambda e: (e["start"], e["end"]))

        free_lanes = []      # heap of available lane indices
        active = []          # heap of (end_time, lane) currently occupied

        for ev in evs:
            # Free lanes whose occupant has ended at or before this
            # event's start.
            while active and active[0][0] <= ev["start"]:
                _, freed_lane = heapq.heappop(active)
                heapq.heappush(free_lanes, freed_lane)

            if free_lanes:
                lane = heapq.heappop(free_lanes)
            else:
                lane = len(active) + len(free_lanes)
                # len(active)+len(free_lanes) undercounts once lanes have
                # been freed and reused; track the true next-lane index
                # explicitly instead.
                lane = max_lanes_per_sm_wg.get((sm, wg), 0)

            heapq.heappush(active, (ev["end"], lane))
            ev["lane"] = lane

            max_lanes_per_sm_wg[(sm, wg)] = max(
                max_lanes_per_sm_wg.get((sm, wg), 0), lane + 1
            )

        for ev in evs:
            trace_events.append(
                {
                    "name": ev["tag_name"],
                    "cat": "range",
                    "ph": "X",
                    "ts": ev["start"],
                    "dur": ev["end"] - ev["start"],
                    "pid": sm,
                    "tid": f"wg{wg}.{ev['lane']}",
                    "args": {
                        "tag": ev["tag"],
                        "tag_name": ev["tag_name"],
                        "sm": sm,
                        "warpgroup": wg,
                        "lane": ev["lane"],
                    },
                }
            )

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
        trace_events.append(
            {
                "name": "process_sort_index",
                "ph": "M",
                "pid": sm,
                "args": {"sort_index": sm},
            }
        )
        for wg in range(NUM_WARPGROUPS):
            n_lanes = max_lanes_per_sm_wg.get((sm, wg), 1)
            for lane in range(n_lanes):
                label = f"Warpgroup {wg}" if lane == 0 else f"Warpgroup {wg} (lane {lane})"
                trace_events.append(
                    {
                        "name": "thread_name",
                        "ph": "M",
                        "pid": sm,
                        "tid": f"wg{wg}.{lane}",
                        "args": {"name": label},
                    }
                )

    trace = {
        "traceEvents": trace_events,
        "displayTimeUnit": "ns",
    }

    with open(out_path, "w") as f:
        json.dump(trace, f)

    return out_path