import json
import torch
import cutlass
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, T
from dataclasses import dataclass

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

def range_start(probe, row, sm_val, tag_val):
    """Record the start of the range identified by *tag_val*.

    Multiple ranges with different tags can be open simultaneously.
    """
    off = PROBE_HEADER + tag_val * PROBE_ENTRY
    probe[row, off + 0] = cutlass.Int64(sm_val)
    probe[row, off + 1] = cutlass.Int64(tag_val)
    probe[row, off + 2] = globaltimer_u64()


def range_stop(probe, row, tag_val):
    """Record the end of the range identified by *tag_val*.

    Stores only the end timestamp (no global memory load).
    Duration is computed on the host side by dump_probe.
    """
    off = PROBE_HEADER + tag_val * PROBE_ENTRY
    probe[row, off + 3] = globaltimer_u64()


def range_finalize(probe, row, tag_bitmask):
    """Write a bitmask of which tag slots contain valid data.

    Pass an integer whose bits correspond to tag values that were used, e.g.:
        range_finalize(probe, row, (1 << TAGS['LOAD_K']) | (1 << TAGS['COMPUTE_QK']))
    or simply pass ~0 / 0xFFFF to mark all 16 slots as potentially valid
    (dump_probe will skip slots with a zero start timestamp).
    """
    probe[row, 0] = cutlass.Int64(tag_bitmask)


def dump_probe(probe: torch.Tensor, num_blocks: int,
               out_path: str = "pipeline_trace.json"):
    """Decode probe data and emit a Perfetto/chrome-tracing JSON trace.

    Each tag slot is valid when its start timestamp is non-zero and the
    corresponding bit is set in the row header bitmask.  Pass tag_bitmask=~0
    from range_finalize to include all tags that have a non-zero timestamp.
    """
    probe_cpu = probe.cpu().contiguous().tolist()
    total_rows = num_blocks * NUM_PROBE_ROLES

    for bid in range(min(num_blocks, 4)):
        for role in range(NUM_PROBE_ROLES):
            row_idx = bid * NUM_PROBE_ROLES + role
            data = probe_cpu[row_idx]
            tag_mask = int(data[0])
            active_tags = [t for t in range(MAX_TAG_SLOTS) if tag_mask & (1 << t)]
            print(f"\n--- Block {bid}, {ROLE_NAMES[role]} warp: "
                  f"{len(active_tags)} active tag(s) ---")
            for tag in active_tags:
                off = PROBE_HEADER + tag * PROBE_ENTRY
                sm_id = int(data[off])
                start, end = int(data[off + 2]), int(data[off + 3])
                dur = end - start if (start > 0 and end > 0) else 0
                print(f"  sm={sm_id} {TAG_NAMES.get(tag, f'tag_{tag}'):20s} "
                      f"start={start} dur={dur} ns")

    events, global_base, sm_seen = [], None, set()
    for row_idx in range(total_rows):
        tag_mask = int(probe_cpu[row_idx][0])
        for tag in range(MAX_TAG_SLOTS):
            if not (tag_mask & (1 << tag)):
                continue
            s = int(probe_cpu[row_idx][PROBE_HEADER + tag * PROBE_ENTRY + 2])
            if s > 0 and (global_base is None or s < global_base):
                global_base = s
    global_base = global_base or 0

    for row_idx in range(total_rows):
        data = probe_cpu[row_idx]
        tag_mask = int(data[0])
        if tag_mask == 0:
            continue
        role = row_idx % NUM_PROBE_ROLES
        for tag in range(MAX_TAG_SLOTS):
            if not (tag_mask & (1 << tag)):
                continue
            off = PROBE_HEADER + tag * PROBE_ENTRY
            sm_id = int(data[off])
            start, end = int(data[off + 2]), int(data[off + 3])
            dur = end - start if (start > 0 and end > 0) else 0
            if start == 0 and end == 0:
                continue
            if (sm_id, role) in sm_seen:
                continue
            events.append(dict(
                name=TAG_NAMES.get(tag, f"tag_{tag}"), ph="X",
                ts=(start - global_base) / 1000.0, dur=dur / 1000.0,
                pid=sm_id, tid=role))
        sm_seen.add((int(data[PROBE_HEADER]), role))

    with open(out_path, "w") as f:
        json.dump({"traceEvents": events}, f)
    num_sms = len({e["pid"] for e in events})
    print(f"\nTrace: {len(events)} events from {num_sms} SMs → {out_path}")
    print("Open with chrome://tracing or https://ui.perfetto.dev")

