import cutlass
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.arch import dsl_user_op
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
def nanosleep(ns: cutlass.Constexpr, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        ir.Type.parse("!llvm.void"),
        [],
        f"nanosleep.u32 {ns};",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
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