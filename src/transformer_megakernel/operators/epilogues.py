import cutlass
import cutlass.cute as cute
from transformer_megakernel.kernel_utils import fence_proxy_async_shared_cta


@cute.jit
def basic_store(*, tiled_mma, tCrC, sC, warpgroup, **_):
    tCrD = cute.make_fragment_like(tCrC, cutlass.BFloat16)
    tCrD.store(tCrC.load().to(cutlass.BFloat16))

    store_atom = cute.make_copy_atom(
        cute.nvgpu.warp.StMatrix8x8x16bOp(num_matrices=4), cutlass.BFloat16
    )
    tiled_copy = cute.make_tiled_copy_C(store_atom, tiled_mma)
    thr_cpy    = tiled_copy.get_slice(warpgroup.group_tidx)

    src = thr_cpy.retile(tCrD)
    dst = thr_cpy.partition_D(sC)
    cute.copy(store_atom, src, dst)
    fence_proxy_async_shared_cta()
    cute.arch.barrier(barrier_id=8 + warpgroup.group_id, number_of_threads=128)


@cute.jit
def residual_add_store(*, tiled_mma, tCrC, sC, warpgroup,
                       gC, gC_tma = None, pid_m, pid_n, bM, bN, use_tma_reduce, **_):
    thr_mma = tiled_mma.get_slice(warpgroup.group_tidx)

    store_atom = cute.make_copy_atom(
        cute.nvgpu.warp.StMatrix8x8x16bOp(num_matrices=4), cutlass.BFloat16
    )
    thr_cpy    = cute.make_tiled_copy_C(store_atom, tiled_mma).get_slice(warpgroup.group_tidx)
    dst        = thr_cpy.partition_D(sC)

    if cutlass.const_expr(use_tma_reduce):
        tCrD = cute.make_fragment_like(tCrC, cutlass.BFloat16)
        tCrD.store(tCrC.load().to(cutlass.BFloat16))
        cute.copy(store_atom, thr_cpy.retile(tCrD), dst)
    else:
        gC_tile = cute.local_tile(gC, (bM, bN), (pid_m, pid_n))
        tCgC    = thr_mma.partition_C(gC_tile)
        rR = cute.make_fragment_like(tCrC, cutlass.BFloat16)
        cute.autovec_copy(tCgC, rR)
        rR.store((tCrC.load() + rR.load().to(cutlass.Float32)).to(cutlass.BFloat16))
        cute.copy(store_atom, thr_cpy.retile(rR), dst)
    fence_proxy_async_shared_cta()
    cute.arch.barrier(barrier_id=8 + warpgroup.group_id, number_of_threads=128)


@cute.jit
def silu_mul(*, tiled_mma, tCrC, sC, warpgroup, gC, gC_tma=None, pid_m, pid_n, bM, bN, **_):
    tid     = warpgroup.group_tidx
    thr_mma = tiled_mma.get_slice(tid)
    tCgC    = thr_mma.partition_C(cute.local_tile(gC, (bM, bN), (pid_m, pid_n)))

    store_atom = cute.make_copy_atom(
        cute.nvgpu.warp.StMatrix8x8x16bOp(num_matrices=4), cutlass.BFloat16
    )
    thr_cpy    = cute.make_tiled_copy_C(store_atom, tiled_mma).get_slice(tid)

    rUp = cute.make_fragment_like(tCrC, cutlass.BFloat16)
    cute.autovec_copy(tCgC, rUp)

    n_tiles = tCrC.shape[2]
    for n in cutlass.range_constexpr(n_tiles):
        g    = tCrC[None, None, n].load()
        u    = rUp[None, None, n].load().to(cutlass.Float32)
        silu = g / (cutlass.Float32(1.0) + cute.math.exp(-g, fastmath=True))
        rUp[None, None, n].store((silu * u).to(cutlass.BFloat16))

    cute.copy(store_atom, thr_cpy.retile(rUp), thr_cpy.partition_D(sC))
    fence_proxy_async_shared_cta()
    cute.arch.barrier(barrier_id=8 + warpgroup.group_id, number_of_threads=128)