"""
Epilogue callables passed into Matmul.run().

Each only stages the result into sC as bf16 and syncs the warpgroup — the
matmul body owns the TMA-store to gC and the next_idx bump. thr_mma is the
already-sliced MMA thread partition from the mainloop; barrier id is per
warpgroup (8 + group_id). gC is the single 3D TMA view (M, N/bN, bN); reads
take it with a bN box and squeeze, the store takes it with bN+PAD.

    * basic_store        - acc                      (QKV, UP)
    * residual_add_store - acc + gC                 (OUT, DOWN)
    * silu_mul           - SiLU(acc) * gC            (GATE)
"""

import cutlass
import cutlass.cute as cute

from megakernel.kernel_utils import fence_proxy_async_shared_cta, fence_proxy_async_global


@cute.jit
def basic_store(*, thr_mma, tCrC, sC, warpgroup, **_):
    tCsC = thr_mma.partition_C(sC)
    tCrD = cute.make_fragment_like(tCrC, cutlass.BFloat16)
    tCrD.store(tCrC.load().to(cutlass.BFloat16))
    cute.autovec_copy(tCrD, tCsC)
    fence_proxy_async_shared_cta()
    cute.arch.barrier(barrier_id = 8 + warpgroup.group_id, number_of_threads = 128)



@cute.jit
def residual_add_store(*, thr_mma, tCrC, sC, warpgroup,
                       gC, gC_tma=None, pid_m, pid_n, bM, bN, use_tma_reduce, **_):
    tCsC = thr_mma.partition_C(sC)
    fence_proxy_async_global()
    if cutlass.const_expr(use_tma_reduce):
        tCrD = cute.make_fragment_like(tCrC, cutlass.BFloat16)
        tCrD.store(tCrC.load().to(cutlass.BFloat16))
        cute.autovec_copy(tCrD, tCsC)
    else:
        gC_tile = cute.local_tile(gC, (bM, bN), (pid_m, pid_n))
        tCgC    = thr_mma.partition_C(gC_tile)
        rR = cute.make_fragment_like(tCrC, cutlass.BFloat16)
        cute.autovec_copy(tCgC, rR)
        rR.store((tCrC.load() + rR.load().to(cutlass.Float32)).to(cutlass.BFloat16))
        cute.autovec_copy(rR, tCsC)
    fence_proxy_async_shared_cta()
    cute.arch.barrier(barrier_id = 8 + warpgroup.group_id, number_of_threads = 128)



@cute.jit
def silu_mul(*, thr_mma, tCrC, sC, warpgroup,
             gC, gC_tma = None, pid_m, pid_n, bM, bN, **_):
    tCsC    = thr_mma.partition_C(sC)
    gC_tile = cute.local_tile(gC, (bM, bN), (pid_m, pid_n))
    tCgC    = thr_mma.partition_C(gC_tile)
    fence_proxy_async_global()
    gate = tCrC.load()
    silu = gate  / (cutlass.Float32(1.0) + cute.math.exp(-gate, fastmath=True))
    rUp = cute.make_fragment_like(tCrC, cutlass.BFloat16)
    cute.autovec_copy(tCgC, rUp)
    rUp.store((silu * rUp.load().to(cutlass.Float32)).to(cutlass.BFloat16))
    cute.autovec_copy(rUp, tCsC)
    fence_proxy_async_shared_cta()
    cute.arch.barrier(barrier_id = 8 + warpgroup.group_id, number_of_threads = 128)
