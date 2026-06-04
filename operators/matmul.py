"""
Matmul op (v2) — config + warpgroup driven.

One warpgroup (128 threads) per call, TMA-async pipelined mainloop over K.
Pipeline steps are @cute.jit methods (read self.<config>, take dynamic state as
args); run() partial-binds the loop-invariant state and drives a rolled dynamic
loop. The epilogue only stages into sC; the store and next_idx bump live here.
"""

import math
from dataclasses import dataclass
from functools import partial
from typing import Callable

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, warp

from operators.kernel_utils import (
    ld_acquire_u32,
    atomic_add_release,
    fence_proxy_async_global,
    WarpgroupMeta,
)

@dataclass
class MatmulConfig:
    bM: int
    bN: int
    bK: int
    num_stages: int
    stage_skip: int
    output_pad: int
    use_tma_store: bool
    use_tma_reduce: bool
    element_type: type

    def __post_init__(self):
        self.stage_size  = (self.bM + self.bN) * self.bK
        self.swizzle_bits = int(math.log2(self.bK)) - 3
        assert self.element_type in (cutlass.Float16, cutlass.BFloat16)
        assert self.bM in (32, 64)
        assert self.bN in (32, 64, 128)
        assert self.bK in (32, 64)
        assert self.num_stages in (2, 3)
        assert self.stage_skip >= self.stage_size
        assert self.output_pad in (0, 8, 16)
        assert self.swizzle_bits in (2, 3)


class Matmul:
    def __init__(self, config: MatmulConfig, epilogue: Callable):
        self.config   = config
        self.epilogue = epilogue

    @cute.jit
    def _get_tiled_mma(self) -> cute.TiledMma:
        return cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.config.element_type, cutlass.Float32, (16, 8, 16)),
            (2, 2, 1),
            permutation_mnk=(self.config.bM, self.config.bN, self.config.bK),
        )

    # ---- pipeline steps: methods, dynamic state passed as args ----

    @cute.jit
    def _gemm(self, stage_idx, *, thr_copy_A, thr_copy_B, sA, sB, tiled_mma, tCrA, tCrB, tCrC):
        tCsA = thr_copy_A.partition_S(sA[None, None, stage_idx])
        tCsB = thr_copy_B.partition_S(sB[None, None, stage_idx])
        cute.copy(thr_copy_A, tCsA, thr_copy_A.retile(tCrA))
        cute.copy(thr_copy_B, tCsB, thr_copy_B.retile(tCrB))
        cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)

    @cute.jit
    def _load_A(self, stage_idx, tile_idx, *, tma_A, tAgA, tAsA, load_bar, warp_id):
        if warp_id == 0:
            cute.copy(tma_A, tAgA[None, tile_idx], tAsA[None, stage_idx],
                      tma_bar_ptr=load_bar + stage_idx)

    @cute.jit
    def _load_B(self, stage_idx, tile_idx, *, tma_B, tBgB, tBsB, load_bar, warp_id):
        if warp_id == 0:
            cute.copy(tma_B, tBgB[None, tile_idx], tBsB[None, stage_idx],
                      tma_bar_ptr=load_bar + stage_idx)

    @cute.jit
    def _expect_tx(self, stage_idx, *, load_bar, warp_id):
        if warp_id == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(load_bar + stage_idx,
                                                        self.config.stage_size * 2)

    @cute.jit
    def _wg_sync(self, *, group_id):
        cute.arch.barrier(barrier_id=8 + group_id, number_of_threads=128)

    @cute.jit
    def _wait_prev(self, *, expected_cnt, mAtomics, atomic_idx, warp_id):
        if warp_id == 0:
            with cute.arch.elect_one():
                ready = cutlass.Int32(0)
                while ready != expected_cnt:
                    ready = ld_acquire_u32((mAtomics.iterator + atomic_idx).toint())

    @cute.jit
    def _wait_stage(self, stage_idx, load_phase, *, load_bar):
        cute.arch.mbarrier_wait(load_bar + stage_idx, (load_phase >> stage_idx) & 1)
        return load_phase ^ (cutlass.Int32(1) << stage_idx)

    @cute.jit
    def run(
        self,
        gA, gB, gC_tma, gC, tma_atom_A, tma_atom_B, tma_atom_C,
        layer_idx, pid_m, pid_n,
        mAtomics, expected_cnt, atomic_idx, next_idx,
        storage,
        warpgroup: WarpgroupMeta,
        compute_mbar_phase, input_mbar_phase, output_mbar_phase,
    ):
        cfg = self.config
        bM, bN, bK, nS = cfg.bM, cfg.bN, cfg.bK, cfg.num_stages
        PAD = cfg.output_pad
        warp_id   = warpgroup.warp_id
        group_id  = warpgroup.group_id
        group_tid = warpgroup.group_tidx
        tiled_mma = self._get_tiled_mma()
        sw = cute.make_swizzle(cfg.swizzle_bits, 4, 3)

        STAGE_ELEMS = cfg.stage_skip
        sA_layout = cute.make_layout((bM, bK, nS), stride=(bK, 1, STAGE_ELEMS))
        sB_layout = cute.make_layout((bN, bK, nS), stride=(bK, 1, STAGE_ELEMS))
        sC_layout     = cute.make_layout((bM, bN),       stride=(bN + PAD, 1))
        sC_tma_layout = cute.make_layout((bM, bN + PAD), stride=(bN + PAD, 1))

        stages_ptr = storage.stages.data_ptr()
        sA = cute.make_tensor(cute.recast_ptr(stages_ptr,           sw, dtype=cfg.element_type), sA_layout)
        sB = cute.make_tensor(cute.recast_ptr(stages_ptr + bM * bK, sw, dtype=cfg.element_type), sB_layout)
        sC     = storage.out.get_tensor(sC_layout)
        sC_tma = storage.out.get_tensor(sC_tma_layout)

        gA_tile = cute.local_tile(gA, (bM, bK),    (pid_m, None))
        gB_tile = cute.local_tile(gB, (1, bN, bK), (layer_idx, pid_n, None))

        sA_g = cute.group_modes(sA, 0, 2)
        sB_g = cute.group_modes(sB, 0, 2)
        gA_g = cute.group_modes(gA_tile, 0, 2)
        gB_g = cute.group_modes(gB_tile, 0, 3)
        tAsA, tAgA = cpasync.tma_partition(tma_atom_A, 0, cute.make_layout(1), sA_g, gA_g)
        tBsB, tBgB = cpasync.tma_partition(tma_atom_B, 0, cute.make_layout(1), sB_g, gB_g)

        thr_mma = tiled_mma.get_slice(group_tid)
        tCrA = thr_mma.make_fragment_A(thr_mma.partition_A(sA[None, None, 0]))
        tCrB = thr_mma.make_fragment_B(thr_mma.partition_B(sB[None, None, 0]))
        acc_shape = thr_mma.partition_shape_C((bM, bN))
        tCrC = cute.make_rmem_tensor(acc_shape, cutlass.Float32)
        tCrC.fill(0.0)

        ld_op = warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4)
        ca = cute.make_copy_atom(ld_op, cfg.element_type)
        cb = cute.make_copy_atom(ld_op, cfg.element_type)
        thr_copy_A = cute.make_tiled_copy_A(ca, tiled_mma).get_slice(group_tid)
        thr_copy_B = cute.make_tiled_copy_B(cb, tiled_mma).get_slice(group_tid)

        load_bar = storage.barriers.load_barrier.data_ptr()
        other = group_id ^ 1
        input_bar_me   = storage.barriers.input_barrier.data_ptr()   + group_id
        input_bar_ot   = storage.barriers.input_barrier.data_ptr()   + other
        compute_bar_me = storage.barriers.compute_barrier.data_ptr() + group_id
        compute_bar_ot = storage.barriers.compute_barrier.data_ptr() + other
        output_bar_me  = storage.barriers.output_barrier.data_ptr()  + group_id
        output_bar_ot  = storage.barriers.output_barrier.data_ptr()  + other

        stage_cell = storage.barriers.stage.get_tensor(cute.make_layout((1,)))
        phase_cell = storage.barriers.phase.get_tensor(cute.make_layout((1,)))

        # partial wrappers: bind loop-invariant dynamic state once
        gemm       = partial(self._gemm, thr_copy_A=thr_copy_A, thr_copy_B=thr_copy_B,
                             sA=sA, sB=sB, tiled_mma=tiled_mma, tCrA=tCrA, tCrB=tCrB, tCrC=tCrC)
        load_A     = partial(self._load_A, tma_A=tma_atom_A, tAgA=tAgA, tAsA=tAsA, load_bar=load_bar, warp_id=warp_id)
        load_B     = partial(self._load_B, tma_B=tma_atom_B, tBgB=tBgB, tBsB=tBsB, load_bar=load_bar, warp_id=warp_id)
        expect_tx  = partial(self._expect_tx, load_bar=load_bar, warp_id=warp_id)
        wg_sync    = partial(self._wg_sync, group_id=group_id)
        wait_prev  = partial(self._wait_prev, expected_cnt=expected_cnt, mAtomics=mAtomics,
                             atomic_idx=atomic_idx, warp_id=warp_id)
        wait_stage = partial(self._wait_stage, load_bar=load_bar)

        k_tiles = gA.shape[1] // bK
        cute.arch.mbarrier_wait(input_bar_me, input_mbar_phase)
        stage      = stage_cell[0]
        load_phase = phase_cell[0]
        next_stage = (stage + 1) % nS

        expect_tx(stage)
        expect_tx(next_stage)

        load_B(stage, 0)
        load_B(next_stage, 1)

        wait_prev()

        load_A(stage, 0)
        load_A(next_stage, 1)

        cute.arch.mbarrier_wait(compute_bar_me, compute_mbar_phase)
        prefetch_stage = (stage + 2) % nS

        for k_tile in cutlass.range(0, k_tiles - 2, unroll=1):
            expect_tx(prefetch_stage)
            load_A(prefetch_stage, k_tile + 2)
            load_B(prefetch_stage, k_tile + 2)
            load_phase = wait_stage(stage, load_phase)
            gemm(stage)
            stage = (stage + 1) % nS
            prefetch_stage = (prefetch_stage + 1) % nS

        load_phase = wait_stage(stage, load_phase)
        gemm(stage)
        in_flight = (stage + 1) % nS

        if warp_id == 1:
            with cute.arch.elect_one():
                stage_cell[0] = (stage + nS - 1) % nS
                phase_cell[0] = load_phase ^ (cutlass.Int32(1) << in_flight)

        cute.arch.mbarrier_arrive(input_bar_ot)
        stage = in_flight

        load_phase = wait_stage(stage, load_phase)
        gemm(stage)

        cute.arch.mbarrier_wait(output_bar_me, output_mbar_phase)
        self.epilogue(
            thr_mma=thr_mma, tCrC=tCrC, sC=sC, warpgroup=warpgroup,
            gC=gC, gC_tma=gC_tma, pid_m=pid_m, pid_n=pid_n,
            bM=bM, bN=bN, use_tma_reduce=cfg.use_tma_reduce,
        )
        cute.arch.mbarrier_arrive(compute_bar_ot)

        if warp_id == 0:
            gC_tma_tile = cute.local_tile(gC_tma, (bM, 1, bN + PAD), (pid_m, pid_n, 0))
            sC_g     = cute.group_modes(sC_tma,      0, cute.rank(sC_tma.layout))
            gC_tma_g = cute.group_modes(gC_tma_tile, 0, cute.rank(gC_tma_tile.layout))
            sC_part, gC_part = cpasync.tma_partition(tma_atom_C, 0, cute.make_layout(1), sC_g, gC_tma_g)
            cute.copy(tma_atom_C, sC_part, gC_part)
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0)
            fence_proxy_async_global()
        cute.arch.mbarrier_arrive(output_bar_ot)

        if group_tid == 0:
            atomic_add_release((mAtomics.iterator + next_idx).toint(), cutlass.Int32(1))