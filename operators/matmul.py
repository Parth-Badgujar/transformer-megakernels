"""
Matmul op (v2) — config + warpgroup driven.

One warpgroup (128 threads) per call, TMA-async pipelined mainloop over K:

    0. spin on atomic[atomic_idx] until == expected_cnt
    1. wait own input pp-barrier
    2. prologue: TMA-load stage 0/1 of A,B
    3. mainloop: prefetch next stage, wait current, ldmatrix + cute.gemm
    4. drain tail stages
    5. epilogue (stages acc into sC), then warp0 TMA-stores sC->gC
    6. bump next_idx

The epilogue is a plain callable; CuTe inlines it at compile time, so the same
mainloop dispatches to basic_store / residual_add_store / silu_mul. The epilogue
only stages into sC — the store and the atomic bump live here.
"""

import math
from dataclasses import dataclass
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
    def __init__(
        self,
        config: MatmulConfig,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gC: cute.Tensor,
        tma_atom_A: cute.CopyAtom,
        tma_atom_B: cute.CopyAtom,
        epilogue: Callable,
        tma_atom_C: cute.CopyAtom,
    ):
        self.config     = config
        self.gA         = gA
        self.gB         = gB
        self.gC         = gC
        self.tma_atom_A = tma_atom_A
        self.tma_atom_B = tma_atom_B
        self.tma_atom_C = tma_atom_C
        self.epilogue   = epilogue
        self.K          = gA.shape[1]

    @cute.jit
    def _get_tiled_mma(self) -> cute.TiledMma:
        return cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.config.element_type, cutlass.Float32, (16, 8, 16)),
            (2, 2, 1),
            permutation_mnk=(self.config.bM, self.config.bN, self.config.bK),
        )

    @cute.jit
    def run(
        self,
        layer_idx, pid_m, pid_n,
        mAtomics, expected_cnt, atomic_idx, next_idx,
        storage,
        warpgroup: WarpgroupMeta,
        compute_mbar_phase, input_mbar_phase, output_mbar_phase,
    ):
        cfg = self.config
        bM, bN, bK, nS = cfg.bM, cfg.bN, cfg.bK, cfg.num_stages
        PAD = cfg.output_pad
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

        gA_tile = cute.local_tile(self.gA, (bM, bK),    (pid_m, None))
        gB_tile = cute.local_tile(self.gB, (1, bN, bK), (layer_idx, pid_n, None))

        sA_g = cute.group_modes(sA, 0, 2)
        sB_g = cute.group_modes(sB, 0, 2)
        gA_g = cute.group_modes(gA_tile, 0, 2)
        gB_g = cute.group_modes(gB_tile, 0, 3)
        tAsA, tAgA = cpasync.tma_partition(self.tma_atom_A, 0, cute.make_layout(1), sA_g, gA_g)
        tBsB, tBgB = cpasync.tma_partition(self.tma_atom_B, 0, cute.make_layout(1), sB_g, gB_g)

        thr_mma = tiled_mma.get_slice(warpgroup.group_tidx)
        tCrA = thr_mma.make_fragment_A(thr_mma.partition_A(sA[None, None, 0]))
        tCrB = thr_mma.make_fragment_B(thr_mma.partition_B(sB[None, None, 0]))
        acc_shape = thr_mma.partition_shape_C((bM, bN))
        tCrC = cute.make_rmem_tensor(acc_shape, cutlass.Float32)
        tCrC.fill(0.0)

        ld_op = warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4)
        ca = cute.make_copy_atom(ld_op, cfg.element_type)
        cb = cute.make_copy_atom(ld_op, cfg.element_type)
        thr_copy_A = cute.make_tiled_copy_A(ca, tiled_mma).get_slice(warpgroup.group_tidx)
        thr_copy_B = cute.make_tiled_copy_B(cb, tiled_mma).get_slice(warpgroup.group_tidx)
        tCrA_cpy = thr_copy_A.retile(tCrA)
        tCrB_cpy = thr_copy_B.retile(tCrB)

        load_bar = storage.barriers.load_barrier.data_ptr()
        other = warpgroup.group_id ^ 1
        input_bar_me   = storage.barriers.input_barrier.data_ptr()   + warpgroup.group_id
        input_bar_ot   = storage.barriers.input_barrier.data_ptr()   + other
        compute_bar_me = storage.barriers.compute_barrier.data_ptr() + warpgroup.group_id
        compute_bar_ot = storage.barriers.compute_barrier.data_ptr() + other
        output_bar_me  = storage.barriers.output_barrier.data_ptr()  + warpgroup.group_id
        output_bar_ot  = storage.barriers.output_barrier.data_ptr()  + other

        stage_cell = storage.barriers.stage.get_tensor(cute.make_layout((1,)))
        phase_cell = storage.barriers.phase.get_tensor(cute.make_layout((1,)))

        @cute.jit
        def gemm(stage_idx):
            tCsA = thr_copy_A.partition_S(sA[None, None, stage_idx])
            tCsB = thr_copy_B.partition_S(sB[None, None, stage_idx])
            cute.copy(thr_copy_A, tCsA, tCrA_cpy)
            cute.copy(thr_copy_B, tCsB, tCrB_cpy)
            cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)

        @cute.jit
        def load_A_async(stage_idx, tile_idx):
            if warpgroup.warp_id == 0:
                cute.copy(self.tma_atom_A, tAgA[None, tile_idx], tAsA[None, stage_idx], tma_bar_ptr=load_bar + stage_idx)

        @cute.jit
        def load_B_async(stage_idx, tile_idx):
            if warpgroup.warp_id == 0:
                cute.copy(self.tma_atom_B, tBgB[None, tile_idx], tBsB[None, stage_idx], tma_bar_ptr=load_bar + stage_idx)

        @cute.jit
        def mbarrier_arrive_expect_tx_stage(stage_idx):
            if warpgroup.warp_id == 0:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(load_bar + stage_idx, cfg.stage_size * 2)

        @cute.jit
        def warpgroup_sync():
            cute.arch.barrier(barrier_id=8 + warpgroup.group_id, number_of_threads=128)

        @cute.jit
        def wait_prev_activations():
            if warpgroup.warp_id == 0:
                with cute.arch.elect_one():
                    ready = cutlass.Int32(0)
                    while ready != expected_cnt:
                        ready = ld_acquire_u32((mAtomics.iterator + atomic_idx).toint())

        k_tiles = self.K // bK
        cute.arch.mbarrier_wait(input_bar_me, input_mbar_phase)
        stage      = stage_cell[0]
        load_phase = phase_cell[0]
        next_stage = (stage + 1) % nS

        @cute.jit
        def mbarrier_wait_stage(stage_idx):
            nonlocal load_phase
            cute.arch.mbarrier_wait(load_bar + stage_idx, (load_phase >> stage_idx) & 1)
            load_phase = load_phase ^ (cutlass.Int32(1) << stage_idx)

        mbarrier_arrive_expect_tx_stage(stage)
        mbarrier_arrive_expect_tx_stage(next_stage)

        load_B_async(stage, 0)
        load_B_async(next_stage, 1)

        wait_prev_activations()
        warpgroup_sync()

        load_A_async(stage, 0)
        load_A_async(next_stage, 1)

        cute.arch.mbarrier_wait(compute_bar_me, compute_mbar_phase)
        prefetch_stage = (stage + 2) % nS

        for k_tile in cutlass.range(0, k_tiles - 2):
            mbarrier_arrive_expect_tx_stage(prefetch_stage)
            load_A_async(prefetch_stage, k_tile + 2)
            load_B_async(prefetch_stage, k_tile + 2)
            mbarrier_wait_stage(stage)
            gemm(stage)
            stage = (stage + 1) % nS
            prefetch_stage = (prefetch_stage + 1) % nS

        mbarrier_wait_stage(stage)
        gemm(stage)
        in_flight = (stage + 1) % nS

        if warpgroup.warp_id == 0:
            with cute.arch.elect_one():
                stage_cell[0] = (stage + nS - 1) % nS
                phase_cell[0] = load_phase ^ (cutlass.Int32(1) << in_flight)

        cute.arch.mbarrier_arrive(input_bar_ot)
        stage = in_flight

        mbarrier_wait_stage(stage)
        gemm(stage)

        cute.arch.mbarrier_wait(output_bar_me, output_mbar_phase)
        self.epilogue(
            thr_mma=thr_mma, tCrC=tCrC, sC=sC, warpgroup=warpgroup,
            gC=self.gC, pid_m=pid_m, pid_n=pid_n,
            bM=bM, bN=bN, use_tma_reduce=cfg.use_tma_reduce,
        )
        cute.arch.mbarrier_arrive(compute_bar_ot)

        if warpgroup.warp_id == 0:
            gC_tma_tile = cute.local_tile(self.gC, (bM, 1, bN + PAD), (pid_m, pid_n, 0))
            sC_g     = cute.group_modes(sC_tma,      0, cute.rank(sC_tma.layout))
            gC_tma_g = cute.group_modes(gC_tma_tile, 0, cute.rank(gC_tma_tile.layout))
            sC_part, gC_part = cpasync.tma_partition(self.tma_atom_C, 0, cute.make_layout(1), sC_g, gC_tma_g)
            cute.copy(self.tma_atom_C, sC_part, gC_part)
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0)
            fence_proxy_async_global()
        cute.arch.mbarrier_arrive(output_bar_ot)

        if warpgroup.group_tidx == 0:
            atomic_add_release((mAtomics.iterator + next_idx).toint(), cutlass.Int32(1))