import math
from dataclasses import dataclass
from functools import partial
from typing import Callable

import cutlass
from cutlass import BFloat16, Float32, Int32
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, warp

from transformer_megakernel.kernel_utils import (
    ld_acquire_u32,
    atomic_add_release,
    fence_proxy_async_global,
    WarpgroupMeta, PipelineMeta, Phases,
    range_start, range_stop, TAGS
)

@dataclass
class MatmulConfig:
    bM: int
    bN: int
    bK: int
    num_stages: int
    stage_elements: int
    output_pad: int
    use_tma_reduce: bool
    group_m = 8

    def __post_init__(self):
        self.stage_size  = (self.bM + self.bN) * self.bK
        self.swizzle_bits = (int(math.log2(self.bK)) - 3, 4, 3)
        assert self.bM in (32, 64, 128)
        assert self.bN in (32, 64, 128)
        assert self.bK in (32, 64)
        assert self.num_stages == 2, "V2 matmul is two-stage"
        assert self.stage_elements >= self.stage_size
        assert self.output_pad in (0, 8, 16)


class Matmul:
    def __init__(self, config: MatmulConfig, epilogue: Callable, profile: bool):
        self.config   = config
        self.epilogue = epilogue
        self.profile = profile

    @cute.jit
    def _get_tiled_mma(self) -> cute.TiledMma:
        cfg = self.config
        warpM = min(4, cfg.bM // 16)
        warpN = 4 // warpM
        return cute.make_tiled_mma(
            warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)),
            (warpM, warpN, 1),
            permutation_mnk = (self.config.bM, self.config.bN, self.config.bK),
        )

    @cute.jit
    def _gemm(self, stage_idx, *, thr_copy_A, thr_copy_B, sA, sB, tiled_mma, tCrA, tCrB, tCrC):
        tCsA = thr_copy_A.partition_S(sA[None, None, stage_idx])
        tCsB = thr_copy_B.partition_S(sB[None, None, stage_idx])
        tCrA_cpy = thr_copy_A.retile(tCrA)
        tCrB_cpy = thr_copy_B.retile(tCrB)
        cute.copy(thr_copy_A, tCsA, tCrA_cpy)
        cute.copy(thr_copy_B, tCsB, tCrB_cpy)
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
    def _wait_stage(self, stage_idx, load_phase, *, load_bar):
        cute.arch.mbarrier_wait(load_bar + stage_idx, (load_phase >> stage_idx) & 1)
        return load_phase ^ (1 << stage_idx)

    @cute.jit
    def run(
        self,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gC_tma: cute.Tensor,
        gC: cute.Tensor,
        tma_A: cute.CopyAtom,
        tma_B: cute.CopyAtom,
        tma_C: cute.CopyAtom,
        layer_idx: Int32,
        pid_m: Int32,
        pid_n: Int32,
        mAtomics: cute.Tensor,
        pipeline: PipelineMeta,
        phases: Phases,
        warpgroup: WarpgroupMeta,
        storage,
        mStart_probe: cute.Tensor,
        mStop_probe: cute.Tensor,
        s_cnt, st_cnt
    ):
        cfg = self.config
        k_tiles = gA.shape[1] // cfg.bK
        tiled_mma = self._get_tiled_mma()
        swizzle = cute.make_swizzle(*cfg.swizzle_bits)

        sA_layout = cute.make_layout(
            shape = (cfg.bM, cfg.bK, cfg.num_stages),
            stride = (cfg.bK, 1, cfg.stage_elements)
        )
        sB_layout = cute.make_layout(
            shape = (cfg.bN, cfg.bK, cfg.num_stages),
            stride = (cfg.bK, 1, cfg.stage_elements)
        )
        sC_layout = cute.make_layout(
            shape = (cfg.bM, cfg.bN),
            stride = (cfg.bN + cfg.output_pad, 1)
        )
        sC_tma_layout = cute.make_layout(
            shape = (cfg.bM, cfg.bN + cfg.output_pad),
            stride = (cfg.bN + cfg.output_pad, 1)
        )

        stages_ptr = storage.stages.data_ptr()

        sA = cute.make_tensor(cute.recast_ptr(stages_ptr, swizzle), sA_layout)
        sB = cute.make_tensor(cute.recast_ptr(stages_ptr + cfg.bM * cfg.bK, swizzle), sB_layout)

        sC = storage.out.get_tensor(sC_layout)
        sC_tma = storage.out.get_tensor(sC_tma_layout)

        gA_tile = cute.local_tile(gA, (cfg.bM, cfg.bK),    (pid_m, None))
        gB_tile = cute.local_tile(gB, (1, cfg.bN, cfg.bK), (layer_idx, pid_n, None))

        sA_g = cute.group_modes(sA, 0, 2)
        sB_g = cute.group_modes(sB, 0, 2)
        gA_g = cute.group_modes(gA_tile, 0, 2)
        gB_g = cute.group_modes(gB_tile, 0, 3)

        tAsA, tAgA = cpasync.tma_partition(tma_A, 0, cute.make_layout(1), sA_g, gA_g)
        tBsB, tBgB = cpasync.tma_partition(tma_B, 0, cute.make_layout(1), sB_g, gB_g)

        thr_mma = tiled_mma.get_slice(warpgroup.group_tidx)
        tCrA = thr_mma.make_fragment_A(thr_mma.partition_A(sA[None, None, 0]))
        tCrB = thr_mma.make_fragment_B(thr_mma.partition_B(sB[None, None, 0]))

        acc_shape = thr_mma.partition_shape_C((cfg.bM, cfg.bN))
        tCrC = cute.make_rmem_tensor(acc_shape, Float32)
        tCrC.fill(0.0)

        ldmatrix = warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4)
        ca = cute.make_copy_atom(ldmatrix, BFloat16)
        cb = cute.make_copy_atom(ldmatrix, BFloat16)
        thr_copy_A = cute.make_tiled_copy_A(ca, tiled_mma).get_slice(warpgroup.group_tidx)
        thr_copy_B = cute.make_tiled_copy_B(cb, tiled_mma).get_slice(warpgroup.group_tidx)

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

        gemm = partial(
            self._gemm,
            thr_copy_A = thr_copy_A, thr_copy_B = thr_copy_B,
            sA = sA, sB = sB, tiled_mma = tiled_mma,
            tCrA = tCrA, tCrB = tCrB, tCrC = tCrC,
        )
        load_A = partial(
            self._load_A,
            tma_A = tma_A,
            tAgA = tAgA, tAsA = tAsA,
            load_bar = load_bar,
            warp_id = warpgroup.warp_id
        )
        load_B = partial(
            self._load_B,
            tma_B = tma_B,
            tBgB = tBgB, tBsB = tBsB,
            load_bar = load_bar,
            warp_id = warpgroup.warp_id
        )
        expect_tx = partial(
            self._expect_tx,
            load_bar = load_bar,
            warp_id = warpgroup.warp_id
        )
        wait_stage = partial(
            self._wait_stage,
            load_bar = load_bar
        )

        cute.arch.mbarrier_wait(input_bar_me, phases.input_phase)
        stage      = stage_cell[0]
        load_phase = phase_cell[0]

        if cutlass.const_expr(self.profile):
            if warpgroup.group_tidx == 0:
                range_start(mStart_probe, s_cnt, cute.arch.block_idx()[0], TAGS["LOAD_B"], warpgroup.group_id)
                s_cnt += 1
        # ---- prologue: prefetch a SINGLE stage ----
        expect_tx(stage)
        load_B(stage, 0)
        if warpgroup.warp_id == 0:
            with cute.arch.elect_one():
                ready = 0
                while ready != pipeline.expected_cnt:
                    ready = ld_acquire_u32((mAtomics.iterator + pipeline.current_idx).toint())
            cute.arch.sync_warp()
        
        if cutlass.const_expr(self.profile):
            if warpgroup.group_tidx == 0:
                range_start(mStart_probe, s_cnt, cute.arch.block_idx()[0], TAGS["LOAD_A"], warpgroup.group_id)
                s_cnt += 1

        load_A(stage, 0)
        prefetch_stage = (stage + 1) % cfg.num_stages
        cute.arch.mbarrier_wait(compute_bar_me, phases.compute_phase)

        if cutlass.const_expr(self.profile):
            if warpgroup.group_tidx == 0:
                range_stop(mStop_probe, st_cnt, cute.arch.block_idx()[0], TAGS["LOAD_A"], warpgroup.group_id)
                st_cnt += 1
                range_stop(mStop_probe, st_cnt, cute.arch.block_idx()[0], TAGS["LOAD_B"], warpgroup.group_id)
                st_cnt += 1

        # ---- mainloop: k_tiles-1 iterations, each prefetches the next tile ----
        for k_tile in cutlass.range(0, k_tiles - 1):
            expect_tx(prefetch_stage)
            if cutlass.const_expr(self.profile):
                if warpgroup.group_tidx == 0:
                    range_start(mStart_probe, s_cnt, cute.arch.block_idx()[0], TAGS["LOAD_A"], warpgroup.group_id)
                    s_cnt += 1
                    range_start(mStart_probe, s_cnt, cute.arch.block_idx()[0], TAGS["LOAD_B"], warpgroup.group_id)
                    s_cnt += 1

            load_A(prefetch_stage, k_tile + 1)
            load_B(prefetch_stage, k_tile + 1)
            load_phase = wait_stage(stage, load_phase)

            if cutlass.const_expr(self.profile):
                if warpgroup.group_tidx == 0:
                    range_stop(mStop_probe, st_cnt, cute.arch.block_idx()[0], TAGS["LOAD_A"], warpgroup.group_id)
                    st_cnt += 1
                    range_stop(mStop_probe, st_cnt, cute.arch.block_idx()[0], TAGS["LOAD_B"], warpgroup.group_id)
                    st_cnt += 1
                    range_start(mStart_probe, s_cnt, cute.arch.block_idx()[0], TAGS["COMPUTE_AB"], warpgroup.group_id)
                    s_cnt += 1

            gemm(stage)

            if cutlass.const_expr(self.profile):
                if warpgroup.group_tidx == 0:
                    range_stop(mStop_probe, st_cnt, cute.arch.block_idx()[0], TAGS["COMPUTE_AB"], warpgroup.group_id)
                    st_cnt += 1

            stage          = (stage + 1) % cfg.num_stages
            prefetch_stage = (prefetch_stage + 1) % cfg.num_stages

        # ---- stage handoff (pre-toggle phase for the final gemm's stage) ----
        if warpgroup.warp_id == 1:
            with cute.arch.elect_one():
                stage_cell[0] = (stage + 1) % cfg.num_stages
                phase_cell[0] = load_phase ^ (1 << stage)

        # ---- single trailing gemm ----
        cute.arch.mbarrier_arrive(input_bar_ot)
        load_phase = wait_stage(stage, load_phase)

        if cutlass.const_expr(self.profile):
            if warpgroup.group_tidx == 0:
                range_start(mStart_probe, s_cnt, cute.arch.block_idx()[0], TAGS["COMPUTE_AB"], warpgroup.group_id)
                s_cnt += 1
        gemm(stage)

        if cutlass.const_expr(self.profile):
            if warpgroup.group_tidx == 0:
                range_stop(mStop_probe, st_cnt, cute.arch.block_idx()[0], TAGS["COMPUTE_AB"], warpgroup.group_id)
                st_cnt += 1

        # ---- epilogue into the output section, after the output barrier ----
        cute.arch.mbarrier_wait(output_bar_me, phases.output_phase)
        cute.arch.mbarrier_arrive(compute_bar_ot)

        if cutlass.const_expr(self.profile):
            if warpgroup.group_tidx == 0:
                range_start(mStart_probe, s_cnt, cute.arch.block_idx()[0], TAGS["STORE_C"], warpgroup.group_id)
                s_cnt += 1

        self.epilogue(
            tiled_mma = tiled_mma,
            tCrC = tCrC, sC = sC,
            warpgroup = warpgroup,
            gC = gC, gC_tma = gC_tma,
            pid_m = pid_m, pid_n = pid_n,
            bM = cfg.bM, bN = cfg.bN,
            use_tma_reduce = cfg.use_tma_reduce,
        )

        if warpgroup.warp_id == 0:
            gC_tma_tile = cute.local_tile(gC_tma, (cfg.bM, 1, cfg.bN + cfg.output_pad), (pid_m, pid_n, 0))
            sC_g     = cute.group_modes(sC_tma,      0, cute.rank(sC_tma.layout))
            gC_tma_g = cute.group_modes(gC_tma_tile, 0, cute.rank(gC_tma_tile.layout))
            sC_part, gC_part = cpasync.tma_partition(tma_C, 0, cute.make_layout(1), sC_g, gC_tma_g)
            cute.copy(tma_C, sC_part, gC_part)
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0)
            fence_proxy_async_global()
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                atomic_add_release((mAtomics.iterator + pipeline.next_idx).toint(), 1)
        cute.arch.mbarrier_arrive(output_bar_ot)

        if cutlass.const_expr(self.profile):
            if warpgroup.group_tidx == 0:
                range_stop(mStop_probe, st_cnt, cute.arch.block_idx()[0], TAGS["STORE_C"], warpgroup.group_id)
                st_cnt += 1
        return s_cnt, st_cnt