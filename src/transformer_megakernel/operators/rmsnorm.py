import math
from functools import partial
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
from cutlass import BFloat16, Float32

from transformer_megakernel.kernel_utils import (
    ld_acquire_u32, atomic_add_release,
    fence_proxy_async_shared_cta, fence_proxy_async_global, WarpgroupMeta, Phases, PipelineMeta,
    range_start, range_stop, TAGS
)

def prev_power_of_2(num):
    return 2 ** math.floor(math.log2(num))


@dataclass
class RMSNormConfig:
    embed_dim: int
    bRMS: int
    stage_elements: int
    warps_per_row: int = 1

    def __post_init__(self):
        assert 4 % self.warps_per_row == 0, "Warps per row should divide 4 (max 4 warps per row)"
        assert (self.bRMS % (4 // self.warps_per_row)) == 0, "bM should be divisible by (4 // warps_per_row)"
        assert self.embed_dim % (32 * self.warps_per_row) == 0, "Embed dim should be divisible by (32 * warps_per_row)"


class RMSNorm:
    def __init__(self, config: RMSNormConfig, profile = False):
        self.config = config
        self.profile = profile


    @cute.jit
    def _make_tv_layout(self):
        cfg = self.config
        lanes_per_row = 32 * cfg.warps_per_row
        num_per_thread = cfg.embed_dim // lanes_per_row
        alligment = num_per_thread & (~(num_per_thread - 1))
        num_sets = (4 // cfg.warps_per_row)
        return cute.make_layout(
            shape = ((lanes_per_row, num_sets), (alligment, num_per_thread // alligment)),
            stride = ((num_sets * alligment, 1), (num_sets, alligment * num_per_thread * num_sets))
        )

    @cute.jit
    def _warpgroup_sync(self, *, group_id):
        cute.arch.barrier(barrier_id = 10 + group_id, number_of_threads = 128)

    @cute.jit
    def _load_activations(self, stage_idx, tile_idx, *, warp_id, load_bar,
                          tma_inp, tEgE, tEsX, off_m, num_sets):
        cfg = self.config
        if warp_id == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(load_bar + stage_idx, num_sets * cfg.embed_dim * 2)
            cute.copy(tma_inp, tEgE[None, off_m + tile_idx], tEsX[None, stage_idx],
                      tma_bar_ptr=load_bar + stage_idx)

    @cute.jit
    def _load_regs(self, stage_idx, *, thr, sX):
        tXsX = thr.partition_S(sX[None, None, stage_idx])
        rX   = cute.make_fragment_like(tXsX)
        cute.autovec_copy(tXsX, rX)
        return rX

    @cute.jit
    def _store_outputs(self, stage_idx, tile_idx, *, warpgroup_sync, warp_id,
                       tma_out, tOsO, tOgO, off_m, num_stages):
        fence_proxy_async_shared_cta()
        warpgroup_sync()
        if warp_id == 0:
            cute.copy(tma_out, tOsO[None, stage_idx], tOgO[None, off_m + tile_idx])
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(num_stages - 1)

    @cute.jit
    def _wait_stage(self, stage_idx, load_phase, *, load_bar):
        cute.arch.mbarrier_wait(load_bar + stage_idx, (load_phase >> stage_idx) & 1)
        return load_phase ^ (1 << stage_idx)

    @cute.jit
    def _compute(self, stage_idx, rX, *, wv, warpgroup, sO, thr,
                           num_sets, warpgroup_sync):
        cfg = self.config
        warps_per_row = cfg.warps_per_row

        xv = rX.load().to(Float32)
        xv_sq = xv * xv
        ssq_per_thr = xv_sq.reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0)
        ssq = cute.arch.warp_reduction_sum(ssq_per_thr)

        if cutlass.const_expr(warps_per_row != 1): # inter warp reduction if only a single warp
            x = warpgroup.warp_id // warps_per_row
            y = warpgroup.warp_id %  warps_per_row
            scratch = cute.make_tensor(
                cute.recast_ptr(sO[None, None, stage_idx].iterator, dtype = Float32),
                cute.make_layout(shape=(num_sets, warps_per_row)),
            )
            if warpgroup.lane_id == 0:
                scratch[(x, y)] = ssq
            warpgroup_sync()
            if y == 0:
                val = scratch[(x, warpgroup.lane_id)] if warpgroup.lane_id < warps_per_row else 0.0
                val = cute.arch.warp_reduction_sum(val)
                if warpgroup.lane_id == 0:
                    scratch[(x, 0)] = val
            warpgroup_sync()
            ssq = scratch[(x, 0)]

        mean_sq     = ssq / cfg.embed_dim
        mean_sq_eps = mean_sq + 1e-6
        scale       = cute.math.rsqrt(mean_sq_eps, fastmath = True)
        yv          = xv * wv * scale

        tXsO = thr.partition_S(sO[None, None, stage_idx])
        rY   = cute.make_fragment_like(tXsO)
        rY.store(yv.to(BFloat16))
        cute.autovec_copy(rY, tXsO)

    @cute.jit
    def run(
        self,
        g_inp: cute.Tensor,
        g_out: cute.Tensor,
        mWeight: cute.Tensor,
        tma_inp: cute.CopyAtom,
        tma_out: cute.CopyAtom,
        rms_w_idx: int,
        pid_m: int,
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
        num_stages = 2 ## Hardcoded to 2 as of now
        warps_per_row = cfg.warps_per_row
        num_sets = 4 // warps_per_row
        chunks = cfg.bRMS // num_sets
        group_id   = warpgroup.group_id
        warp_id    = warpgroup.warp_id
        group_tid  = warpgroup.group_tidx

        warpgroup_sync = partial(self._warpgroup_sync, group_id=group_id)

        other = group_id ^ 1
        input_bar_me   = storage.barriers.input_barrier.data_ptr()   + group_id
        input_bar_ot   = storage.barriers.input_barrier.data_ptr()   + other
        compute_bar_me = storage.barriers.compute_barrier.data_ptr() + group_id
        compute_bar_ot = storage.barriers.compute_barrier.data_ptr() + other
        output_bar_me  = storage.barriers.output_barrier.data_ptr()  + group_id
        output_bar_ot  = storage.barriers.output_barrier.data_ptr()  + other

        sX = cute.make_tensor(
            storage.stages.data_ptr(),
            cute.make_layout(
                shape = (num_sets, cfg.embed_dim, num_stages),
                stride = (cfg.embed_dim, 1, cfg.stage_elements)
            )
        )
        sO = storage.out.get_tensor(
            cute.make_ordered_layout((num_sets, cfg.embed_dim, num_stages), (1, 0, 2))
        )

        load_bar   = storage.barriers.load_barrier.data_ptr()
        stage_cell = storage.barriers.stage.get_tensor(cute.make_layout((1,)))
        phase_cell = storage.barriers.phase.get_tensor(cute.make_layout((1,)))

        gE_chunks = cute.local_tile(g_inp, (num_sets, cfg.embed_dim), (None, 0))
        gO_chunks = cute.local_tile(g_out, (num_sets, cfg.embed_dim), (None, 0))

        sX_g = cute.group_modes(sX, 0, 2)
        sO_g = cute.group_modes(sO, 0, 2)
        gE_g = cute.group_modes(gE_chunks, 0, 2)
        gO_g = cute.group_modes(gO_chunks, 0, 2)

        tEsX, tEgE = cpasync.tma_partition(tma_inp, 0, cute.make_layout(1), sX_g, gE_g)
        tOsO, tOgO = cpasync.tma_partition(tma_out, 0, cute.make_layout(1), sO_g, gO_g)

        atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=128
        )

        if cutlass.const_expr(self.profile):
            if warpgroup.group_tidx == 0:
                range_start(mStart_probe, s_cnt, cute.arch.block_idx()[0], TAGS["LOAD_WEIGHTS"], warpgroup.group_id)
                s_cnt += 1

        tv_layout = self._make_tv_layout()
        thr = cute.make_tiled_copy(atom, tv_layout, (num_sets, cfg.embed_dim)).get_slice(group_tid)
        off_m = pid_m * chunks
        # Load weights gmem -> rmem directly (broadcast across num_sets rows, 0 stride)
        sW_bcast = cute.make_tensor(
            (mWeight.iterator + rms_w_idx * cfg.embed_dim).align(128),
            cute.make_layout((num_sets, cfg.embed_dim), stride=(0, 1)),
        )
        tWsW = thr.partition_S(sW_bcast)
        rW   = cute.make_fragment_like(tWsW)
        cute.autovec_copy(tWsW, rW)
        wv = rW.load().to(Float32)

        if cutlass.const_expr(self.profile):
            if warpgroup.group_tidx == 0:
                range_stop(mStop_probe, st_cnt, cute.arch.block_idx()[0], TAGS["LOAD_WEIGHTS"], warpgroup.group_id)
                st_cnt += 1

        # Bind helpers via partial
        load_activations = partial(
            self._load_activations,
            warp_id = warp_id, load_bar = load_bar,
            tma_inp = tma_inp,
            tEgE = tEgE, tEsX = tEsX,
            off_m = off_m,
            num_sets = num_sets
        )
        load_regs = partial(
            self._load_regs,
            thr = thr, sX = sX
        )
        store_outputs = partial(
            self._store_outputs,
            warpgroup_sync = warpgroup_sync,
            warp_id = warp_id, tma_out = tma_out,
            tOsO = tOsO, tOgO = tOgO,
            off_m = off_m, num_stages = num_stages
        )
        wait_stage = partial(
            self._wait_stage, load_bar = load_bar
        )
        compute = partial(self._compute,
            wv = wv, warpgroup = warpgroup,
            sO = sO, thr = thr,
            num_sets = num_sets,
            warpgroup_sync = warpgroup_sync
        )

        cute.arch.mbarrier_wait(input_bar_me, phases.input_phase)

        stage      = stage_cell[0]
        load_phase = phase_cell[0]

        if warpgroup.warp_id == 0:
            with cute.arch.elect_one():
                ready = cutlass.Int32(0)
                while ready != pipeline.expected_cnt:
                    ready = ld_acquire_u32((mAtomics.iterator + pipeline.current_idx).toint())
            cute.arch.sync_warp()

        # ---- prologue: prefetch the first (num_stages-1) chunks ----
        prev_stage = stage
        for s in cutlass.range_constexpr(num_stages - 1):
            if cutlass.const_expr(self.profile):
                if warpgroup.group_tidx == 0:
                    range_start(mStart_probe, s_cnt, cute.arch.block_idx()[0], TAGS["LOAD_ROW"], warpgroup.group_id)
                    s_cnt += 1
            load_activations(stage, s)
            stage = (stage + 1) % num_stages

        cute.arch.mbarrier_wait(compute_bar_me, phases.compute_phase)

        # ---- steady state: chunks-1 iterations, each prefetches the next chunk ----
        for it in cutlass.range_constexpr(0, chunks - 1):
            if cutlass.const_expr(self.profile):
                if warpgroup.group_tidx == 0:
                    range_start(mStart_probe, s_cnt, cute.arch.block_idx()[0], TAGS["LOAD_ROW"], warpgroup.group_id)
                    s_cnt += 1
            load_activations(stage, it + (num_stages - 1))
            load_phase = wait_stage(prev_stage, load_phase)
            if cutlass.const_expr(self.profile):
                if warpgroup.group_tidx == 0:
                    # Activation tile ready; close load range and open compute range
                    range_stop(mStop_probe, st_cnt, cute.arch.block_idx()[0], TAGS["LOAD_ROW"], warpgroup.group_id)
                    st_cnt += 1
                    range_start(mStart_probe, s_cnt, cute.arch.block_idx()[0], TAGS["COMPUTE_ROW"], warpgroup.group_id)
                    s_cnt += 1
            rX = load_regs(prev_stage)
            if it == 0:
                cute.arch.mbarrier_wait(output_bar_me, phases.output_phase)
            compute(prev_stage, rX)
            if cutlass.const_expr(self.profile):
                if warpgroup.group_tidx == 0:
                    range_stop(mStop_probe, st_cnt, cute.arch.block_idx()[0], TAGS["COMPUTE_ROW"], warpgroup.group_id)
                    st_cnt += 1
                    # range_start(mStart_probe, s_cnt, cute.arch.block_idx()[0], TAGS["STORE_ROW"], warpgroup.group_id)
                    # s_cnt += 1
            store_outputs(prev_stage, it)
            # if cutlass.const_expr(self.profile):
            #     if warpgroup.group_tidx == 0:
            #         range_stop(mStop_probe, st_cnt, cute.arch.block_idx()[0], TAGS["STORE_ROW"], warpgroup.group_id)
            #         st_cnt += 1
                    # Overlap: start loading next activation tile while store is in flight
            stage      = (stage + 1) % num_stages
            prev_stage = (prev_stage + 1) % num_stages

        # ---- stage handoff to the next op (identify the empty stage) ----
        if warp_id == 1:
            with cute.arch.elect_one():
                stage_cell[0] = (prev_stage + num_stages - 1) % num_stages
                phase_cell[0] = load_phase ^ (1 << prev_stage)

        cute.arch.mbarrier_arrive(input_bar_ot)
        load_phase = wait_stage(prev_stage, load_phase)
        if cutlass.const_expr(self.profile):
            if warpgroup.group_tidx == 0:
                range_stop(mStop_probe, st_cnt, cute.arch.block_idx()[0], TAGS["LOAD_ROW"], warpgroup.group_id)
                st_cnt += 1

        rX = load_regs(prev_stage)
        cute.arch.mbarrier_arrive(compute_bar_ot)
        if cutlass.const_expr(self.profile):
            if warpgroup.group_tidx == 0:
                range_start(mStart_probe, s_cnt, cute.arch.block_idx()[0], TAGS["COMPUTE_ROW"], warpgroup.group_id)
                s_cnt += 1
        compute(prev_stage, rX)
        if cutlass.const_expr(self.profile):
            if warpgroup.group_tidx == 0:
                range_stop(mStop_probe, st_cnt, cute.arch.block_idx()[0], TAGS["COMPUTE_ROW"], warpgroup.group_id)
                st_cnt += 1
                range_start(mStart_probe, s_cnt, cute.arch.block_idx()[0], TAGS["STORE_ROW"], warpgroup.group_id)
                s_cnt += 1
        store_outputs(prev_stage, chunks - 1)
        if warp_id == 0:
            cute.arch.cp_async_bulk_wait_group(0)
            fence_proxy_async_global()
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                atomic_add_release((mAtomics.iterator + pipeline.next_idx).toint(), cutlass.Int32(1))
        cute.arch.mbarrier_arrive(output_bar_ot)
        if cutlass.const_expr(self.profile):
            if warpgroup.group_tidx == 0:
                range_stop(mStop_probe, st_cnt, cute.arch.block_idx()[0], TAGS["STORE_ROW"], warpgroup.group_id)
                st_cnt += 1
                
        return s_cnt, st_cnt