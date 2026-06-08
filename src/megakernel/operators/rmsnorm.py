import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
from cutlass import BFloat16, Float32
from dataclasses import dataclass

from megakernel.kernel_utils import (
    ld_acquire_u32, atomic_add_release, nanosleep,
    fence_proxy_async_shared_cta, fence_proxy_async_global, WarpgroupMeta, Phases, PipelineMeta
)


@dataclass
class RMSNormConfig:
    embed_dim: int
    num_stages: int
    bR: int
    rows_per_rms_block: int
    bM: int
    stage_elems: int

    def __post_init__(self):
        assert self.bM % self.rows_per_rms_block == 0
        assert self.rows_per_rms_block % self.bR == 0
        assert self.num_stages >= 2
        assert self.embed_dim % 32 == 0
        assert self.bR == 4, "bR must be 4 (one row per warp in a 4-warp WG)"


class RMSNorm:
    def __init__(self, config: RMSNormConfig):
        self.config = config

    @cute.jit
    def run(
        self, 
        g_inp: cute.Tensor, 
        g_out: cute.Tensor,
        g_wt: cute.Tensor,
        tma_inp: cute.CopyAtom,
        tma_out: cute.CopyAtom,
        tma_w: cute.CopyAtom,
        rms_w_idx: int,
        pid_m: int,
        weight_reuse: bool,
        mAtomics: cute.Tensor,
        pipeline: PipelineMeta,
        phases: Phases,
        warpgroup: WarpgroupMeta,
        storage,
    ):

        cfg        = self.config
        E          = cfg.embed_dim
        nS         = cfg.num_stages
        bR         = cfg.bR
        chunks     = cfg.rows_per_rms_block // cfg.bR
        n_per_thr  = E // 32
        SE         = cfg.stage_elems

        group_id   = warpgroup.group_id
        local_warp = warpgroup.warp_id
        group_tid  = warpgroup.group_tidx

        sW = cute.make_tensor(storage.sW.data_ptr(), cute.make_layout((E,)))
        sX = cute.make_tensor(
            storage.stages.data_ptr(),
            cute.make_layout((bR, E, nS), stride=(E, 1, SE)),
        )
        sO = storage.out.get_tensor(
            cute.make_layout((bR, E, nS), stride=(E, 1, bR * E))
        )

        load_bar = storage.barriers.load_barrier.data_ptr()
        load_stage = storage.barriers.stage.get_tensor(cute.make_layout((1,)))
        phase_cell = storage.barriers.phase.get_tensor(cute.make_layout((1,)))

        other = group_id ^ 1
        input_bar_me   = storage.barriers.input_barrier.data_ptr()   + group_id
        input_bar_ot   = storage.barriers.input_barrier.data_ptr()   + other
        compute_bar_me = storage.barriers.compute_barrier.data_ptr() + group_id
        compute_bar_ot = storage.barriers.compute_barrier.data_ptr() + other
        output_bar_me  = storage.barriers.output_barrier.data_ptr()  + group_id
        output_bar_ot  = storage.barriers.output_barrier.data_ptr()  + other

        gE_chunks = cute.local_tile(g_inp, (bR, E), (None, 0))
        gO_chunks = cute.local_tile(g_out, (bR, E), (None, 0))
        sX_g = cute.group_modes(sX, 0, 2)
        sO_g = cute.group_modes(sO, 0, 2)
        gE_g = cute.group_modes(gE_chunks, 0, 2)
        gO_g = cute.group_modes(gO_chunks, 0, 2)
        tEsX, tEgE = cpasync.tma_partition(tma_inp,  0, cute.make_layout(1), sX_g, gE_g)
        tOsO, tOgO = cpasync.tma_partition(tma_out, 0, cute.make_layout(1), sO_g, gO_g)

        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=128)
        thr_layout = cute.make_layout((bR, 32), stride=(32, 1))
        val_layout = cute.make_layout((1, n_per_thr), stride=(0, 1))
        tc  = cute.make_tiled_copy_tv(atom, thr_layout, val_layout)
        thr = tc.get_slice(group_tid)
        off_m = pid_m * chunks

        @cute.jit
        def load_activations_async(stage, idx):
            if local_warp == 0:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(load_bar + stage, bR * E * 2)
                cute.copy(tma_inp, tEgE[None, off_m + idx], tEsX[None, stage],
                          tma_bar_ptr=load_bar + stage)

        @cute.jit
        def warpgroup_sync():
            cute.arch.barrier(barrier_id=10 + group_id, number_of_threads=128)

        @cute.jit
        def wait_for_prev_activations_sync():
            if local_warp == 0:
                with cute.arch.elect_one():
                    ready = cutlass.Int32(0)
                    while ready != pipeline.expected_cnt:
                        ready = ld_acquire_u32((mAtomics.iterator + pipeline.current_idx).toint()) #ty: ignore
            fence_proxy_async_global()
            warpgroup_sync()

        @cute.jit
        def load_regs(stage):
            tXsX = thr.partition_S(sX[None, None, stage])
            rX   = cute.make_fragment_like(tXsX)
            cute.autovec_copy(tXsX, rX)
            return rX

        @cute.jit
        def store_outputs_async(stage, idx):
            fence_proxy_async_shared_cta()
            warpgroup_sync()
            if local_warp == 0:
                cute.copy(tma_out, tOsO[None, stage], tOgO[None, off_m + idx])
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(nS - 2)

        @cute.jit
        def signal_next_activation():
            if local_warp == 0:
                cute.arch.cp_async_bulk_wait_group(0)
                with cute.arch.elect_one():
                    atomic_add_release((mAtomics.iterator + pipeline.next_idx).toint(), cutlass.Int32(1)) #ty: ignore

        @cute.jit
        def compute_and_store(stage, rX, is_last=False):
            xv = rX.load().to(Float32)
            xv_sq = xv * xv
            ssq_per_thr = xv_sq.reduce(cute.ReductionOp.ADD, init_val = Float32(0.0), reduction_profile=0)
            ssq = cute.arch.warp_reduction_sum(ssq_per_thr)
            mean_sq = ssq / Float32(E)
            mean_sq_eps = mean_sq + Float32(1e-6)
            scale = cute.math.rsqrt(mean_sq_eps, fastmath = True)
            yv = xv * wv * scale
            tXsO = thr.partition_S(sO[None, None, stage])
            rY = cute.make_fragment_like(tXsO)
            rY.store(yv.to(BFloat16))
            if is_last:
                cute.arch.mbarrier_arrive(compute_bar_ot)
            cute.autovec_copy(rY, tXsO)

        cute.arch.mbarrier_wait(input_bar_me, phases.input_phase)

        stage      = load_stage[0]
        load_phase = phase_cell[0]

        @cute.jit
        def wait_for_load_sync(stage):
            nonlocal load_phase
            cute.arch.mbarrier_wait(load_bar + stage, (load_phase >> stage) & 1)
            load_phase ^= (cutlass.Int32(1) << stage)

        if weight_reuse == 0 and local_warp == 0:
            gW_tile = cute.local_tile(g_wt, (1, E), (rms_w_idx, 0))
            sW_g = cute.group_modes(sW, 0, cute.rank(sW.layout))
            gW_g = cute.group_modes(gW_tile, 0, cute.rank(gW_tile.layout))
            sW_part, gW_part = cpasync.tma_partition(tma_w, 0, cute.make_layout(1), sW_g, gW_g)
            with cute.arch.elect_one():
                cute.arch.mbarrier_expect_tx(load_bar + stage, E * 2)
            cute.copy(tma_w, gW_part, sW_part, tma_bar_ptr=load_bar + stage)

        wait_for_prev_activations_sync()
        fence_proxy_async_global()
        warpgroup_sync()

        prev_stage = stage
        for s in cutlass.range_constexpr(nS - 1): #ty: ignore
            load_activations_async(stage, s)
            stage = (stage + 1) % nS

        wait_for_load_sync(prev_stage)

        sW_bcast = cute.make_tensor(sW.iterator, cute.make_layout((bR, E), stride=(0, 1)))
        tWsW = thr.partition_S(sW_bcast)
        rW   = cute.make_fragment_like(tWsW)
        cute.autovec_copy(tWsW, rW)
        wv = rW.load().to(Float32)

        cute.arch.mbarrier_wait(compute_bar_me, phases.compute_phase)
        load_activations_async(stage, nS - 1)
        stage = (stage + 1) % nS
        rX = load_regs(prev_stage)
        cute.arch.mbarrier_wait(output_bar_me, phases.output_phase)
        compute_and_store(prev_stage, rX)
        store_outputs_async(prev_stage, 0)
        prev_stage = (prev_stage + 1) % nS

        for it in cutlass.range_constexpr(1, chunks - (nS - 1)): #ty: ignore
            load_activations_async(stage, it + nS - 1)
            wait_for_load_sync(prev_stage)
            rX = load_regs(prev_stage)
            compute_and_store(prev_stage, rX)
            store_outputs_async(prev_stage, it)
            stage = (stage + 1) % nS
            prev_stage = (prev_stage + 1) % nS

        wait_for_load_sync(prev_stage)
        rX = load_regs(prev_stage)
        in_flight = (prev_stage + 1) % nS
        if local_warp == 1:
            with cute.arch.elect_one():
                load_stage[0] = (prev_stage + nS - 1) % nS
                phase_cell[0] = load_phase ^ (1 << in_flight)
        cute.arch.mbarrier_arrive(input_bar_ot)
        compute_and_store(prev_stage, rX)
        store_outputs_async(prev_stage, chunks - 2)
        prev_stage = in_flight
        wait_for_load_sync(prev_stage)
        rX = load_regs(prev_stage)
        compute_and_store(prev_stage, rX, is_last=True)
        store_outputs_async(prev_stage, chunks - 1)
        if local_warp == 0:
            cute.arch.cp_async_bulk_wait_group(0)
            fence_proxy_async_global()
        cute.arch.mbarrier_arrive(output_bar_ot)
        signal_next_activation()