import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
from cutlass import BFloat16, Float32
from dataclasses import dataclass

from megakernel.kernel_utils import (
    ld_acquire_u32, atomic_add_release, nanosleep,
    fence_proxy_async_shared_cta, fence_proxy_async_global, WarpgroupMeta, Phases, PipelineMeta
)


# =============================================================================
# V2 NOTES (read me)
# -----------------------------------------------------------------------------
# * bR is GONE.  We parameterise by `warps_per_row`: how many of the 4 warps in
#   a warpgroup cooperate on ONE row.  num_sets = 4 // warps_per_row rows are
#   processed concurrently by the warpgroup.
#       warps_per_row = 1  ->  num_sets = 4  (V1-equivalent: 1 warp / row)
#       warps_per_row = 2  ->  num_sets = 2
#       warps_per_row = 4  ->  num_sets = 1  (whole WG on one row, long E)
#
# * TV LAYOUT (verified, see chat): we DO NOT use make_tiled_copy_tv here.
#   make_tiled_copy_tv runs raked_product(thr,val) then a right-inverse, which
#   for the draft layout interleaved the 8-element value mode against the thread
#   mode and produced an overlapping, strided (non-contiguous) map.  Instead we
#   hand-build the (thr,val)->coord layout so that:
#       - each thread owns 8 *contiguous* bf16 (one 128-bit LDS/STS),
#       - lane L owns cols [L*8 .. L*8+7], lane L+1 the next 8, etc (no overlap),
#       - bijective over the (num_sets, E) tile.
#   flat addr = row + num_sets*col ,  col = chunk*(8*L) + lane*8 + elem
#   where L = 32*warps_per_row lanes tile a row.
#       thr = (lane, row) :  stride (8*num_sets, 1)
#       val = (elem, chunk): stride (num_sets, 8*L*num_sets)
#   NOTE: this is conflict-FREE only up to the 2-way pattern inherent to 16B
#   bf16 access without a swizzle; per your call we keep it swizzle-free for now
#   and revisit bank conflicts later.
#
# * STAGE STRIDE: SE is the element-distance between consecutive stages in the
#   shared `stages` region.  It must equal the per-stage section size used by the
#   megakernel (config.stage_elems = (bM+bN)*bK = 16384 bf16 = 32 KiB), NOT a
#   hardcoded 32*1024.  32*1024 elements would be 64 KiB and push stage 1 out of
#   the stages region into the output section.  We take it from the config.
#
# * WEIGHTS: loaded gmem -> rmem DIRECTLY (no smem hop).  We broadcast the
#   weight row across the `num_sets` row-mode with stride 0 and read it with the
#   same `thr` partition so wv[i] lines up with xv[i] elementwise.  The weight
#   row is selected by offsetting the global iterator by rms_w_idx * E.
# =============================================================================


@dataclass
class RMSNormConfig:
    embed_dim: int
    num_stages: int
    rows_per_rms_block: int
    bM: int
    stage_elems: int
    warps_per_row: int = 1

    def __post_init__(self):
        assert self.bM % self.rows_per_rms_block == 0
        self.num_sets = 4 // self.warps_per_row
        assert 4 % self.warps_per_row == 0, "warps_per_row must divide 4"
        assert self.rows_per_rms_block % self.num_sets == 0
        assert self.num_stages >= 2
        assert self.embed_dim % 32 == 0
        n_per_thr = self.embed_dim // (32 * self.warps_per_row)
        assert n_per_thr % 8 == 0, "E/(32*warps_per_row) must be a multiple of 8 (128-bit copies)"


class RMSNorm:
    def __init__(self, config: RMSNormConfig):
        self.config = config

    @cute.jit
    def _make_tv(self):
        cfg = self.config
        L = 32 * cfg.warps_per_row              # lanes tiling one row
        npt = cfg.embed_dim // L                # bf16 per thread
        # (lane,row) x (elem,chunk) -> row + num_sets*col
        return cute.make_layout(
            ((L, cfg.num_sets), (8, npt // 8)),
            stride=((8 * cfg.num_sets, 1), (cfg.num_sets, 8 * L * cfg.num_sets)),
        )

    @cute.jit
    def run(
        self,
        g_inp: cute.Tensor,
        g_out: cute.Tensor,
        mWeight: cute.Tensor,           # global weight tensor (gmem->rmem direct)
        tma_inp: cute.CopyAtom,
        tma_out: cute.CopyAtom,
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
        num_sets   = cfg.num_sets
        wpr        = cfg.warps_per_row
        chunks     = cfg.rows_per_rms_block // num_sets
        n_per_thr  = E // (32 * wpr)
        SE         = cfg.stage_elems       # element stride between stages (32 KiB)

        group_id   = warpgroup.group_id
        local_warp = warpgroup.warp_id
        group_tid  = warpgroup.group_tidx

        sX = cute.make_tensor(
            storage.stages.data_ptr(),
            cute.make_layout((num_sets, E, nS), stride=(E, 1, SE)),
        )
        sO = storage.out.get_tensor(
            cute.make_layout((num_sets, E, nS), stride=(E, 1, num_sets * E))
        )

        load_bar   = storage.barriers.load_barrier.data_ptr()
        load_stage = storage.barriers.stage.get_tensor(cute.make_layout((1,)))
        phase_cell = storage.barriers.phase.get_tensor(cute.make_layout((1,)))

        other = group_id ^ 1
        input_bar_me   = storage.barriers.input_barrier.data_ptr()   + group_id
        input_bar_ot   = storage.barriers.input_barrier.data_ptr()   + other
        compute_bar_me = storage.barriers.compute_barrier.data_ptr() + group_id
        compute_bar_ot = storage.barriers.compute_barrier.data_ptr() + other
        output_bar_me  = storage.barriers.output_barrier.data_ptr()  + group_id
        output_bar_ot  = storage.barriers.output_barrier.data_ptr()  + other

        gE_chunks = cute.local_tile(g_inp, (num_sets, E), (None, 0))
        gO_chunks = cute.local_tile(g_out, (num_sets, E), (None, 0))
        sX_g = cute.group_modes(sX, 0, 2)
        sO_g = cute.group_modes(sO, 0, 2)
        gE_g = cute.group_modes(gE_chunks, 0, 2)
        gO_g = cute.group_modes(gO_chunks, 0, 2)
        tEsX, tEgE = cpasync.tma_partition(tma_inp, 0, cute.make_layout(1), sX_g, gE_g)
        tOsO, tOgO = cpasync.tma_partition(tma_out, 0, cute.make_layout(1), sO_g, gO_g)

        atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=128
        )
        tv_layout = self._make_tv()
        # tiler is (num_sets, E) -- the full per-stage tile
        thr: cute.ThrCopy = cute.make_tiled_copy(atom, tv_layout, (num_sets, E)).get_slice(group_tid)
        off_m = pid_m * chunks

        @cute.jit
        def load_activations_async(stage, idx):
            if local_warp == 0:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(load_bar + stage, num_sets * E * 2)
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
                        ready = ld_acquire_u32((mAtomics.iterator + pipeline.current_idx).toint())  # ty: ignore
            warpgroup_sync()
            fence_proxy_async_global()

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
                cute.arch.cp_async_bulk_wait_group(nS - 2) # change this afterwards

        @cute.jit
        def signal_next_activation():
            if local_warp == 0:
                cute.arch.cp_async_bulk_wait_group(0)
                with cute.arch.elect_one():
                    atomic_add_release((mAtomics.iterator + pipeline.next_idx).toint(), cutlass.Int32(1))  # ty: ignore

        # ---- weights: gmem -> rmem direct, broadcast over the num_sets row-mode ----
        sW_bcast = cute.make_tensor(
            (mWeight.iterator + rms_w_idx * E).align(128),                         # select row
            cute.make_layout((num_sets, E), stride=(0, 1)),
        )
        tWsW = thr.partition_S(sW_bcast)
        rW   = cute.make_fragment_like(tWsW)
        cute.copy(atom, tWsW, rW)
        # cute.autovec_copy(tWsW, rW)
        wv = rW.load().to(Float32)

        @cute.jit
        def compute_and_store(stage, rX, is_last=False):
            xv    = rX.load().to(Float32)
            xv_sq = xv * xv
            ssq_per_thr = xv_sq.reduce(cute.ReductionOp.ADD, init_val=Float32(0.0), reduction_profile=0)
            ssq = cute.arch.warp_reduction_sum(ssq_per_thr)

            if cutlass.const_expr(wpr == 1):
                pass  # one warp owns the whole row; warp_reduction_sum is complete
            else:
                # cross-warp reduction within a row-group via a scratch slot in sO
                x = warpgroup.warp_id // wpr
                y = warpgroup.warp_id %  wpr
                scratch = cute.make_tensor(
                    sO[None, None, stage].iterator,
                    cute.make_layout(shape=(num_sets, wpr)),
                )
                if warpgroup.lane_id == 0:
                    scratch[(x, y)] = ssq
                warpgroup_sync()
                if y == 0:
                    val = scratch[(x, warpgroup.lane_id)] if warpgroup.lane_id < wpr else Float32(0.0)
                    val = cute.arch.warp_reduction_sum(val)
                    if warpgroup.lane_id == 0:
                        scratch[(x, 0)] = val
                warpgroup_sync()
                ssq = scratch[(x, 0)]

            mean_sq     = ssq / Float32(E)
            mean_sq_eps = mean_sq + Float32(1e-6)
            scale       = cute.math.rsqrt(mean_sq_eps, fastmath=True)
            yv          = xv * wv * scale

            tXsO = thr.partition_S(sO[None, None, stage])
            rY   = cute.make_fragment_like(tXsO)
            rY.store(yv.to(BFloat16))
            cute.autovec_copy(rY, tXsO)

        cute.arch.mbarrier_wait(input_bar_me, phases.input_phase)

        stage      = load_stage[0]
        load_phase = phase_cell[0]

        @cute.jit
        def wait_for_load_sync(stage):
            nonlocal load_phase
            cute.arch.mbarrier_wait(load_bar + stage, (load_phase >> stage) & 1)
            load_phase ^= (cutlass.Int32(1) << stage)

        wait_for_prev_activations_sync()

        # ---- prologue: prefetch the first (nS-1) chunks ----
        prev_stage = stage
        for s in cutlass.range_constexpr(nS - 1):  # ty: ignore
            load_activations_async(stage, s)
            stage = (stage + 1) % nS

        cute.arch.mbarrier_wait(compute_bar_me, phases.compute_phase)

        # ---- steady state: chunks-1 iterations, each prefetches the next chunk ----
        for it in cutlass.range_constexpr(0, chunks - 1):  # ty: ignore
            load_activations_async(stage, it + (nS - 1))
            wait_for_load_sync(prev_stage)
            rX = load_regs(prev_stage)
            compute_and_store(prev_stage, rX)
            if it == 0:
                cute.arch.mbarrier_wait(output_bar_me, phases.output_phase)
            store_outputs_async(prev_stage, it)
            stage      = (stage + 1) % nS
            prev_stage = (prev_stage + 1) % nS

        # ---- stage handoff to the next op (identify the empty stage) ----
        # The single trailing compute below waits on `prev_stage`, and that
        # wait_for_load_sync toggles load_phase for `prev_stage` AFTER this write.
        # So pre-toggle the phase bit of `prev_stage` (NOT in_flight) so the value
        # handed to the next op matches load_bar's actual parity. load_stage points
        # the next op at the *other* stage so it doesn't clobber what we still read.
        if local_warp == 1:
            with cute.arch.elect_one():
                load_stage[0] = (prev_stage + nS - 1) % nS
                phase_cell[0] = load_phase ^ (cutlass.Int32(1) << prev_stage)
        cute.arch.mbarrier_arrive(input_bar_ot)
 
        # ---- epilogue: last chunk (already prefetched), single compute, is_last ----
        wait_for_load_sync(prev_stage)
        rX = load_regs(prev_stage)
        cute.arch.mbarrier_arrive(compute_bar_ot)
        compute_and_store(prev_stage, rX, is_last=True)
        store_outputs_async(prev_stage, chunks - 1)
 
        if local_warp == 0:
            cute.arch.cp_async_bulk_wait_group(0)
            fence_proxy_async_global()
        cute.arch.mbarrier_arrive(output_bar_ot)
        signal_next_activation()
 