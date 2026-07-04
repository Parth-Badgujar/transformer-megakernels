
import math
import logging
from functools import partial

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, warp
from dataclasses import dataclass
from cutlass import Float32, BFloat16

from transformer_megakernel.kernel_utils import (
    ld_acquire_u32, atomic_add_release,
    fence_proxy_async_shared_cta, fence_proxy_async_global,
    WarpgroupMeta, PipelineMeta, Phases,
    range_start, range_stop, TAGS
)


@dataclass
class AttentionConfig:
    bQ: int
    bKV: int
    head_dim: int
    num_q_heads: int
    num_kv_heads: int
    q_len: int
    kv_len: int
    output_pad: int
    num_stages: int
    stage_elements: int
    is_causal: bool

    def __post_init__(self):
        assert self.bQ == 64
        assert self.bKV == 64
        assert self.head_dim in (64, 128)
        assert self.num_stages == 2, "V2 attention is two-stage"
        assert self.stage_elements >= 2 * self.bKV * self.head_dim


logger = logging.getLogger(__name__)

class Attention:
    def __init__(self, config: AttentionConfig, profile: bool = False):
        logger.info(f"Initializing Attention operator with config: bQ={config.bQ}, bKV={config.bKV}, head_dim={config.head_dim}, q_heads={config.num_q_heads}, kv_heads={config.num_kv_heads}")
        self.config = config
        self.profile = profile

    @cute.jit
    def get_tiled_mma(self):
        cfg = self.config
        warpM = cfg.bQ // 16
        warpN = 4 // warpM
        tiled_mma_qk = cute.make_tiled_mma(
            warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)),
            (warpM, warpN, 1), permutation_mnk = (cfg.bQ, cfg.bKV, cfg.head_dim),
        )
        tiled_mma_pv = cute.make_tiled_mma(
            warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)),
            (warpM, 1, warpN), permutation_mnk = (cfg.bQ, cfg.head_dim, cfg.bKV),
        )
        return tiled_mma_qk, tiled_mma_pv

    @cute.jit
    def get_tiled_copy_cpasync(self) -> cute.TiledCopy:
        atom_async = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode = cute.nvgpu.LoadCacheMode.GLOBAL),
            cutlass.BFloat16,
            num_bits_per_copy = 128
        )
        async_elems     = 128 // 16
        cols_per_pass   = self.config.head_dim // async_elems
        rows_per_pass   = 128 // cols_per_pass
        tKV_layout      = cute.make_ordered_layout((rows_per_pass, cols_per_pass), order=(1, 0))
        vKV_layout      = cute.make_layout((1, async_elems))
        gmem_tiled_copy = cute.make_tiled_copy_tv(atom_async, tKV_layout, vKV_layout)
        return gmem_tiled_copy

    @cute.jit
    def get_smem_rmem_copy_atom(self) -> tuple[cute.TiledCopy, cute.TiledCopy]:
        ldmatrix   = warp.LdMatrix8x8x16bOp(transpose = False, num_matrices = 4)
        ldmatrix_t = warp.LdMatrix8x8x16bOp(transpose = True,  num_matrices = 4)
        smem_atom_QK = cute.make_copy_atom(ldmatrix,   BFloat16)
        smem_atom_V  = cute.make_copy_atom(ldmatrix_t, BFloat16)
        return smem_atom_QK, smem_atom_V

    @staticmethod
    @cute.jit
    def _reshape_acc_to_mn(acc):
        col_major = cute.make_layout(acc.layout.shape)
        mn_layout = cute.make_layout(
            shape = (
                (col_major.shape[0][1], col_major.shape[1]),
                (col_major.shape[0][0], col_major.shape[2])
            ),
            stride=(
                (col_major.stride[0][1], col_major.stride[1]),
                (col_major.stride[0][0], col_major.stride[2])
            ),
        )
        return cute.make_tensor(acc.iterator, cute.composition(acc.layout, mn_layout))

    @staticmethod
    @cute.jit
    def _reshape_rP_to_mma_A(rP):
        divided = cute.logical_divide(rP.layout, (None, None, 2))
        mma_view = cute.make_layout(
            shape=(
                (divided.shape[0], divided.shape[2][0]),
                divided.shape[1], divided.shape[2][1]
            ),
            stride=(
                (divided.stride[0], divided.stride[2][0]),
                divided.stride[1], divided.stride[2][1]
            ),
        )
        return cute.make_tensor(rP.iterator, mma_view)

    @cute.jit
    def _warpgroup_sync(self, *, group_id):
        cute.arch.barrier(barrier_id = 12 + group_id, number_of_threads = 128)

    @cute.jit
    def load_Q(self, thread_Q_gmem, thread_Q_shared, gmem_tiled_copy):
        cute.copy(gmem_tiled_copy, thread_Q_gmem, thread_Q_shared)
        cute.arch.cp_async_commit_group()

    @cute.jit
    def load_K(self, tile_kv, *, thread_K_gmem, thread_K_shared, gmem_tiled_copy):
        cute.copy(gmem_tiled_copy, thread_K_gmem[None, None, None, tile_kv], thread_K_shared)
        cute.arch.cp_async_commit_group()

    @cute.jit
    def load_V(self, tile_kv, *, thread_V_gmem, thread_V_shared, gmem_tiled_copy):
        cute.copy(gmem_tiled_copy, thread_V_gmem[None, None, None, tile_kv], thread_V_shared)
        cute.arch.cp_async_commit_group()

    @cute.jit
    def gemm_QK(self, *, acc_S, frag_Q, frag_K, thr_cpy_K, sK, tiled_mma_qk, warpgroup_sync):
        """S = Q @ K^T.  Stages K (smem -> rmem via ldmatrix) then runs the QK MMA into acc_S."""
        warpgroup_sync()
        cute.copy(thr_cpy_K, thr_cpy_K.partition_S(sK), thr_cpy_K.retile(frag_K))
        cute.gemm(tiled_mma_qk, acc_S, frag_Q, frag_K, acc_S)
        warpgroup_sync()

    @cute.jit
    def gemm_PV(self, *, acc_O, frag_P, frag_V, thr_cpy_V, sVt, tiled_mma_pv, warpgroup_sync):
        """O += P @ V.  Stages V (smem -> rmem via ldmatrix.trans) then runs the PV MMA into acc_O."""
        warpgroup_sync()
        cute.copy(thr_cpy_V, thr_cpy_V.partition_S(sVt), thr_cpy_V.retile(frag_V))
        cute.gemm(tiled_mma_pv, acc_O, frag_P, frag_V, acc_O)
        warpgroup_sync()

    @cute.jit
    def row_reduce_softmax(self, *, acc_S, acc_O, frag_P, row_max, row_sum,
                           num_rows_per_thr, softmax_scale_log2,
                           cS_mn=None, num_cols_per_thr=0, is_causal=False):
        """One online-softmax step over this KV block: rescale the running output by the
        new max, exponentiate the scores into probabilities, and accumulate the row sums.
        Leaves the bf16 probabilities in frag_P, ready for the PV MMA."""
        scores_mn = Attention._reshape_acc_to_mn(acc_S)
        output_mn = Attention._reshape_acc_to_mn(acc_O)

        if cutlass.const_expr(is_causal):
            for i in cutlass.range_constexpr(num_rows_per_thr):
                row = cS_mn[i, 0][0]
                for j in cutlass.range_constexpr(num_cols_per_thr):
                    col = cS_mn[i, j][1]
                    scores_mn[i, j] = -Float32.inf if col > row else scores_mn[i, j]

        for r in cutlass.range_constexpr(num_rows_per_thr):
            scores      = scores_mn[r, None].load()
            prev_max    = row_max[r]
            block_max   = scores.reduce(cute.ReductionOp.MAX, prev_max, 0)
            block_max   = cute.arch.warp_reduction_max(block_max, threads_in_group=4)
            row_max[r]  = block_max
            safe_max    = block_max if block_max != -Float32.inf else 0.0

            probs       = cute.math.exp2(
                (scores - safe_max) * softmax_scale_log2, fastmath = True)
            rescale     = cute.math.exp2(
                (prev_max - safe_max) * softmax_scale_log2, fastmath = True)

            output_mn[r, None].store(output_mn[r, None].load() * rescale)
            row_sum[r]  = probs.reduce(cute.ReductionOp.ADD, row_sum[r] * rescale, 0)
            scores_mn[r, None].store(probs)

        frag_P.store(acc_S.load().to(BFloat16))

    @cute.jit
    def normalize_output(self, *, acc_O, row_sum, num_rows_per_thr):
        """Divide each output row by its softmax denominator (the accumulated row sum)."""
        output_mn = Attention._reshape_acc_to_mn(acc_O)
        for r in cutlass.range_constexpr(num_rows_per_thr):
            denom     = cute.arch.warp_reduction_sum(row_sum[r], threads_in_group=4)
            inv_denom = cute.arch.rcp_approx(denom if denom != 0.0 else 1.0)
            output_mn[r, None].store(output_mn[r, None].load() * inv_denom)

    @cute.jit
    def run(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        gOut: cute.Tensor,
        tma_out: cute.CopyAtom,
        batch_idx: int,
        q_head_idx: int,
        query_block_idx: int,
        mAtomics: cute.Tensor,
        pipeline: PipelineMeta,
        phases: Phases,
        warpgroup: WarpgroupMeta,
        storage,
        mStart_probe: cute.Tensor,
        mStop_probe: cute.Tensor,
        s_cnt, st_cnt
    ):
        LOG2_E = math.log2(math.e)
        cfg = self.config
        group_id   = warpgroup.group_id
        group_tid  = warpgroup.group_tidx
        kv_head_idx = q_head_idx // (cfg.num_q_heads // cfg.num_kv_heads)

        warpgroup_sync = partial(self._warpgroup_sync, group_id=group_id)

        layout_atom  = cute.make_composed_layout(
            cute.make_swizzle(3, 3, 3),
            0,
            cute.make_layout((8, 64), stride = (64, 1)),
        )

        sKV_layout = cute.tile_to_shape(layout_atom, (cfg.bKV, cfg.head_dim), (0, 1))
        sQ_layout  = cute.tile_to_shape(layout_atom, (cfg.bQ,  cfg.head_dim), (0, 1))

        sO_layout     = cute.make_layout(
            shape = (cfg.bQ, cfg.head_dim),
            stride = (cfg.head_dim + cfg.output_pad, 1)
        )
        sO_tma_layout = cute.make_layout(
            shape = (cfg.bQ, cfg.head_dim + cfg.output_pad),
            stride = (cfg.head_dim + cfg.output_pad, 1)
        )

        gQ  = cute.local_tile(mQ[batch_idx, q_head_idx,  None, None], (cfg.bQ,  cfg.head_dim), (query_block_idx, 0))
        gK  = cute.local_tile(mK[batch_idx, kv_head_idx, None, None], (cfg.bKV, cfg.head_dim), (None, 0))
        gV  = cute.local_tile(mV[batch_idx, kv_head_idx, None, None], (cfg.bKV, cfg.head_dim), (None, 0))

        other = group_id ^ 1
        input_bar_me   = storage.barriers.input_barrier.data_ptr()   + group_id
        input_bar_ot   = storage.barriers.input_barrier.data_ptr()   + other
        compute_bar_me = storage.barriers.compute_barrier.data_ptr() + group_id
        compute_bar_ot = storage.barriers.compute_barrier.data_ptr() + other
        output_bar_me  = storage.barriers.output_barrier.data_ptr()  + group_id
        output_bar_ot  = storage.barriers.output_barrier.data_ptr()  + other
        num_kv_blocks = cfg.kv_len // cfg.bKV

        cute.arch.mbarrier_wait(input_bar_me, phases.input_phase)

        if group_tid == 0:
            ready = 0
            while ready != pipeline.expected_cnt:
                ready = ld_acquire_u32((mAtomics.iterator + pipeline.current_idx).toint())

        load_stage = storage.barriers.stage.get_tensor(cute.make_layout((1,)))
        c = load_stage[0]
        warpgroup_sync()
        if warpgroup.group_tidx == 0:
            load_stage[0] = c ^ 1
        stages_ptr = storage.stages.data_ptr()

        sK  = cute.make_tensor((stages_ptr + c * cfg.stage_elements).align(128), sKV_layout)
        sV  = cute.make_tensor((stages_ptr + c * cfg.stage_elements + cfg.bKV * cfg.head_dim).align(128), sKV_layout)
        sQ  = cute.make_tensor((stages_ptr + c * cfg.stage_elements + cfg.bKV * cfg.head_dim).align(128), sQ_layout)
        sVt = cute.composition(sV, cute.make_layout((cfg.head_dim, cfg.bKV), stride=(cfg.bKV, 1)))
        sO     = storage.out.get_tensor(sO_layout)
        sO_tma = storage.out.get_tensor(sO_tma_layout)

        # MMA Setup
        tiled_mma_qk, tiled_mma_pv = self.get_tiled_mma()
        thr_mma_qk: cute.ThrMma = tiled_mma_qk.get_slice(group_tid)
        thr_mma_pv = tiled_mma_pv.get_slice(group_tid)
        rmem_tensor_Q_S = thr_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sQ))
        rmem_tensor_K_S = thr_mma_qk.make_fragment_B(thr_mma_qk.partition_B(sK))
        rmem_tensor_V_O = thr_mma_pv.make_fragment_B(thr_mma_pv.partition_B(sVt))
        acc_O = thr_mma_pv.make_fragment_C(thr_mma_pv.partition_C(sO))
        acc_S = cute.make_rmem_tensor(thr_mma_qk.partition_shape_C((cfg.bQ,cfg.bKV)), Float32)

        acc_O.fill(0.0)
        acc_S.fill(0.0)

        m_atom_rows      = acc_S.shape[0][1]
        m_outer_acc      = acc_S.shape[1]
        num_rows_per_thr = m_atom_rows * m_outer_acc
        num_cols_per_thr = acc_S.shape[0][0] * acc_S.shape[2]
        softmax_scale      = 1.0 / math.sqrt(cfg.head_dim)
        softmax_scale_log2 = softmax_scale * LOG2_E
        row_max = cute.make_rmem_tensor(cute.make_layout((num_rows_per_thr,)), Float32)
        row_sum = cute.make_rmem_tensor(cute.make_layout((num_rows_per_thr,)), Float32)
        row_max.fill(-Float32.inf)
        row_sum.fill(0.0)
        rmem_tesor_P = cute.make_fragment_like(acc_S, BFloat16)
        rmem_tesor_P_O = Attention._reshape_rP_to_mma_A(rmem_tesor_P)   # MMA-A view of P (refilled each step)

        #GMEM -> SMEM Copy
        gmem_tiled_copy = self.get_tiled_copy_cpasync()
        gmem_thr_copy   = gmem_tiled_copy.get_slice(group_tid)

        thread_Q_global = gmem_thr_copy.partition_S(gQ)
        thread_Q_shared = gmem_thr_copy.partition_D(sQ)

        thread_K_global = gmem_thr_copy.partition_S(gK)
        thread_K_shared = gmem_thr_copy.partition_D(sK)

        thread_V_global = gmem_thr_copy.partition_S(gV)
        thread_V_shared = gmem_thr_copy.partition_D(sV)

        load_Q = partial(self.load_Q,
            gmem_tiled_copy = gmem_thr_copy,
            thread_Q_gmem   = thread_Q_global,
            thread_Q_shared = thread_Q_shared
        )
        load_K = partial(self.load_K,
            gmem_tiled_copy = gmem_thr_copy,
            thread_K_gmem   = thread_K_global,
            thread_K_shared = thread_K_shared
        )
        load_V = partial(self.load_V,
            gmem_tiled_copy = gmem_thr_copy,
            thread_V_gmem   = thread_V_global,
            thread_V_shared = thread_V_shared
        )

        #SMEM -> RMEM Copy
        smem_atom_QK, smem_atom_V = self.get_smem_rmem_copy_atom()
        thr_cpy_Q_V   = cute.make_tiled_copy_A(smem_atom_QK, tiled_mma_qk).get_slice(group_tid)
        thr_cpy_K_V   = cute.make_tiled_copy_B(smem_atom_QK, tiled_mma_qk).get_slice(group_tid)
        thr_cpy_V_P   = cute.make_tiled_copy_B(smem_atom_V,  tiled_mma_pv).get_slice(group_tid)

        # ---- prefetch K_0, Q_0 into stage c ----
        # Load Q and K_0 concurrently (both cp_async, overlap is real)
        if cutlass.const_expr(self.profile):
            if warpgroup.group_tidx == 0:
                range_start(mStart_probe, s_cnt, cute.arch.block_idx()[0], TAGS["LOAD_Q"], warpgroup.group_id)
                s_cnt += 1
                range_start(mStart_probe, s_cnt, cute.arch.block_idx()[0], TAGS["LOAD_K"], warpgroup.group_id)
                s_cnt += 1

        load_Q()
        load_K(0)

        # Q is shared across every KV block -> stage it smem -> rmem exactly once.
        cute.arch.cp_async_wait_group(1)   # K_0 still in flight while Q has landed

        if cutlass.const_expr(self.profile):
            if warpgroup.group_tidx == 0:
                range_stop(mStop_probe, st_cnt, cute.arch.block_idx()[0], TAGS["LOAD_Q"], warpgroup.group_id)
                st_cnt += 1

        warpgroup_sync()
        cute.copy(thr_cpy_Q_V, thr_cpy_Q_V.partition_S(sQ), thr_cpy_Q_V.retile(rmem_tensor_Q_S))

        cute.arch.mbarrier_wait(compute_bar_me, phases.compute_phase)

        cS_mn = Attention._reshape_acc_to_mn(
            thr_mma_qk.partition_C(cute.make_identity_tensor((cfg.bQ, cfg.bKV))))

        # Bind the accumulators / staging buffers so the mainloop reads like the math.
        gemm_QK = partial(self.gemm_QK,
            acc_S=acc_S, frag_Q=rmem_tensor_Q_S, frag_K=rmem_tensor_K_S,
            thr_cpy_K=thr_cpy_K_V, sK=sK, tiled_mma_qk=tiled_mma_qk, warpgroup_sync=warpgroup_sync)
        gemm_PV = partial(self.gemm_PV,
            acc_O=acc_O, frag_P=rmem_tesor_P_O, frag_V=rmem_tensor_V_O,
            thr_cpy_V=thr_cpy_V_P, sVt=sVt, tiled_mma_pv=tiled_mma_pv, warpgroup_sync=warpgroup_sync)
        row_reduce_softmax = partial(self.row_reduce_softmax,
            acc_S=acc_S, acc_O=acc_O, frag_P=rmem_tesor_P, row_max=row_max, row_sum=row_sum,
            num_rows_per_thr=num_rows_per_thr, softmax_scale_log2=softmax_scale_log2,
            cS_mn=cS_mn, num_cols_per_thr=num_cols_per_thr)
        normalize_output = partial(self.normalize_output,
            acc_O=acc_O, row_sum=row_sum, num_rows_per_thr=num_rows_per_thr)

        # ---- mainloop: KV blocks 0 .. N-2, each prefetching the next K ----
        if cutlass.const_expr(cfg.is_causal):
            n_kv = cutlass.min(query_block_idx + 1, num_kv_blocks)
        else:
            n_kv = num_kv_blocks

        last = n_kv - 1
        for n in cutlass.range(last):
            acc_S.fill(0.0)

            # K_n is already in smem; start loading V_n while QK compute runs
            if cutlass.const_expr(self.profile):
                if warpgroup.group_tidx == 0:
                    range_start(mStart_probe, s_cnt, cute.arch.block_idx()[0], TAGS["LOAD_V"], warpgroup.group_id)
                    s_cnt += 1

            load_V(n)

            cute.arch.cp_async_wait_group(1)       # K_n has landed; V_n still in flight
            # Close K load range (opened at end of previous iter, or prologue for n==0)

            if cutlass.const_expr(self.profile):
                if warpgroup.group_tidx == 0:
                    range_stop(mStop_probe, st_cnt, cute.arch.block_idx()[0], TAGS["LOAD_K"], warpgroup.group_id)
                    st_cnt += 1
                    range_start(mStart_probe, s_cnt, cute.arch.block_idx()[0], TAGS["COMPUTE_QK"], warpgroup.group_id)
                    s_cnt += 1

            gemm_QK()

            # Prefetch next K while PV is computed

            if cutlass.const_expr(self.profile):
                if warpgroup.group_tidx == 0:
                    range_stop(mStop_probe, st_cnt, cute.arch.block_idx()[0], TAGS["COMPUTE_QK"], warpgroup.group_id)
                    st_cnt += 1
                    range_start(mStart_probe, s_cnt, cute.arch.block_idx()[0], TAGS["LOAD_K"], warpgroup.group_id)
                    s_cnt += 1

            load_K(n + 1)      # prefetch next K

            if cutlass.const_expr(self.profile):
                if warpgroup.group_tidx == 0:
                    range_start(mStart_probe, s_cnt, cute.arch.block_idx()[0], TAGS["ROW_REDUCE_SOFTMAX"], warpgroup.group_id)
                    s_cnt += 1

            row_reduce_softmax()

            if cutlass.const_expr(self.profile):
                if warpgroup.group_tidx == 0:
                    range_stop(mStop_probe, st_cnt, cute.arch.block_idx()[0], TAGS["ROW_REDUCE_SOFTMAX"], warpgroup.group_id)
                    st_cnt += 1

            cute.arch.cp_async_wait_group(1)        # V_n has landed

            if cutlass.const_expr(self.profile):
                if warpgroup.group_tidx == 0:
                    range_stop(mStop_probe, st_cnt, cute.arch.block_idx()[0], TAGS["LOAD_V"], warpgroup.group_id)
                    st_cnt += 1
                    range_start(mStart_probe, s_cnt, cute.arch.block_idx()[0], TAGS["COMPUTE_PV"], warpgroup.group_id)
                    s_cnt += 1
            #                     cute.arch.block_idx()[0], TAGS["COMPUTE_PV"])
            gemm_PV()

            if cutlass.const_expr(self.profile):
                if warpgroup.group_tidx == 0:
                    range_stop(mStop_probe, st_cnt, cute.arch.block_idx()[0], TAGS["COMPUTE_PV"], warpgroup.group_id)
                    st_cnt += 1

        # ---- tail: last KV block signals the input barrier instead of prefetching ----
        if cutlass.const_expr(self.profile):
            if warpgroup.group_tidx == 0:
                range_start(mStart_probe, s_cnt, cute.arch.block_idx()[0], TAGS["LOAD_V"], warpgroup.group_id)
                s_cnt += 1

        acc_S.fill(0.0)
        load_V(last)
        cute.arch.cp_async_wait_group(1)

        if cutlass.const_expr(self.profile):
            if warpgroup.group_tidx == 0:
                range_stop(mStop_probe, st_cnt, cute.arch.block_idx()[0], TAGS["LOAD_K"], warpgroup.group_id)
                st_cnt += 1
                range_start(mStart_probe, s_cnt, cute.arch.block_idx()[0], TAGS["COMPUTE_QK"], warpgroup.group_id)
                s_cnt += 1

        gemm_QK()
        cute.arch.mbarrier_arrive(input_bar_ot)

        if cutlass.const_expr(self.profile):
            if warpgroup.group_tidx == 0:
                range_stop(mStop_probe, st_cnt, cute.arch.block_idx()[0], TAGS["COMPUTE_QK"], warpgroup.group_id)
                st_cnt += 1
                range_start(mStart_probe, s_cnt, cute.arch.block_idx()[0], TAGS["ROW_REDUCE_SOFTMAX"], warpgroup.group_id)
                s_cnt += 1

        row_reduce_softmax(is_causal=cfg.is_causal)

        if cutlass.const_expr(self.profile):
            if warpgroup.group_tidx == 0:
                range_stop(mStop_probe, st_cnt, cute.arch.block_idx()[0], TAGS["ROW_REDUCE_SOFTMAX"], warpgroup.group_id)
                st_cnt += 1

        cute.arch.cp_async_wait_group(0)

        if cutlass.const_expr(self.profile):
            if warpgroup.group_tidx == 0:
                range_stop(mStop_probe, st_cnt, cute.arch.block_idx()[0], TAGS["LOAD_V"], warpgroup.group_id)
                st_cnt += 1
                range_start(mStart_probe, s_cnt, cute.arch.block_idx()[0], TAGS["COMPUTE_PV"], warpgroup.group_id)
                s_cnt += 1

        gemm_PV()

        if cutlass.const_expr(self.profile):
            if warpgroup.group_tidx == 0:
                range_stop(mStop_probe, st_cnt, cute.arch.block_idx()[0], TAGS["COMPUTE_PV"], warpgroup.group_id)
                st_cnt += 1

        # ---- finalize ----
        cute.arch.mbarrier_arrive(compute_bar_ot)
        normalize_output()

        # ---- write O into the output section, AFTER the output barrier ----
        rO_bf = cute.make_fragment_like(acc_O, BFloat16)
        cute.arch.mbarrier_wait(output_bar_me, phases.output_phase)

        if cutlass.const_expr(self.profile):
            if warpgroup.group_tidx == 0:
                range_start(mStart_probe, s_cnt, cute.arch.block_idx()[0], TAGS["ATTENTION_STORE"], warpgroup.group_id)
                s_cnt += 1

        rO_bf.store(acc_O.load().to(BFloat16))

        smem_store_bf  = cute.make_copy_atom(
            warp.StMatrix8x8x16bOp(num_matrices=4), BFloat16
        )
        smem_thr_Ow = cute.make_tiled_copy_C(smem_store_bf, tiled_mma_pv).get_slice(group_tid)
        smem_O_acc  = smem_thr_Ow.partition_D(sO)
        reg_O_acc   = smem_thr_Ow.retile(rO_bf)
        cute.copy(smem_store_bf, reg_O_acc, smem_O_acc)
        fence_proxy_async_shared_cta()
        warpgroup_sync()

        if warpgroup.warp_id == 0:
            gO_tma = cute.local_tile(gOut, (1, 1, cfg.bQ,  cfg.head_dim + cfg.output_pad), (batch_idx, q_head_idx, query_block_idx, 0))
            sO_g = cute.group_modes(sO_tma, 0, cute.rank(sO_tma.layout))
            gO_g = cute.group_modes(gO_tma, 0, cute.rank(gO_tma.layout))
            sO_part, gO_part = cpasync.tma_partition(tma_out, 0, cute.make_layout(1), sO_g, gO_g)
            cute.copy(tma_out, sO_part, gO_part)
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0)
            fence_proxy_async_global()
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                atomic_add_release((mAtomics.iterator + pipeline.next_idx).toint(), 1)
        cute.arch.mbarrier_arrive(output_bar_ot)

        if cutlass.const_expr(self.profile):
            if warpgroup.group_tidx == 0:
                range_stop(mStop_probe, st_cnt, cute.arch.block_idx()[0], TAGS["ATTENTION_STORE"], warpgroup.group_id)
                st_cnt += 1

        return s_cnt, st_cnt