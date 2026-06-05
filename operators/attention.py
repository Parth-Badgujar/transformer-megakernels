import math
from functools import partial

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, warp
from dataclasses import dataclass
from cutlass import Float32, BFloat16

from operators.kernel_utils import (
    LOG2_E, ld_acquire_u32, atomic_add_release,
    fence_proxy_async_shared_cta, fence_proxy_async_global, WarpgroupMeta, fence_proxy_async
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
    attn_warps_m: int
    output_pad: int
    num_stages: int
    stage_skip: int

    def __post_init__(self):
        assert self.bQ == 64
        assert self.bKV == 64
        assert self.head_dim in (64, 128)


class Attention:
    def __init__(self, config: AttentionConfig):
        self.config = config

    @staticmethod
    @cute.jit
    def _reshape_acc_to_mn(acc):
        col_major = cute.make_layout(acc.layout.shape)
        mn_layout = cute.make_layout(
            ((col_major.shape[0][1], col_major.shape[1]),
             (col_major.shape[0][0], col_major.shape[2])),
            stride=((col_major.stride[0][1], col_major.stride[1]),
                    (col_major.stride[0][0], col_major.stride[2])),
        )
        mn_layout = cute.composition(acc.layout, mn_layout)
        return cute.make_tensor(acc.iterator, mn_layout)

    @staticmethod
    @cute.jit
    def _reshape_rP_to_mma_A(rP):
        divided = cute.logical_divide(rP.layout, (None, None, 2))
        mma_view = cute.make_layout(
            ((divided.shape[0], divided.shape[2][0]), divided.shape[1], divided.shape[2][1]),
            stride=((divided.stride[0], divided.stride[2][0]), divided.stride[1], divided.stride[2][1]),
        )
        return cute.make_tensor(rP.iterator, mma_view)

    @cute.jit
    def _warpgroup_sync(self, *, group_id):
        cute.arch.barrier(barrier_id = 12 + group_id, number_of_threads = 128)

    @cute.jit
    def run(self, mQ, mK, mV, tma_o, gO,
            b, h, qb,
            mAtomics, expected_cnt, atomic_idx, next_idx, storage,
            warpgroup: WarpgroupMeta, compute_mbar_phase, input_mbar_phase, output_bar_phase):

        bQ       = self.config.bQ
        bKV      = self.config.bKV
        d        = self.config.head_dim
        H_q      = self.config.num_q_heads
        H_kv     = self.config.num_kv_heads
        nWarps_m = 4
        kv_len   = self.config.kv_len
        group_id  = warpgroup.group_id
        group_tid = warpgroup.group_tidx
        h_kv = h * H_kv // H_q

        PAD = self.config.output_pad
        stage_skip = self.config.stage_skip
        warpgroup_sync = partial(self._warpgroup_sync, group_id = group_id)

        other = group_id ^ 1
        self.input_bar_me   = storage.barriers.input_barrier.data_ptr()   + group_id
        self.input_bar_ot   = storage.barriers.input_barrier.data_ptr()   + other
        self.compute_bar_me = storage.barriers.compute_barrier.data_ptr() + group_id
        self.compute_bar_ot = storage.barriers.compute_barrier.data_ptr() + other
        self.output_bar_me = storage.barriers.output_barrier.data_ptr()   + group_id
        self.output_bar_ot = storage.barriers.output_barrier.data_ptr()   + other
        cute.arch.mbarrier_wait(self.input_bar_me, input_mbar_phase)

        if group_tid == 0:
            ready = cutlass.Int32(0)
            while ready != expected_cnt:
                ready = ld_acquire_u32((mAtomics.iterator + atomic_idx).toint())
        warpgroup_sync()

        smem_k_block = 64 if d % 64 == 0 else 32
        swizzle_bits = 3 if smem_k_block == 64 else 2
        layout_atom  = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3),
            0,
            cute.make_layout((8, smem_k_block), stride=(smem_k_block, 1)),
        )
        sQ_layout  = cute.tile_to_shape(layout_atom, (bQ,  d), (0, 1))
        sKV_layout = cute.tile_to_shape(layout_atom, (bKV, d), (0, 1))


        sO_layout     = cute.make_layout((bQ, d),       stride=(d + PAD, 1))
        sO_tma_layout = cute.make_layout((bQ, d + PAD), stride=(d + PAD, 1))

        nS = self.config.num_stages
        load_stage = storage.barriers.stage.get_tensor(cute.make_layout((1,)))
        c = load_stage[0]
        stages_ptr = storage.stages.data_ptr()
        sQ     = cute.make_tensor((stages_ptr + c               * stage_skip).align(128), sQ_layout)
        sK     = cute.make_tensor((stages_ptr + ((c + 1) % nS)  * stage_skip).align(128), sKV_layout)
        sV     = cute.make_tensor((stages_ptr + c               * stage_skip).align(128), sKV_layout)
        sO     = storage.out.get_tensor(sO_layout)
        sO_tma = storage.out.get_tensor(sO_tma_layout)

        sVt = cute.composition(
            sV, cute.make_layout((d, bKV), stride=(bKV, 1)),
        )

        tiled_mma_qk = cute.make_tiled_mma(
            warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)),
            (nWarps_m, 1, 1),
        )
        tiled_mma_pv = cute.make_tiled_mma(
            warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)),
            (nWarps_m, 1, 1),
        )

        atom_async = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            BFloat16, num_bits_per_copy=128,
        )
        async_elems     = 128 // 16
        cols_per_pass   = d // async_elems
        rows_per_pass   = 128 // cols_per_pass
        tQ_layout       = cute.make_ordered_layout(
            (rows_per_pass, cols_per_pass), order=(1, 0),
        )
        vQ_layout       = cute.make_layout((1, async_elems))
        gmem_tiled_copy = cute.make_tiled_copy_tv(atom_async, tQ_layout, vQ_layout)
        gmem_thr_copy   = gmem_tiled_copy.get_slice(group_tid)

        ld_op_n      = warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4)
        ld_op_t      = warp.LdMatrix8x8x16bOp(transpose=True,  num_matrices=4)
        smem_atom_QK = cute.make_copy_atom(ld_op_n, BFloat16)
        smem_atom_V  = cute.make_copy_atom(ld_op_t, BFloat16)
        smem_thr_Q   = cute.make_tiled_copy_A(smem_atom_QK, tiled_mma_qk).get_slice(group_tid)
        smem_thr_K   = cute.make_tiled_copy_B(smem_atom_QK, tiled_mma_qk).get_slice(group_tid)
        smem_thr_V   = cute.make_tiled_copy_B(smem_atom_V,  tiled_mma_pv).get_slice(group_tid)

        gQ = cute.local_tile(mQ[b, h,    None, None], (bQ,  d), (qb,   0))
        gK = cute.local_tile(mK[b, h_kv, None, None], (bKV, d), (None, 0))
        gV = cute.local_tile(mV[b, h_kv, None, None], (bKV, d), (None, 0))

        tQgQ = gmem_thr_copy.partition_S(gQ)
        tQsQ = gmem_thr_copy.partition_D(sQ)
        tKgK = gmem_thr_copy.partition_S(gK)
        tKsK = gmem_thr_copy.partition_D(sK)
        tVgV = gmem_thr_copy.partition_S(gV)
        tVsV = gmem_thr_copy.partition_D(sV)

        cute.copy(gmem_tiled_copy, tQgQ, tQsQ)
        cute.arch.cp_async_commit_group()
        cute.copy(gmem_tiled_copy, tKgK[None, None, None, 0], tKsK)
        cute.arch.cp_async_commit_group()

        cute.arch.cp_async_wait_group(1)
        cute.arch.barrier(barrier_id=12 + group_id, number_of_threads=128)

        thr_mma_qk = tiled_mma_qk.get_slice(group_tid)
        thr_mma_pv = tiled_mma_pv.get_slice(group_tid)

        thread_Q_S = thr_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sQ))
        thread_K_S = thr_mma_qk.make_fragment_B(thr_mma_qk.partition_B(sK))
        thread_V_O = thr_mma_pv.make_fragment_B(thr_mma_pv.partition_B(sVt))

        acc_shape_S = thr_mma_qk.partition_shape_C((bQ, bKV))
        acc_shape_O = thr_mma_pv.partition_shape_C((bQ, d))

        acc_O = cute.make_rmem_tensor(acc_shape_O, Float32)
        acc_O.fill(0.0)

        shared_Q_S     = smem_thr_Q.partition_S(sQ)
        thread_Q_S_cpy = smem_thr_Q.retile(thread_Q_S)
        cute.copy(smem_thr_Q, shared_Q_S, thread_Q_S_cpy)

        m_atom_rows      = acc_shape_S[0][1]
        m_outer_acc      = acc_shape_S[1]
        num_rows_per_thr = m_atom_rows * m_outer_acc

        softmax_scale      = Float32(1.0 / math.sqrt(float(d)))
        softmax_scale_log2 = softmax_scale * Float32(LOG2_E)

        row_max = cute.make_rmem_tensor(cute.make_layout((num_rows_per_thr,)), Float32)
        row_sum = cute.make_rmem_tensor(cute.make_layout((num_rows_per_thr,)), Float32)
        row_max.fill(-Float32.inf)
        row_sum.fill(Float32(0.0))

        smem_store_bf = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=32,
        )

        num_kv_blocks = kv_len // bKV
        cute.arch.mbarrier_wait(self.compute_bar_me, compute_mbar_phase)
        fence_proxy_async_shared_cta()
        warpgroup_sync()
        for n in cutlass.range(num_kv_blocks):
            cute.copy(gmem_tiled_copy, tVgV[None, None, None, n], tVsV)
            cute.arch.cp_async_commit_group()

            acc_S = cute.make_rmem_tensor(acc_shape_S, Float32)
            acc_S.fill(0.0)
            shared_K_S      = smem_thr_K.partition_S(sK)
            thread_K_S_cpy  = smem_thr_K.retile(thread_K_S)
            cute.arch.cp_async_wait_group(1)
            warpgroup_sync()

            cute.copy(smem_thr_K, shared_K_S, thread_K_S_cpy)
            cute.gemm(tiled_mma_qk, acc_S, thread_Q_S, thread_K_S, acc_S)

            if n + cutlass.Int32(1) < cutlass.Int32(num_kv_blocks):
                cute.copy(gmem_tiled_copy, tKgK[None, None, None, n + 1], tKsK)
                cute.arch.cp_async_commit_group()

            acc_S_mn = Attention._reshape_acc_to_mn(acc_S)
            acc_O_mn = Attention._reshape_acc_to_mn(acc_O)
            for r in cutlass.range_constexpr(num_rows_per_thr):
                row            = acc_S_mn[r, None].load()
                local_max      = row.reduce(cute.ReductionOp.MAX, row_max[r], 0)
                local_max      = cute.arch.warp_reduction_max(local_max, threads_in_group=4)
                row_max_prev   = row_max[r]
                row_max[r]     = local_max
                local_max_safe = local_max if local_max != -Float32.inf else Float32(0.0)

                P_row = cute.math.exp2(
                    row * softmax_scale_log2 - local_max_safe * softmax_scale_log2,
                    fastmath=True,
                )
                row_scale = cute.math.exp2(
                    (row_max_prev - local_max_safe) * softmax_scale_log2,
                    fastmath=True,
                )

                acc_O_mn[r, None].store(acc_O_mn[r, None].load() * row_scale)
                row_sum[r] = P_row.reduce(
                    cute.ReductionOp.ADD, row_sum[r] * row_scale, 0,
                )
                acc_S_mn[r, None].store(P_row)

            rP_bf = cute.make_fragment_like(acc_S, BFloat16)
            rP_bf.store(acc_S.load().to(BFloat16))
            thread_S_O = Attention._reshape_rP_to_mma_A(rP_bf)

            smem_V_O     = smem_thr_V.partition_S(sVt)
            thread_V_O_cpy = smem_thr_V.retile(thread_V_O)
            cute.arch.cp_async_wait_group(0)
            warpgroup_sync()
            cute.copy(smem_thr_V, smem_V_O, thread_V_O_cpy)
            cute.gemm(tiled_mma_pv, acc_O, thread_S_O, thread_V_O, acc_O)

        cute.arch.mbarrier_arrive(self.input_bar_ot)

        acc_O_mn = Attention._reshape_acc_to_mn(acc_O)
        for r in cutlass.range_constexpr(num_rows_per_thr):
            row_sum[r] = cute.arch.warp_reduction_sum(row_sum[r], threads_in_group=4)
            inv = cute.arch.rcp_approx(
                row_sum[r] if row_sum[r] != 0.0 else Float32(1.0),
            )
            acc_O_mn[r, None].store(acc_O_mn[r, None].load() * inv)

        rO_bf = cute.make_fragment_like(acc_O, BFloat16)
        cute.arch.mbarrier_wait(self.output_bar_me, output_bar_phase)
        cute.arch.mbarrier_arrive(self.compute_bar_ot)
        rO_bf.store(acc_O.load().to(BFloat16))
        smem_thr_Ow = cute.make_tiled_copy_C(smem_store_bf, tiled_mma_pv).get_slice(group_tid)
        smem_O_acc = smem_thr_Ow.partition_D(sO)
        reg_O_acc = smem_thr_Ow.retile(rO_bf)
        cute.copy(smem_store_bf, reg_O_acc, smem_O_acc)
        fence_proxy_async_shared_cta()
        warpgroup_sync()

        if warpgroup.warp_id == 0:
            gO_tma = cute.local_tile(gO, (1, 1, bQ, d + PAD), (b, h, qb, 0))
            sO_g = cute.group_modes(sO_tma, 0, cute.rank(sO_tma.layout))
            gO_g = cute.group_modes(gO_tma, 0, cute.rank(gO_tma.layout))
            sO_part, gO_part = cpasync.tma_partition(
                tma_o, 0, cute.make_layout(1), sO_g, gO_g,
            )
            cute.copy(tma_o, sO_part, gO_part)
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0)
            fence_proxy_async_global()

        warpgroup_sync()
        cute.arch.mbarrier_arrive(self.output_bar_ot)
        if group_tid == 0:
            atomic_add_release((mAtomics.iterator + next_idx).toint(), cutlass.Int32(1))