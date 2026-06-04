import math
from functools import partial

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, warp
from dataclasses import dataclass

from operators.kernel_utils import (
    LOG2_E, ld_acquire_u32, atomic_add_release,
    fence_proxy_async_shared_cta, fence_proxy_async_global, WarpgroupMeta,
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
    def _wg_sync(self, *, group_id):
        cute.arch.barrier(barrier_id=12 + group_id, number_of_threads=128)

    @cute.jit
    def _wait_prev(self, *, expected_cnt, mAtomics, atomic_idx, group_tid):
        if group_tid == 0:
            ready = cutlass.Int32(0)
            while ready != expected_cnt:
                ready = ld_acquire_u32((mAtomics.iterator + atomic_idx).toint())

    @cute.jit
    def run(self, mQ, mK, mV, mO, tma_out, gO,
            b, h, qb,
            mAtomics, expected_cnt, atomic_idx, next_idx, storage,
            warpgroup: WarpgroupMeta, compute_mbar_phase, input_mbar_phase, output_bar_phase):
        cfg      = self.config
        bQ       = cfg.bQ
        bKV      = cfg.bKV
        d        = cfg.head_dim
        H_q      = cfg.num_q_heads
        H_kv     = cfg.num_kv_heads
        nWarps_m = cfg.attn_warps_m
        kv_len   = cfg.kv_len
        nS       = cfg.num_stages
        SE       = cfg.stage_skip
        PAD      = cfg.output_pad
        group_id  = warpgroup.group_id
        group_tid = warpgroup.group_tidx

        h_kv = h * H_kv // H_q
        BF   = cutlass.BFloat16
        F32  = cutlass.Float32

        tiled_mma_qk = cute.make_tiled_mma(warp.MmaF16BF16Op(BF, F32, (16, 8, 16)), (nWarps_m, 1, 1))
        tiled_mma_pv = cute.make_tiled_mma(warp.MmaF16BF16Op(BF, F32, (16, 8, 16)), (nWarps_m, 1, 1))

        atom_async = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), BF, num_bits_per_copy=128)

        smem_k_block = 64 if d % 64 == 0 else 32
        swizzle_bits = 3 if smem_k_block == 64 else 2
        layout_atom  = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3), 0,
            cute.make_layout((8, smem_k_block), stride=(smem_k_block, 1)))

        sO_layout     = cute.make_layout((bQ, d),       stride=(d + PAD, 1))
        sO_tma_layout = cute.make_layout((bQ, d + PAD), stride=(d + PAD, 1))
        sQ_layout  = cute.tile_to_shape(layout_atom, (bQ,  d), (0, 1))
        sKV_layout = cute.tile_to_shape(layout_atom, (bKV, d), (0, 1))

        load_stage = storage.barriers.stage.get_tensor(cute.make_layout((1,)))
        c = load_stage[0]
        stages_ptr = storage.stages.data_ptr()
        sQ     = cute.make_tensor((stages_ptr + c * SE).align(128), sQ_layout)
        sK     = cute.make_tensor((stages_ptr + ((c + 1) % nS) * SE).align(128), sKV_layout)
        sV     = cute.make_tensor((stages_ptr + c * SE).align(128), sKV_layout)
        sO     = storage.out.get_tensor(sO_layout)
        sO_tma = storage.out.get_tensor(sO_tma_layout)

        sVt = cute.composition(sV, cute.make_layout((d, bKV), stride=(bKV, 1)))

        other = group_id ^ 1
        input_bar_me   = storage.barriers.input_barrier.data_ptr()   + group_id
        input_bar_ot   = storage.barriers.input_barrier.data_ptr()   + other
        compute_bar_me = storage.barriers.compute_barrier.data_ptr() + group_id
        compute_bar_ot = storage.barriers.compute_barrier.data_ptr() + other
        output_bar_me  = storage.barriers.output_barrier.data_ptr()  + group_id
        output_bar_ot  = storage.barriers.output_barrier.data_ptr()  + other
        cute.arch.mbarrier_wait(input_bar_me, input_mbar_phase)

        async_elems     = 128 // 16
        cols_per_pass   = d // async_elems
        rows_per_pass   = 128 // cols_per_pass
        tQ_layout       = cute.make_ordered_layout((rows_per_pass, cols_per_pass), order=(1, 0))
        vQ_layout       = cute.make_layout((1, async_elems))
        gmem_tiled_copy = cute.make_tiled_copy_tv(atom_async, tQ_layout, vQ_layout)
        gmem_thr_copy   = gmem_tiled_copy.get_slice(group_tid)

        smem_atom_QK = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), BF)
        smem_atom_V  = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=True,  num_matrices=4), BF)
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

        thr_mma_qk = tiled_mma_qk.get_slice(group_tid)
        thr_mma_pv = tiled_mma_pv.get_slice(group_tid)

        tSrQ = thr_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sQ))
        tSrK = thr_mma_qk.make_fragment_B(thr_mma_qk.partition_B(sK))
        tOrV = thr_mma_pv.make_fragment_B(thr_mma_pv.partition_B(sVt))

        acc_shape_S = thr_mma_qk.partition_shape_C((bQ, bKV))
        acc_shape_O = thr_mma_pv.partition_shape_C((bQ, d))
        acc_O = cute.make_rmem_tensor(acc_shape_O, F32)
        acc_O.fill(0.0)

        tSsQ      = smem_thr_Q.partition_S(sQ)
        tSrQ_view = smem_thr_Q.retile(tSrQ)

        m_atom_rows      = acc_shape_S[0][1]
        m_outer_acc      = acc_shape_S[1]
        num_rows_per_thr = m_atom_rows * m_outer_acc

        softmax_scale      = cutlass.Float32(1.0 / math.sqrt(float(d)))
        softmax_scale_log2 = softmax_scale * cutlass.Float32(LOG2_E)

        row_max = cute.make_rmem_tensor(cute.make_layout((num_rows_per_thr,)), F32)
        row_sum = cute.make_rmem_tensor(cute.make_layout((num_rows_per_thr,)), F32)
        row_max.fill(-F32.inf)
        row_sum.fill(cutlass.Float32(0.0))

        smem_store_bf = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BF, num_bits_per_copy=32)

        num_kv_blocks = kv_len // bKV

        wg_sync   = partial(self._wg_sync, group_id=group_id)
        wait_prev = partial(self._wait_prev, expected_cnt=expected_cnt,
                            mAtomics=mAtomics, atomic_idx=atomic_idx, group_tid=group_tid)

        smem_thr_Ow = cute.make_tiled_copy_C(smem_store_bf, tiled_mma_pv).get_slice(group_tid)

        wait_prev()
        wg_sync()

        cute.copy(gmem_tiled_copy, tQgQ, tQsQ)
        cute.arch.cp_async_commit_group()
        cute.copy(gmem_tiled_copy, tKgK[None, None, None, 0], tKsK)
        cute.arch.cp_async_commit_group()

        cute.arch.cp_async_wait_group(1)
        wg_sync()
        cute.copy(smem_thr_Q, tSsQ, tSrQ_view)

        cute.arch.mbarrier_wait(compute_bar_me, compute_mbar_phase)
        fence_proxy_async_shared_cta()
        wg_sync()

        for n in cutlass.range(num_kv_blocks, unroll=1):
            cute.copy(gmem_tiled_copy, tVgV[None, None, None, n], tVsV)
            cute.arch.cp_async_commit_group()

            acc_S = cute.make_rmem_tensor(acc_shape_S, F32)
            acc_S.fill(0.0)
            tSsK      = smem_thr_K.partition_S(sK)
            tSrK_view = smem_thr_K.retile(tSrK)
            cute.arch.cp_async_wait_group(1)
            wg_sync()

            cute.copy(smem_thr_K, tSsK, tSrK_view)
            cute.gemm(tiled_mma_qk, acc_S, tSrQ, tSrK, acc_S)

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
                local_max_safe = local_max if local_max != -F32.inf else cutlass.Float32(0.0)

                P_row = cute.math.exp2(
                    row * softmax_scale_log2 - local_max_safe * softmax_scale_log2, fastmath=True)
                row_scale = cute.math.exp2(
                    (row_max_prev - local_max_safe) * softmax_scale_log2, fastmath=True)

                acc_O_mn[r, None].store(acc_O_mn[r, None].load() * row_scale)
                row_sum[r] = P_row.reduce(cute.ReductionOp.ADD, row_sum[r] * row_scale, 0)
                acc_S_mn[r, None].store(P_row)

            rP_bf = cute.make_fragment_like(acc_S, BF)
            rP_bf.store(acc_S.load().to(BF))
            tOrS = Attention._reshape_rP_to_mma_A(rP_bf)

            tOsVt     = smem_thr_V.partition_S(sVt)
            tOrV_view = smem_thr_V.retile(tOrV)
            cute.arch.cp_async_wait_group(0)
            wg_sync()
            cute.copy(smem_thr_V, tOsVt, tOrV_view)
            cute.gemm(tiled_mma_pv, acc_O, tOrS, tOrV, acc_O)
            wg_sync()

        cute.arch.mbarrier_arrive(input_bar_ot)
        acc_O_mn = Attention._reshape_acc_to_mn(acc_O)
        for r in cutlass.range_constexpr(num_rows_per_thr):
            row_sum[r] = cute.arch.warp_reduction_sum(row_sum[r], threads_in_group=4)
            inv = cute.arch.rcp_approx(row_sum[r] if row_sum[r] != 0.0 else cutlass.Float32(1.0))
            acc_O_mn[r, None].store(acc_O_mn[r, None].load() * inv)

        cute.arch.mbarrier_wait(output_bar_me, output_bar_phase)
        cute.arch.mbarrier_arrive(compute_bar_ot)

        rO_bf = cute.make_fragment_like(acc_O, BF)
        rO_bf.store(acc_O.load().to(BF))
        taccOsO = smem_thr_Ow.partition_D(sO)
        taccOrO = smem_thr_Ow.retile(rO_bf)
        cute.copy(smem_store_bf, taccOrO, taccOsO)
        wg_sync()
        fence_proxy_async_shared_cta()

        if group_tid == 0:
            gO_tma = cute.local_tile(gO, (1, 1, bQ, d + PAD), (b, h, qb, 0))
            sO_g = cute.group_modes(sO_tma, 0, cute.rank(sO_tma.layout))
            gO_g = cute.group_modes(gO_tma, 0, cute.rank(gO_tma.layout))
            sO_part, gO_part = cpasync.tma_partition(tma_out, 0, cute.make_layout(1), sO_g, gO_g)
            cute.copy(tma_out, sO_part, gO_part)
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0)
            fence_proxy_async_global()

        cute.arch.mbarrier_arrive(output_bar_ot)
        if group_tid == 0:
            atomic_add_release((mAtomics.iterator + next_idx).toint(), cutlass.Int32(1))