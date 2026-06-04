from cutlass.cutlass_dsl import cutlass
import os
os.environ["CUTE_DSL_LINEINFO"] = "1"
from dataclasses import dataclass
from enum import IntEnum

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
from cutlass.utils import SmemAllocator

from operators.rmsnorm import RMSNorm, RMSNormConfig
from operators.matmul import Matmul, MatmulConfig
from operators.attention import Attention, AttentionConfig
from operators.epilogues import basic_store, residual_add_store, silu_mul
from operators.kernel_utils import WarpgroupMeta


@dataclass
class LLMMegaKernelConfig:
    embed_dim:          int
    kv_len:             int
    q_len:              int
    num_q_heads:        int
    num_kv_heads:       int
    num_layers:         int
    ff_dim:             int
    block_rms:          int
    block_q:            int
    block_kv:           int
    num_stages:         int  = 3
    bM:                 int  = 64
    bN:                 int  = 128
    bK:                 int  = 64
    bs:                 int  = 8
    num_sms:            int  = 188
    is_causal:          bool = False
    use_tma_reduce:     bool = True
    output_pad:         int  = 16
    bR:                 int  = 4
    num_stages_rms:     int  = 3
    rows_per_rms_block: int  = 32
    max_works:          int  = 0   # filled in after scheduling


class Op(IntEnum):
    RMS  = 0
    QKV  = 1
    ATTN = 2
    OUT  = 3
    GATE = 4
    UP   = 5
    DOWN = 6


class LLMMegaKernel:
    def __init__(self, config: LLMMegaKernelConfig):
        self.config       = config
        self.embed_dim    = config.embed_dim
        self.kv_len       = config.kv_len
        self.q_len        = config.q_len
        self.num_q_heads  = config.num_q_heads
        self.num_kv_heads = config.num_kv_heads
        self.num_layers   = config.num_layers
        self.ff_dim       = config.ff_dim
        self.num_stages   = config.num_stages
        self.head_dim     = self.embed_dim // self.num_q_heads
        self.bM           = config.bM
        self.bN           = config.bN
        self.bK           = config.bK
        self.bs           = config.bs
        self.num_sms      = config.num_sms
        self.max_works    = config.max_works
        self.bR                 = config.bR
        self.num_stages_rms     = config.num_stages_rms
        self.rows_per_rms_block = config.rows_per_rms_block

        self.qkv_out_dim = (self.num_q_heads + 2 * self.num_kv_heads) * self.head_dim
        self.M_tokens    = self.bs * self.q_len
        self.bQ           = 64
        self.bKV          = config.block_kv
        self.attn_warps_m = 4
        self.use_tma_reduce = config.use_tma_reduce
        self.output_pad     = config.output_pad
        STAGE_ELEMS = max(self.bM * self.bK + self.bN * self.bK, self.bR * self.embed_dim)
        self.stage_elems = STAGE_ELEMS
        self.rms_cfg = RMSNormConfig(
            embed_dim=self.embed_dim, num_stages=self.num_stages,
            num_stages_rms=self.num_stages_rms, bR=self.bR,
            rows_per_rms_block=self.rows_per_rms_block, bM=self.bM,
            stage_elems=STAGE_ELEMS)
        self.mm_cfg = MatmulConfig(
            bM=self.bM, bN=self.bN, bK=self.bK, num_stages=self.num_stages,
            stage_skip=STAGE_ELEMS, output_pad=self.output_pad,
            use_tma_store=True, use_tma_reduce=self.use_tma_reduce,
            element_type=cutlass.BFloat16)
        self.attn_cfg = AttentionConfig(
            bQ=self.bQ, bKV=self.bKV, head_dim=self.head_dim,
            num_q_heads=self.num_q_heads, num_kv_heads=self.num_kv_heads,
            q_len=self.q_len, kv_len=self.kv_len, attn_warps_m=self.attn_warps_m,
            output_pad=self.output_pad, num_stages=self.num_stages, stage_skip=STAGE_ELEMS)

    # -----------------------------------------------------------------------
    # SMEM storage
    # -----------------------------------------------------------------------
    @cute.jit
    def _get_smem_storage(self):
        bM, bN, bK, nS = self.bM, self.bN, self.bK, self.num_stages
        PAD = self.output_pad
        
        E   = self.embed_dim
        bQ  = self.bQ
        bKV = self.bKV
        d   = self.head_dim
        bR     = self.bR
        nS_rms = self.num_stages_rms
        BF  = cutlass.BFloat16
        F32 = cutlass.Float32
        
        # ---- locked 3-stage shared pipeline (nS == nS_rms == 3) ----
        # One STAGE slot = union of a matmul stage (sA+sB) and an rms input stage
        # (sX). Attention overlays Q/K/V one-per-slot (16 KiB each <= 24 KiB slot).
        # Every op OUTPUT (matmul sC, attn sO, rms sO) lives in the single `out`
        # region. The rms weight lives in the persistent per-WG `sW`, OUTSIDE the
        # stage union so a partner op's matmul can't clobber it on weight-reuse.
        #   stages : STAGES * 24 KiB = 72 KiB
        #   out    : 18 KiB     sW : 2 KiB        => 92 KiB, fits 95 KiB budget
        STAGE_ELEMS = max(bM * bK + bN * bK, bR * E)            # 12288 elems = 24 KiB
        STAGES      = nS                                         # 3 (locked)
        OUT_ELEMS   = max(bM * (bN + PAD), bQ * (d + PAD), bR * E * nS_rms)
        SW_ELEMS    = 2 * E                                      # per-WG persistent weight

        @cute.struct
        class BarrierStorage:
            load_barrier:    cute.struct.MemRange[cutlass.Uint64, STAGES]   # SHARED both WGs (was *2)
            pp_barrier:      cute.struct.MemRange[cutlass.Uint64, 2]
            input_barrier:   cute.struct.MemRange[cutlass.Uint64, 2]
            output_barrier:  cute.struct.MemRange[cutlass.Uint64, 2]
            compute_barrier: cute.struct.MemRange[cutlass.Uint64, 2]
            stage: cute.struct.MemRange[cutlass.Int32, 1]    # shared cursor   (matmul + rms)
            phase: cute.struct.MemRange[cutlass.Int32, 1]    # shared load parity (replaces returned load_phase)

        @cute.struct
        class SharedStorage:
            barriers: BarrierStorage
            sW:     cute.struct.Align[cute.struct.MemRange[BF, SW_ELEMS],             128]
            stages: cute.struct.Align[cute.struct.MemRange[BF, STAGES * STAGE_ELEMS], 1024]
            out:    cute.struct.Align[cute.struct.MemRange[BF, OUT_ELEMS],            1024]

        print("Stage elems  : ", STAGE_ELEMS, "(", STAGE_ELEMS * 2 // 1024, " KiB )")
        print("Stages total : ", STAGES * STAGE_ELEMS * 2 // 1024, " KiB")
        print("Out total    : ", OUT_ELEMS * 2 // 1024, " KiB")
        print("Shared Total : ", SharedStorage.__sizeof__())
        return SharedStorage

    @staticmethod
    def _wg_id():
        return cute.arch.warp_idx() // 4

    def _wg_tid(self):
        return cute.arch.thread_idx()[0] - self._wg_id() * 128

    # -----------------------------------------------------------------------
    # Host: build tensors + TMA atoms, launch the kernel
    # -----------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        mSchedule:    cute.Tensor,
        mAtomics:     cute.Tensor,
        mRMS_weights: cute.Tensor,
        mQKV_proj:    cute.Tensor,
        mWorkspace1:  cute.Tensor,
        mWorkspace2:  cute.Tensor,
        mGate_proj:   cute.Tensor,
        mUp_proj:     cute.Tensor,
        mDown_proj:   cute.Tensor,
        mOutProjAttn: cute.Tensor,
        mEmbedding:   cute.Tensor,
    ):
        M    = self.M_tokens
        E    = self.embed_dim
        F    = self.ff_dim
        Qd   = self.qkv_out_dim
        H_q  = self.num_q_heads
        H_kv = self.num_kv_heads
        d    = self.head_dim
        bM, bN, bK = self.bM, self.bN, self.bK

        # ----- workspace1 / workspace2 views -----
        # ws1 holds (M, E)-shaped intermediates (RMS output, attention output,
        # matmul A-operand for QKV/OUT/UP/GATE).
        # ws2 is wider: (M, Qd) for QKV, (M, F) for the FF intermediates.  All
        # views share the same base pointer; we just create the layouts we
        # actually need.
        mWS1_ME = cute.make_tensor(
            mWorkspace1.iterator, cute.make_layout((M, E),  stride=(E,  1)))
        mWS2_MQ = cute.make_tensor(
            mWorkspace2.iterator, cute.make_layout((M, Qd), stride=(Qd, 1)))
        mWS2_MF = cute.make_tensor(
            mWorkspace2.iterator, cute.make_layout((M, F),  stride=(F,  1)))

        # ----- packed Q/K/V views on top of workspace2 -----
        # workspace2 after QKV proj has rows of length Qd = H_q*d + 2*H_kv*d
        # holding [Q | K | V] in interleaved (token-major) order.  Reshape so
        # the attention kernel sees standard (B, H, N, D) layouts.
        q_layout  = cute.make_layout(
            shape=(self.bs, H_q,  self.q_len, d),
            stride=(self.q_len * Qd, d, Qd, 1))
        kv_layout = cute.make_layout(
            shape=(self.bs, H_kv, self.q_len, d),
            stride=(self.q_len * Qd, d, Qd, 1))
        k_off = H_q * d
        v_off = (H_q + H_kv) * d
        mQ = cute.make_tensor(mWorkspace2.iterator,         q_layout)
        mK = cute.make_tensor(mWorkspace2.iterator + k_off, kv_layout)
        mV = cute.make_tensor(mWorkspace2.iterator + v_off, kv_layout)
        mO = cute.make_tensor(
            mWorkspace1.iterator,
            cute.make_layout(
                shape=(self.bs, H_q, self.q_len, d),
                stride=(self.q_len * H_q * d, d, H_q * d, 1)))

        # ----- smem-layout atoms for the TMAs -----
        sw = cute.make_swizzle(3, 4, 3)
        sA_atom = cute.make_layout((bM, bK), stride=(bK, 1))
        # MASKED PADDED C-store box: the smem box innermost is bN+PAD (the full
        # padded row, read contiguously) while the gmem VIEW innermost dim is bN, so
        # boxDim>globalDim and the TMA MASKS the PAD lanes on store. The pad lives in
        # the BOX, not the smem stride -- a strided (gap) smem read corrupts the gmem
        # mapping (wrong output); a masked full-row read writes exactly bN real cols.
        # PAD==0 -> box==bN==gmem inner -> exact tile, plain store (no masking).
        PAD     = self.output_pad
        sC_atom = cute.make_layout((bM, 1, bN + PAD),
                                   stride=(bN + PAD, bN + PAD, 1))
        sW_atom = cute.make_layout((1, bN, bK), stride=(bN * bK, bK, 1))
        sA_swz  = cute.make_composed_layout(sw, 0, sA_atom)
        sW_swz  = cute.make_composed_layout(sw, 0, sW_atom)
        sE_row  = cute.make_layout((1, E),         stride=(E, 1))
        sE_rows = cute.make_layout((self.bR, E),   stride=(E, 1))

        load_op  = cpasync.CopyBulkTensorTileG2SOp()
        store_op = cpasync.CopyBulkTensorTileS2GOp()
        # Residual ops (OUT, DOWN) store into mEmbedding. With the flag on, use the
        # TMA reduce-ADD op so the store does gC[coord] += sC[coord] in gmem (no
        # read-modify-write in the epilogue). MUST match Epilogues(use_tma_reduce=...)
        # or the residual is double-counted / overwritten. ADD only (no MUL), so
        # GATE's silu_mul keeps the classic path regardless.
        if cutlass.const_expr(self.use_tma_reduce):
            emb_store_op = cpasync.CopyReduceBulkTensorTileS2GOp(cute.ReductionOp.ADD)
        else:
            emb_store_op = store_op

        # ----- TMA atoms for weights (layer-indexable: GMEM shape is (L, .., ..)) -----
        # Each weight has an outer L dim, so the matmul does
        #     gB_tile = cute.local_tile(gB, (1, bN, bK), (layer_idx, pid_n, None))
        # to peel off the layer's 2D slice.
        tma_rms_w,  g_rms_w  = cpasync.make_tiled_tma_atom(load_op, mRMS_weights, sE_row, (1, E))
        tma_qkv_w,  g_qkv_w  = cpasync.make_tiled_tma_atom(load_op, mQKV_proj,    sW_swz, (1, bN, bK))
        tma_out_w,  g_out_w  = cpasync.make_tiled_tma_atom(load_op, mOutProjAttn, sW_swz, (1, bN, bK))
        tma_gate_w, g_gate_w = cpasync.make_tiled_tma_atom(load_op, mGate_proj,   sW_swz, (1, bN, bK))
        tma_up_w,   g_up_w   = cpasync.make_tiled_tma_atom(load_op, mUp_proj,     sW_swz, (1, bN, bK))
        tma_down_w, g_down_w = cpasync.make_tiled_tma_atom(load_op, mDown_proj,   sW_swz, (1, bN, bK))

        # ----- TMA atoms for workspaces (same base ptr, multiple shapes) -----
        # SM120: TMA stores must use UNswizzled smem layouts.  Loads can be swizzled.
        # C-store gmem 3D VIEWS: reshape each dense (M,width) workspace row into
        # (M, width/bN, bN) so the descriptor's innermost dim is exactly bN. The
        # store box innermost (bN+PAD) then exceeds it -> the PAD lanes are OOB
        # and masked. width%bN==0 holds: Qd=1536, F=512, E=512 all divide bN=128.
        mWS2_MQ_st = cute.make_tensor(
            mWorkspace2.iterator, cute.make_layout((M, Qd // bN, bN), stride=(Qd, bN, 1)))
        mWS2_MF_st = cute.make_tensor(
            mWorkspace2.iterator, cute.make_layout((M, F  // bN, bN), stride=(F,  bN, 1)))
        mEmb_st    = cute.make_tensor(
            mEmbedding.iterator, cute.make_layout((M, E // bN, bN), stride=(E, bN, 1)))

        tma_ws1_A,    g_ws1_A    = cpasync.make_tiled_tma_atom(load_op,  mWS1_ME, sA_swz,  (bM, bK))
        tma_ws1_rows, g_ws1_rows = cpasync.make_tiled_tma_atom(store_op, mWS1_ME, sE_rows, (self.bR, E))
        tma_ws2_qkv,  g_ws2_qkv  = cpasync.make_tiled_tma_atom(store_op, mWS2_MQ_st, sC_atom, (bM, 1, bN + PAD))
        tma_ws2_ff_s, g_ws2_ff_s = cpasync.make_tiled_tma_atom(store_op, mWS2_MF_st, sC_atom, (bM, 1, bN + PAD))
        tma_ws2_ff_A, g_ws2_ff_A = cpasync.make_tiled_tma_atom(load_op,  mWS2_MF, sA_swz,  (bM, bK))

        # Embedding: load `bR` rows for RMS input, residual store for OUT/DOWN.
        tma_emb_rows, g_emb_rows = cpasync.make_tiled_tma_atom(load_op,  mEmbedding, sE_rows, (self.bR, E))
        tma_emb_red,  g_emb_red  = cpasync.make_tiled_tma_atom(emb_store_op, mEmb_st, sC_atom, (bM, 1, bN + PAD))

        # ----- TMA atom for attention output store (one bQ-row tile per ATTN op) -----
        # Same masked padded store: box innermost is d+PAD (full padded row) but mO's
        # innermost dim is d, so the PAD lanes are masked on store. PAD==0 -> plain.
        sO_tile = cute.make_layout(
            (1, 1, self.bQ, d + PAD),
            stride=(self.bQ * (d + PAD), self.bQ * (d + PAD), d + PAD, 1),
        )
        tma_o, g_o = cpasync.make_tiled_tma_atom(store_op, mO, sO_tile, (1, 1, self.bQ, d + PAD))

        # Ops hold ONLY constexpr config (+ the epilogue callable). All tensors and
        # TMA atoms are passed into run() as arguments from the kernel. This keeps the
        # op objects constexpr so they can be reached through `self` inside the dynamic
        # dispatch loop without being treated as loop-carried values.
        self.rmsnorm = RMSNorm(self.rms_cfg)
        self.qkv  = Matmul(self.mm_cfg, basic_store)
        self.out  = Matmul(self.mm_cfg, residual_add_store)
        self.up   = Matmul(self.mm_cfg, basic_store)
        self.gate = Matmul(self.mm_cfg, silu_mul)
        self.down = Matmul(self.mm_cfg, residual_add_store)
        self.attn = Attention(self.attn_cfg)

        SharedStorage = self._get_smem_storage()

        self.kernel(mSchedule, mAtomics,
                    g_rms_w, g_emb_rows, g_ws1_rows,
                    g_ws1_A, g_qkv_w, g_ws2_qkv,
                    g_out_w, g_emb_red,
                    g_up_w, g_gate_w, g_ws2_ff_s,
                    g_ws2_ff_A, g_down_w,
                    mEmbedding, mWS2_MF,
                    mQ, mK, mV, mO, g_o,
                    tma_rms_w, tma_emb_rows, tma_ws1_rows,
                    tma_ws1_A, tma_qkv_w, tma_ws2_qkv,
                    tma_out_w, tma_emb_red,
                    tma_up_w, tma_gate_w, tma_ws2_ff_s,
                    tma_ws2_ff_A, tma_down_w,
                    tma_o,
                    SharedStorage).launch(
            grid=(self.num_sms,),
            block=(256,),
        )

    # -----------------------------------------------------------------------
    # Device kernel: dispatch loop only
    # -----------------------------------------------------------------------
    @cute.kernel
    def kernel(self, mSchedule, mAtomics,
               g_rms_w, g_emb_rows, g_ws1_rows,
               g_ws1_A, g_qkv_w, g_ws2_qkv,
               g_out_w, g_emb_red,
               g_up_w, g_gate_w, g_ws2_ff_s,
               g_ws2_ff_A, g_down_w,
               mEmbedding, mWS2_MF,
               mQ, mK, mV, mO, g_o,
               tma_rms_w, tma_emb_rows, tma_ws1_rows,
               tma_ws1_A, tma_qkv_w, tma_ws2_qkv,
               tma_out_w, tma_emb_red,
               tma_up_w, tma_gate_w, tma_ws2_ff_s,
               tma_ws2_ff_A, tma_down_w,
               tma_o,
               SharedStorage: cutlass.Constexpr):
        warp_id    = cute.arch.warp_idx()
        group_id   = warp_id // 4
        local_warp = warp_id %  4
        group_tid  = self._wg_tid()
        tidx, _, _ = cute.arch.thread_idx()
        block_id   = cute.arch.block_idx()[0]
        wg = WarpgroupMeta(tidx=tidx, group_tidx=group_tid, group_id=group_id, lane_id=tidx % 32, warp_id=local_warp)

        if local_warp == 0:
            cpasync.prefetch_descriptor(tma_rms_w)
            cpasync.prefetch_descriptor(tma_emb_rows)
            cpasync.prefetch_descriptor(tma_ws1_rows)
            cpasync.prefetch_descriptor(tma_ws1_A)
            cpasync.prefetch_descriptor(tma_qkv_w)
            cpasync.prefetch_descriptor(tma_ws2_qkv)
            cpasync.prefetch_descriptor(tma_out_w)
            cpasync.prefetch_descriptor(tma_emb_red)
            cpasync.prefetch_descriptor(tma_up_w)
            cpasync.prefetch_descriptor(tma_gate_w)
            cpasync.prefetch_descriptor(tma_ws2_ff_s)
            cpasync.prefetch_descriptor(tma_ws2_ff_A)
            cpasync.prefetch_descriptor(tma_down_w)
            cpasync.prefetch_descriptor(tma_o)

        smem    = SmemAllocator()
        storage = smem.allocate(SharedStorage)
        load_barriers   = storage.barriers.load_barrier.data_ptr()
        pp_barriers     = storage.barriers.pp_barrier.data_ptr()
        input_barriers  = storage.barriers.input_barrier.data_ptr()
        output_barriers = storage.barriers.output_barrier.data_ptr()
        compute_barriers = storage.barriers.compute_barrier.data_ptr()
        load_stage = storage.barriers.stage.get_tensor(cute.make_layout((1, )))
        load_phase_cell = storage.barriers.phase.get_tensor(cute.make_layout((1, )))

        if warp_id == 0:
            with cute.arch.elect_one():
                load_stage[0]      = cutlass.Int32(0)
                load_phase_cell[0] = cutlass.Int32(0)
                for i in cutlass.range_constexpr(self.num_stages):   # STAGES, shared (was *2)
                    cute.arch.mbarrier_init(load_barriers + i, 1)
                cute.arch.mbarrier_init(pp_barriers + 0, 128)
                cute.arch.mbarrier_init(pp_barriers + 1, 128)
                for i in cutlass.range_constexpr(2):
                    cute.arch.mbarrier_init(input_barriers + i, 128)
                    cute.arch.mbarrier_init(output_barriers + i, 128)
                    cute.arch.mbarrier_init(compute_barriers + i, 128)
                cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        if group_id == 1:
            cute.arch.mbarrier_arrive(input_barriers + 0)
            cute.arch.mbarrier_arrive(compute_barriers + 0)
            cute.arch.mbarrier_arrive(output_barriers + 0)

        compute_mbar_phase = cutlass.Int32(0)
        input_mbar_phase = cutlass.Int32(0)
        output_mbar_phase = cutlass.Int32(0)
        max_works_local = (self.max_works + (1 - group_id)) // 2

        for local_work_idx in cutlass.range(max_works_local):
            work_idx = local_work_idx * 2 + group_id
            op           = mSchedule[block_id, work_idx, 0]
            pid_m        = mSchedule[block_id, work_idx, 1]
            pid_n        = mSchedule[block_id, work_idx, 2]
            pid_o        = mSchedule[block_id, work_idx, 3]
            expected_cnt = mSchedule[block_id, work_idx, 4]
            atomic_idx   = mSchedule[block_id, work_idx, 5]
            next_idx     = mSchedule[block_id, work_idx, 6]

            op_kind   = op & 0x7
            layer_idx = op >> 3

            if op_kind == cutlass.Int32(int(Op.RMS)):
                rms_w_idx = layer_idx * 2 + pid_o
                self.rmsnorm.run(g_emb_rows, g_ws1_rows, g_rms_w, tma_rms_w, tma_emb_rows, tma_ws1_rows,
                                 rms_w_idx, pid_m, mAtomics, expected_cnt, atomic_idx, next_idx,
                                 storage, pid_n, wg, compute_mbar_phase, input_mbar_phase, output_mbar_phase)
            elif op_kind == cutlass.Int32(int(Op.QKV)):
                self.qkv.run(g_ws1_A, g_qkv_w, g_ws2_qkv, None, tma_ws1_A, tma_qkv_w, tma_ws2_qkv,
                             layer_idx, pid_m, pid_n, mAtomics, expected_cnt, atomic_idx, next_idx,
                             storage, wg, compute_mbar_phase, input_mbar_phase, output_mbar_phase)
            elif op_kind == cutlass.Int32(int(Op.ATTN)):
                self.attn.run(mQ, mK, mV, mO, tma_o, g_o,
                              pid_m, pid_n, pid_o, mAtomics, expected_cnt, atomic_idx, next_idx,
                              storage, wg, compute_mbar_phase, input_mbar_phase, output_mbar_phase)
            elif op_kind == cutlass.Int32(int(Op.OUT)):
                self.out.run(g_ws1_A, g_out_w, g_emb_red, mEmbedding, tma_ws1_A, tma_out_w, tma_emb_red,
                             layer_idx, pid_m, pid_n, mAtomics, expected_cnt, atomic_idx, next_idx,
                             storage, wg, compute_mbar_phase, input_mbar_phase, output_mbar_phase)
            elif op_kind == cutlass.Int32(int(Op.UP)):
                self.up.run(g_ws1_A, g_up_w, g_ws2_ff_s, None, tma_ws1_A, tma_up_w, tma_ws2_ff_s,
                            layer_idx, pid_m, pid_n, mAtomics, expected_cnt, atomic_idx, next_idx,
                            storage, wg, compute_mbar_phase, input_mbar_phase, output_mbar_phase)
            elif op_kind == cutlass.Int32(int(Op.GATE)):
                self.gate.run(g_ws1_A, g_gate_w, g_ws2_ff_s, mWS2_MF, tma_ws1_A, tma_gate_w, tma_ws2_ff_s,
                              layer_idx, pid_m, pid_n, mAtomics, expected_cnt, atomic_idx, next_idx,
                              storage, wg, compute_mbar_phase, input_mbar_phase, output_mbar_phase)
            elif op_kind == cutlass.Int32(int(Op.DOWN)):
                self.down.run(g_ws2_ff_A, g_down_w, g_emb_red, mEmbedding, tma_ws2_ff_A, tma_down_w, tma_emb_red,
                              layer_idx, pid_m, pid_n, mAtomics, expected_cnt, atomic_idx, next_idx,
                              storage, wg, compute_mbar_phase, input_mbar_phase, output_mbar_phase)

            compute_mbar_phase = compute_mbar_phase ^ cutlass.Int32(1)
            input_mbar_phase   = input_mbar_phase ^ cutlass.Int32(1)
            output_mbar_phase  = output_mbar_phase ^ cutlass.Int32(1)