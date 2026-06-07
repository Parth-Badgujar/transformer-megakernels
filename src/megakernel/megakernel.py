from src.megakernel.kernel_utils import PipelineMeta
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
from operators.kernel_utils import WarpgroupMeta, Phases
from cutlass import Int32, BFloat16, Uint64


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
        self.bR           = config.bR
        self.rows_per_rms_block = config.rows_per_rms_block
        self.qkv_out_dim = (self.num_q_heads + 2 * self.num_kv_heads) * self.head_dim
        self.num_tokens    = self.bs * self.q_len
        self.bQ           = config.block_q
        self.bKV          = config.block_kv
        self.use_tma_reduce = config.use_tma_reduce
        self.output_pad     = config.output_pad
        self.stage_elements = max(self.bM * self.bK + self.bN * self.bK, self.bR * self.embed_dim)

        self.rms_config = RMSNormConfig(
            bR = self.bR,
            bM = self.bM,
            embed_dim = self.embed_dim,
            num_stages = self.num_stages,
            rows_per_rms_block = self.rows_per_rms_block,
            stage_elems = self.stage_elements
        )

        self.matmul_config = MatmulConfig(
            bM = self.bM,
            bN = self.bN,
            bK = self.bK,
            num_stages = self.num_stages,
            stage_skip = self.stage_elements,
            use_tma_store = True,
            use_tma_reduce = self.use_tma_reduce,
            element_type = BFloat16, ## add float16 as well
            output_pad = self.output_pad,
        )

        self.attention_config = AttentionConfig(
            bQ = self.bQ,
            bKV = self.bKV,
            q_len = self.q_len,
            kv_len = self.kv_len,
            head_dim = self.head_dim,
            num_q_heads = self.num_q_heads,
            num_kv_heads = self.num_kv_heads,
            num_stages = self.num_stages,
            stage_skip = self.stage_elements,
            output_pad = self.output_pad,
        )

    @cute.jit
    def _get_shared_storage(self):
        num_out_elements   = max(
            self.bM * (self.bN + self.output_pad),
            self.bQ * (self.head_dim + self.output_pad),
            self.bR * self.embed_dim * self.num_stages
        )

        @cute.struct
        class BarrierStorage:
            load_barrier:    cute.struct.MemRange[Uint64, self.num_stages]
            pp_barrier:      cute.struct.MemRange[Uint64, 2]
            input_barrier:   cute.struct.MemRange[Uint64, 2]
            output_barrier:  cute.struct.MemRange[Uint64, 2]
            compute_barrier: cute.struct.MemRange[Uint64, 2]
            stage: cute.struct.MemRange[Int32, 1]
            phase: cute.struct.MemRange[Int32, 1]

        @cute.struct
        class SharedStorage:
            barriers: BarrierStorage
            sW:     cute.struct.Align[cute.struct.MemRange[BFloat16, 2 * self.embed_dim], 128]
            stages: cute.struct.Align[cute.struct.MemRange[BFloat16, self.num_stages * self.stage_elements], 128]
            out:    cute.struct.Align[cute.struct.MemRange[BFloat16, num_out_elements], 128]

        return SharedStorage

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
        PAD = self.output_pad

        mWS1_ME = cute.make_tensor(
            mWorkspace1.iterator, 
            cute.make_ordered_layout(
                shape = (self.num_tokens, self.embed_dim),
                order = (1, 0)
            )
        )

        mWS2_MF = cute.make_tensor(
            mWorkspace2.iterator, 
            cute.make_ordered_layout(
                shape = (self.num_tokens, self.ff_dim),
                order = (1, 0)
            )
        )

        q_layout  = cute.make_layout(
            shape = (self.bs, self.num_q_heads,  self.q_len, self.head_dim),
            stride = (self.q_len * self.qkv_out_dim, self.head_dim, self.qkv_out_dim, 1)
        )
        kv_layout = cute.make_layout(
            shape = (self.bs, self.num_kv_heads, self.kv_len, self.head_dim),
            stride = (self.kv_len * self.qkv_out_dim, self.head_dim, self.qkv_out_dim, 1)
        )

        k_off = self.num_q_heads * self.head_dim
        v_off = (self.num_q_heads + self.num_kv_heads) * self.head_dim

        mQ = cute.make_tensor(mWorkspace2.iterator,         q_layout)
        mK = cute.make_tensor(mWorkspace2.iterator + k_off, kv_layout)
        mV = cute.make_tensor(mWorkspace2.iterator + v_off, kv_layout)
        mO = cute.make_tensor(
            mWorkspace1.iterator, cute.make_ordered_layout(
                shape = (self.bs, self.num_q_heads, self.q_len, self.head_dim),
                order = (3, 1, 2, 0)
            )
        )
        mWS2_MQ_st = cute.make_tensor(
            mWorkspace2.iterator, cute.make_ordered_layout(
                shape = (self.num_tokens, self.qkv_out_dim // self.bN, self.bN),
                order = (2, 1, 0)
            )
        )
        mWS2_MF_st = cute.make_tensor(
            mWorkspace2.iterator, cute.make_ordered_layout(
                shape = (self.num_tokens, self.ff_dim // self.bN, self.bN),
                order = (2, 1, 0)
            )
        )
        mEmb_st    = cute.make_tensor(
            mEmbedding.iterator, cute.make_ordered_layout(
                shape = (self.num_tokens, self.embed_dim // self.bN, self.bN),
                order = (2, 1, 0)
            )
        )

        load_op  = cpasync.CopyBulkTensorTileG2SOp()
        store_op = cpasync.CopyBulkTensorTileS2GOp()

        store_op_red = None
        if cutlass.const_expr(self.use_tma_reduce):
            store_op_red = cpasync.CopyReduceBulkTensorTileS2GOp(cute.ReductionOp.ADD)
        else:
            store_op_red = store_op

        matmul_C_pad = cute.make_ordered_layout(
            shape = (self.bM, 1, self.bN + PAD),
            order = (2, 1, 0)
        ) 
        matmul_A_sw = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3), 0, 
                cute.make_ordered_layout(
                    shape = (self.bM, self.bK),
                    order = (1, 0)
                )
        )
        matmul_B_sw = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3), 0, 
                cute.make_ordered_layout(
                    shape = (1, self.bN, self.bK),
                    order = (2, 1, 0)
                )
        )
        embed_wt  = cute.make_ordered_layout(
            shape = (1, self.embed_dim),
            order = (1, 0)
        )
        embed_act = cute.make_ordered_layout(
            shape = (self.bR, self.embed_dim),
            order = (1, 0)
        )
        '''
        Naming Convention
        g_<Op>_<Tensor Name>
        tma_<Op>_<Tensor Name>
        '''
        # RMS TMA Atoms (Embedding @ mRMS_weights -> WS1)
        tma_RMS_inp, g_RMS_inp   = cpasync.make_tiled_tma_atom(load_op,      mEmbedding,   embed_act,    (self.bR, self.embed_dim))
        tma_RMS_wt,  g_RMS_wt    = cpasync.make_tiled_tma_atom(load_op,      mRMS_weights, embed_wt,     (1, self.embed_dim))
        tma_RMS_out, g_RMS_out   = cpasync.make_tiled_tma_atom(store_op,     mWS1_ME,      embed_act,    (self.bR, self.embed_dim))
        # QKV TMA Atoms (WS1 @ mQKV_Weights -> WS2)
        tma_QKV_inp, g_QKV_inp   = cpasync.make_tiled_tma_atom(load_op,      mWS1_ME,      matmul_A_sw,  (self.bM, self.bK))
        tma_QKV_wt,  g_QKV_wt    = cpasync.make_tiled_tma_atom(load_op,      mQKV_proj,    matmul_B_sw,  (1, self.bN, self.bK))
        tma_QKV_act, g_QKV_act   = cpasync.make_tiled_tma_atom(store_op,     mWS2_MQ_st,   matmul_C_pad, (self.bM, 1, self.bN + PAD))
        # OUT Proj TMA Atoms (WS1 @ mOut_weights -> += Embedding) (Reduction)
        tma_OUT_inp, g_OUT_inp   = cpasync.make_tiled_tma_atom(load_op,      mWS1_ME,      matmul_A_sw,  (self.bM, self.bK))
        tma_OUT_wt,  g_OUT_wt    = cpasync.make_tiled_tma_atom(load_op,      mOutProjAttn, matmul_B_sw,  (1, self.bN, self.bK))
        tma_OUT_out, g_OUT_out   = cpasync.make_tiled_tma_atom(store_op_red, mEmb_st,      matmul_C_pad, (self.bM, 1, self.bN + PAD))
        # UP Proj TMA Atoms (WS1 @ mUp_weights -> WS2)
        tma_UP_inp, g_UP_inp     = cpasync.make_tiled_tma_atom(load_op,      mWS1_ME,      matmul_A_sw,  (self.bM, self.bK))
        tma_UP_wt,  g_UP_wt      = cpasync.make_tiled_tma_atom(load_op,      mUp_proj,     matmul_B_sw,  (1, self.bN, self.bK))
        tma_UP_out, g_UP_out     = cpasync.make_tiled_tma_atom(store_op,     mWS2_MF_st,   matmul_C_pad, (self.bM, 1, self.bN + PAD))
        # GATE Proj TMA AToms (WS1 @ mGate_weights -> (WS2 * Silu(WS1)) -> WS1) (Non TMA multipy reduction)
        tma_GATE_inp, g_GATE_inp = cpasync.make_tiled_tma_atom(load_op,      mWS1_ME,      matmul_A_sw,  (self.bM, self.bK))
        tma_GATE_wt,  g_GATE_wt  = cpasync.make_tiled_tma_atom(load_op,      mGate_proj,   matmul_B_sw,  (1, self.bN, self.bK))
        tma_GATE_out, g_GATE_out = cpasync.make_tiled_tma_atom(store_op,     mWS2_MF_st,   matmul_C_pad, (self.bM, 1, self.bN + PAD))
        # DOWN Proj TMA Atoms (WS1 @ mDown_weights -> += Embedding) (Reduction)
        tma_DOWN_inp, g_DOWN_inp = cpasync.make_tiled_tma_atom(load_op,      mWS1_ME,      matmul_A_sw,  (self.bM, self.bK))
        tma_DOWN_wt,  g_DOWN_wt  = cpasync.make_tiled_tma_atom(load_op,      mDown_proj,   matmul_B_sw,  (1, self.bN, self.bK))
        tma_DOWN_out, g_DOWN_out = cpasync.make_tiled_tma_atom(store_op_red, mEmb_st,      matmul_C_pad, (self.bM, 1, self.bN + PAD))
        

        sO_tile = cute.make_ordered_layout(
            shape = (1, 1, self.bQ, self.head_dim + PAD),
            order = (3, 2, 1, 0)
        )
        tma_o, g_o = cpasync.make_tiled_tma_atom(store_op, mO, sO_tile, (1, 1, self.bQ, self.head_dim + PAD))

        self.rmsnorm = RMSNorm(self.rms_cfg)
        self.qkv     = Matmul(self.matmul_config, basic_store)
        self.out     = Matmul(self.matmul_config, residual_add_store)
        self.up      = Matmul(self.matmul_config, basic_store)
        self.gate    = Matmul(self.matmul_config, silu_mul)
        self.down    = Matmul(self.matmul_config, residual_add_store)
        self.attn    = Attention(self.attention_config)

        SharedStorage = self._get_shared_storage()

        self.kernel(mSchedule, mAtomics,
                    tma_RMS
                    SharedStorage).launch(
            grid=(self.num_sms,),
            block=(256,),
        )

    @cute.kernel
    def kernel(self,
        mSchedule: cute.Tensor,
        mAtomics: cute.Tensor,
        g_rms_w:  cute.Tensor,
        g_emb_rows: cute.Tensor,
        g_ws1_rows: cute.Tensor,
        g_ws1_A: cute.Tensor,
        g_qkv_w: cute.Tensor,
        g_ws2_qkv: cute.Tensor,
        g_out_w: cute.Tensor,
        g_emb_red: cute.Tensor,
        g_up_w: cute.Tensor,
        g_gate_w: cute.Tensor,
        g_ws2_ff_s: cute.Tensor,
        g_ws2_ff_A: cute.Tensor,
        g_down_w: cute.Tensor,
        mEmbedding: cute.Tensor,
        mWS2_MF: cute.Tensor,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        g_o: cute.Tensor,
        tma_rms_w: cute.CopyAtom,
        tma_emb_rows: cute.CopyAtom,
        tma_ws1_rows: cute.CopyAtom,
        tma_ws1_A: cute.CopyAtom,
        tma_qkv_w: cute.CopyAtom,
        tma_ws2_qkv: cute.CopyAtom,
        tma_out_w: cute.CopyAtom,
        tma_emb_red: cute.CopyAtom,
        tma_up_w: cute.CopyAtom,
        tma_gate_w: cute.CopyAtom,
        tma_ws2_ff_s: cute.CopyAtom,
        tma_ws2_ff_A: cute.CopyAtom,
        tma_down_w: cute.CopyAtom,
        tma_o: cute.CopyAtom,
        SharedStorage: cutlass.Constexpr
    ):
        warp_id    = cute.arch.warp_idx()
        group_id   = warp_id // 4
        local_warp = warp_id %  4
        tidx       = cute.arch.thread_idx()[0]
        group_tid  = tidx - group_id * 128
        block_id   = cute.arch.block_idx()[0]

        warpgroup = WarpgroupMeta(
            tidx       = tidx,
            group_tidx = group_tid,
            group_id   = group_id,
            lane_id    = tidx % 32,
            warp_id    = local_warp
        )

        if local_warp == 1:
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
        load_barriers    = storage.barriers.load_barrier.data_ptr()
        input_barriers   = storage.barriers.input_barrier.data_ptr()
        output_barriers  = storage.barriers.output_barrier.data_ptr()
        compute_barriers = storage.barriers.compute_barrier.data_ptr()
        load_stage       = storage.barriers.stage.get_tensor(cute.make_layout((1,)))
        load_phase_cell  = storage.barriers.phase.get_tensor(cute.make_layout((1,)))

        # initialize mbarriers
        if warp_id == 0:
            with cute.arch.elect_one():
                load_stage[0]      = 0
                load_phase_cell[0] = 0
                for i in cutlass.range_constexpr(self.num_stages):
                    cute.arch.mbarrier_init(load_barriers + i, 1)
                for i in cutlass.range_constexpr(2):
                    cute.arch.mbarrier_init(input_barriers + i  , 128)
                    cute.arch.mbarrier_init(output_barriers + i , 128)
                    cute.arch.mbarrier_init(compute_barriers + i, 128)
                cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        # arrive on warpgroup 0 barriers
        if group_id == 1:
            cute.arch.mbarrier_arrive(input_barriers + 0) 
            cute.arch.mbarrier_arrive(compute_barriers + 0)
            cute.arch.mbarrier_arrive(output_barriers + 0)

        phases = Phases(
            compute_phase = 0,
            input_phase = 0,
            output_phase = 0
        )

        max_works_local = (self.max_works + (1 - group_id)) // 2

        for local_work_idx in cutlass.range(max_works_local):
            work_idx = local_work_idx * 2 + group_id
            op           = mSchedule[block_id, work_idx, 0]
            pid_m        = mSchedule[block_id, work_idx, 1]
            pid_n        = mSchedule[block_id, work_idx, 2]
            pid_o        = mSchedule[block_id, work_idx, 3]
            expected_cnt = mSchedule[block_id, work_idx, 4]
            current_idx  = mSchedule[block_id, work_idx, 5]
            next_idx     = mSchedule[block_id, work_idx, 6]
            op_kind   = op & 0x7
            layer_idx = op >> 3

            pipeline = PipelineMeta(
                current_idx = current_idx,
                expected_cnt = expected_cnt,
                next_idx = next_idx
            )

            if op_kind == int(Op.RMS):
                rms_w_idx = layer_idx * 2 + pid_o
                self.rmsnorm.run(g_emb_rows, g_ws1_rows, g_rms_w, tma_rms_w, tma_emb_rows, tma_ws1_rows,
                                 rms_w_idx, pid_m, mAtomics, pipeline,
                                 storage, pid_n, warpgroup, phases)
            elif op_kind == int(Op.QKV):
                self.qkv.run(g_ws1_A, g_qkv_w, g_ws2_qkv, None, tma_ws1_A, tma_qkv_w, tma_ws2_qkv,
                             layer_idx, pid_m, pid_n, mAtomics, pipeline,
                             storage, warpgroup, phases)
            elif op_kind == int(Op.ATTN):
                self.attn.run(mQ, mK, mV, tma_o, g_o,
                              pid_m, pid_n, pid_o, mAtomics, pipeline,
                              storage, warpgroup, phases)
            elif op_kind == int(Op.OUT):
                self.out.run(g_ws1_A, g_out_w, g_emb_red, mEmbedding, tma_ws1_A, tma_out_w, tma_emb_red,
                             layer_idx, pid_m, pid_n, mAtomics, pipeline,
                             storage, warpgroup, phases)
            elif op_kind == int(Op.UP):
                self.up.run(g_ws1_A, g_up_w, g_ws2_ff_s, None, tma_ws1_A, tma_up_w, tma_ws2_ff_s,
                            layer_idx, pid_m, pid_n, mAtomics, pipeline,
                            storage, warpgroup, phases)
            elif op_kind == int(Op.GATE):
                self.gate.run(g_ws1_A, g_gate_w, g_ws2_ff_s, mWS2_MF, tma_ws1_A, tma_gate_w, tma_ws2_ff_s,
                              layer_idx, pid_m, pid_n, mAtomics, pipeline,
                              storage, warpgroup, phases)
            elif op_kind == int(Op.DOWN):
                self.down.run(g_ws2_ff_A, g_down_w, g_emb_red, mEmbedding, tma_ws2_ff_A, tma_down_w, tma_emb_red,
                              layer_idx, pid_m, pid_n, mAtomics, pipeline,
                              storage, warpgroup, phases)

            phases.compute_phase = phases.compute_phase ^ 1
            phases.input_phase   = phases.input_phase   ^ 1
            phases.output_phase  = phases.output_phase  ^ 1