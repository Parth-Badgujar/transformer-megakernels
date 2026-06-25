import os
os.environ["CUTE_DSL_LINEINFO"] = "1"

import math
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
from cutlass.utils import SmemAllocator
from cutlass.cute.runtime import from_dlpack

from transformer_megakernel.operators.rmsnorm import RMSNorm, RMSNormConfig
from transformer_megakernel.operators.matmul import Matmul, MatmulConfig
from transformer_megakernel.operators.attention import Attention, AttentionConfig
from transformer_megakernel.operators.epilogues import basic_store, residual_add_store, silu_mul

from transformer_megakernel.scheduler import OpScheduler
from transformer_megakernel.model import extract_weights, Transformer
from transformer_megakernel.config import InputConfig, KernelConfig, Op
from transformer_megakernel.kernel_utils import (
    WarpgroupMeta, Phases, PipelineMeta,
    PROBE_HEADER, PROBE_ENTRY, MAX_TAG_SLOTS, NUM_PROBE_ROLES,
    ROLE_NAMES, TAG_NAMES, TAGS,
    dump_probe,
)

from cutlass import Int32, BFloat16, Uint64

# Required probe tensor column count: PROBE_HEADER + MAX_TAG_SLOTS * PROBE_ENTRY
PROBE_COLS = PROBE_HEADER + MAX_TAG_SLOTS * PROBE_ENTRY  # = 1 + 16*4 = 65


class Megakernel:
    def __init__(self, input_config: InputConfig, kernel_config: KernelConfig,
                 profile: bool = False):
        self.profile      = profile
        self.embed_dim    = input_config.embed_dim
        self.kv_len       = input_config.kv_len
        self.q_len        = input_config.q_len
        self.num_q_heads  = input_config.num_q_heads
        self.num_kv_heads = input_config.num_kv_heads
        self.num_layers   = input_config.num_layers
        self.ff_dim       = input_config.ff_dim
        self.num_stages   = kernel_config.num_stages
        self.head_dim     = input_config.embed_dim // input_config.num_q_heads
        self.bM           = kernel_config.bM
        self.bN           = kernel_config.bN
        self.bK           = kernel_config.bK
        self.bs           = input_config.bs
        self.num_sms      = kernel_config.num_sms
        self.max_works    = kernel_config.max_works
        self.warps_per_row = kernel_config.warps_per_row
        self.num_sets      = 4 // self.warps_per_row
        self.rows_per_rms_block = kernel_config.rows_per_rms_block
        self.qkv_out_dim  = (input_config.num_q_heads + 2 * input_config.num_kv_heads) * self.head_dim
        self.num_tokens   = input_config.bs * input_config.q_len
        self.bQ           = kernel_config.block_q
        self.bKV          = kernel_config.block_kv
        self.use_tma_reduce = kernel_config.use_tma_reduce
        self.output_pad     = kernel_config.output_pad

        # one stage section must hold the largest of: matmul A+B, attn K+V, rms set
        self.stage_elements = max(
            self.bM * self.bK + self.bN * self.bK,
            2 * self.bKV * self.head_dim,
            self.num_sets * self.embed_dim,
        )

        self.rms_config = RMSNormConfig(
            embed_dim          = self.embed_dim,
            bRMS               = self.rows_per_rms_block,
            stage_elements     = self.stage_elements,
            warps_per_row      = self.warps_per_row,
        )

        self.matmul_config = MatmulConfig(
            bM = self.bM,
            bN = self.bN,
            bK = self.bK,
            num_stages = self.num_stages,
            stage_elements = self.stage_elements,
            use_tma_reduce = self.use_tma_reduce,
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
            stage_elements = self.stage_elements,
            output_pad = self.output_pad,
            is_causal = input_config.is_causal
        )
    @cute.jit
    def _get_shared_storage(self):
        num_out_elements = max(
            self.bM * (self.bN + self.output_pad),
            self.bQ * (self.head_dim + self.output_pad),
            self.num_sets * self.embed_dim * self.num_stages,
        )

        @cute.struct
        class BarrierStorage:
            load_barrier:    cute.struct.MemRange[Uint64, self.num_stages]
            input_barrier:   cute.struct.MemRange[Uint64, 2]
            output_barrier:  cute.struct.MemRange[Uint64, 2]
            compute_barrier: cute.struct.MemRange[Uint64, 2]
            stage:           cute.struct.MemRange[Int32, 1]
            phase:           cute.struct.MemRange[Int32, 1]

        @cute.struct
        class SharedStorage:
            barriers: BarrierStorage
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
        mProbe: cute.Tensor
    ):
        q_layout  = cute.make_layout(
            shape  = (self.bs, self.num_q_heads,  self.q_len, self.head_dim),
            stride = (self.q_len * self.qkv_out_dim, self.head_dim, self.qkv_out_dim, 1),
        )
        kv_layout = cute.make_layout(
            shape  = (self.bs, self.num_kv_heads, self.kv_len, self.head_dim),
            stride = (self.kv_len * self.qkv_out_dim, self.head_dim, self.qkv_out_dim, 1),
        )

        k_off = self.num_q_heads * self.head_dim
        v_off = (self.num_q_heads + self.num_kv_heads) * self.head_dim

        # QKV Split Layout
        mQ = cute.make_tensor(mWorkspace2.iterator, q_layout)
        mK = cute.make_tensor(mWorkspace2.iterator + k_off, kv_layout)
        mV = cute.make_tensor(mWorkspace2.iterator + v_off, kv_layout)

        mAttn_out = cute.make_tensor(
            mWorkspace1.iterator, cute.make_ordered_layout(
                shape = (self.bs, self.num_q_heads, self.q_len, self.head_dim),
                order = (3, 1, 2, 0),
            )
        )
        mQKV_act = cute.make_tensor(
            mWorkspace2.iterator, cute.make_ordered_layout(
                shape = (self.num_tokens, self.qkv_out_dim // self.bN, self.bN),
                order = (2, 1, 0),
            )
        )
        mFF_hidden = cute.make_tensor(
            mWorkspace2.iterator,
            cute.make_ordered_layout(shape = (self.num_tokens, self.ff_dim), order = (1, 0)),
        )
        mFF_hidden_mm = cute.make_tensor(
            mWorkspace2.iterator, cute.make_ordered_layout(
                shape = (self.num_tokens, self.ff_dim // self.bN, self.bN),
                order = (2, 1, 0),
            )
        )

        mEmbed_st = cute.make_tensor(
            mEmbedding.iterator, cute.make_ordered_layout(
                shape = (self.num_tokens, self.embed_dim // self.bN, self.bN),
                order = (2, 1, 0),
            )
        )
        mWS1_embed = cute.make_tensor(
            mWorkspace1.iterator,
            cute.make_ordered_layout(shape = (self.num_tokens, self.embed_dim), order = (1, 0)),
        )

        matmul_A_sw = cute.make_composed_layout(
            cute.make_swizzle(int(math.log2(self.bK)) - 3, 4, 3), 0,
            cute.make_ordered_layout(
                shape = (self.bM, self.bK),
                order = (1, 0)
            ),
        )
        matmul_B_sw = cute.make_composed_layout(
            cute.make_swizzle(int(math.log2(self.bK)) - 3, 4, 3), 0,
            cute.make_ordered_layout(
                shape = (1, self.bN, self.bK),
                order = (2, 1, 0)
            ),
        )
        matmul_C_pad = cute.make_ordered_layout(
            shape = (self.bM, 1, self.bN + self.output_pad), order = (2, 1, 0),
        )

        # RMS activations are tiled by num_sets rows (V2: bR replaced by num_sets)
        embed_act = cute.make_ordered_layout(
            shape = (self.num_sets, self.embed_dim),
            order = (1, 0),
        )
        attn_out = cute.make_ordered_layout(
            shape = (1, 1, self.bQ, self.head_dim + self.output_pad),
            order = (3, 2, 1, 0),
        )

        '''
        TMA Atoms Naming Convention
        g_<Op>_<Tensor Name>
        tma_<Op>_<Tensor Name>
        '''
        load_op  = cpasync.CopyBulkTensorTileG2SOp()
        store_op = cpasync.CopyBulkTensorTileS2GOp()
        if cutlass.const_expr(self.use_tma_reduce):
            store_op_red = cpasync.CopyReduceBulkTensorTileS2GOp(cute.ReductionKind.ADD)
        else:
            store_op_red = store_op

        # RMS TMA Atoms (Embedding -> WS1).  NO weight TMA in V2: weights load
        # gmem->rmem inside the op, mRMS_weights is passed through directly.
        tma_RMS_inp, g_RMS_inp   = cpasync.make_tiled_tma_atom(load_op,  mEmbedding, embed_act, (self.num_sets, self.embed_dim))
        tma_RMS_out, g_RMS_out   = cpasync.make_tiled_tma_atom(store_op, mWS1_embed, embed_act, (self.num_sets, self.embed_dim))
        # Attn Out Store (MHA Output -> WS1)
        tma_ATTN_out, g_ATTN_out = cpasync.make_tiled_tma_atom(store_op, mAttn_out, attn_out, (1, 1, self.bQ, self.head_dim + self.output_pad))
        # QKV (WS1 @ QKV_w -> WS2)
        tma_QKV_inp, g_QKV_inp   = cpasync.make_tiled_tma_atom(load_op,  mWS1_embed, matmul_A_sw,  (self.bM, self.bK))
        tma_QKV_wt,  g_QKV_wt    = cpasync.make_tiled_tma_atom(load_op,  mQKV_proj,  matmul_B_sw,  (1, self.bN, self.bK))
        tma_QKV_act, g_QKV_act   = cpasync.make_tiled_tma_atom(store_op, mQKV_act,   matmul_C_pad, (self.bM, 1, self.bN + self.output_pad))
        # OUT Proj (WS1 @ Out_w -> += Embedding)
        tma_OUT_inp, g_OUT_inp   = cpasync.make_tiled_tma_atom(load_op,      mWS1_embed,   matmul_A_sw,  (self.bM, self.bK))
        tma_OUT_wt,  g_OUT_wt    = cpasync.make_tiled_tma_atom(load_op,      mOutProjAttn, matmul_B_sw,  (1, self.bN, self.bK))
        tma_OUT_out, g_OUT_out   = cpasync.make_tiled_tma_atom(store_op_red, mEmbed_st,    matmul_C_pad, (self.bM, 1, self.bN + self.output_pad))
        g_OUT_out_nt = mEmbedding
        # UP (WS1 @ Up_w -> WS2)
        tma_UP_inp, g_UP_inp     = cpasync.make_tiled_tma_atom(load_op,  mWS1_embed,    matmul_A_sw,  (self.bM, self.bK))
        tma_UP_wt,  g_UP_wt      = cpasync.make_tiled_tma_atom(load_op,  mUp_proj,      matmul_B_sw,  (1, self.bN, self.bK))
        tma_UP_out, g_UP_out     = cpasync.make_tiled_tma_atom(store_op, mFF_hidden_mm, matmul_C_pad, (self.bM, 1, self.bN + self.output_pad))
        # GATE (WS1 @ Gate_w -> SiLU * UP -> WS2)
        tma_GATE_inp, g_GATE_inp = cpasync.make_tiled_tma_atom(load_op,  mWS1_embed,    matmul_A_sw,  (self.bM, self.bK))
        tma_GATE_wt,  g_GATE_wt  = cpasync.make_tiled_tma_atom(load_op,  mGate_proj,    matmul_B_sw,  (1, self.bN, self.bK))
        tma_GATE_out, g_GATE_out = cpasync.make_tiled_tma_atom(store_op, mFF_hidden_mm, matmul_C_pad, (self.bM, 1, self.bN + self.output_pad))
        g_GATE_gate = mFF_hidden  # Non TMA
        # DOWN (FF @ Down_w -> += Embedding)
        tma_DOWN_inp, g_DOWN_inp = cpasync.make_tiled_tma_atom(load_op,      mFF_hidden, matmul_A_sw,  (self.bM, self.bK))
        tma_DOWN_wt,  g_DOWN_wt   = cpasync.make_tiled_tma_atom(load_op,     mDown_proj, matmul_B_sw,  (1, self.bN, self.bK))
        tma_DOWN_out, g_DOWN_out = cpasync.make_tiled_tma_atom(store_op_red, mEmbed_st,  matmul_C_pad, (self.bM, 1, self.bN + self.output_pad))
        g_DOWN_out_nt = mEmbedding

        self.rmsnorm = RMSNorm(self.rms_config,  profile=self.profile)
        self.qkv     = Matmul(self.matmul_config, basic_store,          profile=self.profile)
        self.out     = Matmul(self.matmul_config, residual_add_store,   profile=self.profile)
        self.up      = Matmul(self.matmul_config, basic_store,          profile=self.profile)
        self.gate    = Matmul(self.matmul_config, silu_mul,             profile=self.profile)
        self.down    = Matmul(self.matmul_config, residual_add_store,   profile=self.profile)
        self.attn    = Attention(self.attention_config,                  profile=self.profile)

        SharedStorage = self._get_shared_storage()

        self.kernel(mProbe, mSchedule, mAtomics,
                g_RMS_inp, g_RMS_out, mRMS_weights,
                g_QKV_inp, g_QKV_act, g_QKV_wt,
                g_OUT_inp, g_OUT_out, g_OUT_wt, g_OUT_out_nt,
                g_UP_inp, g_UP_out, g_UP_wt,
                g_GATE_inp, g_GATE_out, g_GATE_wt, g_GATE_gate,
                g_DOWN_inp, g_DOWN_out, g_DOWN_wt, g_DOWN_out_nt,
                tma_RMS_inp,  tma_RMS_out,
                tma_QKV_inp,  tma_QKV_act,  tma_QKV_wt,
                tma_OUT_inp,  tma_OUT_out,  tma_OUT_wt,
                tma_UP_inp,   tma_UP_out,   tma_UP_wt,
                tma_GATE_inp, tma_GATE_out, tma_GATE_wt,
                tma_DOWN_inp, tma_DOWN_out, tma_DOWN_wt,
                mQ, mK, mV,
                tma_ATTN_out, g_ATTN_out,
                SharedStorage).launch(
            grid=(self.num_sms,),
            block=(256,),
        )

    @cute.kernel
    def kernel(self,
        mProbe: cute.Tensor,
        mSchedule: cute.Tensor,
        mAtomics: cute.Tensor,
        g_RMS_inp: cute.Tensor, g_RMS_out: cute.Tensor, mRMS_weights: cute.Tensor,
        g_QKV_inp: cute.Tensor, g_QKV_act: cute.Tensor, g_QKV_wt: cute.Tensor,
        g_OUT_inp: cute.Tensor, g_OUT_out: cute.Tensor, g_OUT_wt: cute.Tensor, g_OUT_out_nt: cute.Tensor,
        g_UP_inp: cute.Tensor, g_UP_out: cute.Tensor, g_UP_wt: cute.Tensor,
        g_GATE_inp: cute.Tensor, g_GATE_out: cute.Tensor, g_GATE_wt: cute.Tensor, g_GATE_gate: cute.Tensor,
        g_DOWN_inp: cute.Tensor, g_DOWN_out: cute.Tensor, g_DOWN_wt: cute.Tensor, g_DOWN_out_nt: cute.Tensor,
        tma_RMS_inp: cute.CopyAtom,  tma_RMS_out: cute.CopyAtom,
        tma_QKV_inp: cute.CopyAtom,  tma_QKV_act: cute.CopyAtom,  tma_QKV_wt: cute.CopyAtom,
        tma_OUT_inp: cute.CopyAtom,  tma_OUT_out: cute.CopyAtom,  tma_OUT_wt: cute.CopyAtom,
        tma_UP_inp: cute.CopyAtom,   tma_UP_out: cute.CopyAtom,   tma_UP_wt: cute.CopyAtom,
        tma_GATE_inp: cute.CopyAtom, tma_GATE_out: cute.CopyAtom, tma_GATE_wt: cute.CopyAtom,
        tma_DOWN_inp: cute.CopyAtom, tma_DOWN_out: cute.CopyAtom, tma_DOWN_wt: cute.CopyAtom,
        mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor,
        tma_ATTN_out: cute.CopyAtom, g_ATTN_out: cute.Tensor,
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
            warp_id    = local_warp,
        )

        # Not sure if it helps
        if local_warp == 1:
            cpasync.prefetch_descriptor(tma_RMS_inp)
            cpasync.prefetch_descriptor(tma_RMS_out)
            cpasync.prefetch_descriptor(tma_QKV_inp)
            cpasync.prefetch_descriptor(tma_QKV_act)
            cpasync.prefetch_descriptor(tma_QKV_wt)
            cpasync.prefetch_descriptor(tma_OUT_inp)
            cpasync.prefetch_descriptor(tma_OUT_out)
            cpasync.prefetch_descriptor(tma_OUT_wt)
            cpasync.prefetch_descriptor(tma_UP_inp)
            cpasync.prefetch_descriptor(tma_UP_out)
            cpasync.prefetch_descriptor(tma_UP_wt)
            cpasync.prefetch_descriptor(tma_GATE_inp)
            cpasync.prefetch_descriptor(tma_GATE_out)
            cpasync.prefetch_descriptor(tma_GATE_wt)
            cpasync.prefetch_descriptor(tma_DOWN_inp)
            cpasync.prefetch_descriptor(tma_DOWN_out)
            cpasync.prefetch_descriptor(tma_DOWN_wt)
            cpasync.prefetch_descriptor(tma_ATTN_out)

        smem    = SmemAllocator()
        storage = smem.allocate(SharedStorage)
        load_barriers    = storage.barriers.load_barrier.data_ptr()
        input_barriers   = storage.barriers.input_barrier.data_ptr()
        output_barriers  = storage.barriers.output_barrier.data_ptr()
        compute_barriers = storage.barriers.compute_barrier.data_ptr()
        load_stage       = storage.barriers.stage.get_tensor(cute.make_layout((1,)))
        load_phase_cell  = storage.barriers.phase.get_tensor(cute.make_layout((1,)))

        if warp_id == 0:
            with cute.arch.elect_one():
                load_stage[0]      = 0
                load_phase_cell[0] = 0
                for i in cutlass.range_constexpr(self.num_stages):
                    cute.arch.mbarrier_init(load_barriers + i, 1)
                for i in cutlass.range_constexpr(2):
                    cute.arch.mbarrier_init(input_barriers + i,   128)
                    cute.arch.mbarrier_init(output_barriers + i,  128)
                    cute.arch.mbarrier_init(compute_barriers + i, 128)
                cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

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
                current_idx  = current_idx,
                expected_cnt = expected_cnt,
                next_idx     = next_idx,
            )

            if op_kind == int(Op.RMS):
                rms_w_idx = layer_idx * 2 + pid_o
                self.rmsnorm.run(
                    g_RMS_inp, g_RMS_out, mRMS_weights,
                    tma_RMS_inp, tma_RMS_out,
                    rms_w_idx, pid_m, 
                    mAtomics, pipeline, phases,
                    warpgroup, storage,
                    mProbe
                )
            elif op_kind == int(Op.QKV):
                self.qkv.run(
                    g_QKV_inp, g_QKV_wt, g_QKV_act, None,
                    tma_QKV_inp, tma_QKV_wt, tma_QKV_act,
                    layer_idx, pid_m, pid_n,
                    mAtomics, pipeline, phases,
                    warpgroup, storage,
                    mProbe
                )
            elif op_kind == int(Op.ATTN):
                self.attn.run(
                    mQ, mK, mV, g_ATTN_out, tma_ATTN_out,
                    pid_m, pid_n, pid_o,
                    mAtomics, pipeline, phases,
                    warpgroup, storage,
                    mProbe
                )
            elif op_kind == int(Op.OUT):
                self.out.run(
                    g_OUT_inp, g_OUT_wt, g_OUT_out, g_OUT_out_nt,
                    tma_OUT_inp, tma_OUT_wt, tma_OUT_out,
                    layer_idx, pid_m, pid_n,
                    mAtomics, pipeline, phases,
                    warpgroup, storage,
                    mProbe
                )
            elif op_kind == int(Op.UP):
                self.up.run(
                    g_UP_inp, g_UP_wt, g_UP_out, None,
                    tma_UP_inp, tma_UP_wt, tma_UP_out,
                    layer_idx, pid_m, pid_n,
                    mAtomics, pipeline, phases,
                    warpgroup, storage,
                    mProbe
                )
            elif op_kind == int(Op.GATE):
                self.gate.run(
                    g_GATE_inp, g_GATE_wt, g_GATE_out, g_GATE_gate,
                    tma_GATE_inp, tma_GATE_wt, tma_GATE_out,
                    layer_idx, pid_m, pid_n,
                    mAtomics, pipeline, phases,
                    warpgroup, storage,
                    mProbe
                )
            elif op_kind == int(Op.DOWN):
                self.down.run(
                    g_DOWN_inp, g_DOWN_wt, g_DOWN_out, g_DOWN_out_nt,
                    tma_DOWN_inp, tma_DOWN_wt, tma_DOWN_out,
                    layer_idx, pid_m, pid_n,
                    mAtomics, pipeline, phases,
                    warpgroup, storage,
                    mProbe
                )

            phases.compute_phase = phases.compute_phase ^ 1
            phases.input_phase   = phases.input_phase   ^ 1
            phases.output_phase  = phases.output_phase  ^ 1


class TransformerMegakernel:
    def __init__(self, model: Transformer, input_config: InputConfig,
                 kernel_config: KernelConfig, profile: bool = False):
        self.profile    = profile
        self.num_sms    = kernel_config.num_sms

        # 1. Extract weights from model
        rms_w, qkv_w, out_w, gate_w, up_w, down_w = extract_weights(model)
        # 2. Build schedule
        scheduler = OpScheduler(input_config, kernel_config)
        sched, atoms, max_works = scheduler.build_schedule()
        kernel_config.max_works = max_works

        self.mSchedule = sched.cuda()
        self.mAtomics  = atoms.cuda()

        # 3. Create DLPack views
        self.cSchedule = from_dlpack(self.mSchedule)
        self.cAtomics  = from_dlpack(self.mAtomics)
        self.cRms_w    = from_dlpack(rms_w,  assumed_align = 16)
        self.cQkv_w    = from_dlpack(qkv_w,  assumed_align = 16)
        self.cGate_w   = from_dlpack(gate_w, assumed_align = 16)
        self.cUp_w     = from_dlpack(up_w,   assumed_align = 16)
        self.cDown_w   = from_dlpack(down_w, assumed_align = 16)
        self.cOut_w    = from_dlpack(out_w,  assumed_align = 16)

        # 4. Allocate workspaces
        self.num_tokens = input_config.bs * input_config.q_len
        self.qkv_dim    = (input_config.num_q_heads + 2 * input_config.num_kv_heads) * (input_config.embed_dim // input_config.num_q_heads)
        self.ws2_dim    = max(self.qkv_dim, input_config.ff_dim)

        self.ws1 = torch.zeros((self.num_tokens, input_config.embed_dim), dtype = torch.bfloat16, device="cuda")
        self.ws2 = torch.zeros((self.num_tokens, self.ws2_dim), dtype = torch.bfloat16, device="cuda")
        self.cWs1 = from_dlpack(self.ws1, assumed_align = 16)
        self.cWs2 = from_dlpack(self.ws2, assumed_align = 16)

        # 5. Allocate probe tensor (always, but only populated when profile=True)
        #    Shape: (num_sms * NUM_PROBE_ROLES, PROBE_COLS)  — int64 so timestamps fit
        num_probe_rows = kernel_config.num_sms * NUM_PROBE_ROLES
        self.mProbe = torch.zeros(
            (num_probe_rows, PROBE_COLS), dtype=torch.int64, device="cuda"
        )
        self.cProbe = from_dlpack(self.mProbe, assumed_align = 16)

        # 6. Compile the kernel
        self.kernel = Megakernel(input_config, kernel_config, profile=profile)

        # Dummy compilation embedding
        dummy_emb = torch.empty((self.num_tokens, input_config.embed_dim), dtype=torch.bfloat16, device="cuda")
        cDummyEmb = from_dlpack(dummy_emb, assumed_align = 16)

        self.compiled_kernel = cute.compile(
            self.kernel,
            self.cSchedule, self.cAtomics,
            self.cRms_w, self.cQkv_w, self.cWs1, self.cWs2,
            self.cGate_w, self.cUp_w, self.cDown_w, self.cOut_w,
            cDummyEmb, self.cProbe
        )

    def __call__(self, input_embedding: torch.Tensor,
                 trace_path: str = "pipeline_trace.json"):
        self.mAtomics.zero_()
        if self.profile:
            self.mProbe.zero_()
        output_embedding = input_embedding.clone()
        cEmbedding = from_dlpack(output_embedding, assumed_align = 16)

        self.compiled_kernel(
            self.cSchedule, self.cAtomics,
            self.cRms_w, self.cQkv_w, self.cWs1, self.cWs2,
            self.cGate_w, self.cUp_w, self.cDown_w, self.cOut_w,
            cEmbedding, self.cProbe
        )

        if self.profile:
            torch.cuda.synchronize()
            dump_probe(self.mProbe, self.num_sms, out_path=trace_path)

        return output_embedding