import math
from collections import defaultdict

import torch

from transformer_megakernel.config import Op, InputConfig, KernelConfig


class OpScheduler:
    def __init__(self, input_config: InputConfig, kernel_config: KernelConfig):
        self.num_sms = kernel_config.num_sms
        self.schedule = [[] for _ in range(self.num_sms)]
        self.atomics = []

        self.bs = input_config.bs
        self.q_len = input_config.q_len
        self.M = self.bs * self.q_len

        self.E = input_config.embed_dim
        self.num_q_heads = input_config.num_q_heads
        self.num_kv_heads = input_config.num_kv_heads
        self.head_dim = input_config.embed_dim // input_config.num_q_heads
        self.Qd = (input_config.num_q_heads + 2 * input_config.num_kv_heads) * self.head_dim
        self.F = input_config.ff_dim
        self.num_layers = input_config.num_layers

        self.bM = kernel_config.bM
        self.bN = kernel_config.bN
        self.block_q = kernel_config.block_q

        self.Mout = int(math.ceil(self.M / self.bM))
        assert self.q_len % self.bM == 0, "bM must divide q_len"

        self.rows_per_rms_block = kernel_config.rows_per_rms_block
        assert self.bM % self.rows_per_rms_block == 0, (
            "rows_per_rms_block must divide bM"
        )
        self.tpg = self.bM // self.rows_per_rms_block
        self.Mtiles = self.Mout * self.tpg

    def _get_prev_sm(self):
        for i in range(len(self.schedule)):
            if len(self.schedule[i]) > len(self.schedule[(i + 1) % self.num_sms]):
                return (i + 1) % self.num_sms
        return 0

    def _compute_pid(self, block_id, total_pid, total_pid_m, group_size_m = 8):
        total_pid_n = total_pid // total_pid_m
        gsm = min(group_size_m, total_pid_m)
        grp = gsm * total_pid_n
        g = block_id // grp
        fm = g * gsm
        sz = min(total_pid_m - fm, gsm)
        pid_m = fm + block_id % sz
        pid_n = (block_id % grp) // sz
        return pid_m, pid_n

    def _enc(self, layer, op):
        return (layer << 3) | int(op.value)

    def schedule_rms1(self, layer, in_idx):
        out_idx = len(self.atomics)
        self.atomics.extend([0] * int(self.Mout))

        Nout = int(math.ceil(self.E / self.bN))
        expected_cnt = 0 if layer == 0 else Nout

        sm_id = self._get_prev_sm()
        for tile_idx in range(self.Mtiles):
            rg = tile_idx // self.tpg
            self.schedule[sm_id].append(
                [
                    self._enc(layer, Op.RMS),
                    tile_idx,
                    tile_idx // self.num_sms,
                    0,
                    expected_cnt,
                    in_idx + rg,
                    out_idx + rg,
                ]
            )
            sm_id = (sm_id + 1) % self.num_sms
        return out_idx

    def schedule_qkv(self, layer, in_idx):
        out_idx = len(self.atomics)
        self.atomics.extend([0] * int(self.bs))

        total_pid_n = int(math.ceil(self.Qd / self.bN))
        total_pid = self.Mout * total_pid_n

        sm_id = self._get_prev_sm()
        for block_id in range(total_pid):
            pid_m, pid_n = self._compute_pid(block_id, total_pid, self.Mout)
            batch_idx = (pid_m * self.bM) // self.q_len
            self.schedule[sm_id].append(
                [
                    self._enc(layer, Op.QKV),
                    pid_m,
                    pid_n,
                    0,
                    self.tpg,
                    in_idx + pid_m,
                    out_idx + batch_idx,
                ]
            )
            sm_id = (sm_id + 1) % self.num_sms
        return out_idx

    def schedule_attn(self, layer, in_idx):
        out_idx = len(self.atomics)
        self.atomics.extend([0] * int(self.bs))

        Mpm = int(math.ceil(self.q_len / self.bM))
        Nqkv = int(math.ceil(self.Qd / self.bN))
        expected_cnt = Mpm * Nqkv
        Qblks = int(math.ceil(self.q_len / self.block_q))

        sm_id = self._get_prev_sm()
        for b in range(self.bs):
            for h in range(self.num_q_heads):
                for qb in range(Qblks):
                    self.schedule[sm_id].append(
                        [
                            self._enc(layer, Op.ATTN),
                            b,
                            h,
                            qb,
                            expected_cnt,
                            in_idx + b,
                            out_idx + b,
                        ]
                    )
                    sm_id = (sm_id + 1) % self.num_sms
        return out_idx

    def schedule_out(self, layer, in_idx):
        out_idx = len(self.atomics)
        self.atomics.extend([0] * int(self.Mout))

        total_pid_n = int(math.ceil(self.E / self.bN))
        total_pid = self.Mout * total_pid_n
        Qblks = int(math.ceil(self.q_len / self.block_q))
        expected_cnt = self.num_q_heads * Qblks

        sm_id = self._get_prev_sm()
        for block_id in range(total_pid):
            pid_m, pid_n = self._compute_pid(block_id, total_pid, self.Mout)
            batch_idx = (pid_m * self.bM) // self.q_len
            self.schedule[sm_id].append(
                [
                    self._enc(layer, Op.OUT),
                    pid_m,
                    pid_n,
                    0,
                    expected_cnt,
                    in_idx + batch_idx,
                    out_idx + pid_m,
                ]
            )
            sm_id = (sm_id + 1) % self.num_sms
        return out_idx

    def schedule_rms2(self, layer, in_idx):
        out_idx = len(self.atomics)
        self.atomics.extend([0] * int(self.Mout))

        Nout = int(math.ceil(self.E / self.bN))
        expected_cnt = Nout

        sm_id = self._get_prev_sm()
        for tile_idx in range(self.Mtiles):
            rg = tile_idx // self.tpg
            self.schedule[sm_id].append(
                [
                    self._enc(layer, Op.RMS),
                    tile_idx,
                    tile_idx // self.num_sms,
                    1,
                    expected_cnt,
                    in_idx + rg,
                    out_idx + rg,
                ]
            )
            sm_id = (sm_id + 1) % self.num_sms
        return out_idx

    def schedule_up(self, layer, in_idx):
        Nf = int(math.ceil(self.F / self.bN))
        out_idx = len(self.atomics)
        self.atomics.extend([0] * int(self.Mout * Nf))

        total_pid = self.Mout * Nf
        sm_id = self._get_prev_sm()
        for block_id in range(total_pid):
            pid_m, pid_n = self._compute_pid(block_id, total_pid, self.Mout)
            self.schedule[sm_id].append(
                [
                    self._enc(layer, Op.UP),
                    pid_m,
                    pid_n,
                    0,
                    self.tpg,
                    in_idx + pid_m,
                    out_idx + (pid_m * Nf + pid_n),
                ]
            )
            sm_id = (sm_id + 1) % self.num_sms
        return out_idx

    def schedule_gate(self, layer, in_idx):
        Nf = int(math.ceil(self.F / self.bN))
        out_idx = len(self.atomics)
        self.atomics.extend([0] * int(self.Mout))

        total_pid = self.Mout * Nf
        sm_id = self._get_prev_sm()
        for block_id in range(total_pid):
            pid_m, pid_n = self._compute_pid(block_id, total_pid, self.Mout)
            self.schedule[sm_id].append(
                [
                    self._enc(layer, Op.GATE),
                    pid_m,
                    pid_n,
                    0,
                    1,
                    in_idx + (pid_m * Nf + pid_n),
                    out_idx + pid_m,
                ]
            )
            sm_id = (sm_id + 1) % self.num_sms
        return out_idx

    def schedule_down(self, layer, in_idx):
        Nout = int(math.ceil(self.E / self.bN))
        Nf = int(math.ceil(self.F / self.bN))
        out_idx = len(self.atomics)
        self.atomics.extend([0] * int(self.Mout))

        total_pid = self.Mout * Nout
        sm_id = self._get_prev_sm()
        for block_id in range(total_pid):
            pid_m, pid_n = self._compute_pid(block_id, total_pid, self.Mout)
            self.schedule[sm_id].append(
                [
                    self._enc(layer, Op.DOWN),
                    pid_m,
                    pid_n,
                    0,
                    Nf,
                    in_idx + pid_m,
                    out_idx + pid_m,
                ]
            )
            sm_id = (sm_id + 1) % self.num_sms
        return out_idx

    def _align(self):
        max_works = max(len(sm) for sm in self.schedule) if self.schedule else 0
        for sm in self.schedule:
            while len(sm) < max_works:
                sm.append([-1, -1, -1, -1, 0, -1, -1])
        return max_works

    def build_schedule(self):
        in_idx = len(self.atomics)
        self.atomics.extend([0] * int(self.Mout))

        for l in range(self.num_layers):
            in_idx = self.schedule_rms1(l, in_idx)
            in_idx = self.schedule_qkv(l, in_idx)
            in_idx = self.schedule_attn(l, in_idx)
            in_idx = self.schedule_out(l, in_idx)
            in_idx = self.schedule_rms2(l, in_idx)
            in_idx = self.schedule_up(l, in_idx)
            in_idx = self.schedule_gate(l, in_idx)
            in_idx = self.schedule_down(l, in_idx)

        self.atomics.append(0)
        max_works = self._align()
        return (
            torch.tensor(self.schedule, dtype=torch.int32),
            torch.tensor(self.atomics, dtype=torch.int32),
            max_works,
        )


def get_attn_schedule(config):
    return OpScheduler(config).build_schedule()


if __name__ == "__main__":
    import sys
    pass
    # cfg = KernelConfig(
    #     embed_dim=512,
    #     kv_len=256,
    #     q_len=256,
    #     num_q_heads=4,
    #     num_kv_heads=4,
    #     num_layers=4,
    #     ff_dim=1024,
    #     block_rms=1,
    #     block_q=64,
    #     block_kv=64,
    #     num_stages=2,
    #     bM=128,
    #     bN=128,
    #     bK=64,
    #     bs=16,
    #     num_sms=188,
    #     output_pad=8,
    #     warps_per_row=1,
    #     rows_per_rms_block=int(sys.argv[1]) if len(sys.argv) > 1 else 32,
    # )

    # scheduler = OpScheduler(cfg)
    # schedule_tensor, atomics_tensor, max_works = scheduler.build_schedule()
    # print(
    #     f"M={scheduler.M}, bM={scheduler.bM}, "
    #     f"rows_per_rms_block={scheduler.rows_per_rms_block}, "
    #     f"tpg={scheduler.tpg}, Mout={scheduler.Mout}, "
    #     f"Mtiles_per_RMS={scheduler.Mtiles}"
    # )
    # print(
    #     f"num_sms={scheduler.num_sms}, max_works_per_SM={max_works}, "
    #     f"total_atomics={len(scheduler.atomics)}"
    # )

    # schedule = schedule_tensor.tolist()
    # next_idx_map = defaultdict(list)
    # for li, lst in enumerate(schedule):
    #     for ii, node in enumerate(lst):
    #         next_idx_map[node[-1]].append((li, ii))

    # violations = []
    # for li, lst in enumerate(schedule):
    #     for ii, node in enumerate(lst):
    #         atomic_cnt, prev_idx = node[-3], node[-2]
    #         if atomic_cnt > 0:
    #             actual = len(next_idx_map.get(prev_idx, []))
    #             if actual != atomic_cnt:
    #                 violations.append(
    #                     f"  list={li} item={ii} op={node[0] & 0x7} layer={node[0] >> 3} "
    #                     f"pid_m={node[1]} pid_n={node[2]} pid_o={node[3]}: "
    #                     f"expected_cnt={atomic_cnt} but {actual} predecessors "
    #                     f"write to atomic[{prev_idx}]"
    #                 )
    # if violations:
    #     print(f"FAIL: {len(violations)} dependency violations")
    #     for v in violations[:10]:
    #         print(v)
    # else:
    #     print("SUCCESS: dependency graph valid.")
# import math
# from collections import defaultdict
# from dataclasses import dataclass
# from enum import IntEnum
# import torch

# # from transformer_megakernel.config import Op, InputConfig
# # from transformer_megakernel.operators.matmul import MatmulConfig

# @dataclass
# class InputConfig:
#     bs: int
#     embed_dim: int
#     kv_len: int
#     q_len: int
#     num_q_heads: int
#     num_kv_heads: int
#     num_layers: int
#     ff_dim: int
#     is_causal: bool

# class Op(IntEnum):
#     RMS = 0
#     QKV = 1
#     ATTN = 2
#     OUT = 3
#     GATE = 4
#     UP = 5
#     DOWN = 6
#     NOP  = 99


# class MatmulConfig:
#     bM: int
#     bN: int
#     bK: int

#     num_stages: int
#     stage_elements: int
#     output_pad: int
#     use_tma_reduce: bool
#     group_m = 8

#     def __post_init__(self):
#         self.stage_size  = (self.bM + self.bN) * self.bK
#         self.swizzle_bits = (int(math.log2(self.bK)) - 3, 4, 3)
#         assert self.bM in (32, 64, 128)
#         assert self.bN in (32, 64, 128)
#         assert self.bK in (32, 64)
#         assert self.num_stages == 2, "V2 matmul is two-stage"
#         assert self.stage_elements >= self.stage_size
#         assert self.output_pad in (0, 8, 16)

# @dataclass
# class AttentionConfig:
#     bQ: int
#     bKV: int
#     head_dim: int
#     num_q_heads: int
#     num_kv_heads: int
#     q_len: int
#     kv_len: int
#     output_pad: int
#     num_stages: int
#     stage_elements: int
#     is_causal: bool

#     def __post_init__(self):
#         assert self.bQ == 64
#         assert self.bKV == 64
#         assert self.head_dim in (64, 128)
#         assert self.num_stages == 2, "V2 attention is two-stage"
#         assert self.stage_elements >= 2 * self.bKV * self.head_dim


# @dataclass
# class RMSNormConfig:
#     embed_dim: int
#     bRMS: int
#     stage_elements: int
#     warps_per_row: int = 1

#     def __post_init__(self):
#         assert 4 % self.warps_per_row == 0, (
#             "warps_per_row must divide 4 (max 4 warps per row)"
#         )
#         assert (self.bRMS % (4 // self.warps_per_row)) == 0, (
#             "bRMS must be divisible by (4 // warps_per_row)"
#         )
#         assert self.embed_dim % (32 * self.warps_per_row) == 0, (
#             "embed_dim must be divisible by (32 * warps_per_row)"
#         )


# @dataclass
# class LayerOpConfigs:
#     """Per-op MatmulConfigs for one transformer layer.

#     Constraints enforced at OpScheduler construction:
#       - qkv.bM == down.bM          : cross-layer alignment (down output → rms1 input)
#       - ffn_up.bM <= out.bM        : rms2 maps its M-tiles into the out atomic slots;
#                                      if ffn_up.bM < out.bM several rms2 tiles share
#                                      one out slot (fine); if ffn_up.bM > out.bM a
#                                      single rms2 tile would span multiple out slots
#                                      (not supported by the single-atomic design).
#       - ffn_up.bM % qkv.bM == 0   : gate→down M-tile mapping requires ffn_up.bM to
#         OR qkv.bM % ffn_up.bM == 0  be a multiple of down.bM (== qkv.bM), or vice
#                                      versa, so the integer-division slot mapping is
#                                      lossless.
#     """

#     qkv: MatmulConfig    # projects hidden → Q K V
#     out: MatmulConfig    # projects attn output → hidden  (bM may differ from qkv)
#     ffn_up: MatmulConfig # FFN up-projection  (bM may differ; see constraints above)
#     ffn_dn: MatmulConfig # FFN down-projection (bM must equal qkv.bM)


# class OpScheduler:
#     """Builds a static work schedule and atomic-dependency graph for one
#     transformer stack, given per-op tile configs.

#     Each work item in the schedule is a 7-element list:
#       [enc(layer, op), pid_m, pid_n, pid_o, expected_cnt, in_atomic, out_atomic]

#     * expected_cnt : number of predecessors that must increment *in_atomic*
#                      before this tile may run (0 = unconditionally ready).
#     * in_atomic    : index into the shared atomics array this tile waits on.
#     * out_atomic   : index into the shared atomics array this tile increments
#                      when it finishes.
#     """

#     def __init__(
#         self,
#         input_config: InputConfig,
#         rms_config: RMSNormConfig,
#         attn_config: AttentionConfig,
#         layer_configs: LayerOpConfigs,
#         num_sms: int,
#     ):
#         self.num_sms = num_sms
#         self.schedule = [[] for _ in range(self.num_sms)]
#         self.atomics: list[int] = []

#         # ── input dimensions ────────────────────────────────────────────────
#         self.bs = input_config.bs
#         self.q_len = input_config.q_len
#         self.M = self.bs * self.q_len
#         self.E = input_config.embed_dim
#         self.num_q_heads = input_config.num_q_heads
#         self.num_kv_heads = input_config.num_kv_heads
#         self.head_dim = input_config.embed_dim // input_config.num_q_heads
#         # Concatenated Q-K-V projection width
#         self.Qd = (input_config.num_q_heads + 2 * input_config.num_kv_heads) * self.head_dim
#         self.F = input_config.ff_dim
#         self.num_layers = input_config.num_layers

#         # ── per-op configs ──────────────────────────────────────────────────
#         self.rms = rms_config
#         self.attn = attn_config
#         self.ops = layer_configs

#         # ── derived tile parameters ─────────────────────────────────────────
#         self.bRMS = rms_config.bRMS
#         self.bQ = attn_config.bQ

#         self._validate_configs()

#     # ────────────────────────────────────────────────────────────────────────
#     # Helpers
#     # ────────────────────────────────────────────────────────────────────────

#     def _validate_configs(self):
#         ops = self.ops
#         assert ops.qkv.bM == ops.ffn_dn.bM, (
#             f"qkv.bM ({ops.qkv.bM}) must equal ffn_dn.bM ({ops.ffn_dn.bM}) "
#             f"for cross-layer down→rms1 alignment"
#         )
#         assert ops.ffn_up.bM <= ops.out.bM, (
#             f"ffn_up.bM ({ops.ffn_up.bM}) must be <= out.bM ({ops.out.bM}); "
#             f"a single rms2 tile must not span more than one out-atomic slot"
#         )
#         bM_dn = ops.ffn_dn.bM
#         bM_up = ops.ffn_up.bM
#         assert bM_up % bM_dn == 0 or bM_dn % bM_up == 0, (
#             f"ffn_up.bM ({bM_up}) and ffn_dn.bM ({bM_dn}) must be multiples "
#             f"of each other for the gate→down atomic-slot mapping"
#         )
#         assert ops.qkv.bM % self.bRMS == 0, (
#             f"qkv.bM ({ops.qkv.bM}) must be divisible by bRMS ({self.bRMS})"
#         )
#         assert ops.ffn_up.bM % self.bRMS == 0, (
#             f"ffn_up.bM ({ops.ffn_up.bM}) must be divisible by bRMS ({self.bRMS})"
#         )

#     def _mout(self, bM: int) -> int:
#         """Number of M-dimension tiles."""
#         return math.ceil(self.M / bM)

#     def _tpg(self, bM: int) -> int:
#         """RMS sub-tiles per M-tile (thread-groups per M-group)."""
#         return bM // self.bRMS

#     def _get_next_sm(self) -> int:
#         """Round-robin: return the SM with the shortest current queue."""
#         for i in range(self.num_sms):
#             if len(self.schedule[i]) > len(self.schedule[(i + 1) % self.num_sms]):
#                 return (i + 1) % self.num_sms
#         return 0

#     def _enc(self, layer: int, op: Op) -> int:
#         return (layer << 3) | int(op.value)

#     @staticmethod
#     def _compute_pid(block_id: int, total_pid: int, total_pid_m: int,
#                      group_size_m: int = 8) -> tuple[int, int]:
#         """L2-locality-aware 2-D block → (pid_m, pid_n) mapping."""
#         total_pid_n = total_pid // total_pid_m
#         gsm = min(group_size_m, total_pid_m)
#         grp = gsm * total_pid_n
#         g = block_id // grp
#         fm = g * gsm
#         sz = min(total_pid_m - fm, gsm)
#         pid_m = fm + block_id % sz
#         pid_n = (block_id % grp) // sz
#         return pid_m, pid_n

#     def _alloc(self, n: int) -> int:
#         """Allocate *n* atomic slots; return the base index."""
#         idx = len(self.atomics)
#         self.atomics.extend([0] * n)
#         return idx

#     # ────────────────────────────────────────────────────────────────────────
#     # Per-operator schedulers
#     # ────────────────────────────────────────────────────────────────────────

#     def _schedule_rms(
#         self,
#         layer: int,
#         rms_pass: int,
#         in_idx: int,
#         bM: int,
#         bM_prev: int,
#         expected_cnt: int,
#     ) -> int:
#         """Schedule RMSNorm tiles.

#         Args:
#             rms_pass:     0 = pre-attention RMS, 1 = pre-FFN RMS.
#             bM:           Row-tile size for *this* RMS (matches its consumer).
#             bM_prev:      Row-tile size of the *producer* (whose atomics live at
#                           in_idx).  If bM < bM_prev, multiple RMS tiles share one
#                           producer atomic slot; the slot mapping is
#                           ``(rg * bM) // bM_prev``.
#             expected_cnt: Number of producer tiles that must signal in_atomic
#                           before this tile can start.
#         """
#         Mo = self._mout(bM)
#         t = self._tpg(bM)
#         out_idx = self._alloc(Mo)

#         sm_id = self._get_next_sm()
#         for tile_idx in range(Mo * t):
#             rg = tile_idx // t
#             # Map rg (at bM granularity) to the producer's atomic slot granularity.
#             in_slot = (rg * bM) // bM_prev if bM != bM_prev else rg
#             self.schedule[sm_id].append([
#                 self._enc(layer, Op.RMS),
#                 tile_idx,
#                 tile_idx // self.num_sms,
#                 rms_pass,
#                 expected_cnt,
#                 in_idx + in_slot,
#                 out_idx + rg,
#             ])
#             sm_id = (sm_id + 1) % self.num_sms
#         return out_idx

#     def _schedule_qkv(self, layer: int, in_idx: int) -> int:
#         """Schedule QKV projection tiles.

#         Waits on:  rms1 output atomic[pid_m]  (one per M-tile).
#                    expected_cnt = tpg (RMS sub-tiles per M-tile).
#         Signals:   per-batch atomic (one per batch element).
#         """
#         cfg = self.ops.qkv
#         Mo = self._mout(cfg.bM)
#         Nqd = math.ceil(self.Qd / cfg.bN)
#         total = Mo * Nqd
#         out_idx = self._alloc(self.bs)

#         expected_cnt = self._tpg(cfg.bM)
#         sm_id = self._get_next_sm()
#         for block_id in range(total):
#             pid_m, pid_n = self._compute_pid(block_id, total, Mo, cfg.group_m)
#             batch_idx = (pid_m * cfg.bM) // self.q_len
#             self.schedule[sm_id].append([
#                 self._enc(layer, Op.QKV),
#                 pid_m, pid_n, 0,
#                 expected_cnt,
#                 in_idx + pid_m,
#                 out_idx + batch_idx,
#             ])
#             sm_id = (sm_id + 1) % self.num_sms
#         return out_idx

#     def _schedule_attn(self, layer: int, in_idx: int) -> int:
#         """Schedule attention tiles.

#         Waits on:  per-batch QKV atomic.
#                    expected_cnt = (q_len/qkv.bM) * ceil(Qd/qkv.bN)
#                    (all QKV tiles for this batch must complete).

#         Signals:   per-batch atomic.
#         """
#         cfg_qkv = self.ops.qkv
#         Mo_q = math.ceil(self.q_len / cfg_qkv.bM)
#         Nqd = math.ceil(self.Qd / cfg_qkv.bN)
#         expected_cnt = Mo_q * Nqd
#         Qblks = math.ceil(self.q_len / self.bQ)
#         out_idx = self._alloc(self.bs)

#         sm_id = self._get_next_sm()
#         for b in range(self.bs):
#             for h in range(self.num_q_heads):
#                 for qb in range(Qblks):
#                     self.schedule[sm_id].append([
#                         self._enc(layer, Op.ATTN),
#                         b, h, qb,
#                         expected_cnt,
#                         in_idx + b,
#                         out_idx + b,
#                     ])
#                     sm_id = (sm_id + 1) % self.num_sms
#         return out_idx

#     def _schedule_out(self, layer: int, in_idx: int) -> int:
#         """Schedule output projection tiles.

#         Waits on:  per-batch attention atomic.
#                    expected_cnt = num_q_heads * ceil(q_len/bQ)
#                    (all attention tiles for this batch must complete).
#         Signals:   per-M-tile atomic (at out.bM granularity).
#         """
#         cfg = self.ops.out
#         Mo = self._mout(cfg.bM)
#         Nout = math.ceil(self.E / cfg.bN)
#         total = Mo * Nout
#         expected_cnt = self.num_q_heads * math.ceil(self.q_len / self.bQ)
#         out_idx = self._alloc(Mo)

#         sm_id = self._get_next_sm()
#         for block_id in range(total):
#             pid_m, pid_n = self._compute_pid(block_id, total, Mo, cfg.group_m)
#             batch_idx = (pid_m * cfg.bM) // self.q_len
#             self.schedule[sm_id].append([
#                 self._enc(layer, Op.OUT),
#                 pid_m, pid_n, 0,
#                 expected_cnt,
#                 in_idx + batch_idx,
#                 out_idx + pid_m,
#             ])
#             sm_id = (sm_id + 1) % self.num_sms
#         return out_idx

#     def _schedule_up(self, layer: int, in_idx: int) -> int:
#         """Schedule FFN up-projection tiles.

#         Waits on:  per-M-tile rms2 atomic[pid_m].
#                    expected_cnt = tpg (RMS sub-tiles per M-tile).
#         Signals:   per-(M-tile, N-tile) atomic  (one per up tile, feeds gate 1:1).
#         """
#         cfg = self.ops.ffn_up
#         Mo = self._mout(cfg.bM)
#         Nf = math.ceil(self.F / cfg.bN)
#         total = Mo * Nf
#         out_idx = self._alloc(Mo * Nf)

#         expected_cnt = self._tpg(cfg.bM)
#         sm_id = self._get_next_sm()
#         for block_id in range(total):
#             pid_m, pid_n = self._compute_pid(block_id, total, Mo, cfg.group_m)
#             self.schedule[sm_id].append([
#                 self._enc(layer, Op.UP),
#                 pid_m, pid_n, 0,
#                 expected_cnt,
#                 in_idx + pid_m,
#                 out_idx + pid_m * Nf + pid_n,
#             ])
#             sm_id = (sm_id + 1) % self.num_sms
#         return out_idx

#     def _schedule_gate(self, layer: int, in_idx: int) -> int:
#         """Schedule gate (SiLU·gate) tiles.

#         Waits on:  single up-tile atomic (1:1 mapping with up tiles).
#                    expected_cnt = 1.
#         Signals:   per-M-tile atomic (at ffn_up.bM granularity).
#         """
#         cfg = self.ops.ffn_up
#         Mo = self._mout(cfg.bM)
#         Nf = math.ceil(self.F / cfg.bN)
#         total = Mo * Nf
#         out_idx = self._alloc(Mo)

#         sm_id = self._get_next_sm()
#         for block_id in range(total):
#             pid_m, pid_n = self._compute_pid(block_id, total, Mo, cfg.group_m)
#             self.schedule[sm_id].append([
#                 self._enc(layer, Op.GATE),
#                 pid_m, pid_n, 0,
#                 1,                              # exactly 1 up tile feeds each gate tile
#                 in_idx + pid_m * Nf + pid_n,
#                 out_idx + pid_m,
#             ])
#             sm_id = (sm_id + 1) % self.num_sms
#         return out_idx

#     def _schedule_down(self, layer: int, in_idx: int) -> int:
#         """Schedule FFN down-projection tiles.

#         Waits on:  per-M-tile gate atomic (at ffn_up.bM granularity).
#                    expected_cnt = ceil(F / ffn_up.bN)
#                    (all gate tiles for this M-row must complete).

#         When ffn_dn.bM < ffn_up.bM the down M-tile is finer than the gate
#         M-tile; the mapping is  gate_slot = (pid_m * ffn_dn.bM) // ffn_up.bM.

#         Signals:   per-M-tile atomic (at ffn_dn.bM granularity).
#         """
#         cfg_dn = self.ops.ffn_dn
#         cfg_up = self.ops.ffn_up
#         Mo = self._mout(cfg_dn.bM)
#         Nout = math.ceil(self.E / cfg_dn.bN)
#         total = Mo * Nout
#         Nf_gate = math.ceil(self.F / cfg_up.bN)   # gate tiles per gate M-slot
#         out_idx = self._alloc(Mo)

#         sm_id = self._get_next_sm()
#         for block_id in range(total):
#             pid_m, pid_n = self._compute_pid(block_id, total, Mo, cfg_dn.group_m)
#             gate_slot = (pid_m * cfg_dn.bM) // cfg_up.bM
#             self.schedule[sm_id].append([
#                 self._enc(layer, Op.DOWN),
#                 pid_m, pid_n, 0,
#                 Nf_gate,
#                 in_idx + gate_slot,
#                 out_idx + pid_m,
#             ])
#             sm_id = (sm_id + 1) % self.num_sms
#         return out_idx

#     # ────────────────────────────────────────────────────────────────────────
#     # Pad & build
#     # ────────────────────────────────────────────────────────────────────────

#     def _align(self) -> int:
#         """Pad every SM queue to the same length with no-op sentinel entries."""
#         max_works = max(len(sm) for sm in self.schedule) if self.schedule else 0
#         sentinel = [-1, -1, -1, -1, 0, -1, -1]
#         for sm in self.schedule:
#             while len(sm) < max_works:
#                 sm.append(sentinel[:])
#         return max_works

#     def build_schedule(self):
#         """Build and return (schedule_tensor, atomics_tensor, max_works).

#         schedule_tensor : int32[num_sms, max_works, 7]
#         atomics_tensor  : int32[total_atomics]
#         max_works       : int
#         """
#         cfg_qkv = self.ops.qkv
#         cfg_out = self.ops.out

#         # Initial token-embedding M-tile signals (one per qkv M-tile, all pre-set).
#         Mo_init = self._mout(cfg_qkv.bM)
#         in_idx = self._alloc(Mo_init)

#         for layer in range(self.num_layers):
#             Nn_out = math.ceil(self.E / cfg_out.bN)
#             Nn_dn = math.ceil(self.E / self.ops.ffn_dn.bN)

#             # Layer 0: rms1 waits on nothing (input is the embedding, already ready).
#             # Layer l>0: rms1 waits for all down-projection N-tiles of the prev layer
#             #            to signal the per-M-tile atomic (Nn_dn writers per slot).
#             rms1_exp = 0 if layer == 0 else Nn_dn

#             in_idx = self._schedule_rms(
#                 layer, 0, in_idx,
#                 bM=cfg_qkv.bM, bM_prev=cfg_qkv.bM,  # same granularity (prev = down = qkv.bM)
#                 expected_cnt=rms1_exp,
#             )
#             in_idx = self._schedule_qkv(layer, in_idx)
#             in_idx = self._schedule_attn(layer, in_idx)
#             in_idx = self._schedule_out(layer, in_idx)

#             # rms2 reads from out (at out.bM granularity) but runs at ffn_up.bM.
#             in_idx = self._schedule_rms(
#                 layer, 1, in_idx,
#                 bM=self.ops.ffn_up.bM, bM_prev=cfg_out.bM,
#                 expected_cnt=Nn_out,
#             )
#             in_idx = self._schedule_up(layer, in_idx)
#             in_idx = self._schedule_gate(layer, in_idx)
#             in_idx = self._schedule_down(layer, in_idx)

#         # Trailing sentinel slot to keep index arithmetic safe.
#         self.atomics.append(0)

#         max_works = self._align()
#         return (
#             torch.tensor(self.schedule, dtype=torch.int32),
#             torch.tensor(self.atomics, dtype=torch.int32),
#             max_works,
#         )

#     # ────────────────────────────────────────────────────────────────────────
#     # Dependency verification  (mirrors the __main__ check in the original)
#     # ────────────────────────────────────────────────────────────────────────

#     def verify(self) -> list[str]:
#         """Verify that every (expected_cnt, in_atomic) pair is consistent with
#         the actual number of tiles that write to that atomic slot.

#         Returns a list of human-readable violation strings (empty = OK).
#         """
#         schedule = self.schedule

#         # Map atomic index → list of (sm_id, work_idx) that write to it.
#         writers: dict[int, list] = defaultdict(list)
#         for sm_id, sm in enumerate(schedule):
#             for w_idx, node in enumerate(sm):
#                 if node[0] == -1:
#                     continue
#                 out_at = node[6]
#                 if out_at >= 0:
#                     writers[out_at].append((sm_id, w_idx))

#         violations: list[str] = []
#         n_atomics = len(self.atomics)

#         for sm_id, sm in enumerate(schedule):
#             for w_idx, node in enumerate(sm):
#                 if node[0] == -1:
#                     continue
#                 op_enc, pid_m, pid_n, pid_o, exp_cnt, in_at, out_at = node
#                 op = op_enc & 0x7
#                 layer = op_enc >> 3

#                 # Check atomic index bounds.
#                 for label, idx in (("in_atomic", in_at), ("out_atomic", out_at)):
#                     if idx >= 0 and idx >= n_atomics:
#                         violations.append(
#                             f"  sm={sm_id} w={w_idx} op={op} layer={layer} "
#                             f"pm={pid_m} pn={pid_n}: {label}={idx} out of range "
#                             f"(total={n_atomics})"
#                         )

#                 # Check predecessor count.
#                 if exp_cnt > 0 and in_at >= 0:
#                     actual = len(writers.get(in_at, []))
#                     if actual != exp_cnt:
#                         violations.append(
#                             f"  sm={sm_id} w={w_idx} op={op} layer={layer} "
#                             f"pm={pid_m} pn={pid_n} po={pid_o}: "
#                             f"expected_cnt={exp_cnt} but {actual} predecessors "
#                             f"write to atomic[{in_at}]"
#                         )

#         return violations


# # ────────────────────────────────────────────────────────────────────────────
# # Convenience factory
# # ────────────────────────────────────────────────────────────────────────────

# def build_schedule(
#     input_config: InputConfig,
#     rms_config: RMSNormConfig,
#     attn_config: AttentionConfig,
#     layer_configs: LayerOpConfigs,
#     num_sms: int,
#     verify: bool = True,
# ):
#     """Build a schedule and optionally run the dependency verifier.

#     Returns (schedule_tensor, atomics_tensor, max_works).
#     Raises RuntimeError if verify=True and violations are found.
#     """
#     scheduler = OpScheduler(input_config, rms_config, attn_config, layer_configs, num_sms)
#     result = scheduler.build_schedule()

#     if verify:
#         violations = scheduler.verify()
#         if violations:
#             msg = f"{len(violations)} dependency violations:\n" + "\n".join(violations[:20])
#             raise RuntimeError(msg)

#     return result


# # ────────────────────────────────────────────────────────────────────────────
# # Self-test
# # ────────────────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     import sys
#     from types import SimpleNamespace

#     # Minimal InputConfig stand-in (replace with your real dataclass if needed).
#     def make_input(bs, q_len, embed_dim, num_q_heads, num_kv_heads, ff_dim, num_layers):
#         return SimpleNamespace(
#             bs=bs, q_len=q_len, embed_dim=embed_dim,
#             num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
#             ff_dim=ff_dim, num_layers=num_layers,
#         )

#     def make_mm(bM, bN, bK=64):
#         return SimpleNamespace(bM=bM, bN=bN, bK=bK, num_stages=2,
#                                stage_elements=(bM+bN)*bK, output_pad=0,
#                                use_tma_reduce=False, group_m=8)

#     TESTS = [
#         dict(name="all_bM64",
#              bs=2, q_len=256, E=512, Qh=8, KVh=8, F=1024, L=2, sms=108,
#              bM_q=64, bN_q=64, bM_o=64, bN_o=128,
#              bM_u=64, bN_u=128, bM_d=64, bN_d=128, bRMS=32, bQ=64),
#         dict(name="bM_up128",
#              bs=2, q_len=256, E=512, Qh=8, KVh=8, F=1024, L=2, sms=108,
#              bM_q=64, bN_q=128, bM_o=128, bN_o=128,
#              bM_u=128, bN_u=128, bM_d=64, bN_d=128, bRMS=32, bQ=64),
#         dict(name="bM_up32",
#              bs=2, q_len=256, E=512, Qh=8, KVh=8, F=1024, L=2, sms=108,
#              bM_q=64, bN_q=128, bM_o=64, bN_o=128,
#              bM_u=32, bN_u=128, bM_d=64, bN_d=128, bRMS=32, bQ=64),
#         dict(name="diff_bN",
#              bs=2, q_len=256, E=512, Qh=8, KVh=8, F=1024, L=2, sms=108,
#              bM_q=64, bN_q=64, bM_o=64, bN_o=128,
#              bM_u=64, bN_u=128, bM_d=64, bN_d=64, bRMS=32, bQ=64),
#         dict(name="bRMS64",
#              bs=2, q_len=256, E=512, Qh=8, KVh=8, F=1024, L=2, sms=108,
#              bM_q=128, bN_q=128, bM_o=128, bN_o=128,
#              bM_u=128, bN_u=128, bM_d=128, bN_d=128, bRMS=64, bQ=64),
#         dict(name="large_model",
#              bs=4, q_len=512, E=1024, Qh=16, KVh=16, F=4096, L=4, sms=132,
#              bM_q=64, bN_q=128, bM_o=128, bN_o=128,
#              bM_u=128, bN_u=128, bM_d=64, bN_d=128, bRMS=32, bQ=64),
#     ]

#     all_pass = True
#     for t in TESTS:
#         inp = make_input(t["bs"], t["q_len"], t["E"], t["Qh"], t["KVh"], t["F"], t["L"])
#         rms = SimpleNamespace(embed_dim=t["E"], bRMS=t["bRMS"], stage_elements=0, warps_per_row=1)
#         attn = SimpleNamespace(bQ=t["bQ"], bKV=64, head_dim=t["E"]//t["Qh"],
#                                num_q_heads=t["Qh"], num_kv_heads=t["KVh"],
#                                q_len=t["q_len"], kv_len=t["q_len"],
#                                output_pad=0, num_stages=2, stage_elements=0, is_causal=False)
#         layer_cfgs = SimpleNamespace(
#             qkv=make_mm(t["bM_q"], t["bN_q"]),
#             out=make_mm(t["bM_o"], t["bN_o"]),
#             ffn_up=make_mm(t["bM_u"], t["bN_u"]),
#             ffn_dn=make_mm(t["bM_d"], t["bN_d"]),
#         )

#         try:
#             sched = OpScheduler(inp, rms, attn, layer_cfgs, t["sms"])
#             stensor, atomics, mw = sched.build_schedule()
#             violations = sched.verify()
#             if violations:
#                 print(f"[FAIL] {t['name']}: {len(violations)} violations")
#                 for v in violations[:5]:
#                     print(v)
#                 all_pass = False
#             else:
#                 print(f"[PASS] {t['name']:20s}  atomics={len(sched.atomics):5d}  "
#                       f"max_works={mw:4d}  total_slots={t['sms']*mw}")
#         except Exception as e:
#             print(f"[ERR ] {t['name']}: {e}")
#             all_pass = False

#     print()
#     print("All tests passed." if all_pass else "Some tests FAILED.")
#     sys.exit(0 if all_pass else 1)