import math
import logging

import torch

from transformer_megakernel.config import Op, InputConfig, KernelConfig

logger = logging.getLogger(__name__)

class OpScheduler:
    def __init__(self, input_config: InputConfig, kernel_config: KernelConfig):
        logger.info(f"Initializing OpScheduler with num_sms={kernel_config.num_sms}, bs={input_config.bs}, q_len={input_config.q_len}, embed_dim={input_config.embed_dim}, layers={input_config.num_layers}")
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

        self.Mout = math.ceil(self.M / self.bM)
        assert self.q_len % self.bM == 0, "bM must divide q_len"

        self.rows_per_rms_block = kernel_config.rows_per_rms_block
        assert self.bM % self.rows_per_rms_block == 0, (
            "rows_per_rms_block must divide bM"
        )
        self.tpg = self.bM // self.rows_per_rms_block
        self.Mtiles = self.Mout * self.tpg

    # ── Helpers ──────────────────────────────────────────────────────────

    def _get_prev_sm(self):
        for i in range(len(self.schedule)):
            if len(self.schedule[i]) > len(self.schedule[(i + 1) % self.num_sms]):
                return (i + 1) % self.num_sms
        return 0

    def _alloc(self, n):
        """Allocate *n* atomic slots; return the base index."""
        idx = len(self.atomics)
        self.atomics.extend([0] * int(n))
        return idx

    def _emit(self, layer, op_kind, pid_m, pid_n, pid_o, expected_cnt, current_idx, next_idx):
        """Append one instruction to the least-loaded SM."""
        sm_id = self._get_prev_sm()
        self.schedule[sm_id].append(
            [layer, op_kind, pid_m, pid_n, pid_o, expected_cnt, current_idx, next_idx]
        )

    def _compute_pid(self, block_id, total_pid, total_pid_m, group_size_m=8):
        total_pid_n = total_pid // total_pid_m
        gsm = min(group_size_m, total_pid_m)
        grp = gsm * total_pid_n
        g = block_id // grp
        fm = g * gsm
        sz = min(total_pid_m - fm, gsm)
        pid_m = fm + block_id % sz
        pid_n = (block_id % grp) // sz
        return pid_m, pid_n

    # ── Per-operator schedulers ──────────────────────────────────────────

    def schedule_rms1(self, layer, in_idx):
        out_idx = self._alloc(self.Mout)
        Nout = math.ceil(self.E / self.bN)
        expected_cnt = 0 if layer == 0 else Nout

        for tile_idx in range(self.Mtiles):
            rg = tile_idx // self.tpg
            self._emit(
                layer       = layer,
                op_kind     = int(Op.RMS.value),
                pid_m        = tile_idx,
                pid_n        = tile_idx // self.num_sms,
                pid_o        = 0,
                expected_cnt = expected_cnt,
                current_idx  = in_idx + rg,
                next_idx     = out_idx + rg,
            )
        return out_idx

    def schedule_qkv(self, layer, in_idx):
        out_idx = self._alloc(self.bs)
        total_pid_n = math.ceil(self.Qd / self.bN)
        total_pid = self.Mout * total_pid_n

        for block_id in range(total_pid):
            pid_m, pid_n = self._compute_pid(block_id, total_pid, self.Mout)
            self._emit(
                layer        = layer,
                op_kind      = int(Op.QKV.value),
                pid_m        = pid_m,
                pid_n        = pid_n,
                pid_o        = 0,
                expected_cnt = self.tpg,
                current_idx  = in_idx + pid_m,
                next_idx     = out_idx + (pid_m * self.bM) // self.q_len,
            )
        return out_idx

    def schedule_attn(self, layer, in_idx):
        out_idx = self._alloc(self.bs)
        Mpm = math.ceil(self.q_len / self.bM)
        Nqkv = math.ceil(self.Qd / self.bN)
        expected_cnt = Mpm * Nqkv
        Qblks = math.ceil(self.q_len / self.block_q)

        for b in range(self.bs):
            for h in range(self.num_q_heads):
                for qb in range(Qblks):
                    self._emit(
                        layer        = layer,
                        op_kind      = int(Op.ATTN.value),
                        pid_m        = b,
                        pid_n        = h,
                        pid_o        = qb,
                        expected_cnt = expected_cnt,
                        current_idx  = in_idx + b,
                        next_idx     = out_idx + b,
                    )
        return out_idx

    def schedule_out(self, layer, in_idx):
        out_idx = self._alloc(self.Mout)
        total_pid_n = math.ceil(self.E / self.bN)
        total_pid = self.Mout * total_pid_n
        Qblks = math.ceil(self.q_len / self.block_q)
        expected_cnt = self.num_q_heads * Qblks

        for block_id in range(total_pid):
            pid_m, pid_n = self._compute_pid(block_id, total_pid, self.Mout)
            self._emit(
                layer        = layer,
                op_kind      = int(Op.OUT.value),
                pid_m        = pid_m,
                pid_n        = pid_n,
                pid_o        = 0,
                expected_cnt = expected_cnt,
                current_idx  = in_idx + (pid_m * self.bM) // self.q_len,
                next_idx     = out_idx + pid_m,
            )
        return out_idx

    def schedule_rms2(self, layer, in_idx):
        out_idx = self._alloc(self.Mout)
        Nout = math.ceil(self.E / self.bN)
        expected_cnt = Nout

        for tile_idx in range(self.Mtiles):
            rg = tile_idx // self.tpg
            self._emit(
                layer        = layer,
                op_kind      = int(Op.RMS.value),
                pid_m        = tile_idx,
                pid_n        = tile_idx // self.num_sms,
                pid_o        = 1,
                expected_cnt = expected_cnt,
                current_idx  = in_idx + rg,
                next_idx     = out_idx + rg,
            )
        return out_idx

    def schedule_up(self, layer, in_idx):
        Nf = math.ceil(self.F / self.bN)
        out_idx = self._alloc(self.Mout * Nf)
        total_pid = self.Mout * Nf

        for block_id in range(total_pid):
            pid_m, pid_n = self._compute_pid(block_id, total_pid, self.Mout)
            self._emit(
                layer        = layer,
                op_kind      = int(Op.UP.value),
                pid_m        = pid_m,
                pid_n        = pid_n,
                pid_o        = 0,
                expected_cnt = self.tpg,
                current_idx  = in_idx + pid_m,
                next_idx     = out_idx + (pid_m * Nf + pid_n),
            )
        return out_idx

    def schedule_gate(self, layer, in_idx):
        Nf = math.ceil(self.F / self.bN)
        out_idx = self._alloc(self.Mout)
        total_pid = self.Mout * Nf

        for block_id in range(total_pid):
            pid_m, pid_n = self._compute_pid(block_id, total_pid, self.Mout)
            self._emit(
                layer        = layer,
                op_kind      = int(Op.GATE.value),
                pid_m        = pid_m,
                pid_n        = pid_n,
                pid_o        = 0,
                expected_cnt = 1,
                current_idx  = in_idx + (pid_m * Nf + pid_n),
                next_idx     = out_idx + pid_m,
            )
        return out_idx

    def schedule_down(self, layer, in_idx):
        Nout = math.ceil(self.E / self.bN)
        Nf = math.ceil(self.F / self.bN)
        out_idx = self._alloc(self.Mout)
        total_pid = self.Mout * Nout

        for block_id in range(total_pid):
            pid_m, pid_n = self._compute_pid(block_id, total_pid, self.Mout)
            self._emit(
                layer        = layer,
                op_kind      = int(Op.DOWN.value),
                pid_m        = pid_m,
                pid_n        = pid_n,
                pid_o        = 0,
                expected_cnt = Nf,
                current_idx  = in_idx + pid_m,
                next_idx     = out_idx + pid_m,
            )
        return out_idx
    # ── Pad & build ──────────────────────────────────────────────────────

    def _align(self):
        max_works = max(len(sm) for sm in self.schedule) if self.schedule else 0
        for sm in self.schedule:
            while len(sm) < max_works:
                sm.append([-1, -1, -1, -1, -1, 0, -1, -1])
        return max_works

    def build_schedule(self):
        in_idx = self._alloc(self.Mout)

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
        logger.info(f"Schedule built with max_works={max_works}, num_instructions per SM (padded)")
        return (
            torch.tensor(self.schedule, dtype=torch.int32),
            torch.tensor(self.atomics, dtype=torch.int32),
            max_works,
        )