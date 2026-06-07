import math
from collections import defaultdict

import torch

from src.megakernel.megakernel import LLMMegaKernelConfig, Op


class OpScheduler:
    def __init__(self, config: LLMMegaKernelConfig):
        self.num_sms  = config.num_sms
        self.schedule = [[] for _ in range(self.num_sms)]
        self.atomics  = []

        self.bs    = config.bs
        self.q_len = config.q_len
        self.M     = self.bs * self.q_len

        self.E            = config.embed_dim
        self.num_q_heads  = config.num_q_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim     = config.embed_dim // config.num_q_heads
        self.Qd           = (config.num_q_heads + 2 * config.num_kv_heads) * self.head_dim
        self.F            = config.ff_dim
        self.num_layers   = config.num_layers

        self.bM      = config.bM
        self.bN      = config.bN
        self.block_q = config.block_q

        self.Mout = int(math.ceil(self.M / self.bM))
        assert self.q_len % self.bM == 0, "bM must divide q_len"

        self.rows_per_rms_block = config.rows_per_rms_block
        assert self.bM % self.rows_per_rms_block == 0, "rows_per_rms_block must divide bM"
        self.tpg    = self.bM // self.rows_per_rms_block
        self.Mtiles = self.Mout * self.tpg

    def _get_prev_sm(self):
        for i in range(len(self.schedule)):
            if len(self.schedule[i]) > len(self.schedule[(i + 1) % self.num_sms]):
                return (i + 1) % self.num_sms
        return 0

    def _compute_pid(self, block_id, total_pid, total_pid_m, group_size_m=4):
        total_pid_n = total_pid // total_pid_m
        gsm = min(group_size_m, total_pid_m)
        grp = gsm * total_pid_n
        g   = block_id // grp
        fm  = g * gsm
        sz  = min(total_pid_m - fm, gsm)
        pid_m = fm + block_id % sz
        pid_n = (block_id % grp) // sz
        return pid_m, pid_n

    def _enc(self, layer, op):
        return (layer << 3) | int(op.value)

    def schedule_rms1(self, layer, in_idx):
        out_idx = len(self.atomics)
        self.atomics.extend([0] * int(self.Mout))

        Nout         = int(math.ceil(self.E / self.bN))
        expected_cnt = 0 if layer == 0 else Nout

        sm_id = self._get_prev_sm()
        for tile_idx in range(self.Mtiles):
            rg = tile_idx // self.tpg
            self.schedule[sm_id].append([
                self._enc(layer, Op.RMS), tile_idx, tile_idx // self.num_sms, 0,
                expected_cnt, in_idx + rg, out_idx + rg,
            ])
            sm_id = (sm_id + 1) % self.num_sms
        return out_idx

    def schedule_qkv(self, layer, in_idx):
        out_idx     = len(self.atomics)
        self.atomics.extend([0] * int(self.bs))

        total_pid_n = int(math.ceil(self.Qd / self.bN))
        total_pid   = self.Mout * total_pid_n

        sm_id = self._get_prev_sm()
        for block_id in range(total_pid):
            pid_m, pid_n = self._compute_pid(block_id, total_pid, self.Mout)
            batch_idx = (pid_m * self.bM) // self.q_len
            self.schedule[sm_id].append([
                self._enc(layer, Op.QKV), pid_m, pid_n, 0,
                self.tpg, in_idx + pid_m, out_idx + batch_idx,
            ])
            sm_id = (sm_id + 1) % self.num_sms
        return out_idx

    def schedule_attn(self, layer, in_idx):
        out_idx = len(self.atomics)
        self.atomics.extend([0] * int(self.bs))

        Mpm          = int(math.ceil(self.q_len / self.bM))
        Nqkv         = int(math.ceil(self.Qd   / self.bN))
        expected_cnt = Mpm * Nqkv
        Qblks        = int(math.ceil(self.q_len / self.block_q))

        sm_id = self._get_prev_sm()
        for b in range(self.bs):
            for h in range(self.num_q_heads):
                for qb in range(Qblks):
                    self.schedule[sm_id].append([
                        self._enc(layer, Op.ATTN), b, h, qb,
                        expected_cnt, in_idx + b, out_idx + b,
                    ])
                    sm_id = (sm_id + 1) % self.num_sms
        return out_idx

    def schedule_out(self, layer, in_idx):
        out_idx = len(self.atomics)
        self.atomics.extend([0] * int(self.Mout))

        total_pid_n  = int(math.ceil(self.E / self.bN))
        total_pid    = self.Mout * total_pid_n
        Qblks        = int(math.ceil(self.q_len / self.block_q))
        expected_cnt = self.num_q_heads * Qblks

        sm_id = self._get_prev_sm()
        for block_id in range(total_pid):
            pid_m, pid_n = self._compute_pid(block_id, total_pid, self.Mout)
            batch_idx = (pid_m * self.bM) // self.q_len
            self.schedule[sm_id].append([
                self._enc(layer, Op.OUT), pid_m, pid_n, 0,
                expected_cnt, in_idx + batch_idx, out_idx + pid_m,
            ])
            sm_id = (sm_id + 1) % self.num_sms
        return out_idx

    def schedule_rms2(self, layer, in_idx):
        out_idx = len(self.atomics)
        self.atomics.extend([0] * int(self.Mout))

        Nout         = int(math.ceil(self.E / self.bN))
        expected_cnt = Nout

        sm_id = self._get_prev_sm()
        for tile_idx in range(self.Mtiles):
            rg = tile_idx // self.tpg
            self.schedule[sm_id].append([
                self._enc(layer, Op.RMS), tile_idx, tile_idx // self.num_sms, 1,
                expected_cnt, in_idx + rg, out_idx + rg,
            ])
            sm_id = (sm_id + 1) % self.num_sms
        return out_idx

    def schedule_up(self, layer, in_idx):
        Nf      = int(math.ceil(self.F / self.bN))
        out_idx = len(self.atomics)
        self.atomics.extend([0] * int(self.Mout * Nf))

        total_pid = self.Mout * Nf
        sm_id = self._get_prev_sm()
        for block_id in range(total_pid):
            pid_m, pid_n = self._compute_pid(block_id, total_pid, self.Mout)
            self.schedule[sm_id].append([
                self._enc(layer, Op.UP), pid_m, pid_n, 0,
                self.tpg, in_idx + pid_m, out_idx + (pid_m * Nf + pid_n),
            ])
            sm_id = (sm_id + 1) % self.num_sms
        return out_idx

    def schedule_gate(self, layer, in_idx):
        Nf      = int(math.ceil(self.F / self.bN))
        out_idx = len(self.atomics)
        self.atomics.extend([0] * int(self.Mout))

        total_pid = self.Mout * Nf
        sm_id = self._get_prev_sm()
        for block_id in range(total_pid):
            pid_m, pid_n = self._compute_pid(block_id, total_pid, self.Mout)
            self.schedule[sm_id].append([
                self._enc(layer, Op.GATE), pid_m, pid_n, 0,
                1, in_idx + (pid_m * Nf + pid_n), out_idx + pid_m,
            ])
            sm_id = (sm_id + 1) % self.num_sms
        return out_idx

    def schedule_down(self, layer, in_idx):
        Nout    = int(math.ceil(self.E / self.bN))
        Nf      = int(math.ceil(self.F / self.bN))
        out_idx = len(self.atomics)
        self.atomics.extend([0] * int(self.Mout))

        total_pid = self.Mout * Nout
        sm_id = self._get_prev_sm()
        for block_id in range(total_pid):
            pid_m, pid_n = self._compute_pid(block_id, total_pid, self.Mout)
            self.schedule[sm_id].append([
                self._enc(layer, Op.DOWN), pid_m, pid_n, 0,
                Nf, in_idx + pid_m, out_idx + pid_m,
            ])
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
            torch.tensor(self.atomics,  dtype=torch.int32),
            max_works,
        )


def get_attn_schedule(config: LLMMegaKernelConfig):
    return OpScheduler(config).build_schedule()


if __name__ == "__main__":
    import sys

    cfg = LLMMegaKernelConfig(
        embed_dim    = 512,
        kv_len       = 256,
        q_len        = 256,
        num_q_heads  = 4,
        num_kv_heads = 4,
        num_layers   = 4,
        ff_dim       = 1024,
        block_rms    = 1,
        block_q      = 64,
        block_kv     = 64,
        bM           = 64,
        bN           = 128,
        bK           = 128,
        bs           = 16,
        num_sms      = 188,
        rows_per_rms_block = int(sys.argv[1]) if len(sys.argv) > 1 else 64,
    )

    scheduler = OpScheduler(cfg)
    schedule_tensor, atomics_tensor, max_works = scheduler.build_schedule()
    print(f"M={scheduler.M}, bM={scheduler.bM}, "
          f"rows_per_rms_block={scheduler.rows_per_rms_block}, "
          f"tpg={scheduler.tpg}, Mout={scheduler.Mout}, "
          f"Mtiles_per_RMS={scheduler.Mtiles}")
    print(f"num_sms={scheduler.num_sms}, max_works_per_SM={max_works}, "
          f"total_atomics={len(scheduler.atomics)}")

    schedule = schedule_tensor.tolist()
    next_idx_map = defaultdict(list)
    for li, lst in enumerate(schedule):
        for ii, node in enumerate(lst):
            next_idx_map[node[-1]].append((li, ii))

    violations = []
    for li, lst in enumerate(schedule):
        for ii, node in enumerate(lst):
            atomic_cnt, prev_idx = node[-3], node[-2]
            if atomic_cnt > 0:
                actual = len(next_idx_map.get(prev_idx, []))
                if actual != atomic_cnt:
                    violations.append(
                        f"  list={li} item={ii} op={node[0]&0x7} layer={node[0]>>3} "
                        f"pid_m={node[1]} pid_n={node[2]} pid_o={node[3]}: "
                        f"expected_cnt={atomic_cnt} but {actual} predecessors "
                        f"write to atomic[{prev_idx}]"
                    )
    if violations:
        print(f"FAIL: {len(violations)} dependency violations")
        for v in violations[:10]:
            print(v)
    else:
        print("SUCCESS: dependency graph valid.")