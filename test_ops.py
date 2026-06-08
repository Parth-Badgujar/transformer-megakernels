"""
test_ops.py  --  op-by-op single-layer validation of the megakernel.

Runs ONE transformer layer, stopping after each op (RMS1, QKV, ATTN, OUT,
RMS2, UP, GATE, DOWN) via a *prefix schedule*, reads back the workspace that op
wrote, and compares it against:
    * an fp32 PyTorch reference  (the truth)
    * a bf16 PyTorch reference   (the rounding floor the kernel can't beat)

The first op whose error >> the bf16 floor is where the bug lives.

Why prefix schedules: the megakernel reuses ws1 (RMS1->ATTN) and ws2 (QKV->UP/
GATE), so a single full run only leaves the final state. Stopping after op K
freezes the intermediate. All prefixes are padded to the full layer's max_works
and atomics length, so the kernel is compiled ONCE and the schedule/atomics GPU
tensors are overwritten per op.

Run:  python test_ops.py
Adjust the import below to your scheduler filename (scheduler.py / new_scheduler.py).
"""
import os
os.environ["CUTE_DSL_LINEINFO"] = "1"
import math
import torch
import torch.nn.functional as F
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from megakernel import LLMMegaKernel, LLMMegaKernelConfig
from scheduler import OpScheduler          # <-- rename if your file is new_scheduler.py

torch.manual_seed(1436)
dtype, device = torch.bfloat16, "cuda"

# ---------------------------------------------------------------------------
# Hyperparameters (match bench.py, but ONE layer)
# ---------------------------------------------------------------------------
head_dim, bs, seq_len = 128, 16, 256
num_q_heads = num_kv_heads = 4
ff_dim      = 512
num_layers  = 1                              # SINGLE layer
embed_dim   = num_q_heads * head_dim         # 512
Qd          = (num_q_heads + 2 * num_kv_heads) * head_dim   # 1536
M           = bs * seq_len
num_sms     = torch.cuda.get_device_properties(0).multi_processor_count

def get_rms_block(seq, nsm):
    return min(64, 2 ** math.floor(math.log2(seq / nsm)))
rows_per_rms_block = get_rms_block(M, num_sms)

cfg = LLMMegaKernelConfig(
    embed_dim=embed_dim, kv_len=seq_len, q_len=seq_len,
    num_q_heads=num_q_heads, num_kv_heads=num_kv_heads, num_layers=num_layers,
    ff_dim=ff_dim, block_rms=1, block_q=64, block_kv=64,
    bM=64, bN=128, bK=64, bs=bs, num_sms=num_sms, is_causal=False,
    num_stages=3, bR=4, rows_per_rms_block=rows_per_rms_block,
    use_tma_reduce=True, output_pad=16,
)

# ---------------------------------------------------------------------------
# Prefix scheduler: first K ops of layer 0, padded to common (atomics, works)
# ---------------------------------------------------------------------------
class PrefixScheduler(OpScheduler):
    def build_prefix(self, n_ops, pad_atomics=None, pad_works=None):
        builders = [self.schedule_rms1, self.schedule_qkv, self.schedule_attn,
                    self.schedule_out,  self.schedule_rms2, self.schedule_up,
                    self.schedule_gate, self.schedule_down]
        in_idx = len(self.atomics)
        self.atomics.extend([0] * int(self.Mout))     # initial "embedding ready" slots
        for k in range(n_ops):
            in_idx = builders[k](0, in_idx)            # layer 0 only
        self.atomics.append(0)
        if pad_atomics is not None:
            self.atomics += [0] * (pad_atomics - len(self.atomics))
        mw = pad_works if pad_works is not None else max((len(s) for s in self.schedule), default=0)
        for s in self.schedule:
            s += [[-1, -1, -1, -1, 0, -1, -1]] * (mw - len(s))
        return (torch.tensor(self.schedule, dtype=torch.int32),
                torch.tensor(self.atomics,  dtype=torch.int32), mw)

OP_NAMES = ["RMS1", "QKV", "ATTN", "OUT", "RMS2", "UP", "GATE", "DOWN"]

# full layer first -> common shapes
full = PrefixScheduler(cfg)
_, full_atoms, W_FULL = full.build_prefix(8)
L_FULL = full_atoms.numel()
cfg.max_works = W_FULL
print(f"num_sms={num_sms} rows_per_rms_block={rows_per_rms_block} "
      f"max_works={W_FULL} atomics_len={L_FULL}")

prefixes = []
for k in range(1, 9):
    s, a, _ = PrefixScheduler(cfg).build_prefix(k, pad_atomics=L_FULL, pad_works=W_FULL)
    prefixes.append((s, a))

# ---------------------------------------------------------------------------
# Weights (bf16 for kernel, .float() for reference). RMS weights RANDOMIZED so
# the weight-multiply path is actually exercised (bench uses ones, which hides
# a weight-application bug).
# ---------------------------------------------------------------------------
def lin(out_f, in_f):                          # nn.Linear-style init, [out, in]
    return (torch.empty(out_f, in_f, dtype=dtype, device=device)
            .uniform_(-1, 1) * (in_f ** -0.5))

rms_w  = (1.0 + 0.1 * torch.randn(2, embed_dim, dtype=dtype, device=device))   # [norm1, norm2]
qkv_w  = lin(Qd, embed_dim)[None]
out_w  = lin(embed_dim, embed_dim)[None]
gate_w = lin(ff_dim, embed_dim)[None]
up_w   = lin(ff_dim, embed_dim)[None]
down_w = lin(embed_dim, ff_dim)[None]

x = torch.randn(M, embed_dim, dtype=dtype, device=device)

# ---------------------------------------------------------------------------
# References (fp32 truth + bf16 floor). `cast` rounds at each kernel rounding pt.
# ---------------------------------------------------------------------------
def attention(qkv):                            # qkv: (M, Qd) float
    qd, kd = num_q_heads * head_dim, num_kv_heads * head_dim
    q = qkv[:, :qd].view(bs, seq_len, num_q_heads,  head_dim).transpose(1, 2)
    k = qkv[:, qd:qd+kd].view(bs, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    v = qkv[:, qd+kd:].view(bs, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    o = F.scaled_dot_product_attention(q, k, v, is_causal=False,
                                       enable_gqa=(num_q_heads != num_kv_heads))
    return o.transpose(1, 2).contiguous().view(M, num_q_heads * head_dim)

def chain(bf16, eps=1e-6):                      # returns list[8] of reference tensors
    c = (lambda t: t.to(torch.bfloat16).float()) if bf16 else (lambda t: t)
    W = {k: v.float() for k, v in
         dict(rms1=rms_w[0], rms2=rms_w[1], qkv=qkv_w[0], out=out_w[0],
              gate=gate_w[0], up=up_w[0], down=down_w[0]).items()}
    xf = x.float()
    def rms(t, w): return t * torch.rsqrt((t * t).mean(-1, keepdim=True) + eps) * w
    h1      = c(rms(xf, W["rms1"]))
    qkv     = c(h1 @ W["qkv"].T)
    attn    = c(attention(qkv))
    attn_x  = c(xf + c(attn @ W["out"].T))
    h2      = c(rms(attn_x, W["rms2"]))
    up      = c(h2 @ W["up"].T)
    gate    = c(F.silu(h2 @ W["gate"].T) * up)
    final   = c(attn_x + c(gate @ W["down"].T))
    return [h1, qkv, attn, attn_x, h2, up, gate, final]

ref_fp32 = chain(bf16=False)
ref_bf16 = chain(bf16=True)

# op K (1-indexed) -> (workspace getter, column count). emb for OUT/DOWN.
WS_OF_OP = {  # name -> ("ws1"|"ws2"|"emb", ncols)
    "RMS1": ("ws1", embed_dim), "QKV": ("ws2", Qd),  "ATTN": ("ws1", embed_dim),
    "OUT":  ("emb", embed_dim), "RMS2": ("ws1", embed_dim), "UP": ("ws2", ff_dim),
    "GATE": ("ws2", ff_dim),    "DOWN": ("emb", embed_dim),
}

# ---------------------------------------------------------------------------
# Persistent GPU tensors (fixed shape -> compile once)
# ---------------------------------------------------------------------------
ws2_dim   = max(Qd, ff_dim)
sched_g   = prefixes[7][0].cuda()              # shape (num_sms, W_FULL, 7)
atoms_g   = prefixes[7][1].cuda()              # shape (L_FULL,)
ws1_g     = torch.zeros(M, embed_dim, dtype=dtype, device=device)
ws2_g     = torch.zeros(M, ws2_dim,   dtype=dtype, device=device)
emb_g     = x.clone()

cS  = from_dlpack(sched_g); cA = from_dlpack(atoms_g)
cR  = from_dlpack(rms_w, assumed_align=16);  cQ = from_dlpack(qkv_w, assumed_align=16)
cW1 = from_dlpack(ws1_g, assumed_align=16);  cW2 = from_dlpack(ws2_g, assumed_align=16)
cG  = from_dlpack(gate_w, assumed_align=16);  cU = from_dlpack(up_w, assumed_align=16)
cD  = from_dlpack(down_w, assumed_align=16);  cO = from_dlpack(out_w, assumed_align=16)
cE  = from_dlpack(emb_g, assumed_align=16)

kernel   = LLMMegaKernel(cfg)
print("[compile] ...", flush=True)
compiled = cute.compile(kernel, cS, cA, cR, cQ, cW1, cW2, cG, cU, cD, cO, cE)
print("[compile] done", flush=True)

# ---------------------------------------------------------------------------
# Run each prefix, compare
# ---------------------------------------------------------------------------
def stats(a, b):
    d = (a - b).abs()
    return d.max().item(), d.mean().item(), (d / (b.abs() + 1e-6)).mean().item()

print(f"\n{'op':5} {'k_vs_fp32 max':>14} {'mean':>10} {'rel':>9} "
      f"{'bf16_floor':>11} {'ratio':>7}  verdict")
for k in range(1, 9):
    name = OP_NAMES[k - 1]
    sched_g.copy_(prefixes[k - 1][0].cuda())
    atoms_g.copy_(prefixes[k - 1][1].cuda())
    ws1_g.zero_(); ws2_g.zero_(); emb_g.copy_(x)        # fresh state each run
    torch.cuda.synchronize()
    compiled(cS, cA, cR, cQ, cW1, cW2, cG, cU, cD, cO, cE)
    torch.cuda.synchronize()

    src, ncol = WS_OF_OP[name]
    if name in ("UP", "GATE"):
        # FFN intermediate is a COMPACT (M, F) view (stride F), packed into the
        # first M*F elements of the (M, ws2_dim) buffer -- NOT ws2_g[:, :F].
        got = ws2_g.reshape(-1)[: M * ff_dim].reshape(M, ff_dim).float()
    else:
        got = {"ws1": ws1_g, "ws2": ws2_g, "emb": emb_g}[src][:, :ncol].float()
    rf, rb = ref_fp32[k - 1][:, :ncol], ref_bf16[k - 1][:, :ncol]
    kmax, kmean, krel = stats(got, rf)
    floor = (rb - rf).abs().max().item()
    ratio = kmax / (floor + 1e-9)
    verdict = "OK" if ratio < 3.0 else "*** SUSPECT ***"
    print(f"{name:5} {kmax:14.6f} {kmean:10.6f} {krel:9.4f} "
          f"{floor:11.6f} {ratio:7.1f}  {verdict}")

print("\nThe first *** SUSPECT *** op is where the kernel diverges beyond bf16; "
      "ops above it are clean. If all are OK, the depth-linear error is pure bf16 "
      "accumulation, not a logic bug.")