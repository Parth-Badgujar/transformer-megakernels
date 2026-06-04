import os 
os.environ["CUTE_DSL_LINEINFO"] = "1"
os.environ["CUTE_DSL_KEEP_PTX"] = "1"
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import argparse


from megakernel import LLMMegaKernel, LLMMegaKernelConfig
from scheduler import get_attn_schedule

parser = argparse.ArgumentParser()
parser.add_argument("--n_iters", type = int, default = 1000)
args = parser.parse_args()
n_iters = args.n_iters

dtype  = torch.bfloat16
device = "cuda"

# ---------------------------------------------------------------------------
# PyTorch reference model
# ---------------------------------------------------------------------------
class RMSNormLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype, device=device))

    def forward(self, x):
        x_f   = x.float()
        var   = (x_f * x_f).mean(dim=-1, keepdim=True)
        scale = torch.rsqrt(var)
        return (x_f * scale * self.weight.float()).to(x.dtype)


class GroupedQueryAttention(nn.Module):
    def __init__(self, embed_dim, num_q_heads, num_kv_heads, batch_size, seq_len, is_causal=False):
        super().__init__()
        self.embed_dim    = embed_dim
        self.num_q_heads  = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim     = embed_dim // num_q_heads
        self.batch_size   = batch_size
        self.seq_len      = seq_len
        self.is_causal    = is_causal

        self.q_proj_dim  = num_q_heads  * self.head_dim
        self.kv_proj_dim = num_kv_heads * self.head_dim
        packed_qkv_dim   = self.q_proj_dim + 2 * self.kv_proj_dim

        self.norm     = RMSNormLayer(embed_dim)
        self.qkv_proj = nn.Linear(embed_dim, packed_qkv_dim, bias=False, dtype=dtype, device=device)
        self.out_proj = nn.Linear(embed_dim, embed_dim,      bias=False, dtype=dtype, device=device)

    def forward(self, x):
        residual = x
        h        = self.norm(x)

        qkv = self.qkv_proj(h)
        q_flat, k_flat, v_flat = qkv.split(
            [self.q_proj_dim, self.kv_proj_dim, self.kv_proj_dim], dim=-1
        )

        q = q_flat.view(self.batch_size, self.seq_len, self.num_q_heads,  self.head_dim).transpose(1, 2)
        k = k_flat.view(self.batch_size, self.seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v_flat.view(self.batch_size, self.seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal, enable_gqa=True)
        attn_flat = attn_out.transpose(1, 2).contiguous().view(self.batch_size * self.seq_len, self.embed_dim)
        return residual + self.out_proj(attn_flat)


class SwiGLU(nn.Module):
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, ff_dim, bias=False, dtype=dtype, device=device)
        self.up_proj   = nn.Linear(dim, ff_dim, bias=False, dtype=dtype, device=device)
        self.down_proj = nn.Linear(ff_dim, dim, bias=False, dtype=dtype, device=device)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_q_heads, num_kv_heads, batch_size, seq_len, ff_dim, is_causal=False):
        super().__init__()
        self.attn  = GroupedQueryAttention(embed_dim, num_q_heads, num_kv_heads, batch_size, seq_len, is_causal)
        self.norm2 = RMSNormLayer(embed_dim)
        self.ffn   = SwiGLU(embed_dim, ff_dim)

    def forward(self, x):
        x        = self.attn(x)
        residual = x
        return residual + self.ffn(self.norm2(x))


class MultiLayerTransformer(nn.Module):
    def __init__(self, embed_dim, num_q_heads, num_kv_heads, batch_size, seq_len, ff_dim, num_layers, is_causal=False):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(embed_dim, num_q_heads, num_kv_heads, batch_size, seq_len, ff_dim, is_causal)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Hyperparameters + config
# ---------------------------------------------------------------------------
head_dim     = 128
batch_size   = 16
seq_len      = 256
num_q_heads  = 4
num_kv_heads = 4
num_stages   = 3
is_causal    = False
num_layers   = 4
ff_dim       = 512

embed_dim  = num_q_heads * head_dim
qkv_dim    = (num_q_heads + 2 * num_kv_heads) * head_dim
num_tokens = batch_size * seq_len

torch.manual_seed(1436)
device_prop = torch.cuda.get_device_properties(0)
num_sms = device_prop.multi_processor_count

import math 

def get_rms_block(seq, num_sms, bM):
    x = seq / num_sms
    print("num seq per block :", x)
    return min(bM, 2 ** math.floor(math.log2(x)))

rows_per_rms_block = get_rms_block(batch_size * seq_len, num_sms, bM = 64)
print("SM Count =", num_sms)
print("Using BLOCK RMS =", rows_per_rms_block)

cfg = LLMMegaKernelConfig(
    embed_dim    = embed_dim,
    kv_len       = seq_len,
    q_len        = seq_len,
    num_q_heads  = num_q_heads,
    num_kv_heads = num_kv_heads,
    num_layers   = num_layers,
    ff_dim       = ff_dim,
    block_rms    = 1,
    block_q      = 64,
    block_kv     = 64,
    bM           = 64,
    bN           = 128,
    bK           = 64,
    bs           = batch_size,
    num_sms      = num_sms,
    is_causal    = is_causal,
    num_stages   = num_stages,
    bR                 = 4,
    num_stages_rms     = 3,
    rows_per_rms_block = rows_per_rms_block,
    use_tma_reduce = True,
    output_pad = 16
)

# ---------------------------------------------------------------------------
# Build schedule (sets cfg.max_works)
# ---------------------------------------------------------------------------
sched, atoms, max_works = get_attn_schedule(cfg)
cfg.max_works = max_works
print(f"Max works per SM: {max_works}")

# ---------------------------------------------------------------------------
# Reference model + weight tensors
# ---------------------------------------------------------------------------
model = MultiLayerTransformer(
    embed_dim, num_q_heads, num_kv_heads, batch_size, seq_len, ff_dim, num_layers, is_causal=is_causal
).cuda()

with torch.no_grad():
    for layer in model.layers:
        for w in [layer.attn.qkv_proj, layer.attn.out_proj,
                  layer.ffn.gate_proj, layer.ffn.up_proj, layer.ffn.down_proj]:
            w.weight.mul_(1)
model.eval()

x = torch.randn(num_tokens, embed_dim, dtype=dtype, device=device)

rms_w_list, qkv_w_list, out_w_list = [], [], []
gate_w_list, up_w_list, down_w_list = [], [], []

for layer in model.layers:
    rms_w_list.extend([
        layer.attn.norm.weight.detach().clone(),
        layer.norm2.weight.detach().clone(),
    ])
    qkv_w_list.append(layer.attn.qkv_proj.weight.detach().clone())
    out_w_list.append(layer.attn.out_proj.weight.detach().clone())
    gate_w_list.append(layer.ffn.gate_proj.weight.detach().clone())
    up_w_list.append(layer.ffn.up_proj.weight.detach().clone())
    down_w_list.append(layer.ffn.down_proj.weight.detach().clone())

rms_w  = torch.stack(rms_w_list)
qkv_w  = torch.stack(qkv_w_list)
out_w  = torch.stack(out_w_list)
gate_w = torch.stack(gate_w_list)
up_w   = torch.stack(up_w_list)
down_w = torch.stack(down_w_list)

# ---------------------------------------------------------------------------
# Workspaces + megakernel config
# ---------------------------------------------------------------------------
ws2_dim   = max(qkv_dim, ff_dim)
embedding = x.clone()
ws1 = torch.full((num_tokens, embed_dim), 0.5,  dtype=dtype, device=device)
ws2 = torch.full((num_tokens, ws2_dim),   0.25, dtype=dtype, device=device)

mSchedule = sched.cuda()
mAtomics  = atoms.cuda()

cSchedule  = from_dlpack(mSchedule)
cAtomics   = from_dlpack(mAtomics)
cRms_w     = from_dlpack(rms_w,     assumed_align=16)
cQkv_w     = from_dlpack(qkv_w,     assumed_align=16)
cWs1       = from_dlpack(ws1,       assumed_align=16)
cWs2       = from_dlpack(ws2,       assumed_align=16)
cGate_w    = from_dlpack(gate_w,    assumed_align=16)
cUp_w      = from_dlpack(up_w,      assumed_align=16)
cDown_w    = from_dlpack(down_w,    assumed_align=16)
cOut_w     = from_dlpack(out_w,     assumed_align=16)
cEmbedding = from_dlpack(embedding, assumed_align=16)

kernel = LLMMegaKernel(cfg)

# ---------------------------------------------------------------------------
# Compile + first run
# ---------------------------------------------------------------------------
ref = model(x)

os.system("rm *.ptx")
os.system("rm *.cubin")

print("[compile] starting", flush=True)
t0 = time.time()
compiled = cute.compile(
    kernel,
    cSchedule, cAtomics, cRms_w, cQkv_w, cWs1, cWs2,
    cGate_w, cUp_w, cDown_w, cOut_w, cEmbedding,
)
print(f"[compile] done in {time.time() - t0:.1f}s", flush=True)

compiled(cSchedule, cAtomics, cRms_w, cQkv_w, cWs1, cWs2,
         cGate_w, cUp_w, cDown_w, cOut_w, cEmbedding)
torch.cuda.synchronize()
if os.path.exists("/opt/watchdog/users/parth/cuda13.2/bin/ptxas"):
    os.system("/opt/watchdog/users/parth/cuda13.2/bin/ptxas --gpu-name sm_120a --output-file megakernel.cubin --verbose cutlass*.ptx")
else:
    os.system("ptxas --gpu-name sm_120a --output-file megakernel.cubin --verbose cutlass*.ptx")

torch.save(embedding.cpu(), "embeddings.pt")
torch.save(ref.cpu(), "ref.pt")

diff     = (embedding.float() - ref.float()).abs()
max_err  = diff.max().item()
mean_err = diff.mean().item()
rel_err  = (diff / (ref.float().abs() + 1e-6)).mean().item()

print(f"\nResults:")
print(f"  Max  abs error    : {max_err:.6f}")
print(f"  Mean abs error    : {mean_err:.6f}")
print(f"  Mean rel error    : {rel_err:.6f}")
print(f"  Kernel output norm: {embedding.float().norm():.4f}")
print(f"  Ref    output norm: {ref.float().norm():.4f}\n")

# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
# n_iters = 1
warmup  = 5

model_compile = torch.compile(model)
for _ in range(warmup):
    model_compile(x)

torch.cuda.nvtx.range_push("Compiled")
model_compile(x)
torch.cuda.nvtx.range_pop()

t0 = time.perf_counter_ns()
for _ in range(n_iters):
    model_compile(x)
torch.cuda.synchronize()
print(f"[compile] {((time.perf_counter_ns() - t0) / 1e6) / n_iters:.3f}ms", flush=True)

t0 = time.perf_counter_ns()
for _ in range(n_iters):
    mAtomics.zero_()
    compiled(cSchedule, cAtomics, cRms_w, cQkv_w, cWs1, cWs2,
             cGate_w, cUp_w, cDown_w, cOut_w, cEmbedding)
torch.cuda.synchronize()
print(f"[mega]    {((time.perf_counter_ns() - t0) / 1e6) / n_iters:.3f}ms", flush=True)

t0 = time.perf_counter_ns()
for _ in range(n_iters):
    model(x)
torch.cuda.synchronize()
print(f"[eager]   {((time.perf_counter_ns() - t0) / 1e6) / n_iters:.3f}ms", flush=True)