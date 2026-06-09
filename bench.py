import os
os.environ["CUTE_DSL_LINEINFO"]  = "1"
os.environ["CUTE_DSL_KEEP_PTX"]  = "1"

import time
import math
import glob
import argparse
import subprocess

import torch
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from megakernel.scheduler import get_attn_schedule
from megakernel import LLMMegaKernel, LLMMegaKernelConfig
from megakernel.model import MultiLayerTransformer, extract_weights


# -----------------------------------------------------------------------------
# Args / setup
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--n_iters",    type=int, default=100)
parser.add_argument("--num_rounds", type=int, default=1, help="correctness rounds")
args = parser.parse_args()
n_iters    = args.n_iters
num_rounds = args.num_rounds

dtype  = torch.bfloat16
device = "cuda"
torch.manual_seed(42)

device_prop = torch.cuda.get_device_properties(0)
num_sms     = device_prop.multi_processor_count


def get_rms_block(seq, num_sms, bM):
    x = seq / num_sms
    print("num seq per block :", x)
    return min(bM, 2 ** math.floor(math.log2(x)))


# -----------------------------------------------------------------------------
# Problem / config  (V2: two-stage, bM=64 for now)
# -----------------------------------------------------------------------------
head_dim      = 128
batch_size    = 16
seq_len       = 256
num_q_heads   = 4
num_kv_heads  = 4
num_stages    = 2          # V2 is two-stage
is_causal     = False
num_layers    = 4
ff_dim        = 512
warps_per_row = 1          # V2: replaces bR; num_sets = 4 // warps_per_row

embed_dim  = num_q_heads * head_dim
qkv_dim    = (num_q_heads + 2 * num_kv_heads) * head_dim
num_tokens = batch_size * seq_len

bM = 64
num_sets = 4 // warps_per_row
rows_per_rms_block = get_rms_block(num_tokens, num_sms, bM=bM)
# RMS tile must be a multiple of num_sets and divide bM
rows_per_rms_block = max(rows_per_rms_block, num_sets)

print("SM Count =", num_sms)
print("Using BLOCK RMS =", rows_per_rms_block)

cfg = LLMMegaKernelConfig(
    embed_dim          = embed_dim,
    kv_len             = seq_len,
    q_len              = seq_len,
    num_q_heads        = num_q_heads,
    num_kv_heads       = num_kv_heads,
    num_layers         = num_layers,
    ff_dim             = ff_dim,
    block_rms          = 1,
    block_q            = 64,
    block_kv           = 64,
    num_stages         = num_stages,
    bM                 = bM,
    bN                 = 128,
    bK                 = 64,
    bs                 = batch_size,
    num_sms            = num_sms,
    is_causal          = is_causal,
    warps_per_row      = warps_per_row,
    rows_per_rms_block = rows_per_rms_block,
    use_tma_reduce     = True,
    output_pad         = 8,
)

sched, atoms, max_works = get_attn_schedule(cfg)
cfg.max_works = max_works
print(f"Max works per SM: {max_works}")


# -----------------------------------------------------------------------------
# Reference models (bf16 + fp32) and weights
# -----------------------------------------------------------------------------
model = MultiLayerTransformer(
    embed_dim, num_q_heads, num_kv_heads, ff_dim, num_layers, is_causal=is_causal
).cuda()
model.eval()

model_f32 = MultiLayerTransformer(
    embed_dim, num_q_heads, num_kv_heads, ff_dim, num_layers,
    is_causal=is_causal, dtype=torch.float32,
).cuda()
model_f32.eval()

for n, p in model.named_parameters():
    model_f32.get_parameter(n).data.copy_(p.data.float())

total_params = sum(p.numel() for p in model.parameters())
print("Total Parameters :", total_params)

rms_w, qkv_w, out_w, gate_w, up_w, down_w = extract_weights(model)

sample_input     = torch.randn(batch_size, seq_len, embed_dim, dtype=dtype, device=device)
sample_input_f32 = sample_input.float()

ws2_dim   = max(qkv_dim, ff_dim)
embedding = sample_input.view(num_tokens, -1)
ws1 = torch.zeros((num_tokens, embed_dim), dtype=dtype, device=device)
ws2 = torch.zeros((num_tokens, ws2_dim),   dtype=dtype, device=device)

mSchedule = sched.cuda()
mAtomics  = atoms.cuda()

# DLPack views
cSchedule = from_dlpack(mSchedule)
cAtomics  = from_dlpack(mAtomics)
cRms_w    = from_dlpack(rms_w,  assumed_align=16)
cQkv_w    = from_dlpack(qkv_w,  assumed_align=16)
cWs1      = from_dlpack(ws1,    assumed_align=16)
cWs2      = from_dlpack(ws2,    assumed_align=16)
cGate_w   = from_dlpack(gate_w, assumed_align=16)
cUp_w     = from_dlpack(up_w,   assumed_align=16)
cDown_w   = from_dlpack(down_w, assumed_align=16)
cOut_w    = from_dlpack(out_w,  assumed_align=16)

# fresh embedding buffer per correctness round (kernel writes in place)
embeddings_arr   = [embedding.clone() for _ in range(num_rounds)]
c_embedding_arr  = [from_dlpack(e, assumed_align=16) for e in embeddings_arr]
cEmbedding       = from_dlpack(embedding, assumed_align=16)


# -----------------------------------------------------------------------------
# Compile
# -----------------------------------------------------------------------------
kernel = LLMMegaKernel(cfg)

# clear stale artifacts (shell glob needs shell=True)
subprocess.run("rm -f *.ptx *.cubin", shell=True)

ref     = model(sample_input)
ref_f32 = model_f32(sample_input_f32)

print("[compile] starting", flush=True)
t0 = time.time()
compiled = cute.compile(
    kernel,
    cSchedule, cAtomics, cRms_w, cQkv_w, cWs1, cWs2,
    cGate_w, cUp_w, cDown_w, cOut_w, cEmbedding,
)
print(f"[compile] done in {time.time() - t0:.2f}s", flush=True)

# ptxas pass (glob in python so we don't depend on a shell)
ptxas = "/opt/watchdog/users/parth/cuda13.2/bin/ptxas"
if not os.path.exists(ptxas):
    ptxas = "ptxas"
ptx_files = sorted(glob.glob("cutlass*.ptx"))
if ptx_files:
    subprocess.run(
        [ptxas, "--gpu-name", "sm_120a", "--output-file", "megakernel.cubin",
         "--verbose", *ptx_files]
    )
else:
    print("[ptxas] no cutlass*.ptx found (set CUTE_DSL_KEEP_PTX=1); skipping")


# -----------------------------------------------------------------------------
# Correctness
# -----------------------------------------------------------------------------
max_errs, mean_errs, rel_errs = [], [], []
for i in range(num_rounds):
    mAtomics.zero_()
    compiled(cSchedule, cAtomics, cRms_w, cQkv_w, cWs1, cWs2,
             cGate_w, cUp_w, cDown_w, cOut_w, c_embedding_arr[i])
    torch.cuda.synchronize()

    out = embeddings_arr[i].view(batch_size, seq_len, -1).float()
    diff = (out - ref.float()).abs()

    max_errs.append(diff.max().item())
    mean_errs.append(diff.mean().item())
    rel_errs.append((diff / (ref.float().abs() + 1e-6)).mean().item())

    if i == 0:
        max_err_f32_out = (out - ref_f32.float()).abs().max().item()
        max_err_f32_ref = (ref.float() - ref_f32.float()).abs().max().item()
        print("\nResults:")
        print(f"  Max  abs error (bf16 ref vs f32 ref) : {max_err_f32_ref:.6f}")
        print(f"  Max  abs error (bf16 out vs f32 ref) : {max_err_f32_out:.6f}")
        print(f"  Max  abs error (bf16 out vs bf16 ref): {max_errs[0]:.6f}")
        print(f"  Mean abs error    : {mean_errs[0]:.6f}")
        print(f"  Mean rel error    : {rel_errs[0]:.6f}")
        print(f"  Kernel output norm: {out.norm():.4f}")
        print(f"  Ref bf16 norm     : {ref.float().norm():.4f}")
        print(f"  Ref f32  norm     : {ref_f32.norm():.4f}\n")

print("Max  (max errs):",  max(max_errs))
print("Max  (mean errs):", max(mean_errs))
print("Max  (rel errs):",  max(rel_errs))
print("Min  (max errs):",  min(max_errs))
print("Min  (mean errs):", min(mean_errs))
print("Min  (rel errs):",  min(rel_errs))

# histograms are only meaningful across many rounds
if num_rounds > 1:
    import numpy as np
    np.save("max_errs.npy",  np.array(max_errs))
    np.save("mean_errs.npy", np.array(mean_errs))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for name, data in (("max_errs", max_errs), ("mean_errs", mean_errs)):
            plt.figure()
            _, _, patches = plt.hist(np.array(data))
            plt.bar_label(patches)
            plt.savefig(f"{name}.png")
            plt.close()
    except ImportError:
        print("[plot] matplotlib not available; saved .npy only")


# -----------------------------------------------------------------------------
# Benchmark: torch.compile vs megakernel vs eager
# -----------------------------------------------------------------------------
warmup = 5

model_compile = torch.compile(model)
for _ in range(warmup):
    model_compile(sample_input)
torch.cuda.synchronize()

t0 = time.perf_counter_ns()
for _ in range(n_iters):
    model_compile(sample_input)
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
    model(sample_input)
torch.cuda.synchronize()
print(f"[eager]   {((time.perf_counter_ns() - t0) / 1e6) / n_iters:.3f}ms", flush=True)