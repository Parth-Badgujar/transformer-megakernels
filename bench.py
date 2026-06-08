import os
os.environ["CUTE_DSL_LINEINFO"] = "1"
os.environ["CUTE_DSL_KEEP_PTX"] = "1"
import time
import math
import torch
import argparse
import subprocess
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from scheduler import get_attn_schedule
from megakernel import LLMMegaKernel, LLMMegaKernelConfig
from megakernel.model import MultiLayerTransformer, extract_weights

parser = argparse.ArgumentParser()
parser.add_argument("--n_iters", type = int, default = 100)
args = parser.parse_args()
n_iters = args.n_iters

dtype  = torch.bfloat16
device = "cuda"
torch.manual_seed(42)
device_prop = torch.cuda.get_device_properties(0)
num_sms = device_prop.multi_processor_count

def get_rms_block(seq, num_sms, bM):
    x = seq / num_sms
    print("num seq per block :", x)
    return min(bM, 2 ** math.floor(math.log2(x)))


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
    rows_per_rms_block = rows_per_rms_block,
    use_tma_reduce = True,
    output_pad = 16
)

sched, atoms, max_works = get_attn_schedule(cfg)
cfg.max_works = max_works
print(f"Max works per SM: {max_works}")


model = MultiLayerTransformer(
    embed_dim, num_q_heads, num_kv_heads, ff_dim, num_layers, is_causal=is_causal
).cuda()
model.eval()

model_f32 = MultiLayerTransformer(
    embed_dim, num_q_heads, num_kv_heads, ff_dim, num_layers, is_causal=is_causal, dtype = torch.float32
).cuda()
model_f32.eval()

for n,p in model.named_parameters():
    p_f32 = model_f32.get_parameter(n)
    p_f32.data.copy_(p.data.float())

s = 0
for p in model.parameters():
    s += p.numel()
print("Total Parameters :", s)

rms_w, qkv_w, out_w, gate_w, up_w, down_w = extract_weights(model)

sample_input = torch.randn(batch_size, seq_len, embed_dim, dtype=dtype, device=device)
sample_input_f32 = sample_input.float()

ws2_dim   = max(qkv_dim, ff_dim)
embedding = sample_input.view(batch_size * seq_len, -1)
ws1 = torch.full((num_tokens, embed_dim), 0,  dtype=dtype, device=device)
ws2 = torch.full((num_tokens, ws2_dim),   0, dtype=dtype, device=device)

mSchedule = sched.cuda()
mAtomics  = atoms.cuda()

cSchedule  = from_dlpack(mSchedule)
cAtomics   = from_dlpack(mAtomics)
cRms_w     = from_dlpack(rms_w,     assumed_align = 16)
cQkv_w     = from_dlpack(qkv_w,     assumed_align = 16)
cWs1       = from_dlpack(ws1,       assumed_align = 16)
cWs2       = from_dlpack(ws2,       assumed_align = 16)
cGate_w    = from_dlpack(gate_w,    assumed_align = 16)
cUp_w      = from_dlpack(up_w,      assumed_align = 16)
cDown_w    = from_dlpack(down_w,    assumed_align = 16)
cOut_w     = from_dlpack(out_w,     assumed_align = 16)
embeddings_arr = []
num_rounds = 1
for i in range(num_rounds):
    embeddings_arr.append(embedding.clone())
c_embedding_arr = []
for i in range(num_rounds):
    c_embedding_arr.append(from_dlpack(embeddings_arr[i], assumed_align = 16))
cEmbedding = from_dlpack(embedding, assumed_align = 16)

kernel = LLMMegaKernel(cfg)
subprocess.run(["rm", "*.ptx"])
subprocess.run(["rm", "*.cubin"])
ref = model(sample_input)
ref_f32 = model_f32(sample_input_f32)
print("[compile] starting", flush=True)
t0 = time.time()
compiled = cute.compile(
    kernel,
    cSchedule, cAtomics, cRms_w, cQkv_w, cWs1, cWs2,
    cGate_w, cUp_w, cDown_w, cOut_w, cEmbedding,
)
print(f"[compile] done in {time.time() - t0:.2f}s", flush=True)


if os.path.exists("/opt/watchdog/users/parth/cuda13.2/bin/ptxas"):
    subprocess.run(["/opt/watchdog/users/parth/cuda13.2/bin/ptxas", "--gpu-name", "sm_120a", "--output-file", "megakernel.cubin", "--verbose", "cutlass*.ptx"])
else:
    subprocess.run(["ptxas", "--gpu-name", "sm_120a", "--output-file", "megakernel.cubin", "--verbose", "cutlass*.ptx"])

max_errs = []
mean_errs = []
rel_errs = []
for i in range(num_rounds):
    mAtomics.zero_()
    compiled(cSchedule, cAtomics, cRms_w, cQkv_w, cWs1, cWs2,
            cGate_w, cUp_w, cDown_w, cOut_w, c_embedding_arr[i])
    torch.cuda.synchronize()
    torch.save(ref, f"ref_v{i}.pt")
    embeddings_arr[i] = embeddings_arr[i].view(batch_size, seq_len, -1)
    torch.save(embeddings_arr[i], f"out_v{i}.pt")
    diff     = (embeddings_arr[i].float() - ref.float()).abs()
    diff_f32_out = (embeddings_arr[i].float() - ref_f32.float()).abs()
    diff_f32_ref = (ref.float() - ref_f32.float()).abs()
    max_err  = diff.max().item()
    max_err_f32_out = diff_f32_out.max().item()
    max_err_f32_ref = diff_f32_ref.max().item()
    
    mean_err = diff.mean().item()
    rel_err  = (diff / (ref.float().abs() + 1e-6)).mean().item()
    max_errs.append(max_err)
    mean_errs.append(mean_err)
    rel_errs.append(rel_err)
    if i == 0:
        print(f"\nResults:")
        print(f"  Max  abs error (bf16 ref vs f32 ref) : {max_err_f32_ref:.6f}")
        print(f"  Max  abs error (bf16 out vs f32 ref)  : {max_err_f32_out:.6f}")
        print(f"  Max  abs error (bf16 out vs bf16 ref): {max_err:.6f}")
        print(f"  Mean abs error    : {mean_err:.6f}")
        print(f"  Mean rel error    : {rel_err:.6f}")
        print(f"  Kernel output norm: {embeddings_arr[i].float().norm():.4f}")
        print(f"  Ref bf16 output norm: {ref.float().norm():.4f}")
        print(f"  Ref f32 output norm: {ref_f32.norm():.4f}\n")

import numpy as np
import matplotlib.pyplot as plt
arr = np.array(max_errs)
np.save("max_errs.npy", arr)
fig = plt.figure()
_, _, patches = plt.hist(arr)
plt.bar_label(patches)
plt.savefig("max_errs.png")

mean_arr = np.array(mean_errs)
np.save("mean_errs.npy", mean_arr)
fig = plt.figure()
_, _, patches  =plt.hist(mean_arr)
plt.bar_label(patches)
plt.savefig("mean_errs.png")


print("Max (max errs)", max(max_errs))
print("Max (mean errs)", max(mean_errs))
print("Max (rel errs)", max(rel_errs))

print("Min (max errs)", min(max_errs))
print("Min (mean errs)", min(mean_errs))
print("Min (rel errs)", min(rel_errs))



warmup  = 5

model_compile = torch.compile(model)
for _ in range(warmup):
    model_compile(sample_input)

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