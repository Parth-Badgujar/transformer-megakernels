"""
Run bench.py with all 4 combinations of (use_tma_reduce, output_pad).
Patches the config in-place, runs, and saves results.
"""
import os
os.environ["CUTE_DSL_LINEINFO"] = "1"
import time
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from scheduler import get_attn_schedule
from transformer_megakernel import LLMMegaKernel, LLMMegaKernelConfig
from transformer_megakernel.model import MultiLayerTransformer, extract_weights

torch.manual_seed(42)
dtype  = torch.bfloat16
device = "cuda"
device_prop = torch.cuda.get_device_properties(0)
num_sms = device_prop.multi_processor_count

head_dim     = 128
batch_size   = 16
seq_len      = 256
num_q_heads  = 4
num_kv_heads = 4
num_stages   = 3
is_causal    = False
num_layers   = 4
ff_dim       = 512
embed_dim    = num_q_heads * head_dim
qkv_dim      = (num_q_heads + 2 * num_kv_heads) * head_dim
num_tokens   = batch_size * seq_len

def get_rms_block(seq, nsm, bM):
    return min(bM, 2 ** math.floor(math.log2(seq / nsm)))

rows_per_rms_block = get_rms_block(batch_size * seq_len, num_sms, bM=64)

model = MultiLayerTransformer(
    embed_dim, num_q_heads, num_kv_heads, ff_dim, num_layers, is_causal=is_causal
).cuda().eval()

rms_w, qkv_w, out_w, gate_w, up_w, down_w = extract_weights(model)
sample_input = torch.randn(batch_size, seq_len, embed_dim, dtype=dtype, device=device)
ref = model(sample_input)

num_rounds = 500

configs = [
    (False, 0,  "reduce=F_pad=0"),
    (False, 16, "reduce=F_pad=16"),
    (True,  0,  "reduce=T_pad=0"),
    (True,  16, "reduce=T_pad=16"),
]

results = {}

for use_tma_reduce, output_pad, label in configs:
    print(f"\n{'='*60}")
    print(f"  Config: use_tma_reduce={use_tma_reduce}, output_pad={output_pad}")
    print(f"{'='*60}")

    cfg = LLMMegaKernelConfig(
        embed_dim=embed_dim, kv_len=seq_len, q_len=seq_len,
        num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
        num_layers=num_layers, ff_dim=ff_dim, block_rms=1,
        block_q=64, block_kv=64, bM=64, bN=128, bK=64,
        bs=batch_size, num_sms=num_sms, is_causal=is_causal,
        num_stages=num_stages, bR=4,
        rows_per_rms_block=rows_per_rms_block,
        use_tma_reduce=use_tma_reduce,
        output_pad=output_pad,
    )
    sched, atoms, max_works = get_attn_schedule(cfg)
    cfg.max_works = max_works

    ws2_dim   = max(qkv_dim, ff_dim)
    embedding = sample_input.view(num_tokens, -1)
    ws1 = torch.zeros(num_tokens, embed_dim, dtype=dtype, device=device)
    ws2 = torch.zeros(num_tokens, ws2_dim,   dtype=dtype, device=device)

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

    embeddings_arr = [embedding.clone() for _ in range(num_rounds)]
    c_emb_arr = [from_dlpack(e, assumed_align=16) for e in embeddings_arr]

    kernel = LLMMegaKernel(cfg)
    print("[compile] ...", flush=True)
    compiled = cute.compile(
        kernel, cSchedule, cAtomics, cRms_w, cQkv_w, cWs1, cWs2,
        cGate_w, cUp_w, cDown_w, cOut_w, c_emb_arr[0],
    )
    print("[compile] done", flush=True)

    max_errs = []
    mean_errs = []
    for i in range(num_rounds):
        mAtomics.zero_()
        compiled(cSchedule, cAtomics, cRms_w, cQkv_w, cWs1, cWs2,
                 cGate_w, cUp_w, cDown_w, cOut_w, c_emb_arr[i])
        torch.cuda.synchronize()
        out = embeddings_arr[i].view(batch_size, seq_len, -1)
        diff = (out.float() - ref.float()).abs()
        max_errs.append(diff.max().item())
        mean_errs.append(diff.mean().item())

    arr = np.array(max_errs)
    mean_arr = np.array(mean_errs)
    results[label] = (arr, mean_arr)

    mode_val = np.median(arr)  # approximate mode
    correct = np.sum(arr <= mode_val * 1.5)
    print(f"  Correct (within 1.5x median): {correct}/{num_rounds}")
    print(f"  Max err range: [{arr.min():.6f}, {arr.max():.6f}]")
    print(f"  Mean err range: [{mean_arr.min():.6f}, {mean_arr.max():.6f}]")

# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, (label, (max_arr, mean_arr)) in enumerate(results.items()):
    ax = axes[idx // 2][idx % 2]
    _, _, patches = ax.hist(max_arr, bins=20)
    ax.bar_label(patches)
    ax.set_title(f"Max Errors: {label}")
    ax.set_xlabel("Max Abs Error")
    ax.set_ylabel("Count")

plt.tight_layout()
plt.savefig("experiment_comparison.png", dpi=100)
print("\nSaved experiment_comparison.png")
