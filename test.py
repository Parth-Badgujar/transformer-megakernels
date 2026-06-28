import time
import torch
import logging
import argparse
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from transformer_megakernel import InputConfig, KernelConfig, TransformerMegakernel
from transformer_megakernel.model import Transformer

parser = argparse.ArgumentParser()
parser.add_argument("--num_iters", type = int, default = 100)
parser.add_argument("--num_rounds", type = int, default = 100)
parser.add_argument("--warmup", type = int, default = 10)
parser.add_argument("--profile", action="store_true", default=False,
                    help="Enable intrakernel profiler and dump JSON trace")
parser.add_argument("--trace_path", type=str, default="pipeline_trace.json",
                    help="Output path for the profiler JSON trace")

args = parser.parse_args()



torch.manual_seed(42)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

input_config = InputConfig(
    bs = 8,
    embed_dim = 1024,
    kv_len = 256, 
    q_len = 256,
    num_q_heads = 8,
    num_kv_heads = 8,
    num_layers = 8,
    is_causal = True,
    ff_dim = 4096,
)

kernel_config = KernelConfig(
    use_tma_reduce = True,
    output_pad = 8,
    warps_per_row = 1,
    rows_per_rms_block = 16,
    max_works = 0,
    block_q = 64,
    block_kv = 64,
    num_stages = 2,
    bM = 128,
    bN = 128,
    bK = 64,
    num_sms = 188
)

model = Transformer(
    embed_dim = input_config.embed_dim,
    num_q_heads = input_config.num_q_heads,
    num_kv_heads = input_config.num_kv_heads,
    ff_dim = input_config.ff_dim,
    num_layers = input_config.num_layers,
    is_causal = input_config.is_causal,
    dtype = torch.bfloat16,
    device = "cuda"
).eval()

model_f32 = Transformer(
    embed_dim = input_config.embed_dim,
    num_q_heads = input_config.num_q_heads,
    num_kv_heads = input_config.num_kv_heads,
    ff_dim = input_config.ff_dim,
    num_layers = input_config.num_layers,
    is_causal = input_config.is_causal,
    dtype = torch.float32,
    device = "cuda"
).eval()


for n, p in model.named_parameters():
    model_f32.get_parameter(n).data.copy_(p.data.float())

input_embeddings = torch.randn(
    input_config.bs, input_config.q_len, input_config.embed_dim,
    device = "cuda", dtype = torch.bfloat16
)
input_embeddings_f32 = input_embeddings.float()

with torch.no_grad():
    ref_bf16 = model(input_embeddings)
    ref_fp32 = model_f32(input_embeddings_f32)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters : {total_params}", flush = True)

# ---- Build the megakernel (non-profiled for correctness + benchmarks) ----
start = time.time()
megakernel = TransformerMegakernel(model, input_config = input_config, kernel_config = kernel_config)
stop = time.time()
print(f"time taken: {stop - start} seconds", flush = True)

print("Starting Check Rounds", flush = True)
max_errs = []
mean_errs = []
rel_errs = []
max_err_f32_ref = (ref_bf16.float() - ref_fp32.float()).abs().max().item()

for rounds in tqdm(range(args.num_rounds)):
    output = megakernel(input_embeddings.view(-1, input_config.embed_dim)).float()
    output = output.view(input_config.bs, input_config.q_len, -1).float()
    torch.cuda.synchronize()

    diff = (output - ref_bf16.float()).abs()
    max_errs.append(diff.max().item())
    mean_errs.append(diff.mean().item())
    rel_errs.append((diff / (ref_bf16.float().abs() + 1e-6)).mean().item())

    max_err_f32_out = (output - ref_fp32.float()).abs().max().item()

    if rounds == 0:
        print("\nResults:")
        print(f"  Max  abs error (bf16 ref vs f32 ref) : {max_err_f32_ref:.6f}", flush = True)
        print(f"  Max  abs error (bf16 out vs f32 ref) : {max_err_f32_out:.6f}", flush = True)
        print(f"  Max  abs error (bf16 out vs bf16 ref): {max_errs[0]:.6f}", flush = True)
        print(f"  Mean abs error    : {mean_errs[0]:.6f}", flush = True)
        print(f"  Mean rel error    : {rel_errs[0]:.6f}", flush = True)
        print(f"  Kernel output norm: {output.norm():.4f}", flush = True)
        print(f"  Ref bf16 norm     : {ref_bf16.float().norm():.4f}", flush = True)
        print(f"  Ref f32  norm     : {ref_fp32.norm():.4f}\n", flush = True)
    
print("Max  (max errs):",  max(max_errs), flush = True)
print("Max  (mean errs):", max(mean_errs), flush = True)
print("Max  (rel errs):",  max(rel_errs), flush = True)
print("Min  (max errs):",  min(max_errs), flush = True)
print("Min  (mean errs):", min(mean_errs), flush = True)
print("Min  (rel errs):",  min(rel_errs), flush = True)


if args.num_rounds > 1:
    import numpy as np
    np.save("max_errs.npy",  np.array(max_errs))
    np.save("mean_errs.npy", np.array(mean_errs))
    for name, data in (("max_errs", max_errs), ("mean_errs", mean_errs)):
        plt.figure()
        _, _, patches = plt.hist(np.array(data))
        plt.bar_label(patches)
        plt.savefig(f"{name}.png")
        plt.close()
        
# Benchmark: torch.compile vs megakernel vs eager
# -----------------------------------------------------------------------------

start = torch.cuda.Event(enable_timing = True)
stop = torch.cuda.Event(enable_timing = True)
model_compile = torch.compile(model)
for _ in range(args.warmup):
    model_compile(input_embeddings)
start.record()
for _ in range(args.num_iters):
    model_compile(input_embeddings)
stop.record()
torch.cuda.synchronize()
time_taken = start.elapsed_time(stop)
print(f"[compile] {time_taken / args.num_iters:.3f}ms", flush = True)


start = torch.cuda.Event(enable_timing = True)
stop = torch.cuda.Event(enable_timing = True)
for _ in range(args.warmup):
    megakernel(input_embeddings.view(-1, input_config.embed_dim))
start.record()
for _ in range(args.num_iters):
    megakernel(input_embeddings.view(-1, input_config.embed_dim))
stop.record()
torch.cuda.synchronize()
time_taken = start.elapsed_time(stop)
print(f"[mega]    {time_taken / args.num_iters:.3f}ms", flush = True)


start = torch.cuda.Event(enable_timing = True)
stop = torch.cuda.Event(enable_timing = True)
for _ in range(args.warmup):
    model(input_embeddings)
start.record()
for _ in range(args.num_iters):
    model(input_embeddings)
stop.record()
torch.cuda.synchronize()
time_taken = start.elapsed_time(stop)
print(f"[eager]   {time_taken / args.num_iters:.3f}ms", flush = True)


# ---- Intrakernel profiler pass ----
# This runs a separate profiled megakernel after benchmarks are done so that
# the profiler instrumentation doesn't affect benchmark timings.
if args.profile:
    print("\n" + "="*60, flush=True)
    print("INTRAKERNEL PROFILER", flush=True)
    print("="*60, flush=True)
    print(f"Building profiled megakernel (profile=True)...", flush=True)

    profiled_megakernel = TransformerMegakernel(
        model, input_config=input_config, kernel_config=kernel_config,
        profile=True
    )

    # Warmup the profiled kernel
    print(f"Warmup Profiler pass, trace → {args.trace_path}", flush=True)

    for _ in range(3):
        profiled_megakernel(input_embeddings.view(-1, input_config.embed_dim))
    torch.cuda.synchronize()

    # Run the profiled pass — dump_probe is called inside __call__ when profile=True
    print(f"Running profiled pass, trace → {args.trace_path}", flush=True)
    profiled_output = profiled_megakernel(
        input_embeddings.view(-1, input_config.embed_dim),
        trace_path=args.trace_path,
    )
    torch.cuda.synchronize()

    # Verify the profiled kernel produces the same output
    profiled_output_reshaped = profiled_output.view(
        input_config.bs, input_config.q_len, -1
    ).float()
    profiled_diff = (profiled_output_reshaped - ref_bf16.float()).abs()
    print(f"  Profiled max abs error vs bf16 ref: {profiled_diff.max().item():.6f}",
          flush=True)
    print(f"  Trace written to: {args.trace_path}", flush=True)
    print("  Open with: chrome://tracing or https://ui.perfetto.dev", flush=True)
    print("="*60, flush=True)