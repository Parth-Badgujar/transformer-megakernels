import os
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS"] = "ATEN,TRITON"

import time
import torch
import logging
import argparse
from omegaconf import OmegaConf
from transformer_megakernel import InputConfig, KernelConfig, TransformerMegakernel
from transformer_megakernel.model import Transformer

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/default.yaml")
parser.add_argument("--num_iters", type=int, default=100)
parser.add_argument("--warmup", type=int, default=10)
args = parser.parse_args()

num_sms = torch.cuda.get_device_properties(0).multi_processor_count
torch.manual_seed(42)

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("transformer_megakernel").setLevel(logging.INFO)

# Suppress verbose PyTorch / TensorRT compiler logs
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("torch._dynamo").setLevel(logging.WARNING)
logging.getLogger("torch._inductor").setLevel(logging.WARNING)
logging.getLogger("torch_tensorrt").setLevel(logging.WARNING)

logger.info(f"Loading config from {args.config}")
conf = OmegaConf.load(args.config)

input_config = InputConfig(**conf.input_config)
kernel_config_dict = OmegaConf.to_container(conf.kernel_config, resolve=True)
kernel_config_dict['num_sms'] = num_sms
kernel_config_dict['max_works'] = 0
kernel_config = KernelConfig(**kernel_config_dict)

model = Transformer(
    embed_dim=input_config.embed_dim,
    num_q_heads=input_config.num_q_heads,
    num_kv_heads=input_config.num_kv_heads,
    ff_dim=input_config.ff_dim,
    num_layers=input_config.num_layers,
    is_causal=input_config.is_causal,
    dtype=torch.bfloat16,
    device="cuda"
).eval()

num_embeddings = 10
input_embeddings_list = [
    torch.randn(
        input_config.bs, input_config.q_len, input_config.embed_dim,
        device="cuda", dtype=torch.bfloat16
    ) for _ in range(num_embeddings)
]

# ---- Build the megakernel ----
logger.info("Building megakernel for benchmarking...")
megakernel = TransformerMegakernel(model, input_config=input_config, kernel_config=kernel_config)

# Benchmark: torch.compile vs megakernel vs eager
# -----------------------------------------------------------------------------
logger.info("Starting benchmarks...")

start = torch.cuda.Event(enable_timing=True)
stop = torch.cuda.Event(enable_timing=True)
model_compile = torch.compile(model)
for i in range(args.warmup):
    model_compile(input_embeddings_list[i % num_embeddings])
start.record()
for i in range(args.num_iters):
    model_compile(input_embeddings_list[i % num_embeddings])
stop.record()
torch.cuda.synchronize()
time_taken = start.elapsed_time(stop)
logger.info(f"[compile_inductor] time taken per iter: {time_taken / args.num_iters:.3f}ms")

start = torch.cuda.Event(enable_timing=True)
stop = torch.cuda.Event(enable_timing=True)
model_compile_autotune = torch.compile(model, mode="max-autotune")
for i in range(args.warmup):
    model_compile_autotune(input_embeddings_list[i % num_embeddings])
start.record()
for i in range(args.num_iters):
    model_compile_autotune(input_embeddings_list[i % num_embeddings])
stop.record()
torch.cuda.synchronize()
time_taken = start.elapsed_time(stop)
logger.info(f"[compile_max_autotune] time taken per iter: {time_taken / args.num_iters:.3f}ms")

try:
    import torch_tensorrt
    has_trt = True
except ImportError:
    has_trt = False
    logger.warning("torch_tensorrt not found. Skipping TensorRT benchmark. Run `uv pip install torch-tensorrt` to enable.")

if has_trt:
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    model_compile_trt = torch.compile(model, backend="tensorrt")
    for i in range(args.warmup):
        model_compile_trt(input_embeddings_list[i % num_embeddings])
    start.record()
    for i in range(args.num_iters):
        model_compile_trt(input_embeddings_list[i % num_embeddings])
    stop.record()
    torch.cuda.synchronize()
    time_taken = start.elapsed_time(stop)
    logger.info(f"[compile_tensorrt] time taken per iter: {time_taken / args.num_iters:.3f}ms")

start = torch.cuda.Event(enable_timing=True)
stop = torch.cuda.Event(enable_timing=True)
for i in range(args.warmup):
    megakernel(input_embeddings_list[i % num_embeddings].view(-1, input_config.embed_dim))
start.record()
for i in range(args.num_iters):
    megakernel(input_embeddings_list[i % num_embeddings].view(-1, input_config.embed_dim))
stop.record()
torch.cuda.synchronize()
time_taken = start.elapsed_time(stop)
logger.info(f"[mega]    time taken per iter: {time_taken / args.num_iters:.3f}ms")

start = torch.cuda.Event(enable_timing=True)
stop = torch.cuda.Event(enable_timing=True)
for i in range(args.warmup):
    model(input_embeddings_list[i % num_embeddings])
start.record()
for i in range(args.num_iters):
    model(input_embeddings_list[i % num_embeddings])
stop.record()
torch.cuda.synchronize()
time_taken = start.elapsed_time(stop)
logger.info(f"[eager]   time taken per iter: {time_taken / args.num_iters:.3f}ms")
