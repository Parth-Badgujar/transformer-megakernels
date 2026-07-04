import torch
import logging
import argparse
from omegaconf import OmegaConf
from transformer_megakernel import InputConfig, KernelConfig, TransformerMegakernel
from transformer_megakernel.model import Transformer

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/default.yaml")
parser.add_argument("--trace_path", type=str, default="pipeline_trace.json")
args = parser.parse_args()

num_sms = torch.cuda.get_device_properties(0).multi_processor_count
torch.manual_seed(42)

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

input_embeddings = torch.randn(
    input_config.bs, input_config.q_len, input_config.embed_dim,
    device="cuda", dtype=torch.bfloat16
)

logger.info("="*60)
logger.info("INTRAKERNEL PROFILER")
logger.info("="*60)
logger.info("Building profiled megakernel (profile=True)...")

profiled_megakernel = TransformerMegakernel(
    model, input_config=input_config, kernel_config=kernel_config, profile=True
)

logger.info(f"Warmup Profiler pass, trace -> {args.trace_path}")
for _ in range(3):
    profiled_megakernel(input_embeddings.view(-1, input_config.embed_dim))
torch.cuda.synchronize()

logger.info(f"Running profiled pass, trace -> {args.trace_path}")
profiled_output = profiled_megakernel(
    input_embeddings.view(-1, input_config.embed_dim),
    trace_path=args.trace_path,
)
torch.cuda.synchronize()
logger.info(f"Trace written to: {args.trace_path}")
logger.info("Open with: chrome://tracing or https://ui.perfetto.dev")
logger.info("="*60)