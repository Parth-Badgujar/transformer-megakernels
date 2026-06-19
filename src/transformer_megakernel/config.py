from enum import IntEnum
from dataclasses import dataclass

@dataclass
class KernelConfig:
    use_tma_reduce: bool
    output_pad: int
    warps_per_row: int
    rows_per_rms_block: int
    max_works: int
    block_q: int
    block_kv: int
    num_stages: int
    bM: int
    bN: int
    bK: int
    num_sms: int

@dataclass
class InputConfig:
    bs: int
    embed_dim: int
    kv_len: int
    q_len: int
    num_q_heads: int
    num_kv_heads: int
    num_layers: int
    ff_dim: int
    is_causal: bool

class Op(IntEnum):
    RMS = 0
    QKV = 1
    ATTN = 2
    OUT = 3
    GATE = 4
    UP = 5
    DOWN = 6
    NOP  = 99