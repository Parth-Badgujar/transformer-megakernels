from enum import IntEnum
from dataclasses import dataclass

@dataclass
class MegakernelConfig:
    embed_dim:          int
    kv_len:             int
    q_len:              int
    num_q_heads:        int
    num_kv_heads:       int
    num_layers:         int
    ff_dim:             int
    block_rms:          int
    block_q:            int
    block_kv:           int
    num_stages:         int  = 2          # V2: two-stage
    bM:                 int  = 128        # V2: larger M tile
    bN:                 int  = 64
    bK:                 int  = 64
    bs:                 int  = 8
    num_sms:            int  = 188
    is_causal:          bool = False
    use_tma_reduce:     bool = False
    output_pad:         int  = 8          # V2: 8 so 128*(128+8)*2 = 34 KiB fits
    warps_per_row:      int  = 1          # V2: replaces bR; num_sets = 4//warps_per_row
    rows_per_rms_block: int  = 32
    max_works:          int  = 0          # filled in after scheduling

class Op(IntEnum):
    RMS  = 0
    QKV  = 1
    ATTN = 2
    OUT  = 3
    GATE = 4
    UP   = 5
    DOWN = 6
    NOP  = 99