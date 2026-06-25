from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.lib.stride_tricks import as_strided
from enum import IntEnum
import math
import islpy as isl

# from transformer_megakernel.config import Op
# from transformer_megakernel.operators.matmul import MatmulConfig

def build_isl_map_str(shape: tuple, stride: tuple, tiler: tuple, index: tuple, offset: int) -> str:
    """
    Dynamically generates an ISL polyhedral map string for an N-dimensional tile,
    incorporating an absolute memory offset.
    """
    ndims = len(shape)
    
    # 1. Generate variable names: c0, c1, c2, ...
    vars_list = [f"c{i}" for i in range(ndims)]
    vars_str = ", ".join(vars_list)
    
    # 2. Generate the flat memory offset equation, starting with the base offset
    offset_terms = [f"{vars_list[i]}*{stride[i]}" for i in range(ndims)]
    offset_expr = str(offset)
    if offset_terms:
        offset_expr += " + " + " + ".join(offset_terms)
    
    # 3. Generate the strict tile boundaries
    bounds = []
    for i in range(ndims):
        start = index[i] * tiler[i]
        end = min((index[i] + 1) * tiler[i], shape[i]) # Prevent out-of-bounds
        
        # If the tile index is completely outside the parent shape, return an empty set
        if start >= end:
            return f"{{ [{vars_str}] -> [offset] : 1 = 0 }}"
            
        bounds.append(f"{start} <= {vars_list[i]} < {end}")
        
    bounds_str = " and ".join(bounds) if bounds else "1 = 1"
    
    # 4. Assemble the final ISL string
    return f"{{ [{vars_str}] -> [offset] : offset = {offset_expr} and {bounds_str} }}"

class Op(IntEnum):
    RMS = 0
    QKV = 1
    ATTN = 2
    OUT = 3
    GATE = 4
    UP = 5
    DOWN = 6
    NOP  = 99


@dataclass
class AttentionConfig:
    bQ: int
    bKV: int
    head_dim: int
    num_q_heads: int
    num_kv_heads: int
    q_len: int
    kv_len: int
    output_pad: int
    num_stages: int
    stage_elements: int
    is_causal: bool

    def __post_init__(self):
        assert self.bQ == 64
        assert self.bKV == 64
        assert self.head_dim in (64, 128)
        assert self.num_stages == 2, "V2 attention is two-stage"
        assert self.stage_elements >= 2 * self.bKV * self.head_dim



@dataclass
class MatmulConfig:
    bM: int
    bN: int
    bK: int
    num_stages: int
    stage_elements: int
    output_pad: int
    use_tma_reduce: bool
    group_m = 8

    def __post_init__(self):
        self.stage_size  = (self.bM + self.bN) * self.bK
        self.swizzle_bits = (int(math.log2(self.bK)) - 3, 4, 3)
        assert self.bM in (32, 64, 128)
        assert self.bN in (32, 64, 128)
        assert self.bK in (32, 64)
        assert self.num_stages == 2, "V2 matmul is two-stage"
        assert self.stage_elements >= self.stage_size
        assert self.output_pad in (0, 8, 16)


@dataclass
class RMSNormConfig:
    embed_dim: int
    bRMS: int
    stage_elements: int
    warps_per_row: int = 1

    def __post_init__(self):
        assert 4 % self.warps_per_row == 0, "Warps per row should divide 4 (max 4 warps per row)"
        assert (self.bRMS % (4 // self.warps_per_row)) == 0, "bM should be divisible by (4 // warps_per_row)"
        assert self.embed_dim % (32 * self.warps_per_row) == 0, "Embed dim should be divisible by (32 * warps_per_row)"

import copy

@dataclass
class AtomicState:
    atomic_index: int

@dataclass
class TensorPartition:
    parent: Tensor
    tiler: tuple[int, ...]
    index: tuple[int, ...]
    offset: int = 0

    def intersect(self, other: TensorPartition) -> bool:
        map_a_str = build_isl_map_str(
            shape=self.parent.shape, 
            stride=self.parent.stride, 
            tiler=self.tiler, 
            index=self.index,
            offset=self.offset
        )
        map_b_str = build_isl_map_str(
            shape=other.parent.shape, 
            stride=other.parent.stride, 
            tiler=other.tiler, 
            index=other.index,
            offset=other.offset
        )
        
        map_a = isl.Map(map_a_str)
        map_b = isl.Map(map_b_str)
        
        intersection = map_a.range().intersect(map_b.range())
        
        return not intersection.is_empty()
    
    def __repr__(self):
        return (
            f"TensorPartition(\n"
            f"    base={hex(self.parent._base.ctypes.data)},\n"
            f"    shape={self.parent.shape},\n"
            f"    stride={self.parent.stride},\n"
            f"    tiler={self.tiler},\n"
            f"    index={self.index},\n"
            f"    offset={self.offset}\n"
            f")"
        )

class Tensor:
    def __init__(self, shape, stride, is_weight = True):
        self.shape = shape
        self.stride = stride
        self._base = np.arange(1)
        self.is_weight = is_weight
        self._input_op = None
        self._output_op = None

    def partition(self, tiler, index):
        return TensorPartition(self, tiler, index)

    def set_input_op(self, op):
        self._input_op = op

    def set_output_op(self, op):
        self._output_op = op

    def get_output_op(self):
        return self._output_op

    def get_input_op(self):
        return self._input_op

    def reshape(self, shape, stride):
        new_tensor = copy.deepcopy(self)
        new_tensor.shape = shape
        new_tensor.stride = stride
        return new_tensor

@dataclass
class ComputeTile:
    op: Op
    input_tile: TensorPartition
    output_tile: TensorPartition
    pid_m: int
    pid_n: int
    pid_k: int
    layer_idx: int
    prev_atomic = None
    next_atomic = None
    atomic_count = 0

    def get_instr(self):
        return [
            self.layer_idx << 3 | int(self.op.value),
            self.pid_m, self.pid_n, self.pid_k,
            self.atomic_count,
            self.next_atomic if self.next_atomic is not None else -1,
            self.prev_atomic if self.prev_atomic is not None else -1
        ]

class TileOP:
    def __init__(self, *args, **kwargs):
        self.compute_tiles: list[ComputeTile]
        self.op: Op
    def __call__(self, *args, **kwargs) -> Tensor:
        pass

    def add_atomic(self, curr_compute_tile, next_compute_tile, atomics: AtomicState):
        if curr_compute_tile.next_atomic is not None:
            next_compute_tile.prev_atomic = curr_compute_tile.next_atomic
        elif next_compute_tile.prev_atomic is not None:
            curr_compute_tile.next_atomic = next_compute_tile.prev_atomic
        else:
            curr_compute_tile.next_atomic = atomics.atomic_index
            next_compute_tile.prev_atomic = atomics.atomic_index
            atomics.atomic_index += 1


class Matmul(TileOP):
    def __init__(self,
        M: int, N: int, K: int,
        layer_idx: int,
        config: MatmulConfig,
        op_kind: Op
    ):
        self.M = M
        self.N = N
        self.K = K
        self.bM = config.bM
        self.bN = config.bN
        self.bK = config.bK
        self.layer_idx = layer_idx
        self.group_m = config.group_m
        self.op = op_kind
        self.compute_tiles: list[ComputeTile] = []

    def _compute_pid(self, total_pid, total_pid_m, group_size_m = 8):
        block_id = 0
        while block_id < total_pid:
            total_pid_n = total_pid // total_pid_m
            gsm = min(group_size_m, total_pid_m)
            grp = gsm * total_pid_n
            g = block_id // grp
            fm = g * gsm
            sz = min(total_pid_m - fm, gsm)
            pid_m = fm + block_id % sz
            pid_n = (block_id % grp) // sz
            yield pid_m, pid_n
            block_id += 1

    def __call__(self,
        A: Tensor,
        B: Tensor,
        atomic_state: AtomicState,
        C: Tensor = None
    ):
        M, K = A.shape[0], A.shape[1]
        assert K == B.shape[0], "Incompatible matmul shapes"
        N = B.shape[1]
        out = Tensor(shape = (M, N), stride = (N, 1), is_weight = False)

        A.set_input_op(self)
        B.set_input_op(self)
        if C is not None:
            C.set_input_op(self)
        out.set_output_op(self)

        tiler_D = (self.bM, self.bN)
        tiler_A = (self.bM, K)
        total_pid_m = M // self.bM
        total_pid_n = N // self.bN
        total_pid = total_pid_m * total_pid_n

        for pid_m, pid_n in self._compute_pid(total_pid, total_pid_m, self.group_m):
            self.compute_tiles.append(
                ComputeTile(
                    op = self.op,
                    input_tile = TensorPartition(A, tiler_A, (pid_m, 0)),
                    output_tile = TensorPartition(out, tiler_D, (pid_m, pid_n)),
                    pid_m = pid_m,
                    pid_n = pid_n,
                    pid_k = 0,
                    layer_idx = self.layer_idx
                )
            )

        prev_op = A._output_op
        if prev_op is None:
            return out
        for curr_tile in self.compute_tiles:
            for prev_tile in prev_op.compute_tiles:
                if curr_tile.input_tile.intersect(prev_tile.output_tile):
                    curr_tile.atomic_count += 1
                    prev_op.add_atomic(prev_tile, curr_tile, atomic_state)
        return out

class RMSNorm(TileOP):
    def __init__(self, layer_idx: int, config: RMSNormConfig, op_kind: Op):
        self.embed_dim = config.embed_dim
        self.bRMS = config.bRMS
        self.op = op_kind
        self.compute_tiles: list[ComputeTile] = []
        self.layer_idx = layer_idx

    def __call__(self, A: Tensor, W: Tensor, atomic_state: AtomicState):
        N = A.shape[0]
        assert A.shape[1] == self.embed_dim
        out = Tensor(shape = (N, self.embed_dim), stride = (self.embed_dim, 1), is_weight = False)
        A.set_input_op(self)
        W.set_input_op(self)
        out.set_output_op(self)

        for pid_m in range(0, N // self.bRMS):
            self.compute_tiles.append(
                ComputeTile(
                    op = self.op,
                    input_tile = TensorPartition(A, (self.bRMS, self.embed_dim), (pid_m, 0)),
                    output_tile = TensorPartition(out, (self.bRMS, self.embed_dim), (pid_m, 0)),
                    pid_m = pid_m,
                    pid_n = 0,
                    pid_k = 0,
                    layer_idx = self.layer_idx
                )
            )

        prev_op = A.get_output_op()
        if prev_op is None:
            return out
        for curr_tile in self.compute_tiles:
            for prev_tile in prev_op.compute_tiles:
                if curr_tile.input_tile.intersect(prev_tile.output_tile):
                    curr_tile.atomic_count += 1
                    prev_op.add_atomic(prev_tile, curr_tile, atomic_state)
        return out

class Attention(TileOP):
    def __init__(self, layer_idx: int, config: AttentionConfig, op_kind: Op):
        self.bQ = config.bQ
        self.bKV = config.bKV
        self.head_dim = config.head_dim
        self.num_q_heads = config.num_q_heads
        self.num_kv_heads = config.num_kv_heads
        self.q_len = config.q_len
        self.kv_len = config.kv_len
        self.output_pad = config.output_pad
        self.num_stages = config.num_stages
        self.stage_elements = config.stage_elements
        self.is_causal = config.is_causal
        self.op = op_kind
        self.compute_tiles: list[ComputeTile] = []
        self.layer_idx = layer_idx

    def __call__(self, QKV: Tensor, atomic_state: AtomicState):
        batch_size = QKV.shape[0]
        seq_len    = QKV.shape[1]
        packed_qkv = QKV.shape[2]
        num_heads  = self.num_q_heads
        embed_dim  = num_heads * self.head_dim
        
        out = Tensor(
            shape = (batch_size, seq_len, embed_dim),
            stride = (seq_len * embed_dim, embed_dim, 1),
            is_weight = False
        )

        QKV.set_input_op(self)
        out.set_output_op(self)

        for pid_b in range(batch_size):
            for pid_h in range(num_heads):
                for pid_q in range(seq_len // self.bQ):
                    self.compute_tiles.append(
                        ComputeTile(
                            op = self.op,
                            input_tile = TensorPartition(QKV, (1, self.bQ, packed_qkv), (pid_b, pid_q, 0)),
                            output_tile = TensorPartition(out, (1, self.bQ, embed_dim), (pid_b, pid_q, 0)),
                            pid_m = pid_b,
                            pid_n = pid_h,
                            pid_k = pid_q,
                            layer_idx = self.layer_idx
                        )
                    )

        prev_op = QKV.get_output_op()
        if prev_op is None:
            return out
        for curr_tile in self.compute_tiles:
            for prev_tile in prev_op.compute_tiles:
                if curr_tile.input_tile.intersect(prev_tile.output_tile):
                    curr_tile.atomic_count += 1
                    prev_op.add_atomic(prev_tile, curr_tile, atomic_state)
        return out


matmul_config = MatmulConfig(
    bM = 64,
    bN = 64,
    bK = 64,
    num_stages = 2,
    stage_elements = 2 * 64 * 64 * 2,
    output_pad = 8,
    use_tma_reduce = True
)

M = 128
N = 128
K = 128
matmul1 = Matmul(
    M=M, N=N, K=K, layer_idx=0, config=matmul_config, op_kind=Op.DOWN
) # m1024 k512 n2048

matmul2 = Matmul(
    M=N, N=M, K=N, layer_idx=0, config=matmul_config, op_kind=Op.DOWN
) # m2048 k2048 n4096

atomic_state = AtomicState(atomic_index = 0)

input_tensor = Tensor(
    shape = (M, K),
    stride = (K, 1),
    is_weight = False
)

weight1_tensor = Tensor(
    shape = (K, N),
    stride = (1, K),
    is_weight = True
)

weight2_tensor = Tensor(
    shape = (N, M),
    stride = (1, N),
    is_weight = True
)

rms_config = RMSNormConfig(
    embed_dim = 128,
    bRMS = 16,
    stage_elements = 2 * 64 * 64,
    warps_per_row = 1
)

rmsnorm1 = RMSNorm(layer_idx = 0, config = rms_config, op_kind = Op.RMS)
weight3_tensor = Tensor(
    shape = (N, N),
    stride = (0, 1),
    is_weight = True
)

matmul3 = Matmul(
    M=N, N=M, K=N, layer_idx=0, config=matmul_config, op_kind=Op.DOWN
) # m2048 k2048 n4096



output_tensor = matmul1(input_tensor, weight1_tensor, atomic_state)
output2_tensor = matmul2(output_tensor, weight2_tensor, atomic_state)
output3_tensor = rmsnorm1(output2_tensor, weight3_tensor, atomic_state)
output4_tensor = matmul3(output3_tensor, weight2_tensor, atomic_state)
output5_tensor =
# for tile in matmul1.compute_tiles:
#     print(tile)
#     print("Next atomic :", tile.next_atomic)
#     print("Previous atomic :", tile.prev_atomic)
#     print("Atomic count :", tile.atomic_count)
#     print("Instruction :", tile.get_instr())
#     print("-------------------")
for tile in rmsnorm1.compute_tiles:
    print(tile)
    print("Next atomic :", tile.next_atomic)
    print("Previous atomic :", tile.prev_atomic)
    print("Atomic count :", tile.atomic_count)
    print("-------------------")

for tile in matmul3.compute_tiles:
    print(tile)
    print("Next atomic :", tile.next_atomic)
    print("Previous atomic :", tile.prev_atomic)
    print("Atomic count :", tile.atomic_count)
    print("-------------------")