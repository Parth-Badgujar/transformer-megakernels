import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNormLayer(nn.Module):
    def __init__(self, dim, dtype = torch.bfloat16, device = "cuda"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype = dtype, device = device))

    def forward(self, x: torch.Tensor):
        x_f   = x.float()
        var   = (x_f * x_f).mean(dim = -1, keepdim = True)
        scale = torch.rsqrt(var)
        return (x_f * scale * self.weight.float()).to(x.dtype)


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        is_causal: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda"
    ):
        super().__init__()
        self.embed_dim    = embed_dim
        self.num_q_heads  = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim     = embed_dim // num_q_heads
        self.is_causal    = is_causal
        self.enable_gqa   = (num_q_heads != num_kv_heads)
        self.q_proj_dim  = num_q_heads  * self.head_dim
        self.kv_proj_dim = num_kv_heads * self.head_dim
        packed_qkv_dim   = self.q_proj_dim + 2 * self.kv_proj_dim
        self.qkv_proj = nn.Linear(embed_dim, packed_qkv_dim, bias = False, dtype = dtype, device = device)
        self.out_proj = nn.Linear(embed_dim, embed_dim,      bias = False, dtype = dtype, device = device)

    def forward(self, x: torch.Tensor):
        bs, seq_len = x.shape[0], x.shape[1]
        qkv = self.qkv_proj(x)
        q_flat, k_flat, v_flat = qkv.split(
            [self.q_proj_dim, self.kv_proj_dim, self.kv_proj_dim], dim=-1
        )
        q = q_flat.view(bs, seq_len, self.num_q_heads,  self.head_dim).transpose(1, 2)
        k = k_flat.view(bs, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v_flat.view(bs, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal = self.is_causal, enable_gqa = self.enable_gqa)
        attn_flat = attn_out.transpose(1, 2).contiguous().view(bs, seq_len, self.embed_dim)
        return self.out_proj(attn_flat)


class SwiGLU(nn.Module):
    def __init__(
        self,
        dim: int, 
        ff_dim: int,
        bias: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device  = "cuda"
    ):
        super().__init__()
        self.gate_proj = nn.Linear(dim, ff_dim, bias = bias, dtype = dtype, device = device)
        self.up_proj   = nn.Linear(dim, ff_dim, bias = bias, dtype = dtype, device = device)
        self.down_proj = nn.Linear(ff_dim, dim, bias = bias, dtype = dtype, device = device)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int, 
        num_q_heads: int,
        num_kv_heads: int,
        ff_dim: int,
        is_causal: bool = False,
        dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        self.norm1 = RMSNormLayer(embed_dim, dtype = dtype)
        self.attention  = GroupedQueryAttention(embed_dim, num_q_heads, num_kv_heads, is_causal, dtype = dtype)
        self.norm2 = RMSNormLayer(embed_dim, dtype = dtype)
        self.feedforward   = SwiGLU(embed_dim, ff_dim, dtype = dtype)

    def forward(self, x: torch.Tensor):
        residual = x
        x = self.norm1(x)
        x = self.attention(x)
        x = residual + x
        residual = x
        x = self.norm2(x)
        x = self.feedforward(x)
        return residual + x


class MultiLayerTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        ff_dim: int,
        num_layers: int,
        is_causal: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda"
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(
                embed_dim, num_q_heads, num_kv_heads, ff_dim, is_causal, dtype = dtype
            )
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x


def extract_weights(model: MultiLayerTransformer):
    rms_w = []
    qkv_w = []
    out_w = []
    gate_w = []
    up_w = []
    down_w = []
    
    get_raw_clone = lambda x: x.weight.detach().clone()
    
    for layer in model.layers:
        rms_w.append(get_raw_clone(layer.norm1))
        rms_w.append(get_raw_clone(layer.norm2))
        qkv_w.append(get_raw_clone(layer.attention.qkv_proj))
        out_w.append(get_raw_clone(layer.attention.out_proj))
        gate_w.append(get_raw_clone(layer.feedforward.gate_proj))
        up_w.append(get_raw_clone(layer.feedforward.up_proj))
        down_w.append(get_raw_clone(layer.feedforward.down_proj))

    rms_w  = torch.stack(rms_w)
    qkv_w  = torch.stack(qkv_w)
    out_w  = torch.stack(out_w)
    gate_w = torch.stack(gate_w)
    up_w   = torch.stack(up_w)
    down_w = torch.stack(down_w)
    return rms_w, qkv_w, out_w, gate_w, up_w, down_w