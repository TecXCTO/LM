'''
This is a professional-grade implementation of a modern Decoder-only Transformer block.
I have integrated 2025/2026 industry standards: RoPE (Rotary Positional Embeddings), RMSNorm,
and Grouped Query Attention (GQA). '''


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    """Modern scaling for training stability (used by Llama 3/4 & DeepSeek)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

class RotaryEmbedding(nn.Module):
    """Enables the model to understand relative word positions efficiently."""
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, seq_len):
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, :, None, :], emb.sin()[None, :, None, :]

def apply_rotary_pos_emb(q, k, cos, sin):
    """Applies the rotation to queries and keys."""
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class ModernAttention(nn.Module):
    """Grouped Query Attention (GQA) for faster inference/lower memory."""
    def __init__(self, dim, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // n_kv_heads

        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(self, x, cos, sin):
        batch, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)

        # Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Repeat KV heads for GQA compatibility
        k = k.repeat_interleave(self.n_rep, dim=2)
        v = v.repeat_interleave(self.n_rep, dim=2)

        # Efficient Scaled Dot-Product Attention (uses FlashAttention if available)
        output = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), 
            is_causal=True
        )
        
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(output)

class SwiGLU(nn.Module):
    """The gold standard for LLM activation functions."""
    def __init__(self, dim, intermediate_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class TransformerBlock(nn.Module):
    """A single modular block of our LLM."""
    def __init__(self, dim, n_heads, n_kv_heads, mlp_dim):
        super().__init__()
        self.attention_norm = RMSNorm(dim)
        self.attention = ModernAttention(dim, n_heads, n_kv_heads)
        self.ffn_norm = RMSNorm(dim)
        self.feed_forward = SwiGLU(dim, mlp_dim)

    def forward(self, x, cos, sin):
        # Residual Connection 1: Attention
        x = x + self.attention(self.attention_norm(x), cos, sin)
        # Residual Connection 2: Feed Forward
        x = x + self.feed_forward(self.ffn_norm(x))
        return x
