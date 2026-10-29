"""Masked-diffusion Transformer (LLaDA-style).

A standard bidirectional Transformer that learns to predict tokens masked by a
random mask ratio (the MDLM / LLaDA objective). Generation is done by
iteratively unmasking tokens in parallel (see generate.py).

Notes:
- No causal mask: attention is bidirectional over the whole sequence.
- RoPE rotary embeddings, pre-norm RMSNorm, SwiGLU MLPs, tied embeddings.
- Uses PyTorch SDPA so it works on Turing GPUs (T4/P100/1660 Ti) without FlashAttention.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def _precompute_rope(dim: int, max_len: int, theta: float, dtype=torch.float32):
    """Return (cos, sin) tensors of shape [1, 1, max_len, dim // 2]."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=dtype) / dim))
    t = torch.arange(max_len, dtype=dtype)
    angles = torch.outer(t, freqs)  # [L, dim/2]
    cos = torch.cos(angles).unsqueeze(0).unsqueeze(0)
    sin = torch.sin(angles).unsqueeze(0).unsqueeze(0)
    return cos, sin


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Rotate q/k by precomputed frequencies (pairwise (i, i+half) rotation).

    q, k: [B, H, L, head_dim]; cos/sin: [1, 1, L, head_dim // 2].
    """
    half = q.shape[-1] // 2
    q1, q2 = q[..., :half], q[..., half:]
    k1, k2 = k[..., :half], k[..., half:]
    q1_rot = q1 * cos - q2 * sin
    q2_rot = q2 * cos + q1 * sin
    k1_rot = k1 * cos - k2 * sin
    k2_rot = k2 * cos + k1 * sin
    q = torch.cat([q1_rot, q2_rot], dim=-1)
    k = torch.cat([k1_rot, k2_rot], dim=-1)
    return q, k


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.wq = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        q = self.wq(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin)
        if self.n_kv_heads != self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).reshape(B, L, -1)
        return self.wo(out)


class MLP(nn.Module):
    def __init__(self, d_model: int, dim_ff: int):
        super().__init__()
        self.gate = nn.Linear(d_model, dim_ff, bias=False)
        self.up = nn.Linear(d_model, dim_ff, bias=False)
        self.down = nn.Linear(dim_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int, dim_ff: int):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = Attention(d_model, n_heads, n_kv_heads, head_dim)
        self.ffn_norm = RMSNorm(d_model)
        self.mlp = MLP(d_model, dim_ff)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.mlp(self.ffn_norm(x))
        return x


@dataclass
class ModelConfig:
    vocab_size: int = 32768
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 8
    dim_ff: int = 2048
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    dropout: float = 0.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class MaskedDiffusionTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [
                Block(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.dim_ff)
                for _ in range(cfg.n_layers)
            ]
        )
        self.ln_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Tied embeddings: output projection shares the input embedding weights.
        self.lm_head.weight = self.wte.weight

        cos, sin = _precompute_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: [B, L] of token ids (masked positions hold MASK id).
        Returns logits [B, L, vocab_size]."""
        B, L = input_ids.shape
        x = self.wte(input_ids)
        x = self.drop(x)
        cos = self.cos_cached[..., :L, :]
        sin = self.sin_cached[..., :L, :]
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.ln_f(x)
        return self.lm_head(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
