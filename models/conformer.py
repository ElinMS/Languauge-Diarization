"""
conformer.py
────────────
Full Conformer block implementation, following
  "Conformer: Convolution-augmented Transformer for Speech Recognition"
  Gulati et al. 2020  (https://arxiv.org/abs/2005.08100)

Architecture of one Conformer block:
    x → FF (½-step) → MHSA → Conv Module → FF (½-step) → LayerNorm → x'

Key design choices
──────────────────
* Pre-norm (LayerNorm before each sub-module, more stable training).
* Rotary position embeddings (RoPE) on the MHSA to handle variable lengths.
* Depthwise separable convolution with GLU gating.
* Relative positional encoding is handled by the attention module.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ─────────────────────────────────────────────────────────────────────────────
#  Rotary Position Embedding (RoPE)
# ─────────────────────────────────────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    """
    Rotary position embedding.
    Applies a rotation matrix based on position to Q and K,
    replacing additive sinusoidal embeddings.
    """

    def __init__(self, dim: int):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        t     = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, self.inv_freq)           # (T, dim/2)
        emb   = torch.cat([freqs, freqs], dim=-1)       # (T, dim)
        return emb                                       # cos/sin applied outside


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor, emb: torch.Tensor):
    """Apply RoPE to Q and K tensors. emb: (T, head_dim)."""
    cos = emb.cos()[None, None, :, :]   # (1, 1, T, D)
    sin = emb.sin()[None, None, :, :]
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-Head Self-Attention with RoPE
# ─────────────────────────────────────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.h       = num_heads
        self.d_k     = d_model // num_heads
        self.scale   = self.d_k ** -0.5

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model,     bias=False)
        self.dropout  = nn.Dropout(dropout)
        self.rope     = RotaryEmbedding(self.d_k)

    def forward(
        self,
        x: torch.Tensor,                    # (B, T, D)
        mask: Optional[torch.Tensor] = None # (B, 1, T, T) bool True=ignore
    ) -> torch.Tensor:
        B, T, D = x.shape

        # project & split heads
        qkv = self.qkv_proj(x)              # (B, T, 3D)
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape to (B, H, T, d_k)
        def split_heads(t):
            return rearrange(t, "b t (h d) -> b h t d", h=self.h)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # apply RoPE
        rope_emb = self.rope(T, x.device)  # (T, d_k)
        q, k = apply_rotary_emb(q, k, rope_emb)

        # scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, T, T)
        if mask is not None:
            attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)         # (B, H, T, d_k)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.out_proj(out)


# ─────────────────────────────────────────────────────────────────────────────
#  Feed-Forward Module (Macaron-style half-step)
# ─────────────────────────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    def __init__(self, d_model: int, expansion_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        d_ff = d_model * expansion_factor
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),                  # Swish / SiLU
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.5 * self.net(self.norm(x))


# ─────────────────────────────────────────────────────────────────────────────
#  Convolution Module
# ─────────────────────────────────────────────────────────────────────────────

class ConvolutionModule(nn.Module):
    """
    Conformer convolution module:
        LayerNorm → pointwise expand → GLU → depthwise → BatchNorm → SiLU → pointwise → dropout
    """

    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size must be odd"
        self.norm         = nn.LayerNorm(d_model)
        self.pw_expand    = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.dw_conv      = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=d_model,            # depthwise
        )
        self.bn           = nn.BatchNorm1d(d_model)
        self.pw_contract  = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout      = nn.Dropout(dropout)
        self.act          = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        residual = x
        x = self.norm(x)

        # → (B, D, T) for Conv1d
        x = x.transpose(1, 2)

        x = self.pw_expand(x)              # (B, 2D, T)
        x = F.glu(x, dim=1)               # (B, D,  T)  — GLU gate

        x = self.dw_conv(x)               # (B, D, T)
        x = self.bn(x)
        x = self.act(x)

        x = self.pw_contract(x)           # (B, D, T)
        x = self.dropout(x)

        # → (B, T, D)
        x = x.transpose(1, 2)
        return residual + x


# ─────────────────────────────────────────────────────────────────────────────
#  Single Conformer Block
# ─────────────────────────────────────────────────────────────────────────────

class ConformerBlock(nn.Module):
    """
    Full Conformer block following Gulati 2020 Fig. 1:

        x → FF½ → MHSA → Conv → FF½ → LayerNorm → output
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_expansion_factor: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ff1  = FeedForward(d_model, ff_expansion_factor, dropout)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.conv = ConvolutionModule(d_model, conv_kernel_size, dropout)
        self.ff2  = FeedForward(d_model, ff_expansion_factor, dropout)
        self.norm = nn.LayerNorm(d_model)

        self.attn_norm = nn.LayerNorm(d_model)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.ff1(x)
        # MHSA with pre-norm + residual
        x = x + self.attn_drop(self.attn(self.attn_norm(x), mask))
        x = self.conv(x)
        x = self.ff2(x)
        x = self.norm(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
#  Conv2D Subsampling (reduces time by factor 4)
# ─────────────────────────────────────────────────────────────────────────────

class Conv2dSubsampling(nn.Module):
    """
    Two stacked 3×3 Conv2d layers with stride=2 → 4× time compression.
    Input : (B, T, n_mels)
    Output: (B, T//4, encoder_dim)
    """

    def __init__(self, n_mels: int, encoder_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, encoder_dim // 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(encoder_dim // 4, encoder_dim // 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # after two stride-2 convs on freq axis: out_freq = ceil(n_mels/4)
        out_freq = math.ceil(math.ceil(n_mels / 2) / 2)
        self.proj = nn.Linear(encoder_dim // 4 * out_freq, encoder_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x : (B, T, n_mels)
        Returns (out, mask_lens) where mask_lens = input_lens // 4
        """
        # → (B, 1, T, n_mels)  — treat n_mels as "channels" dim for Conv2d
        x = x.unsqueeze(1)
        x = self.conv(x)                      # (B, C, T//4, n_mels//4)
        B, C, T, F = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(B, T, C * F)
        x = self.proj(x)                      # (B, T//4, encoder_dim)
        return x
