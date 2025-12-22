"""
Encoder-only transformer for chess move prediction.

Architecture as specified in plan.md §2:
- Bidirectional self-attention (no causal mask)
- [CLS] token readout for move prediction
- Positional embeddings only for board squares (not CLS or metadata)

Uses RMSNorm and QK-norm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..data.tokenization import VOCAB_SIZE, NUM_MOVE_CLASSES, NUM_PROMO_CLASSES


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class SelfAttention(nn.Module):
    """Multi-head self-attention with QK-norm."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        B, L, D = x.shape

        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        # Apply QK-norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, L, D)

        return self.out(attn)


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with RMSNorm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads)
        self.ln2 = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class ChessTransformer(nn.Module):
    """
    Encoder-only transformer for human-like chess move prediction.

    Input: token sequence of shape (batch, 68)
    Output: move logits (batch, 4096), promotion logits (batch, 4)
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
    ):
        super().__init__()
        self.d_model = d_model

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(64, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])

        self.ln_final = RMSNorm(d_model)
        self.move_head = nn.Linear(d_model, NUM_MOVE_CLASSES)
        self.promo_head = nn.Linear(d_model, NUM_PROMO_CLASSES)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)

    def forward(self, tokens: Tensor) -> dict[str, Tensor]:
        x = self.token_emb(tokens)
        x[:, 1:65] = x[:, 1:65] + self.pos_emb(torch.arange(64, device=tokens.device))

        for block in self.blocks:
            x = block(x)

        cls = self.ln_final(x[:, 0])

        return {
            'move_logits': self.move_head(cls),
            'promo_logits': self.promo_head(cls),
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(size: str = 'small', **kwargs) -> ChessTransformer:
    """Create model with preset sizes: tiny, small, medium, large."""
    presets = {
        'tiny': {'d_model': 128, 'n_heads': 4, 'n_layers': 4, 'd_ff': 512},
        'small': {'d_model': 256, 'n_heads': 8, 'n_layers': 6, 'd_ff': 1024},
        'medium': {'d_model': 512, 'n_heads': 8, 'n_layers': 8, 'd_ff': 2048},
        'large': {'d_model': 768, 'n_heads': 12, 'n_layers': 12, 'd_ff': 3072},
    }
    if size not in presets:
        raise ValueError(f"Unknown size '{size}'. Choose from: {list(presets.keys())}")
    return ChessTransformer(**{**presets[size], **kwargs})
