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

from ..data.tokenization import (
    VOCAB_SIZE, NUM_MOVE_CLASSES, NUM_PROMO_CLASSES, NO_MOVE_ID,
)


class SelfAttention(nn.Module):
    """Multi-head self-attention with QK-norm."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
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
        self.rmsnorm1 = nn.RMSNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads)
        self.rmsnorm2 = nn.RMSNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.rmsnorm1(x))
        x = x + self.ffn(self.rmsnorm2(x))
        return x


class ChessTransformer(nn.Module):
    """
    Encoder-only transformer for human-like chess move prediction.

    Input: token sequence of shape (batch, 74)
           - positions 0-67: board tokens (CLS + 64 squares + castling + elo + time_left)
           - positions 68-73: move history (6 most recent moves, normalized)
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
        # Move history embedding: 4096 moves + 1 padding token (NO_MOVE_ID)
        self.move_history_emb = nn.Embedding(NO_MOVE_ID + 1, d_model)
        # History position embedding (6 slots: opponent, you, opponent, you, opponent, you)
        self.history_pos_emb = nn.Embedding(6, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])

        # Cache position indices as buffers (avoids torch.arange each forward)
        self.register_buffer('board_pos_idx', torch.arange(64), persistent=False)
        self.register_buffer('history_pos_idx', torch.arange(6), persistent=False)

        self.rmsnorm_final = nn.RMSNorm(d_model)
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
            elif isinstance(module, nn.RMSNorm):
                nn.init.ones_(module.weight)

    def forward(self, tokens: Tensor) -> dict[str, Tensor]:
        # Split tokens: board (0-67) and history (68-73)
        board_tokens = tokens[:, :68]
        history_tokens = tokens[:, 68:74]

        # Embed board tokens
        board_emb = self.token_emb(board_tokens)
        board_emb[:, 1:65] = board_emb[:, 1:65] + self.pos_emb(self.board_pos_idx)

        # Embed move history with positional encoding
        history_emb = self.move_history_emb(history_tokens)
        history_emb = history_emb + self.history_pos_emb(self.history_pos_idx)

        # Concatenate
        x = torch.cat([board_emb, history_emb], dim=1)  # (batch, 74, d_model)

        for block in self.blocks:
            x = block(x)

        cls = self.rmsnorm_final(x[:, 0])

        return {
            'move_logits': self.move_head(cls),
            'promo_logits': self.promo_head(cls),
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

