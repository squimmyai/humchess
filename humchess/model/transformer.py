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


class RMSNorm(nn.Module):
    """Root-mean-square normalization that always computes in fp32.

    The built-in nn.RMSNorm keeps its weight in fp32 while autocast feeds it
    bf16 activations, which disables the fused kernel and emits a dtype-mismatch
    warning. This implementation promotes the input to fp32, normalizes, applies
    the weight, and casts back, so it works cleanly with fp32 master weights +
    bf16 autocast (the standard mixed-precision recipe).
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(rms + self.eps)
        x = x * self.weight.float()
        return x.to(dtype)


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


class SquarePairMoveHead(nn.Module):
    """Score moves as q(from_square) . k(to_square).

    Instead of collapsing the whole position into [CLS] and applying a flat
    Linear(d_model, 4096), score each move from the hidden states of the two
    squares it touches. The 4096 logits become a 64x64 from/to score matrix,
    flattened as from*64 + to - exactly the existing move-ID convention, so the
    legal mask, targets, and sampling are unchanged.

    Why: a 64x64 matrix has rank <= 64, so with d_head >= 64 the bilinear form
    can represent any from/to score matrix (no bottleneck), while sharing
    parameters across squares and giving the head a board-geometry inductive
    bias. It is also far smaller than Linear(384, 4096).

    White normalization guarantees the side to move is always White, so the
    geometry is always from White's perspective - no color conditioning needed.
    """

    def __init__(self, d_model: int, d_head: int = 64, use_rel_bias: bool = True):
        super().__init__()
        self.d_head = d_head
        self.norm = RMSNorm(d_model)
        self.from_proj = nn.Linear(d_model, d_head, bias=False)
        self.to_proj = nn.Linear(d_model, d_head, bias=False)
        self.scale = d_head ** -0.5
        # Relative-displacement bias: index by (dfile+7, drank+7), so moves with
        # the same geometry (e.g. any knight move) share a bias. Cheap prior.
        self.use_rel_bias = use_rel_bias
        if use_rel_bias:
            self.rel_bias = nn.Parameter(torch.zeros(15, 15))

    def forward(self, square_hidden: Tensor) -> Tensor:
        # square_hidden: (B, 64, d_model), squares in a1..h8 order.
        h = self.norm(square_hidden)
        q = self.from_proj(h)                       # (B, 64, d_head)
        k = self.to_proj(h)                         # (B, 64, d_head)
        scores = (q @ k.transpose(1, 2)) * self.scale  # (B, 64, 64)
        if self.use_rel_bias:
            scores = scores + self._rel_bias_matrix(scores.device, scores.dtype)
        return scores.reshape(scores.shape[0], 64 * 64)

    def _rel_bias_matrix(self, device, dtype) -> Tensor:
        idx = torch.arange(64, device=device)
        from_file, from_rank = idx % 8, idx // 8
        to_file, to_rank = idx % 8, idx // 8
        dfile = (to_file.view(1, 64) - from_file.view(64, 1)) + 7
        drank = (to_rank.view(1, 64) - from_rank.view(64, 1)) + 7
        return self.rel_bias[dfile, drank].to(dtype)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with RMSNorm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.rmsnorm1 = RMSNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads)
        self.rmsnorm2 = RMSNorm(d_model)
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
        d_head: int = 64,
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

        self.rmsnorm_final = RMSNorm(d_model)
        self.move_head = SquarePairMoveHead(d_model, d_head=d_head)
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

        x = self.rmsnorm_final(x)
        cls = x[:, 0]
        squares = x[:, 1:65]  # (B, 64, d_model), a1..h8

        return {
            'move_logits': self.move_head(squares),
            'promo_logits': self.promo_head(cls),
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

