# HumChess: Human-Like Chess Move Prediction

A chess model that emulates human moves conditioned on Elo rating and time remaining, providing an AI opponent that plays realistically at different skill levels.

---

## 1. Problem Statement

We learn a human-like chess policy:

```
π(move | position, Elo, TimeLeft)
```

Where:
- **position**: board state (piece placement + castling rights)
- **Elo**: player skill bucket for the side to move
- **TimeLeft**: clock time remaining for the side to move
- **move**: a legal chess move

**Inference**: Sample from π (multinomial sampling after legality masking). No temperature/top-k/top-p tuning.

---

## 2. Model Architecture

### 2.1 Overview

**Encoder-only transformer** with bidirectional self-attention (no causal mask).

| Component | Value |
|-----------|-------|
| Normalization | RMSNorm (pre-norm) |
| Attention | Multi-head with QK-norm |
| FFN activation | SiLU |
| Positional embeddings | Learned, board squares only |

### 2.2 Input Processing

Input sequence: 74 tokens

```
[CLS, SQ_0, ..., SQ_63, CASTLING, ELO_BUCKET, TL_BUCKET, HIST_1, ..., HIST_6]
```

- Token embeddings: `nn.Embedding(66, d_model)` for positions 0-67
- Move history embeddings: `nn.Embedding(4097, d_model)` for positions 68-73 (4096 moves + padding)
- Positional embeddings: `nn.Embedding(64, d_model)` added only to board squares (indices 1-64)
- CLS, metadata, and history tokens receive no positional embedding

### 2.3 Transformer Blocks

Each block (pre-norm):
```
x = x + Attention(RMSNorm(x))
x = x + FFN(RMSNorm(x))
```

**Attention** with QK-norm:
- Project to Q, K, V via single linear: `(B, L, D) → (B, L, 3, H, D/H)`
- Apply RMSNorm to Q and K per-head before attention
- Use `scaled_dot_product_attention` (Flash Attention when available)

**FFN**:
```
FFN(x) = W2(SiLU(W1(x)))
```

### 2.4 Output Heads

The final [CLS] hidden state (after RMSNorm) feeds two linear heads:

1. **Move head**: `Linear(d_model, 4096)` — from×to classification
2. **Promotion head**: `Linear(d_model, 4)` — piece type when promoting

### 2.5 Default Hyperparameters

| Parameter | Default |
|-----------|---------|
| d_model | 256 |
| n_heads | 8 |
| n_layers | 6 |
| d_ff | 1024 |

---

## 3. Token Vocabulary

**Total vocabulary size: 66 tokens**

### 3.1 Piece Tokens (0-12)

| ID | Token | Description |
|----|-------|-------------|
| 0 | EMPTY | Empty square |
| 1-6 | WP, WN, WB, WR, WQ, WK | White pieces |
| 7-12 | BP, BN, BB, BR, BQ, BK | Black pieces |

### 3.2 Special Token (13)

| ID | Token |
|----|-------|
| 13 | CLS |

### 3.3 Castling Rights Tokens (14-29)

16 tokens encoding all combinations of castling rights as a 4-bit value.

Bit order: `[WK, WQ, BK, BQ]` (MSB to LSB)

Token ID = 14 + (WK×8 + WQ×4 + BK×2 + BQ)

### 3.4 Elo Bucket Tokens (30-46)

17 buckets:

| Bucket | Elo Range |
|--------|-----------|
| 0 | < 1000 |
| 1-15 | 1000-1100, 1100-1200, ..., 2400-2500 |
| 16 | ≥ 2500 |

### 3.5 Time-Left Bucket Tokens (47-65)

19 buckets:

| Bucket | Time Range |
|--------|------------|
| 0 | < 10 seconds |
| 1 | 10-30 seconds |
| 2 | 30s - 1 minute |
| 3-16 | 1-2m, 2-3m, ..., 14-15m |
| 17 | ≥ 15 minutes |
| 18 | Unknown |

---

## 4. Square Indexing

Square order: **A1→H1, A2→H2, ..., A8→H8**

```
Index = (rank - 1) × 8 + file
      = (rank - 1) × 8 + (ord(file) - ord('a'))
```

Examples:
- A1 = 0, H1 = 7
- A2 = 8, H8 = 63

---

## 5. White Normalization

All positions are normalized so **white is always to move**. This halves the effective state space.

### 5.1 Transform (when black to move)

1. **Board rotation**: Rotate 180° → `sq' = 63 - sq`
2. **Color swap**: Exchange white↔black pieces
3. **Castling swap**: `(WK, WQ, BK, BQ) → (BK, BQ, WK, WQ)`
4. **Move transform**: `from' = 63 - from`, `to' = 63 - to`

### 5.2 Implementation

```python
def normalize_position(tokens, move_uci, is_black_to_move):
    if not is_black_to_move:
        return tokens, move_uci

    # Rotate board and swap colors
    board = tokens[1:65]
    new_board = [COLOR_SWAP[board[63 - sq]] for sq in range(64)]

    # Swap castling rights
    wk, wq, bk, bq = parse_castling(tokens[65])
    new_castling = encode_castling(bk, bq, wk, wq)

    # Transform move squares
    new_move = transform_move(move_uci)

    return new_tokens, new_move
```

---

## 6. Move Encoding

### 6.1 Move ID (0-4095)

```
move_id = from_square × 64 + to_square
```

Castling is encoded as king moves: e1g1, e1c1, etc.

### 6.2 Promotion ID (0-3)

| ID | Piece |
|----|-------|
| 0 | Queen |
| 1 | Rook |
| 2 | Bishop |
| 3 | Knight |

### 6.3 Detecting Promotions

A move is a promotion iff (after normalization):
- The moving piece is a white pawn (token = WP)
- The destination square is on rank 8 (to_sq ≥ 56)

---

## 7. Legality Masking

### 7.1 Training and Inference

Illegal moves must have zero probability. Before softmax:

```python
masked_logits = logits.clone()
masked_logits[~legal_mask] = float('-inf')
loss = cross_entropy(masked_logits, target)
```

### 7.2 Legal Mask Caching

Legal move indices are cached with an LRU cache (default 100k entries).

**Cache key**: `(tuple(board_tokens), ep_square)`

The en passant square is included because the same piece configuration can have different legal moves depending on EP availability.

---

## 8. Data Pipeline

### 8.1 Data Sources

Two modes:
1. **PGN files**: Parsed on-the-fly with `python-chess`
2. **Parquet shards**: Pre-tokenized for faster training

### 8.2 PGN Indexing

Fast indexing (~100x faster than parsing) by scanning for game boundaries:
```bash
uv run python -m humchess.data.build_parquet index --pgn data/games.pgn
```

Creates `.idx.json` with sparse checkpoints every 10k games for random access.

### 8.3 Parquet Conversion

```bash
uv run python -m humchess.data.build_parquet build \
    --pgn data/games.pgn \
    --out-dir data/tokenized \
    --num-workers 8
```

Each Parquet row contains:
- `tokens`: List[int] of length 74 (68 board + 6 history)
- `move_id`: int16 (0-4095)
- `promo_id`: int8 (-1 if not promotion, 0-3 otherwise)
- `is_promotion`: bool
- `legal_mask`: bytes (512 bytes = 4096 bits, packed)

### 8.4 Dataset Sharding

**Two-level sharding** for distributed training:

1. **DDP rank sharding**: Files divided by `file_idx % world_size == rank`
2. **DataLoader worker sharding**: Remaining files divided by `file_idx % num_workers == worker_id`

This ensures each (rank, worker) pair processes unique data.

### 8.5 Batch Contents

```python
{
    'tokens': Tensor[B, 74],      # Input token IDs (68 board + 6 history)
    'move_id': Tensor[B],         # Target move (0-4095)
    'promo_id': Tensor[B],        # Target promotion (-1 or 0-3)
    'legal_mask': Tensor[B, 4096], # True for legal moves
    'is_promotion': Tensor[B],    # Whether move is a promotion
}
```

---

## 9. Training

### 9.1 Loss Function

```python
# Move loss (always computed)
move_loss = masked_cross_entropy(move_logits, move_targets, legal_mask)

# Promotion loss (only for promotion moves)
if is_promotion.any():
    promo_loss = cross_entropy(promo_logits[is_promotion], promo_targets[is_promotion])
else:
    promo_loss = 0

total_loss = move_loss + promo_loss
```

### 9.2 Optimizer

AdamW with configurable learning rate (default 1e-4).

### 9.3 Mixed Precision

Supported modes:
- `bf16`: bfloat16 autocast (recommended for Ampere+)
- `fp16`: float16 autocast with GradScaler
- `fp32`: full precision

Optional TF32 matmul acceleration (`--tf32`).

### 9.4 Distributed Training

```bash
# Single GPU
uv run python -m humchess.train --config configs/model.yml

# Multi-GPU
uv run torchrun --nproc_per_node=4 -m humchess.train --config configs/model.yml
```

Uses `DistributedDataParallel` with `find_unused_parameters=True` (needed because promo_head is unused when batch has no promotions).

### 9.5 Checkpointing

End-of-epoch checkpoints saved to `checkpoint_dir`:
```python
{
    'epoch': int,              # Next epoch to start from
    'model_state_dict': dict,
    'optimizer_state_dict': dict,
    'stats': dict,             # Loss, accuracy metrics
}
```

Resume with `--resume path/to/checkpoint.pt`.

### 9.6 Metrics

Logged during training:
- `loss`: Combined move + promotion loss
- `move_loss`: Move head cross-entropy
- `move_accuracy`: Top-1 accuracy (with masking)
- `promo_loss`: Promotion head cross-entropy (when applicable)
- `promo_accuracy`: Promotion prediction accuracy
- `samples_per_sec`: Throughput

---

## 10. Inference

1. Tokenize position with Elo and time-left (68 tokens)
2. Apply white normalization if black to move
3. Normalize and append move history (6 tokens, padded with NO_MOVE_ID if insufficient)
4. Forward pass through model
5. Apply legality mask to move logits
6. Sample move_id from masked softmax distribution
7. If promotion (pawn to rank 8): sample promo_id from promotion head
8. Denormalize move if originally black to move
9. Return UCI move string

---

## 11. Configuration

Example `configs/model.yml`:

```yaml
model:
  d_model: 256
  n_heads: 8
  n_layers: 6
  d_ff: 1024

data:
  parquet:
    - data/tokenized/*.parquet
  # Or for raw PGN:
  # pgn:
  #   - data/games.pgn
  # min_elo: 1000
  # max_elo: 2500

training:
  batch_size: 256
  lr: 1e-4
  epochs: 1
  num_workers: 4
  precision: bf16
  checkpoint_dir: checkpoints
  log_interval: 100
```

---

## 12. Future Work (v2)

Potential enhancements not in current implementation:

### 12.1 Adversarial Training

**Motivation**: The BC model predicts individual moves well, but Elo is a *game-level* property. It emerges from error profiles, conversion ability, consistency under pressure, blunder rates, etc. A model good at predicting single moves might still produce trajectories that feel "too strong" or "too weak" for the target Elo.

**Concept**:
1. **Generator**: Treat π_θ as a generator of game trajectories (self-play or vs fixed opponent)
2. **Discriminator**: Train D to distinguish:
   - Real human trajectories from the dataset
   - Generated trajectories from π_θ
   - Both conditioned on the same (Elo, TimeLeft) context
3. **RL Fine-tuning**: Update π_θ via RL to "fool" D into classifying generated games as real human games at that Elo

**Stabilization**: Use a KL divergence penalty to the supervised BC policy (analogous to RLHF's KL-to-SFT tether). This prevents reward hacking and preserves move-level human realism while improving game-level Elo calibration.

### 12.2 Other Enhancements

- Opponent Elo conditioning
- Time control / increment conditioning (not just time remaining)
- Auxiliary heads (value/WDL/eval/time-spent prediction)
- Skip forced/trivial positions during training
