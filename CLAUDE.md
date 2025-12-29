# HumChess

Human-like chess move prediction conditioned on Elo and time remaining.

## Overview

- **Model**: Encoder-only transformer (RMSNorm, QK-norm, SiLU). Reads 74 tokens, outputs move logits (4096 classes) + promotion logits (4 classes).
- **Input**: `[CLS, 64 squares, castling, elo_bucket, time_left_bucket, 6 history moves]`
- **White normalization**: All positions normalized so white is to move (rotate 180° + swap colors when black to move)
- **Legality masking**: Illegal moves set to -inf before softmax during training and inference
- **Inference**: Sample from masked distribution (no temperature tuning)

## Key Files

!!!Before doing anything, read these files to understand the codebase!!!:

- `humchess/data/pgn_dataset.py` - Data loading, PGN parsing, Parquet streaming, sharding
- `humchess/data/tokenization.py` - Token vocabulary, board encoding, white normalization, move encoding
- `humchess/model/transformer.py` - Model architecture
- `humchess/train.py` - Training loop, DDP, mixed precision, checkpointing

## Commands

```bash
# Index a PGN file (required before building parquet)
uv run python -m humchess.data.build_parquet index --pgn data/games.pgn

# Convert PGN to Parquet shards
uv run python -m humchess.data.build_parquet build --pgn data/games.pgn --out-dir data/tokenized

# Train (single GPU)
uv run python -m humchess.train --config configs/model.yml

# Train (multi-GPU)
uv run torchrun --nproc_per_node=4 -m humchess.train --config configs/model.yml
```
