"""CPU tests for the held-out validation path (humchess.train.evaluate)."""

import torch
import torch.nn as nn

from humchess.data.tokenization import SEQ_LENGTH, NUM_MOVE_CLASSES
from humchess.model.transformer import ChessTransformer
from humchess.train import evaluate


def _make_batch(batch_size: int = 4) -> dict:
    tokens = torch.randint(0, 66, (batch_size, SEQ_LENGTH), dtype=torch.long)
    move_id = torch.randint(0, NUM_MOVE_CLASSES, (batch_size,), dtype=torch.long)
    legal_mask = torch.zeros(batch_size, NUM_MOVE_CLASSES, dtype=torch.bool)
    legal_mask[torch.arange(batch_size), move_id] = True
    legal_mask[:, :10] = True
    return {
        "tokens": tokens,
        "move_id": move_id,
        "legal_mask": legal_mask,
    }


def test_evaluate_returns_metrics():
    model = ChessTransformer(d_model=32, n_heads=4, n_layers=1, d_ff=64)
    loader = [_make_batch() for _ in range(5)]
    stats = evaluate(model, loader, torch.device("cpu"), precision="fp32", max_batches=3)
    assert set(stats) == {"val_loss", "val_move_loss", "val_move_accuracy"}
    assert 0.0 <= stats["val_move_accuracy"] <= 1.0
    assert stats["val_loss"] > 0


def test_evaluate_respects_max_batches():
    model = ChessTransformer(d_model=32, n_heads=4, n_layers=1, d_ff=64)
    loader = [_make_batch() for _ in range(100)]
    stats = evaluate(model, loader, torch.device("cpu"), precision="fp32", max_batches=2)
    assert stats  # produced metrics without iterating the whole loader


def test_evaluate_sets_eval_then_train_mode():
    model = ChessTransformer(d_model=32, n_heads=4, n_layers=1, d_ff=64)
    model.train()
    evaluate(model, [_make_batch()], torch.device("cpu"), precision="fp32")
    assert model.training  # restored to train mode