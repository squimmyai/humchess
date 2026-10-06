"""CPU regression tests for the multi-GPU deadlock fixes in humchess.train.

The original bug: each rank iterated its own dataloader, and ranks receive
different numbers of shards/batches. Collectives issued from inside the loop
(logging all_reduce, checkpoint all_reduce, final all_gather_object) were then
called a different number of times per rank, which deadlocks.

These tests run real 2-process DDP with gloo on CPU, giving each rank a
deliberately different number of batches. They must complete without hanging.

A short process-group timeout is set so that, if the equal-step logic regresses,
the mismatch surfaces as an exception rather than an indefinite hang.
"""

import datetime
import os

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp

from humchess.data.tokenization import (
    SEQ_LENGTH,
    NUM_MOVE_CLASSES,
)
from humchess.model.transformer import ChessTransformer
from humchess.train import train_epoch

PG_TIMEOUT = datetime.timedelta(seconds=30)
BATCH = 4
PORT = 29517


def _make_batch(batch_size: int = BATCH) -> dict:
    """Create a synthetic pre-batched sample dict (as the dataset yields)."""
    tokens = torch.randint(0, 66, (batch_size, SEQ_LENGTH), dtype=torch.long)
    move_id = torch.randint(0, NUM_MOVE_CLASSES, (batch_size,), dtype=torch.long)
    legal_mask = torch.zeros(batch_size, NUM_MOVE_CLASSES, dtype=torch.bool)
    legal_mask[torch.arange(batch_size), move_id] = True  # ensure target is legal
    legal_mask[:, :10] = True
    return {
        "tokens": tokens,
        "move_id": move_id,
        "promo_id": torch.full((batch_size,), -1, dtype=torch.long),
        "legal_mask": legal_mask,
        "is_promotion": torch.zeros(batch_size, dtype=torch.bool),
        "shard_path": "shard.parquet",
        "shard_complete": True,
    }


def _run(rank: int, world_size: int, batches: list[int], log_interval: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(PORT)
    dist.init_process_group("gloo", rank=rank, world_size=world_size, timeout=PG_TIMEOUT)
    try:
        torch.manual_seed(0)
        model = ChessTransformer(d_model=32, n_heads=4, n_layers=1, d_ff=64)
        model = nn.parallel.DistributedDataParallel(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        dataloader = [_make_batch() for _ in range(batches[rank])]
        stats = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=torch.device("cpu"),
            precision="fp32",
            scaler=None,
            rank=rank,
            world_size=world_size,
            log_interval=log_interval,
            is_distributed=True,
            checkpoint_dir=None,
            checkpoint_every_n_shards=None,
        )
        assert "loss" in stats
        assert stats["completed_shards"] == {"shard.parquet"}
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "batches,log_interval",
    [
        ([5, 3], 2),
        ([3, 5], 2),
        ([1, 4], 2),
        ([4, 1], 2),
        ([7, 2], 3),
        ([2, 2], 2),  # equal counts (control)
        ([6, 1], 100),  # shorter than log_interval
    ],
)
def test_uneven_batches_do_not_deadlock(batches, log_interval):
    """Ranks with different batch counts must finish together, not hang."""
    mp.spawn(
        _run,
        args=(2, batches, log_interval),
        nprocs=2,
        join=True,
    )