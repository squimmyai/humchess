"""Tests for drop_last in PGNDataset.from_parquet.

Under torch.compile(fullgraph=True) each distinct input shape triggers a fresh
graph, and shard row counts differ, so trailing partial batches used to overflow
the Dynamo recompile cache and hard-fail mid-training (seen at step 116,400).
drop_last=True keeps every emitted batch at exactly batch_size.
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from humchess.data.pgn_dataset import (  # noqa: E402
    PGNDataset,
    _pack_legal_mask,
    collate_fn,
)
from humchess.data.tokenization import SEQ_LENGTH, NUM_MOVE_CLASSES  # noqa: E402


def _write_shard(path: Path, n: int) -> Path:
    """Write a parquet shard with `n` synthetic-but-valid rows."""
    tokens = np.zeros((n, SEQ_LENGTH), dtype=np.int64)
    tokens[:, 0] = 1  # CLS
    move_id = np.zeros(n, dtype=np.int16)
    promo_id = np.full(n, -1, dtype=np.int8)
    is_promotion = np.zeros(n, dtype=bool)
    mask = torch.zeros(NUM_MOVE_CLASSES, dtype=torch.bool)
    mask[0] = True
    legal = [_pack_legal_mask(mask) for _ in range(n)]
    table = pa.table({
        "tokens": pa.array(tokens.tolist(), type=pa.list_(pa.int64())),
        "move_id": pa.array(move_id),
        "promo_id": pa.array(promo_id),
        "is_promotion": pa.array(is_promotion),
        "legal_mask": pa.array(legal, type=pa.binary()),
    })
    pq.write_table(table, path)
    return path


@pytest.fixture
def shards(tmp_path):
    # Deliberately unequal row counts, some not multiples of the batch size.
    rows = (13, 7, 20, 5)
    return [str(_write_shard(tmp_path / f"s{i}.parquet", n)) for i, n in enumerate(rows)]


@pytest.mark.parametrize("batch_size", [3, 4, 5])
def test_drop_last_single_shape(shards, batch_size):
    ds = PGNDataset.from_parquet(
        parquet_paths=shards, batch_size=batch_size, drop_last=True
    )
    batches = list(ds)
    assert batches, "no batches emitted"
    sizes = {b["tokens"].shape[0] for b in batches}
    assert sizes == {batch_size}, f"non-static shapes: {sizes}"


def test_drop_last_marks_shard_complete(shards):
    ds = PGNDataset.from_parquet(parquet_paths=shards, batch_size=3, drop_last=True)
    per_shard_last = {}
    for b in ds:
        per_shard_last[b["shard_path"]] = b["shard_complete"]
    # Every shard that yielded batches must have its final batch flagged complete.
    assert per_shard_last and all(per_shard_last.values()), per_shard_last


def test_no_drop_last_keeps_partial(shards):
    ds = PGNDataset.from_parquet(parquet_paths=shards, batch_size=3, drop_last=False)
    sizes = [b["tokens"].shape[0] for b in ds]
    assert any(s < 3 for s in sizes), sizes
