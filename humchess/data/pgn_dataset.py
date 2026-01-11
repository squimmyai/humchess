"""
Streaming PGN dataset for training.

Implements an IterableDataset that streams positions from PGN files,
applying white normalization and computing legality masks on-the-fly.

Supports S3-compatible storage (including Cloudflare R2) via PyArrow's S3FileSystem.
Set environment variables:
  - R2_ACCESS_KEY_ID: Access key
  - R2_SECRET_ACCESS_KEY: Secret key
  - R2_ENDPOINT_URL: Endpoint (e.g., https://<account>.r2.cloudflarestorage.com)

Use s3:// URLs in config paths, e.g.:
  parquet:
    - s3://bucket-name/path/to/shards/*.parquet
"""

import json
import os
import re
import time
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Optional
from urllib.parse import urlparse

import chess
import chess.pgn
import torch
from torch.utils.data import IterableDataset, get_worker_info


def _get_s3_filesystem(endpoint_url: str | None = None):
    """
    Get a PyArrow S3FileSystem for the given endpoint.

    Credentials are read from environment variables:
      - R2_ACCESS_KEY_ID / AWS_ACCESS_KEY_ID
      - R2_SECRET_ACCESS_KEY / AWS_SECRET_ACCESS_KEY
      - R2_ENDPOINT_URL (if endpoint_url not provided)

    Note: Not cached because each DataLoader worker process needs its own instance.
    """
    import pyarrow.fs as pafs

    access_key = os.environ.get('R2_ACCESS_KEY_ID') or os.environ.get('AWS_ACCESS_KEY_ID')
    secret_key = os.environ.get('R2_SECRET_ACCESS_KEY') or os.environ.get('AWS_SECRET_ACCESS_KEY')
    endpoint = endpoint_url or os.environ.get('R2_ENDPOINT_URL')

    if not access_key or not secret_key:
        raise ValueError(
            "S3 credentials not found. Set R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY "
            "(or AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY) environment variables."
        )

    # R2 and most S3-compatible storage require path-style addressing
    # (bucket in path, not subdomain)
    return pafs.S3FileSystem(
        endpoint_override=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        force_virtual_addressing=False,  # Use path-style for R2 compatibility
        request_timeout=30,
        connect_timeout=10,
    )


def _parse_s3_path(path: str) -> tuple[str, str]:
    """Parse s3://bucket/key into (bucket, key)."""
    parsed = urlparse(path)
    bucket = parsed.netloc
    key = parsed.path.lstrip('/')
    return bucket, key


def expand_s3_glob(pattern: str) -> list[str]:
    """
    Expand an S3 glob pattern like s3://bucket/path/*.parquet.

    Returns list of full s3:// URLs.
    """
    import pyarrow.fs as pafs

    bucket, key_pattern = _parse_s3_path(pattern)
    fs = _get_s3_filesystem()

    # PyArrow's glob doesn't work on S3, so we list and filter
    # Find the prefix before any glob characters
    prefix_parts = []
    for part in key_pattern.split('/'):
        if '*' in part or '?' in part:
            break
        prefix_parts.append(part)
    prefix = '/'.join(prefix_parts)

    # List all files under prefix
    selector = pafs.FileSelector(f"{bucket}/{prefix}", recursive=True)
    try:
        file_infos = fs.get_file_info(selector)
    except Exception as e:
        raise ValueError(f"Failed to list S3 path s3://{bucket}/{prefix}: {e}")

    # Filter by glob pattern
    import fnmatch
    results = []
    for info in file_infos:
        if info.type == pafs.FileType.File:
            # info.path is "bucket/key"
            full_key = info.path[len(bucket) + 1:]  # Remove "bucket/" prefix
            if fnmatch.fnmatch(full_key, key_pattern):
                results.append(f"s3://{bucket}/{full_key}")

    return sorted(results)

from .tokenization import (
    SEQ_LENGTH, NUM_MOVE_CLASSES, NUM_HISTORY_PLIES, NO_MOVE_ID,
    board_to_tokens, normalize_position, move_to_ids, is_promotion_move,
    normalize_move_id,
)


class LRUCache(OrderedDict):
    """Simple LRU cache with a maximum size."""

    def __init__(self, maxsize: int = 100_000):
        super().__init__()
        self.maxsize = maxsize

    def get(self, key, default=None):
        if key in self:
            self.move_to_end(key)
            return self[key]
        return default

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            self.popitem(last=False)


def parse_time_control(tc_str: str) -> Optional[tuple[int, int]]:
    """
    Parse time control string to (initial_seconds, increment_seconds).

    Examples: "300+0", "180+2", "600"
    Returns None if unparseable.
    """
    if not tc_str or tc_str == '-':
        return None

    match = re.match(r'(\d+)\+?(\d+)?', tc_str)
    if not match:
        return None

    initial = int(match.group(1))
    increment = int(match.group(2)) if match.group(2) else 0
    return initial, increment


def parse_clock(clock_str: str) -> Optional[float]:
    """
    Parse clock string to seconds remaining.

    Examples: "0:05:00", "0:00:30.5"
    Returns None if unparseable.
    """
    if not clock_str:
        return None

    match = re.match(r'(\d+):(\d+):(\d+(?:\.\d+)?)', clock_str)
    if not match:
        return None

    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = float(match.group(3))
    return hours * 3600 + minutes * 60 + seconds


class PGNDataset(IterableDataset):
    """
    Streaming dataset that yields positions from PGN files.

    Each item is a dict containing:
        - tokens: tensor of shape (SEQ_LENGTH,) - input token IDs
        - move_id: int - target move class (0-4095)
        - promo_id: int or -1 - promotion class (0-3) or -1 if not a promotion
        - legal_mask: tensor of shape (NUM_MOVE_CLASSES,) - 1 for legal, 0 for illegal
        - is_promotion: bool - whether this move is a promotion
    """

    def __init__(
        self,
        pgn_paths: Optional[list[str | Path]] = None,
        *,
        parquet_paths: Optional[list[str | Path]] = None,
        min_elo: int = 0,
        max_elo: int = 4000,
        skip_first_n_plies: int = 0,
        start_game: int = 0,
        end_game: Optional[int] = None,
        include_metadata: bool = False,
        rank: int = 0,
        world_size: int = 1,
        skip_shards: Optional[set[str]] = None,
    ):
        """
        Args:
            pgn_paths: List of paths to PGN files.
            min_elo: Minimum Elo to include (filters both players).
            max_elo: Maximum Elo to include.
            skip_first_n_plies: Skip opening moves (e.g., 10 to skip first 5 moves each).
            start_game: First game index to include per PGN file.
            end_game: End game index (exclusive) per PGN file.
            skip_shards: Set of shard paths to skip (for resuming from checkpoint).
        """
        self.pgn_paths = [Path(p) for p in pgn_paths or []]
        # Keep S3 URLs as strings (Path mangles s3:// to s3:/)
        self.parquet_paths: list[Path | str] = [
            p if str(p).startswith('s3://') else Path(p)
            for p in parquet_paths or []
        ]
        self.min_elo = min_elo
        self.max_elo = max_elo
        self.skip_first_n_plies = skip_first_n_plies
        self.start_game = start_game
        self.end_game = end_game
        self.include_metadata = include_metadata
        self.rank = rank
        self.world_size = world_size
        self.skip_shards = skip_shards or set()
        self._legal_mask_cache: LRUCache = LRUCache(maxsize=100_000)
        self._pgn_index_cache: dict[Path, dict[str, object]] = {}
        self._batch_size: int = 1024  # Default, overridden by from_parquet

    @classmethod
    def from_parquet(
        cls,
        parquet_paths: list[str | Path],
        rank: int = 0,
        world_size: int = 1,
        batch_size: int = 1024,
        skip_shards: Optional[set[str]] = None,
    ) -> "PGNDataset":
        """Create a dataset that reads pre-tokenized Parquet shards."""
        dataset = cls(
            pgn_paths=None,
            parquet_paths=parquet_paths,
            rank=rank,
            world_size=world_size,
            skip_shards=skip_shards,
        )
        dataset._batch_size = batch_size
        return dataset

    def __iter__(self) -> Iterator[dict]:
        worker_info = get_worker_info()

        if self.parquet_paths:
            yield from self._iter_parquet(worker_info)
            return

        # Shard files across workers
        if worker_info is None:
            # Single-process loading
            paths = self.pgn_paths
            shard_by_game = False
        else:
            # Multi-process: shard by file when possible, otherwise shard by game.
            if len(self.pgn_paths) == 1:
                paths = self.pgn_paths
                shard_by_game = True
            else:
                paths = [
                    p for i, p in enumerate(self.pgn_paths)
                    if i % worker_info.num_workers == worker_info.id
                ]
                shard_by_game = False

        for path in paths:
            yield from self._iter_pgn(path, worker_info, shard_by_game)

    def _iter_parquet(self, worker_info) -> Iterator[dict]:
        import pyarrow.parquet as pq

        # Two-level sharding: by DDP rank first, then by DataLoader worker
        # This ensures each (rank, worker) pair gets a unique subset of files
        if self.world_size > 1:
            # First shard by rank
            rank_paths = [
                p for i, p in enumerate(self.parquet_paths)
                if i % self.world_size == self.rank
            ]
        else:
            rank_paths = list(self.parquet_paths)

        if worker_info is None:
            paths = rank_paths
        else:
            # Then shard by worker within this rank's files
            paths = [
                p for i, p in enumerate(rank_paths)
                if i % worker_info.num_workers == worker_info.id
            ]

        # Filter out already-completed shards (for resume)
        if self.skip_shards:
            paths = [p for p in paths if str(p) not in self.skip_shards]

        # Check if using S3 paths (check first path, assume all are same type)
        use_s3 = paths and str(paths[0]).startswith('s3://')
        fs = _get_s3_filesystem() if use_s3 else None

        for path in paths:
            path_str = str(path)
            if use_s3:
                # For S3, strip the s3:// prefix - PyArrow expects "bucket/key"
                bucket, key = _parse_s3_path(path_str)
                # Use read_table for R2 compatibility (avoids range request issues)
                table = pq.read_table(f"{bucket}/{key}", filesystem=fs)
                num_rows = table.num_rows
                for i in range(0, num_rows, self._batch_size):
                    batch = table.slice(i, min(self._batch_size, num_rows - i))
                    is_last_batch = (i + self._batch_size >= num_rows)
                    try:
                        for result in self._process_parquet_batch(batch):
                            result['shard_path'] = path_str
                            result['shard_complete'] = is_last_batch
                            yield result
                    except ValueError as e:
                        raise ValueError(f"{e} (shard: {path_str}, batch offset: {i})") from e
            else:
                parquet_file = pq.ParquetFile(path)
                num_rows = parquet_file.metadata.num_rows
                rows_yielded = 0
                for batch in parquet_file.iter_batches(batch_size=self._batch_size):
                    batch_offset = rows_yielded
                    rows_yielded += batch.num_rows
                    is_last_batch = (rows_yielded >= num_rows)
                    try:
                        for result in self._process_parquet_batch(batch):
                            result['shard_path'] = path_str
                            result['shard_complete'] = is_last_batch
                            yield result
                    except ValueError as e:
                        raise ValueError(f"{e} (shard: {path_str}, batch offset: {batch_offset})") from e

    def _process_parquet_batch(self, batch) -> Iterator[dict]:
        """Process a single parquet batch into training samples."""
        import numpy as np

        # Convert tokens - numpy will fail or create object array if jagged
        tokens_np = np.array(batch.column("tokens").to_pylist(), dtype=np.int64)
        if tokens_np.ndim != 2 or tokens_np.shape[1] != SEQ_LENGTH:
            raise ValueError(f"Corrupt parquet: tokens shape {tokens_np.shape}, expected (N, {SEQ_LENGTH})")

        tokens = torch.from_numpy(tokens_np)
        move_ids = torch.from_numpy(batch.column("move_id").to_numpy().astype(np.int64))
        promo_ids = torch.from_numpy(batch.column("promo_id").to_numpy().astype(np.int64))
        is_promos = torch.from_numpy(batch.column("is_promotion").to_numpy(zero_copy_only=False))

        # Validate token ranges - catches corrupt parquet data
        board_tokens = tokens[:, :68]
        history_tokens = tokens[:, 68:74]
        if (board_tokens < 0).any() or (board_tokens >= 66).any():
            bad_mask = (board_tokens < 0) | (board_tokens >= 66)
            bad_idx = bad_mask.nonzero()
            raise ValueError(f"Corrupt parquet: invalid board token at row {bad_idx[0, 0].item()}, "
                           f"col {bad_idx[0, 1].item()}, value {board_tokens[bad_idx[0, 0], bad_idx[0, 1]].item()}")
        if (history_tokens < 0).any() or (history_tokens > NO_MOVE_ID).any():
            bad_mask = (history_tokens < 0) | (history_tokens > NO_MOVE_ID)
            bad_idx = bad_mask.nonzero()
            raise ValueError(f"Corrupt parquet: invalid history token at row {bad_idx[0, 0].item()}, "
                           f"col {bad_idx[0, 1].item()}, value {history_tokens[bad_idx[0, 0], bad_idx[0, 1]].item()}")

        # Vectorized legal mask unpacking
        mask_bytes_list = batch.column("legal_mask").to_pylist()
        legal_masks = _unpack_legal_masks_batch(mask_bytes_list)

        # Yield pre-batched dict
        yield {
            'tokens': tokens,
            'move_id': move_ids,
            'promo_id': promo_ids,
            'legal_mask': legal_masks,
            'is_promotion': is_promos,
        }
                    
    def _iter_pgn(self, path: Path, worker_info, shard_by_game: bool) -> Iterator[dict]:
        """Iterate over all positions in a single PGN file."""
        worker_id = worker_info.id if worker_info is not None else None
        if worker_info is None and (self.start_game > 0 or self.end_game is not None):
            try:
                index = self._load_pgn_index(path)
            except FileNotFoundError:
                index = None
                raise

            if index is not None:
                offsets = index["offsets"]
                stride = index["stride"]
                total_games = index["total_games"]
                if total_games == 0:
                    return

                global_start = max(self.start_game, 0)
                global_end = total_games if self.end_game is None else min(self.end_game, total_games)
                if global_start >= global_end:
                    return

                start_checkpoint = global_start // stride
                if start_checkpoint >= len(offsets):
                    return
                checkpoint_offset = offsets[start_checkpoint]
                game_idx = start_checkpoint * stride

                with open(path, 'r', errors='replace') as f:
                    f.seek(checkpoint_offset)
                    while game_idx < global_end:
                        game = chess.pgn.read_game(f)
                        if game is None:
                            break
                        if game_idx >= global_start:
                            yield from self._iter_game(game, path, game_idx, worker_id)
                        game_idx += 1
                return

        with open(path, 'r', errors='replace') as f:
            game_idx = 0
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                if game_idx < self.start_game:
                    game_idx += 1
                    continue
                if self.end_game is not None and game_idx >= self.end_game:
                    break

                yield from self._iter_game(game, path, game_idx, worker_id)
                game_idx += 1

    def _iter_game(
        self,
        game: chess.pgn.Game,
        path: Path,
        game_idx: int,
        worker_id: Optional[int],
    ) -> Iterator[dict]:
        """Iterate over positions in a single game."""
        # Extract headers
        headers = game.headers
        white_elo = self._parse_elo(headers.get('WhiteElo'))
        black_elo = self._parse_elo(headers.get('BlackElo'))

        # Filter by Elo
        if white_elo is None or black_elo is None:
            return
        if not (self.min_elo <= white_elo <= self.max_elo):
            return
        if not (self.min_elo <= black_elo <= self.max_elo):
            return

        # Replay game
        board = game.board()
        node = game

        # Track move history (raw move IDs before normalization)
        move_history: list[int] = []

        ply = 0
        for node in game.mainline():
            move = node.move

            # Skip opening plies (but still track history)
            if ply < self.skip_first_n_plies:
                # Record raw move ID for history
                raw_move_id = move.from_square * 64 + move.to_square
                move_history.append(raw_move_id)
                if len(move_history) > NUM_HISTORY_PLIES:
                    move_history.pop(0)
                board.push(move)
                ply += 1
                continue

            # Get Elo for side to move
            elo = white_elo if board.turn == chess.WHITE else black_elo

            # Get time remaining from clock comment if available
            time_left = self._get_clock_from_node(node)

            # Get position before move
            move_uci = move.uci()

            # Convert to tokens (returns tuple with is_black_to_move flag)
            tokens, is_black = board_to_tokens(board, elo, time_left)

            # Apply white normalization
            tokens, move_uci = normalize_position(tokens, move_uci, is_black)

            # Build normalized history (most recent first)
            # Pad with NO_MOVE_ID if insufficient history
            history_ids = []
            for i in range(NUM_HISTORY_PLIES):
                if i < len(move_history):
                    # Get from end of list (most recent)
                    raw_id = move_history[-(i + 1)]
                    history_ids.append(normalize_move_id(raw_id, is_black))
                else:
                    history_ids.append(NO_MOVE_ID)

            # Append history to tokens (now 74 elements total)
            tokens = tokens + history_ids

            # Convert move to IDs
            move_id, promo_id = move_to_ids(move_uci)

            # Create legality mask (using normalized position)
            # We need to recreate the board for the normalized position
            # For simplicity, we use the original board's legal moves and transform them
            # Include EP square in cache key - same pieces can have different legal moves
            # depending on whether en passant is available
            cache_key = (tuple(tokens[1:66]), board.ep_square)
            legal_mask = self._legal_mask_cache.get(cache_key)
            if legal_mask is None:
                legal_move_ids = self._get_normalized_legal_moves(board, is_black)
                legal_mask = torch.zeros(NUM_MOVE_CLASSES, dtype=torch.bool)
                legal_mask[legal_move_ids] = True
                self._legal_mask_cache[cache_key] = legal_mask

            # Check if promotion
            board_tokens = tokens[1:65]  # squares only
            is_promo = is_promotion_move(move_id, board_tokens)

            yield {
                'tokens': torch.tensor(tokens, dtype=torch.long),
                'move_id': move_id,
                'promo_id': promo_id if promo_id is not None else -1,
                'legal_mask': legal_mask,
                'is_promotion': is_promo,
                **(
                    {
                        'meta': {
                            'file': str(path),
                            'game': game_idx,
                            'ply': ply,
                            **({'worker': worker_id} if worker_id is not None else {}),
                        }
                    } if self.include_metadata else {}
                ),
            }

            # Record raw move ID for history (before advancing)
            raw_move_id = move.from_square * 64 + move.to_square
            move_history.append(raw_move_id)
            if len(move_history) > NUM_HISTORY_PLIES:
                move_history.pop(0)

            # Advance position
            board.push(move)
            ply += 1

    def _parse_elo(self, elo_str: Optional[str]) -> Optional[int]:
        """Parse Elo string to int, returning None if invalid."""
        if not elo_str or elo_str == '?':
            return None
        try:
            return int(elo_str)
        except ValueError:
            return None

    def _get_clock_from_node(self, node: chess.pgn.ChildNode) -> Optional[float]:
        """Extract clock time from node comment if present."""
        comment = node.comment
        if not comment:
            return None

        # Look for [%clk H:MM:SS] format
        match = re.search(r'\[%clk\s+(\d+:\d+:\d+(?:\.\d+)?)\]', comment)
        if match:
            return parse_clock(match.group(1))
        return None

    def _get_normalized_legal_moves(
        self,
        board: chess.Board,
        was_black: bool,
    ) -> list[int]:
        """Get legal move IDs, normalized if black to move."""
        move_ids = []
        for move in board.legal_moves:
            from_sq = move.from_square
            to_sq = move.to_square

            if was_black:
                # Apply 180° rotation to squares
                from_sq = 63 - from_sq
                to_sq = 63 - to_sq

            move_id = from_sq * 64 + to_sq
            move_ids.append(move_id)

        return list(set(move_ids))  # dedupe (promotions -> same from/to)

    def _load_pgn_index(self, path: Path) -> dict[str, object]:
        cached = self._pgn_index_cache.get(path)
        if cached is not None:
            return cached

        index_path = path.with_suffix(path.suffix + ".idx.json")
        stat = path.stat()
        index_data: Optional[dict[str, object]] = None

        if index_path.exists():
            try:
                data = json.loads(index_path.read_text(encoding="utf-8"))
                if data.get("size") == stat.st_size and data.get("mtime") == stat.st_mtime:
                    offsets = data.get("offsets")
                    stride = int(data.get("stride", 1))
                    if stride <= 0:
                        stride = 1
                    if offsets is not None:
                        offsets = list(offsets)
                    total_games = int(
                        data.get(
                            "total_games",
                            len(offsets) * stride if offsets else 0,
                        )
                    )
                    if offsets is not None:
                        index_data = {
                            "offsets": offsets,
                            "stride": stride,
                            "total_games": total_games,
                        }
            except (OSError, ValueError, TypeError):
                index_data = None

        if index_data is None:
            raise FileNotFoundError(
                f"Missing PGN index for {path}. "
                f"Build it with: uv run python -m humchess.data.build_parquet index --pgn {path}"
            )

        self._pgn_index_cache[path] = index_data
        return index_data


def _pack_legal_mask(mask: torch.Tensor) -> bytes:
    """Pack a 4096-bit legal mask into 512 bytes."""
    import numpy as np

    packed = np.packbits(mask.cpu().numpy().astype(np.uint8), bitorder="little")
    return packed.tobytes()


def _unpack_legal_mask(data: bytes) -> torch.Tensor:
    """Unpack 512 bytes into a 4096-bit legal mask tensor."""
    import numpy as np

    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="little")[:NUM_MOVE_CLASSES]
    return torch.from_numpy(bits.astype(bool))


def _unpack_legal_masks_batch(mask_bytes_list: list[bytes]) -> torch.Tensor:
    """Unpack a batch of legal masks efficiently."""
    import numpy as np

    batch_size = len(mask_bytes_list)
    # Concatenate all bytes, unpack once
    all_bytes = b''.join(mask_bytes_list)
    all_bits = np.unpackbits(np.frombuffer(all_bytes, dtype=np.uint8), bitorder="little")
    # Reshape to (batch, 4096) - each mask is 512 bytes = 4096 bits
    masks = all_bits.reshape(batch_size, -1)[:, :NUM_MOVE_CLASSES]
    return torch.from_numpy(masks.astype(bool))


def _split_range(total: int, worker_id: int, num_workers: int) -> tuple[int, int]:
    base = total // num_workers
    extra = total % num_workers
    start = worker_id * base + min(worker_id, extra)
    end = start + base + (1 if worker_id < extra else 0)
    return start, end


def write_parquet_from_pgn(
    pgn_path: str | Path,
    out_dir: str | Path,
    start_game: int = 0,
    end_game: Optional[int] = None,
    max_games_per_shard: int = 1000,
    max_plies_per_shard: Optional[int] = None,
    log_every_plies: int = 0,
    min_elo: int = 0,
    max_elo: int = 4000,
    skip_first_n_plies: int = 0,
    num_workers: int = 0,
    batch_size: int = 1,
    prefetch_factor: int = 2,
    progress_queue=None,
    worker_id: Optional[int] = None,
) -> list[Path]:
    """
    Write Parquet shards from a PGN file without splitting games across shards.

    If num_workers > 0, worker processes stream samples to the main process,
    which writes shards.
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    pgn_path = Path(pgn_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if num_workers > 0:
        index_path = pgn_path.with_suffix(pgn_path.suffix + ".idx.json")
        if not index_path.exists():
            raise FileNotFoundError(
                f"Missing PGN index for {pgn_path}. Build it with build_pgn_index.py or the Rust indexer."
            )

    dataset = PGNDataset(
        pgn_paths=[pgn_path],
        min_elo=min_elo,
        max_elo=max_elo,
        skip_first_n_plies=skip_first_n_plies,
        start_game=start_game,
        end_game=end_game,
        include_metadata=True,
    )

    shard_paths: list[Path] = []
    shard_tokens: list[list[int]] = []
    shard_move_ids: list[int] = []
    shard_promo_ids: list[int] = []
    shard_is_promos: list[bool] = []
    shard_masks: list[bytes] = []
    shard_game_start: Optional[int] = None
    shard_game_end: Optional[int] = None
    shard_games: int = 0

    current_game = None
    game_tokens: list[list[int]] = []
    game_move_ids: list[int] = []
    game_promo_ids: list[int] = []
    game_is_promos: list[bool] = []
    game_masks: list[bytes] = []

    def flush_shard():
        nonlocal shard_tokens, shard_move_ids, shard_promo_ids, shard_is_promos, shard_masks
        nonlocal shard_game_start, shard_game_end, shard_games
        if not shard_tokens:
            return

        stem = pgn_path.stem
        shard_path = out_dir / f"{stem}_games_{shard_game_start}-{shard_game_end}.parquet"

        table = pa.table({
            "tokens": shard_tokens,
            "move_id": np.array(shard_move_ids, dtype=np.int16),
            "promo_id": np.array(shard_promo_ids, dtype=np.int8),
            "is_promotion": np.array(shard_is_promos, dtype=bool),
            "legal_mask": shard_masks,
        })

        pq.write_table(table, shard_path)
        shard_paths.append(shard_path)

        shard_tokens = []
        shard_move_ids = []
        shard_promo_ids = []
        shard_is_promos = []
        shard_masks = []
        shard_game_start = None
        shard_game_end = None
        shard_games = 0

    def append_game_buffers(
        game_idx: int,
        tokens: list[list[int]],
        move_ids: list[int],
        promo_ids: list[int],
        is_promos: list[bool],
        masks: list[bytes],
    ):
        nonlocal shard_tokens, shard_move_ids, shard_promo_ids, shard_is_promos, shard_masks
        nonlocal shard_game_start, shard_game_end, shard_games
        if not tokens:
            return

        would_exceed = False
        if max_games_per_shard is not None and shard_games >= max_games_per_shard:
            would_exceed = True
        if max_plies_per_shard is not None and (len(shard_tokens) + len(tokens)) > max_plies_per_shard:
            would_exceed = True

        if would_exceed and shard_tokens:
            flush_shard()

        if shard_game_start is None:
            shard_game_start = game_idx
        shard_game_end = game_idx
        shard_games += 1

        shard_tokens.extend(tokens)
        shard_move_ids.extend(move_ids)
        shard_promo_ids.extend(promo_ids)
        shard_is_promos.extend(is_promos)
        shard_masks.extend(masks)

    total_games = 0
    total_plies = 0
    start_time = time.perf_counter()
    last_report_time = start_time
    last_report_plies = 0

    def maybe_report_progress():
        nonlocal last_report_time, last_report_plies
        if progress_queue is None:
            return
        if total_plies == last_report_plies:
            return
        now = time.perf_counter()
        if log_every_plies and total_plies % log_every_plies == 0:
            last_report_time = now
        elif now - last_report_time < 2.0:
            return
        last_report_time = now
        last_report_plies = total_plies
        progress_queue.put(
            {
                "type": "progress",
                "worker": worker_id,
                "plies": total_plies,
                "games": total_games,
            }
        )

    if num_workers <= 0:
        for sample in dataset:
            meta = sample['meta']
            game_idx = meta['game']

            if current_game is None:
                current_game = game_idx

            if game_idx != current_game:
                append_game_buffers(
                    current_game,
                    game_tokens,
                    game_move_ids,
                    game_promo_ids,
                    game_is_promos,
                    game_masks,
                )
                game_tokens = []
                game_move_ids = []
                game_promo_ids = []
                game_is_promos = []
                game_masks = []
                current_game = game_idx
                total_games += 1

            game_tokens.append(sample['tokens'].tolist())
            game_move_ids.append(int(sample['move_id']))
            game_promo_ids.append(int(sample['promo_id']))
            game_is_promos.append(bool(sample['is_promotion']))
            game_masks.append(_pack_legal_mask(sample['legal_mask']))
            total_plies += 1

            if log_every_plies and total_plies % log_every_plies == 0:
                elapsed = time.perf_counter() - start_time
                tokens_total = total_plies * SEQ_LENGTH
                plies_per_s = total_plies / elapsed if elapsed > 0 else 0.0
                games_per_s = total_games / elapsed if elapsed > 0 else 0.0
                tokens_per_s = tokens_total / elapsed if elapsed > 0 else 0.0
                if progress_queue is None:
                    print(
                        f"progress plies={total_plies} tokens={tokens_total} "
                        f"plies_per_s={plies_per_s:.2f} games_per_s={games_per_s:.4f} "
                        f"tokens_per_s={tokens_per_s:.2f}"
                    )
            maybe_report_progress()

        if current_game is not None:
            append_game_buffers(
                current_game,
                game_tokens,
                game_move_ids,
                game_promo_ids,
                game_is_promos,
                game_masks,
            )
            total_games += 1

        flush_shard()
        if progress_queue is not None:
            progress_queue.put(
                {
                    "type": "done",
                    "worker": worker_id,
                    "plies": total_plies,
                    "games": total_games,
                    "shards": [str(p) for p in shard_paths],
                }
            )
        return shard_paths

    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=lambda x: x,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    in_flight: dict[int, dict[str, object]] = {}

    def new_state(game_idx: int) -> dict[str, object]:
        return {
            "game_idx": game_idx,
            "tokens": [],
            "move_ids": [],
            "promo_ids": [],
            "is_promos": [],
            "masks": [],
        }

    def flush_state(state: dict[str, object]):
        nonlocal total_games
        tokens = state["tokens"]
        if not tokens:
            return
        append_game_buffers(
            state["game_idx"],
            tokens,
            state["move_ids"],
            state["promo_ids"],
            state["is_promos"],
            state["masks"],
        )
        total_games += 1

    for batch in loader:
        for sample in batch:
            meta = sample["meta"]
            game_idx = int(meta["game"])
            worker_id = int(meta.get("worker", 0))

            state = in_flight.get(worker_id)
            if state is None:
                state = new_state(game_idx)
                in_flight[worker_id] = state
            elif game_idx != state["game_idx"]:
                flush_state(state)
                state = new_state(game_idx)
                in_flight[worker_id] = state

            state["tokens"].append(sample["tokens"].tolist())
            state["move_ids"].append(int(sample["move_id"]))
            state["promo_ids"].append(int(sample["promo_id"]))
            state["is_promos"].append(bool(sample["is_promotion"]))
            state["masks"].append(_pack_legal_mask(sample["legal_mask"]))
            total_plies += 1

            if log_every_plies and total_plies % log_every_plies == 0:
                elapsed = time.perf_counter() - start_time
                tokens_total = total_plies * SEQ_LENGTH
                plies_per_s = total_plies / elapsed if elapsed > 0 else 0.0
                games_per_s = total_games / elapsed if elapsed > 0 else 0.0
                tokens_per_s = tokens_total / elapsed if elapsed > 0 else 0.0
                if progress_queue is None:
                    print(
                        f"progress plies={total_plies} tokens={tokens_total} "
                        f"plies_per_s={plies_per_s:.2f} games_per_s={games_per_s:.4f} "
                        f"tokens_per_s={tokens_per_s:.2f}"
                    )
            maybe_report_progress()

    for state in in_flight.values():
        flush_state(state)

    flush_shard()
    if progress_queue is not None:
        progress_queue.put(
            {
                "type": "done",
                "worker": worker_id,
                "plies": total_plies,
                "games": total_games,
                "shards": [str(p) for p in shard_paths],
            }
        )
    return shard_paths


def collate_fn(batch: list[dict]) -> dict:
    """
    Collate function for DataLoader.

    Returns batched tensors.
    """
    result = {
        'tokens': torch.stack([item['tokens'] for item in batch]),
        'move_id': torch.tensor([item['move_id'] for item in batch], dtype=torch.long),
        'promo_id': torch.tensor([item['promo_id'] for item in batch], dtype=torch.long),
        'legal_mask': torch.stack([item['legal_mask'] for item in batch]),
        'is_promotion': torch.tensor([item['is_promotion'] for item in batch], dtype=torch.bool),
    }
    if 'shard_path' in batch[0]:
        result['shard_paths'] = [item['shard_path'] for item in batch]
    return result
