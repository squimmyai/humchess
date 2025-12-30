"""Tests for Parquet dataset generation and loading."""

import tempfile
from pathlib import Path

import pytest
import torch
import numpy as np

from humchess.data.pgn_dataset import (
    PGNDataset,
    write_parquet_from_pgn,
    collate_fn,
    _pack_legal_mask,
    _unpack_legal_mask,
)
from humchess.data.tokenization import (
    SEQ_LENGTH,
    NUM_MOVE_CLASSES,
    NUM_HISTORY_PLIES,
    NO_MOVE_ID,
    normalize_move_id,
)


# Sample PGN with multiple games for testing
SAMPLE_PGN = '''[Event "Rated Blitz game"]
[Site "https://lichess.org/tGpzk7yJ"]
[White "calvinmaster"]
[Black "dislikechess"]
[Result "1-0"]
[UTCDate "2017.03.31"]
[UTCTime "22:00:01"]
[WhiteElo "2186"]
[BlackElo "1907"]
[WhiteRatingDiff "+4"]
[BlackRatingDiff "-4"]
[ECO "C34"]
[Opening "King's Gambit Accepted, Schallopp Defense"]
[TimeControl "180+0"]
[Termination "Normal"]

1. e4 { [%clk 0:03:00] } e5 { [%clk 0:03:00] } 2. f4 { [%clk 0:02:58] } exf4 { [%clk 0:02:58] } 3. Nf3 { [%clk 0:02:57] } Nf6 { [%clk 0:02:57] } 4. e5 { [%clk 0:02:55] } Nh5 { [%clk 0:02:55] } 5. Bc4 { [%clk 0:02:54] } g5 { [%clk 0:02:54] } 6. h4 { [%clk 0:02:50] } Ng3 { [%clk 0:02:31] } 7. Nxg5 { [%clk 0:02:41] } Nxh1 { [%clk 0:02:29] } 8. Bxf7+ { [%clk 0:02:25] } Ke7 { [%clk 0:02:28] } 9. Nc3 { [%clk 0:02:25] } c6 { [%clk 0:02:25] } 10. d4 { [%clk 0:02:19] } h6 { [%clk 0:02:20] } 11. Qh5 { [%clk 0:02:16] } Bg7 { [%clk 0:02:15] } 12. Nge4 { [%clk 0:02:08] } Qf8 { [%clk 0:02:06] } 13. Nd6 { [%clk 0:02:03] } Na6 { [%clk 0:01:53] } 14. Bxf4 { [%clk 0:02:01] } Nb4 { [%clk 0:01:50] } 15. Kd2 { [%clk 0:01:49] } Nf2 { [%clk 0:01:46] } 16. Rf1 { [%clk 0:01:45] } Rh7 { [%clk 0:01:32] } 17. Rxf2 { [%clk 0:01:41] } Bh8 { [%clk 0:01:24] } 18. Bg5+ { [%clk 0:01:31] } hxg5 { [%clk 0:01:15] } 19. Qxg5+ { [%clk 0:01:31] } 1-0

[Event "Rated Bullet game"]
[Site "https://lichess.org/LzvBtZ93"]
[White "Gregster101"]
[Black "flavietta"]
[Result "1-0"]
[UTCDate "2017.03.31"]
[UTCTime "22:00:00"]
[WhiteElo "1385"]
[BlackElo "1339"]
[WhiteRatingDiff "+10"]
[BlackRatingDiff "-9"]
[ECO "C34"]
[Opening "King's Gambit Accepted, King's Knight Gambit"]
[TimeControl "120+1"]
[Termination "Time forfeit"]

1. e4 { [%clk 0:02:00] } e5 { [%clk 0:02:00] } 2. f4 { [%clk 0:02:00] } exf4 { [%clk 0:02:00] } 3. Nf3 { [%clk 0:01:59] } Bc5 { [%clk 0:01:59] } 4. d4 { [%clk 0:01:59] } Bb6 { [%clk 0:01:58] } 5. Bxf4 { [%clk 0:01:59] } Nf6 { [%clk 0:01:58] } 6. e5 { [%clk 0:01:58] } Qe7 { [%clk 0:01:56] } 7. Bc4 { [%clk 0:01:54] } O-O { [%clk 0:01:53] } 8. O-O { [%clk 0:01:53] } Ng4 { [%clk 0:01:49] } 9. c3 { [%clk 0:01:39] } d6 { [%clk 0:01:39] } 10. Nbd2 { [%clk 0:01:32] } Nc6 { [%clk 0:01:36] } 11. Re1 { [%clk 0:01:30] } Bf5 { [%clk 0:01:28] } 12. exd6 { [%clk 0:01:26] } cxd6 { [%clk 0:01:27] } 13. Rxe7 { [%clk 0:01:24] } Nxe7 { [%clk 0:01:23] } 14. Nh4 { [%clk 0:01:21] } Bd7 { [%clk 0:01:18] } 15. Bxd6 { [%clk 0:01:16] } Rfe8 { [%clk 0:01:12] } 16. Qf3 { [%clk 0:01:13] } Nf6 { [%clk 0:00:42] } 17. Re1 { [%clk 0:01:07] } Ned5 { [%clk 0:00:33] } 18. Rxe8+ { [%clk 0:01:05] } Rxe8 { [%clk 0:00:32] } 1-0

[Event "Rated Bullet tournament"]
[Site "https://lichess.org/pwao4UdD"]
[White "Farnese66"]
[Black "TryHardStopping"]
[Result "0-1"]
[UTCDate "2017.03.31"]
[UTCTime "22:00:02"]
[WhiteElo "1609"]
[BlackElo "1599"]
[WhiteRatingDiff "-13"]
[BlackRatingDiff "+13"]
[ECO "A06"]
[Opening "Zukertort Opening: Tennison Gambit"]
[TimeControl "60+0"]
[Termination "Normal"]

1. e4 { [%clk 0:01:00] } d5 { [%clk 0:01:00] } 2. Nf3 { [%clk 0:01:00] } dxe4 { [%clk 0:00:59] } 3. Ng5 { [%clk 0:00:59] } Bf5 { [%clk 0:00:58] } 4. d3 { [%clk 0:00:57] } exd3 { [%clk 0:00:56] } 5. Bxd3 { [%clk 0:00:55] } Bxd3 { [%clk 0:00:55] } 6. cxd3 { [%clk 0:00:55] } Nf6 { [%clk 0:00:54] } 7. O-O { [%clk 0:00:52] } h6 { [%clk 0:00:53] } 8. Nf3 { [%clk 0:00:49] } e6 { [%clk 0:00:51] } 9. Re1 { [%clk 0:00:49] } Bd6 { [%clk 0:00:48] } 10. Nc3 { [%clk 0:00:49] } O-O { [%clk 0:00:47] } 11. Be3 { [%clk 0:00:48] } Nc6 { [%clk 0:00:45] } 12. Qd2 { [%clk 0:00:48] } Ne5 { [%clk 0:00:44] } 13. a3 { [%clk 0:00:48] } Nxf3+ { [%clk 0:00:42] } 14. gxf3 { [%clk 0:00:46] } Nh5 { [%clk 0:00:41] } 15. Bxh6 { [%clk 0:00:41] } Qh4 { [%clk 0:00:39] } 16. Qe2 { [%clk 0:00:40] } Qxh2+ { [%clk 0:00:38] } 17. Kf1 { [%clk 0:00:37] } Qh1# { [%clk 0:00:38] } 0-1
'''


@pytest.fixture
def sample_pgn_file():
    """Create a temporary PGN file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pgn', delete=False) as f:
        f.write(SAMPLE_PGN)
        return Path(f.name)


# =============================================================================
# Legal mask packing/unpacking tests
# =============================================================================


class TestLegalMaskPacking:
    """Test legal mask packing and unpacking."""

    def test_pack_unpack_round_trip(self):
        """Test that pack -> unpack returns original mask."""
        mask = torch.zeros(NUM_MOVE_CLASSES, dtype=torch.bool)
        # Set some legal moves
        mask[0] = True
        mask[100] = True
        mask[4095] = True

        packed = _pack_legal_mask(mask)
        unpacked = _unpack_legal_mask(packed)

        assert unpacked.shape == (NUM_MOVE_CLASSES,)
        assert (unpacked == mask).all()

    def test_packed_size(self):
        """Test that packed mask is 512 bytes (4096 bits)."""
        mask = torch.zeros(NUM_MOVE_CLASSES, dtype=torch.bool)
        packed = _pack_legal_mask(mask)
        assert len(packed) == 512

    def test_all_zeros(self):
        """Test packing all zeros."""
        mask = torch.zeros(NUM_MOVE_CLASSES, dtype=torch.bool)
        packed = _pack_legal_mask(mask)
        unpacked = _unpack_legal_mask(packed)
        assert not unpacked.any()

    def test_all_ones(self):
        """Test packing all ones."""
        mask = torch.ones(NUM_MOVE_CLASSES, dtype=torch.bool)
        packed = _pack_legal_mask(mask)
        unpacked = _unpack_legal_mask(packed)
        assert unpacked.all()

    def test_random_mask(self):
        """Test packing random mask."""
        mask = torch.randint(0, 2, (NUM_MOVE_CLASSES,), dtype=torch.bool)
        packed = _pack_legal_mask(mask)
        unpacked = _unpack_legal_mask(packed)
        assert (unpacked == mask).all()

    def test_typical_legal_moves_count(self):
        """Test with typical number of legal moves (~20-40)."""
        mask = torch.zeros(NUM_MOVE_CLASSES, dtype=torch.bool)
        # Set ~30 random legal moves
        legal_indices = torch.randperm(NUM_MOVE_CLASSES)[:30]
        mask[legal_indices] = True

        packed = _pack_legal_mask(mask)
        unpacked = _unpack_legal_mask(packed)
        assert (unpacked == mask).all()
        assert unpacked.sum() == 30


# =============================================================================
# Parquet round-trip tests
# =============================================================================


class TestParquetRoundTrip:
    """Test parquet generation and loading."""

    def test_parquet_round_trip(self, sample_pgn_file, tmp_path):
        """Test that PGN -> Parquet -> Dataset preserves data."""
        shards = write_parquet_from_pgn(
            pgn_path=sample_pgn_file,
            out_dir=tmp_path,
            max_games_per_shard=10,
        )
        assert len(shards) == 1

        pgn_samples = list(PGNDataset(pgn_paths=[sample_pgn_file]))
        parquet_samples = list(PGNDataset.from_parquet(parquet_paths=shards))

        assert len(pgn_samples) == len(parquet_samples)

        # Check first sample in detail
        first_pgn = pgn_samples[0]
        first_parquet = parquet_samples[0]

        assert first_pgn['move_id'] == first_parquet['move_id']
        assert first_pgn['promo_id'] == first_parquet['promo_id']
        assert first_pgn['is_promotion'] == first_parquet['is_promotion']
        assert first_pgn['tokens'].tolist() == first_parquet['tokens'].tolist()
        assert first_pgn['legal_mask'].shape == first_parquet['legal_mask'].shape
        assert (first_pgn['legal_mask'] == first_parquet['legal_mask']).all()

    def test_parquet_token_shape(self, sample_pgn_file, tmp_path):
        """Test that parquet samples have correct token shape including history."""
        shards = write_parquet_from_pgn(
            pgn_path=sample_pgn_file,
            out_dir=tmp_path,
            max_games_per_shard=10,
        )

        samples = list(PGNDataset.from_parquet(parquet_paths=shards))
        for sample in samples[:10]:
            assert sample['tokens'].shape == (SEQ_LENGTH,)
            assert sample['tokens'].dtype == torch.long

    def test_parquet_history_tokens(self, sample_pgn_file, tmp_path):
        """Test that history tokens are included in parquet data."""
        shards = write_parquet_from_pgn(
            pgn_path=sample_pgn_file,
            out_dir=tmp_path,
            max_games_per_shard=10,
        )

        parquet_samples = list(PGNDataset.from_parquet(parquet_paths=shards))

        # First position should have NO_MOVE_ID padding for history
        first_sample = parquet_samples[0]
        history_tokens = first_sample['tokens'][68:74]
        # All should be NO_MOVE_ID (4096) since no history at start
        assert all(t == 4096 for t in history_tokens.tolist())

        # Later positions should have real history
        # Skip first few plies to get some with history
        later_sample = parquet_samples[10]  # Should have some history
        history = later_sample['tokens'][68:74]
        # At least some should not be NO_MOVE_ID
        assert not all(t == 4096 for t in history.tolist())

    def test_parquet_history_values_correct(self, sample_pgn_file, tmp_path):
        """Test that history tokens contain correct move IDs from previous plies."""
        import chess
        import chess.pgn
        import io

        shards = write_parquet_from_pgn(
            pgn_path=sample_pgn_file,
            out_dir=tmp_path,
            max_games_per_shard=10,
        )

        parquet_samples = list(PGNDataset.from_parquet(parquet_paths=shards))

        # Parse the first game to get ground truth moves
        with open(sample_pgn_file, 'r') as f:
            game = chess.pgn.read_game(f)

        board = game.board()
        moves = list(game.mainline_moves())

        # Track move history as we replay
        move_history = []

        for ply, move in enumerate(moves):
            sample = parquet_samples[ply]
            history_tokens = sample['tokens'][68:74].tolist()

            # Determine if this position is black to move (for normalization)
            is_black = board.turn == chess.BLACK

            # Check each history slot
            for i in range(NUM_HISTORY_PLIES):
                expected_token = history_tokens[i]

                if i < len(move_history):
                    # Get raw move ID from history (most recent first)
                    raw_id = move_history[-(i + 1)]
                    # Apply normalization
                    normalized_id = normalize_move_id(raw_id, is_black)
                    assert expected_token == normalized_id, (
                        f"Ply {ply}, history slot {i}: expected {normalized_id}, got {expected_token}"
                    )
                else:
                    # Should be padding
                    assert expected_token == NO_MOVE_ID, (
                        f"Ply {ply}, history slot {i}: expected NO_MOVE_ID, got {expected_token}"
                    )

            # Add current move to history for next iteration
            raw_move_id = move.from_square * 64 + move.to_square
            move_history.append(raw_move_id)
            if len(move_history) > NUM_HISTORY_PLIES:
                move_history.pop(0)

            board.push(move)

    def test_parquet_legal_mask_shape(self, sample_pgn_file, tmp_path):
        """Test that legal masks from parquet have correct shape."""
        shards = write_parquet_from_pgn(
            pgn_path=sample_pgn_file,
            out_dir=tmp_path,
            max_games_per_shard=10,
        )

        samples = list(PGNDataset.from_parquet(parquet_paths=shards))
        for sample in samples[:10]:
            assert sample['legal_mask'].shape == (NUM_MOVE_CLASSES,)
            assert sample['legal_mask'].dtype == torch.bool
            # Should have at least one legal move
            assert sample['legal_mask'].sum() > 0

    def test_parquet_move_is_legal(self, sample_pgn_file, tmp_path):
        """Test that target move is always legal."""
        shards = write_parquet_from_pgn(
            pgn_path=sample_pgn_file,
            out_dir=tmp_path,
            max_games_per_shard=10,
        )

        samples = list(PGNDataset.from_parquet(parquet_paths=shards))
        for sample in samples:
            assert sample['legal_mask'][sample['move_id']].item() is True


# =============================================================================
# Multi-shard tests
# =============================================================================


class TestMultiShardParquet:
    """Test multi-shard parquet generation and loading."""

    def test_multiple_shards_by_games(self, sample_pgn_file, tmp_path):
        """Test that multiple shards are created with max_games_per_shard=1."""
        shards = write_parquet_from_pgn(
            pgn_path=sample_pgn_file,
            out_dir=tmp_path,
            max_games_per_shard=1,
        )
        # Should create 3 shards (one per game)
        assert len(shards) == 3

        # All samples should be loadable
        all_samples = list(PGNDataset.from_parquet(parquet_paths=shards))
        pgn_samples = list(PGNDataset(pgn_paths=[sample_pgn_file]))
        assert len(all_samples) == len(pgn_samples)

    def test_multiple_shards_by_plies(self, sample_pgn_file, tmp_path):
        """Test shard splitting by max_plies_per_shard."""
        shards = write_parquet_from_pgn(
            pgn_path=sample_pgn_file,
            out_dir=tmp_path,
            max_plies_per_shard=20,
        )
        # Should create multiple shards
        assert len(shards) > 1

        # All samples should be loadable
        all_samples = list(PGNDataset.from_parquet(parquet_paths=shards))
        pgn_samples = list(PGNDataset(pgn_paths=[sample_pgn_file]))
        assert len(all_samples) == len(pgn_samples)

    def test_shard_names_contain_game_range(self, sample_pgn_file, tmp_path):
        """Test that shard filenames contain game range."""
        shards = write_parquet_from_pgn(
            pgn_path=sample_pgn_file,
            out_dir=tmp_path,
            max_games_per_shard=1,
        )
        for shard in shards:
            assert 'games_' in shard.name
            assert shard.suffix == '.parquet'


# =============================================================================
# DataLoader integration tests
# =============================================================================


class TestParquetDataLoader:
    """Test parquet loading with DataLoader."""

    def test_dataloader_batching(self, sample_pgn_file, tmp_path):
        """Test that parquet data works with DataLoader."""
        from torch.utils.data import DataLoader

        shards = write_parquet_from_pgn(
            pgn_path=sample_pgn_file,
            out_dir=tmp_path,
            max_games_per_shard=10,
        )

        dataset = PGNDataset.from_parquet(parquet_paths=shards)
        loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
        batch = next(iter(loader))

        assert batch['tokens'].shape == (4, SEQ_LENGTH)
        assert batch['move_id'].shape == (4,)
        assert batch['promo_id'].shape == (4,)
        assert batch['legal_mask'].shape == (4, NUM_MOVE_CLASSES)
        assert batch['is_promotion'].shape == (4,)

    def test_dataloader_dtypes(self, sample_pgn_file, tmp_path):
        """Test tensor dtypes in batched data."""
        from torch.utils.data import DataLoader

        shards = write_parquet_from_pgn(
            pgn_path=sample_pgn_file,
            out_dir=tmp_path,
            max_games_per_shard=10,
        )

        dataset = PGNDataset.from_parquet(parquet_paths=shards)
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        batch = next(iter(loader))

        assert batch['tokens'].dtype == torch.long
        assert batch['move_id'].dtype == torch.long
        assert batch['promo_id'].dtype == torch.long
        assert batch['legal_mask'].dtype == torch.bool
        assert batch['is_promotion'].dtype == torch.bool

    def test_shard_path_in_batch(self, sample_pgn_file, tmp_path):
        """Test that shard_path is included in batches."""
        from torch.utils.data import DataLoader

        shards = write_parquet_from_pgn(
            pgn_path=sample_pgn_file,
            out_dir=tmp_path,
            max_games_per_shard=10,
        )

        dataset = PGNDataset.from_parquet(parquet_paths=shards)
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        batch = next(iter(loader))

        assert 'shard_paths' in batch
        assert len(batch['shard_paths']) == 2
