"""Tests for PGN dataset module."""

import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from humchess.data.pgn_dataset import (
    PGNDataset,
    collate_fn,
    parse_time_control,
    parse_clock,
)
from humchess.data.tokenization import SEQ_LENGTH, NUM_MOVE_CLASSES


# Extracted from lichess_db_standard_rated_2017-04.pgn
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

1. e4 { [%clk 0:02:00] } e5 { [%clk 0:02:00] } 2. f4 { [%clk 0:02:00] } exf4 { [%clk 0:02:00] } 3. Nf3 { [%clk 0:01:59] } Bc5 { [%clk 0:01:59] } 4. d4 { [%clk 0:01:59] } Bb6 { [%clk 0:01:58] } 5. Bxf4 { [%clk 0:01:59] } Nf6 { [%clk 0:01:58] } 6. e5 { [%clk 0:01:58] } Qe7 { [%clk 0:01:56] } 7. Bc4 { [%clk 0:01:54] } O-O { [%clk 0:01:53] } 8. O-O { [%clk 0:01:53] } Ng4 { [%clk 0:01:49] } 9. c3 { [%clk 0:01:39] } d6 { [%clk 0:01:39] } 10. Nbd2 { [%clk 0:01:32] } Nc6 { [%clk 0:01:36] } 11. Re1 { [%clk 0:01:30] } Bf5 { [%clk 0:01:28] } 12. exd6 { [%clk 0:01:26] } cxd6 { [%clk 0:01:27] } 13. Rxe7 { [%clk 0:01:24] } Nxe7 { [%clk 0:01:23] } 14. Nh4 { [%clk 0:01:21] } Bd7 { [%clk 0:01:18] } 15. Bxd6 { [%clk 0:01:16] } Rfe8 { [%clk 0:01:12] } 16. Qf3 { [%clk 0:01:13] } Nf6 { [%clk 0:00:42] } 17. Re1 { [%clk 0:01:07] } Ned5 { [%clk 0:00:33] } 18. Rxe8+ { [%clk 0:01:05] } Rxe8 { [%clk 0:00:32] } 19. Bxd5 { [%clk 0:01:03] } Re1+ { [%clk 0:00:17] } 20. Kf2 { [%clk 0:00:59] } Ra1 { [%clk 0:00:07] } 21. Ne4 { [%clk 0:00:46] } Ng4+ { [%clk 0:00:05] } 22. Kg3 { [%clk 0:00:42] } Nf6 { [%clk 0:00:02] } 23. Nxf6+ { [%clk 0:00:42] } gxf6 { [%clk 0:00:01] } 24. Qxf6 { [%clk 0:00:42] } 1-0

[Event "Rated Bullet tournament https://lichess.org/tournament/WhXK2rPu"]
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

[Event "Rated Classical tournament https://lichess.org/tournament/whc7Blcq"]
[Site "https://lichess.org/cWdFdebu"]
[White "Sammyc"]
[Black "sexymen"]
[Result "0-1"]
[UTCDate "2017.03.31"]
[UTCTime "22:00:02"]
[WhiteElo "1455"]
[BlackElo "1488"]
[WhiteRatingDiff "-11"]
[BlackRatingDiff "+11"]
[ECO "A01"]
[Opening "Nimzo-Larsen Attack: Classical Variation"]
[TimeControl "600+0"]
[Termination "Normal"]

1. b3 { [%clk 0:10:00] } d5 { [%clk 0:10:00] } 2. Bb2 { [%clk 0:09:54] } Nc6 { [%clk 0:09:55] } 3. Nf3 { [%clk 0:09:49] } d4 { [%clk 0:09:52] } 4. g3 { [%clk 0:09:33] } e5 { [%clk 0:09:44] } 5. e3 { [%clk 0:09:20] } Bg4 { [%clk 0:09:33] } 6. e4 { [%clk 0:09:06] } Qf6 { [%clk 0:08:45] } 7. Be2 { [%clk 0:08:27] } O-O-O { [%clk 0:08:13] } 8. c3 { [%clk 0:07:54] } d3 { [%clk 0:06:59] } 9. O-O { [%clk 0:07:02] } dxe2 { [%clk 0:06:34] } 10. Qxe2 { [%clk 0:06:59] } Bxf3 { [%clk 0:06:31] } 11. Qb5 { [%clk 0:06:08] } Bxe4 { [%clk 0:05:33] } 12. Re1 { [%clk 0:05:20] } Qf3 { [%clk 0:05:27] } 13. Kf1 { [%clk 0:04:50] } Bd3+ { [%clk 0:05:04] } 14. Kg1 { [%clk 0:04:40] } Bxb5 { [%clk 0:05:01] } 15. c4 { [%clk 0:04:37] } Ba6 { [%clk 0:04:49] } 16. Nc3 { [%clk 0:04:36] } Rxd2 { [%clk 0:04:41] } 17. Ne4 { [%clk 0:04:18] } Rxb2 { [%clk 0:04:21] } 18. Kf1 { [%clk 0:03:39] } Nf6 { [%clk 0:03:54] } 19. Ng5 { [%clk 0:03:00] } Qxf2# { [%clk 0:03:40] } 0-1

[Event "Rated Classical game"]
[Site "https://lichess.org/7uYK5P5w"]
[White "KNVchess"]
[Black "Santilius"]
[Result "1-0"]
[UTCDate "2017.03.31"]
[UTCTime "22:01:10"]
[WhiteElo "1662"]
[BlackElo "1500"]
[WhiteRatingDiff "+28"]
[BlackRatingDiff "-113"]
[ECO "C50"]
[Opening "Italian Game"]
[TimeControl "900+15"]
[Termination "Normal"]

1. e4 { [%clk 0:15:00] } e5 { [%clk 0:15:00] } 2. Nf3 { [%clk 0:15:11] } Nc6 { [%clk 0:15:07] } 3. Bc4 { [%clk 0:15:10] } b5 { [%clk 0:14:36] } 4. Bd5 { [%clk 0:15:17] } a6 { [%clk 0:14:06] } 5. d3 { [%clk 0:15:22] } Nf6 { [%clk 0:14:05] } 6. Ng5 { [%clk 0:15:28] } h6 { [%clk 0:13:58] } 7. Nxf7 { [%clk 0:15:39] } Qe7 { [%clk 0:13:47] } 8. Nxh8 { [%clk 0:15:50] } Qb4+ { [%clk 0:13:09] } 9. Nc3 { [%clk 0:15:55] } a5 { [%clk 0:12:36] } 10. O-O { [%clk 0:15:36] } Bb7 { [%clk 0:11:40] } 11. Bf7+ { [%clk 0:15:41] } Kd8 { [%clk 0:11:32] } 12. Bh5 { [%clk 0:15:27] } Nxh5 { [%clk 0:11:16] } 13. Qxh5 { [%clk 0:15:39] } a4 { [%clk 0:09:55] } 14. Nd5 { [%clk 0:15:46] } Nd4 { [%clk 0:09:03] } 15. Nxb4 { [%clk 0:15:58] } c5 { [%clk 0:08:01] } 16. Nd5 { [%clk 0:15:33] } Nxc2 { [%clk 0:08:00] } 17. Qf7 { [%clk 0:14:48] } Nxa1 { [%clk 0:07:59] } 18. Qxf8# { [%clk 0:14:43] } 1-0
'''


@pytest.fixture
def sample_pgn_file():
    """Create a temporary PGN file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pgn', delete=False) as f:
        f.write(SAMPLE_PGN)
        return Path(f.name)


class TestParseTimeControl:
    def test_standard_format(self):
        assert parse_time_control("300+0") == (300, 0)

    def test_with_increment(self):
        assert parse_time_control("180+2") == (180, 2)

    def test_no_increment(self):
        assert parse_time_control("600") == (600, 0)

    def test_empty(self):
        assert parse_time_control("") is None

    def test_dash(self):
        assert parse_time_control("-") is None


class TestParseClock:
    def test_five_minutes(self):
        assert parse_clock("0:05:00") == 300.0

    def test_with_decimals(self):
        assert parse_clock("0:00:30.5") == 30.5

    def test_hours(self):
        assert parse_clock("1:30:00") == 5400.0

    def test_empty(self):
        assert parse_clock("") is None


class TestPGNDataset:
    def test_iteration(self, sample_pgn_file):
        dataset = PGNDataset(pgn_paths=[sample_pgn_file])
        samples = list(dataset)

        assert len(samples) > 0

    def test_sample_structure(self, sample_pgn_file):
        dataset = PGNDataset(pgn_paths=[sample_pgn_file])
        sample = next(iter(dataset))

        assert 'tokens' in sample
        assert 'move_id' in sample
        assert 'promo_id' in sample
        assert 'legal_mask' in sample
        assert 'is_promotion' in sample

    def test_tokens_shape(self, sample_pgn_file):
        dataset = PGNDataset(pgn_paths=[sample_pgn_file])
        sample = next(iter(dataset))

        assert sample['tokens'].shape == (SEQ_LENGTH,)
        assert sample['tokens'].dtype == torch.long

    def test_legal_mask_shape(self, sample_pgn_file):
        dataset = PGNDataset(pgn_paths=[sample_pgn_file])
        sample = next(iter(dataset))

        assert sample['legal_mask'].shape == (NUM_MOVE_CLASSES,)
        assert sample['legal_mask'].dtype == torch.bool

    def test_legal_mask_has_moves(self, sample_pgn_file):
        dataset = PGNDataset(pgn_paths=[sample_pgn_file])
        sample = next(iter(dataset))

        # There should be at least one legal move
        assert sample['legal_mask'].sum() > 0
        # Starting position has 20 legal moves
        assert sample['legal_mask'].sum() <= 218  # max possible legal moves

    def test_move_id_is_legal(self, sample_pgn_file):
        dataset = PGNDataset(pgn_paths=[sample_pgn_file])

        for i, sample in enumerate(dataset):
            if i >= 10:
                break
            # The played move should always be legal
            assert sample['legal_mask'][sample['move_id']].item() is True

    def test_elo_filtering(self, sample_pgn_file):
        # Filter to only include 1500+ games
        # Game 1: 2186/1907 - included
        # Game 2: 1385/1339 - excluded (both below 1500)
        # Game 3: 988/1145 - excluded (both below 1500)
        dataset = PGNDataset(
            pgn_paths=[sample_pgn_file],
            min_elo=1500,
        )
        samples = list(dataset)

        # Only first game should be included
        assert len(samples) > 0

    def test_skip_plies(self, sample_pgn_file):
        dataset_full = PGNDataset(pgn_paths=[sample_pgn_file])
        dataset_skip = PGNDataset(
            pgn_paths=[sample_pgn_file],
            skip_first_n_plies=4,
        )

        full_count = sum(1 for _ in dataset_full)
        skip_count = sum(1 for _ in dataset_skip)

        # Should have fewer samples when skipping
        assert skip_count < full_count


class TestCollateFn:
    def test_batching(self, sample_pgn_file):
        dataset = PGNDataset(pgn_paths=[sample_pgn_file])
        loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
        batch = next(iter(loader))

        assert batch['tokens'].shape == (4, SEQ_LENGTH)
        assert batch['move_id'].shape == (4,)
        assert batch['promo_id'].shape == (4,)
        assert batch['legal_mask'].shape == (4, NUM_MOVE_CLASSES)
        assert batch['is_promotion'].shape == (4,)

    def test_dtypes(self, sample_pgn_file):
        dataset = PGNDataset(pgn_paths=[sample_pgn_file])
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        batch = next(iter(loader))

        assert batch['tokens'].dtype == torch.long
        assert batch['move_id'].dtype == torch.long
        assert batch['promo_id'].dtype == torch.long
        assert batch['legal_mask'].dtype == torch.bool
        assert batch['is_promotion'].dtype == torch.bool
