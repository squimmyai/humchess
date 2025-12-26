"""Tests for Parquet PGN dumps."""

import tempfile
from pathlib import Path

import pytest

from humchess.data.pgn_dataset import PGNDataset, write_parquet_from_pgn


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
'''


@pytest.fixture
def sample_pgn_file():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pgn', delete=False) as f:
        f.write(SAMPLE_PGN)
        return Path(f.name)


def test_parquet_round_trip(sample_pgn_file, tmp_path):
    shards = write_parquet_from_pgn(
        pgn_path=sample_pgn_file,
        out_dir=tmp_path,
        max_games_per_shard=10,
    )
    assert len(shards) == 1

    pgn_samples = list(PGNDataset(pgn_paths=[sample_pgn_file]))
    parquet_samples = list(PGNDataset.from_parquet(parquet_paths=shards))

    assert len(pgn_samples) == len(parquet_samples)

    first_pgn = pgn_samples[0]
    first_parquet = parquet_samples[0]

    assert first_pgn['move_id'] == first_parquet['move_id']
    assert first_pgn['promo_id'] == first_parquet['promo_id']
    assert first_pgn['is_promotion'] == first_parquet['is_promotion']
    assert first_pgn['tokens'].tolist() == first_parquet['tokens'].tolist()
    assert first_pgn['legal_mask'].shape == first_parquet['legal_mask'].shape
    assert (first_pgn['legal_mask'] == first_parquet['legal_mask']).all()
