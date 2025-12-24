#!/usr/bin/env python3
"""
Benchmark PGN processing speed per ply/game/file using PGNDataset.
"""

from __future__ import annotations

import argparse
import time

from .pgn_dataset import PGNDataset
from .tokenization import SEQ_LENGTH


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark PGN processing speed per ply/game/file.",
    )
    parser.add_argument("pgn", nargs="+", help="PGN file paths")
    parser.add_argument("--min-elo", type=int, default=0)
    parser.add_argument("--max-elo", type=int, default=4000)
    parser.add_argument("--skip-first-n-plies", type=int, default=0)
    parser.add_argument("--max-plies", type=int, default=None,
                        help="Max plies to process total")
    parser.add_argument("--log-every", type=int, default=0,
                        help="Log progress every N plies (0 disables)")
    args = parser.parse_args()

    dataset = PGNDataset(
        pgn_paths=args.pgn,
        min_elo=args.min_elo,
        max_elo=args.max_elo,
        skip_first_n_plies=args.skip_first_n_plies,
        include_metadata=True,
    )

    total_plies = 0
    total_time = 0.0
    file_plies = 0
    file_time = 0.0
    game_plies = 0
    game_time = 0.0
    games_processed = 0
    last_file = None
    last_game = None

    prev_time = time.perf_counter()
    for sample in dataset:
        now = time.perf_counter()
        elapsed = now - prev_time
        prev_time = now

        meta = sample['meta']
        file_label = meta['file']
        game_idx = meta['game']
        ply = meta['ply']

        if last_file is None:
            last_file = file_label
            last_game = game_idx
            games_processed = 1

        if file_label != last_file:
            if file_plies > 0:
                print(
                    f"{last_file} file-summary plies={file_plies} "
                    f"elapsed_s={file_time:.2f} ms_per_ply={file_time * 1000.0 / file_plies:.3f}"
                )
            file_plies = 0
            file_time = 0.0
            game_plies = 0
            game_time = 0.0
            last_file = file_label
            last_game = game_idx

        if game_idx != last_game:
            game_plies = 0
            game_time = 0.0
            last_game = game_idx
            games_processed += 1

        total_plies += 1
        total_time += elapsed
        file_plies += 1
        file_time += elapsed
        game_plies += 1
        game_time += elapsed

        if args.log_every and total_plies % args.log_every == 0:
            tokens_total = total_plies * SEQ_LENGTH
            tokens_per_s = tokens_total / total_time if total_time > 0 else 0.0
            print(
                f"progress plies={total_plies} tokens={tokens_total} "
                f"plies_per_s={(total_plies / total_time):.2f} "
                f"games_per_s={(games_processed / total_time):.4f} "
                f"tokens_per_s={tokens_per_s:.2f} "
                f"ms_per_ply={(total_time * 1000.0 / total_plies):.3f}"
            )

        if args.max_plies is not None and total_plies >= args.max_plies:
            break

    if file_plies > 0:
        print(
            f"{last_file} file-summary plies={file_plies} "
            f"elapsed_s={file_time:.2f} ms_per_ply={file_time * 1000.0 / file_plies:.3f}"
        )
    if total_plies > 0:
        tokens_total = total_plies * SEQ_LENGTH
        tokens_per_s = tokens_total / total_time if total_time > 0 else 0.0
        print(
            f"total summary plies={total_plies} tokens={tokens_total} "
            f"plies_per_s={(total_plies / total_time):.2f} "
            f"games_per_s={(games_processed / total_time):.4f} "
            f"tokens_per_s={tokens_per_s:.2f} "
            f"elapsed_s={total_time:.2f} ms_per_ply={total_time * 1000.0 / total_plies:.3f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
