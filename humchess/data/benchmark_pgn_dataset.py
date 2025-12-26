#!/usr/bin/env python3
"""
Benchmark PGN processing speed per ply/game/file using PGNDataset.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from torch.utils.data import DataLoader

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
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader worker processes (0 disables)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for DataLoader iteration")
    parser.add_argument("--prefetch-factor", type=int, default=2,
                        help="Number of batches to prefetch per worker")
    args = parser.parse_args()

    if args.num_workers > 0 and len(args.pgn) == 1:
        pgn_path = Path(args.pgn[0])
        index_path = pgn_path.with_suffix(pgn_path.suffix + ".idx.json")
        if not index_path.exists():
            raise FileNotFoundError(
                f"Missing PGN index for {pgn_path}. Build it with build_pgn_index.py or the Rust indexer."
            )

    dataset = PGNDataset(
        pgn_paths=args.pgn,
        min_elo=args.min_elo,
        max_elo=args.max_elo,
        skip_first_n_plies=args.skip_first_n_plies,
        include_metadata=True,
    )

    use_dataloader = args.num_workers > 0 or args.batch_size > 1
    if not use_dataloader:
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

    total_plies = 0
    total_time = 0.0
    games_processed = 0
    last_game_by_file: dict[str, int] = {}

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=lambda x: x,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    prev_time = time.perf_counter()
    for batch in loader:
        now = time.perf_counter()
        elapsed = now - prev_time
        prev_time = now
        total_time += elapsed

        for sample in batch:
            meta = sample['meta']
            file_label = meta['file']
            game_idx = meta['game']

            last_game = last_game_by_file.get(file_label)
            if last_game is None or game_idx != last_game:
                last_game_by_file[file_label] = game_idx
                games_processed += 1

            total_plies += 1

            if args.max_plies is not None and total_plies >= args.max_plies:
                break

        if args.log_every and total_plies and total_plies % args.log_every == 0:
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
