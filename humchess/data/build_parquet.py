#!/usr/bin/env python3
"""
Build, validate, and combine Parquet shards from PGN files.

Commands
--------

index
    Build a PGN index file (.idx.json) by scanning for game boundaries.
    Much faster than parsing-based indexing (~100x speedup) since it only
    looks for structural patterns, not game content. Creates a sparse index
    with checkpoints every --stride games (default 10000).

    Example:
        uv run python -m humchess.data.build_parquet index \\
            --pgn data/lichess.pgn

build
    Build parquet shards from a PGN file. Each shard contains tokenized
    positions from a range of games. Requires an index file to exist.

    Example:
        uv run python -m humchess.data.build_parquet build \\
            --pgn data/lichess.pgn \\
            --out-dir data/tokenized \\
            --num-workers 8 \\
            --max-games-per-shard 10000

validate
    Check that parquet shards cover a contiguous game range with no gaps
    or overlaps. Returns exit code 0 if valid, 1 if issues found.

    Example:
        uv run python -m humchess.data.build_parquet validate \\
            --input-dir data/tokenized

combine
    Combine multiple parquet shards into a single file. Output filename is
    derived from shard names (e.g., "stem_games_0-100.parquet" -> "stem.parquet").
    Optionally validates game ranges and can delete source shards after.

    Example:
        uv run python -m humchess.data.build_parquet combine \\
            --input-dir data/tokenized

        # Skip validation and delete shards after:
        uv run python -m humchess.data.build_parquet combine \\
            --input-dir data/tokenized \\
            --no-validate \\
            --delete-shards
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import time
from queue import Empty
from pathlib import Path

from .pgn_dataset import SEQ_LENGTH, _split_range, write_parquet_from_pgn


def build_pgn_index(
    path: Path,
    stride: int = 10_000,
    log_every: int = 100_000,
) -> Path:
    """
    Build a PGN index file by scanning for game boundaries without parsing.

    This is much faster than the chess.pgn-based approach since it only looks
    for structural patterns (blank lines followed by '[') rather than parsing
    moves.

    Args:
        path: Path to PGN file
        stride: Store every Nth game offset (1 = all games, 10000 = sparse)
        log_every: Print progress every N games

    Returns:
        Path to the created index file
    """
    start_time = time.perf_counter()
    offsets: list[int] = []
    total_games = 0

    file_size = path.stat().st_size

    with open(path, "rb") as f:
        in_blank_region = True  # Start of file counts as "after blank"

        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break

            # Strip whitespace (handles \r\n, \n, spaces)
            stripped = line.strip()

            if not stripped:
                in_blank_region = True
                continue

            # New game: non-blank line starting with '[' after blank region
            if in_blank_region and stripped.startswith(b"["):
                if stride == 1 or total_games % stride == 0:
                    offsets.append(offset)
                total_games += 1
                in_blank_region = False

                if log_every and total_games % log_every == 0:
                    elapsed = time.perf_counter() - start_time
                    games_per_s = total_games / elapsed if elapsed > 0 else 0.0
                    pct = 100 * offset / file_size if file_size > 0 else 0
                    print(
                        f"indexing {path.name} games={total_games:,} "
                        f"progress={pct:.1f}% games_per_s={games_per_s:,.0f}"
                    )
            else:
                in_blank_region = False

    elapsed = time.perf_counter() - start_time
    games_per_s = total_games / elapsed if elapsed > 0 else 0.0
    print(
        f"indexing done {path.name} total_games={total_games:,} "
        f"offsets={len(offsets):,} elapsed={elapsed:.2f}s "
        f"games_per_s={games_per_s:,.0f}"
    )

    # Write index file
    stat = path.stat()
    index_data = {
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "stride": stride,
        "total_games": total_games,
        "offsets": offsets,
    }

    index_path = path.with_suffix(path.suffix + ".idx.json")
    tmp_path = index_path.with_suffix(index_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(index_data, ensure_ascii=True), encoding="utf-8")
    tmp_path.replace(index_path)

    print(f"wrote {index_path}")
    return index_path


def _parse_shard_filename(path: Path) -> tuple[str, int, int] | None:
    """Parse a shard filename to extract stem and game range.

    Expected format: {stem}_games_{start}-{end}.parquet
    Returns (stem, start, end) or None if parsing fails.
    """
    match = re.match(r"(.+)_games_(\d+)-(\d+)\.parquet$", path.name)
    if not match:
        return None
    stem = match.group(1)
    start = int(match.group(2))
    end = int(match.group(3))
    return stem, start, end


def _collect_shards(input_dir: Path) -> tuple[list[tuple[Path, int, int]], list[Path]]:
    """Collect and parse all shard files in a directory.

    Returns (all_shards, unparseable) where all_shards is sorted by start game.
    """
    shard_paths = sorted(input_dir.glob("*.parquet"))
    if not shard_paths:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")

    shards_by_stem: dict[str, list[tuple[Path, int, int]]] = {}
    unparseable: list[Path] = []

    for path in shard_paths:
        parsed = _parse_shard_filename(path)
        if parsed is None:
            unparseable.append(path)
            continue
        stem, start, end = parsed
        if stem not in shards_by_stem:
            shards_by_stem[stem] = []
        shards_by_stem[stem].append((path, start, end))

    if unparseable:
        print(f"Warning: {len(unparseable)} files with unparseable names (skipped)")
        for p in unparseable[:5]:
            print(f"  {p.name}")
        if len(unparseable) > 5:
            print(f"  ... and {len(unparseable) - 5} more")

    if len(shards_by_stem) > 1:
        print(f"Found {len(shards_by_stem)} different stems:")
        for stem in shards_by_stem:
            print(f"  {stem}: {len(shards_by_stem[stem])} shards")

    # Sort each stem's shards by start game, then flatten
    all_shards: list[tuple[Path, int, int]] = []
    for stem in sorted(shards_by_stem.keys()):
        shards = sorted(shards_by_stem[stem], key=lambda x: x[1])
        all_shards.extend(shards)

    # Sort globally by start game
    all_shards.sort(key=lambda x: x[1])

    return all_shards, unparseable


def validate_shards(input_dir: Path) -> bool:
    """Validate that parquet shards cover a contiguous game range.

    Returns True if validation passes (no gaps or overlaps).
    """
    all_shards, _ = _collect_shards(input_dir)

    print(f"Validating {len(all_shards)} shards...")
    gaps: list[tuple[int, int]] = []
    overlaps: list[tuple[int, int, int, int]] = []

    for i in range(1, len(all_shards)):
        _, prev_start, prev_end = all_shards[i - 1]
        _, curr_start, curr_end = all_shards[i]

        expected_start = prev_end + 1
        if curr_start > expected_start:
            gaps.append((prev_end, curr_start))
        elif curr_start < expected_start:
            overlaps.append((prev_start, prev_end, curr_start, curr_end))

    if gaps:
        print(f"WARNING: Found {len(gaps)} gaps in game coverage:")
        for prev_end, curr_start in gaps[:10]:
            print(f"  Games {prev_end + 1} to {curr_start - 1} missing")
        if len(gaps) > 10:
            print(f"  ... and {len(gaps) - 10} more gaps")

    if overlaps:
        print(f"WARNING: Found {len(overlaps)} overlapping ranges:")
        for ps, pe, cs, ce in overlaps[:10]:
            print(f"  [{ps}-{pe}] overlaps with [{cs}-{ce}]")
        if len(overlaps) > 10:
            print(f"  ... and {len(overlaps) - 10} more overlaps")

    if not gaps and not overlaps:
        first_start = all_shards[0][1]
        last_end = all_shards[-1][2]
        print(f"OK: Game ranges are contiguous from {first_start} to {last_end}")
        print(f"    Total games covered: {last_end - first_start + 1}")
        return True

    return False


def _derive_output_path(input_dir: Path, shards: list[tuple[Path, int, int]]) -> Path:
    """Derive output filename from shard names by removing the game range."""
    if not shards:
        raise ValueError("No shards to derive output path from")

    # Get stem from first shard, remove _games_X-Y suffix
    first_shard = shards[0][0]
    stem = first_shard.stem  # e.g. "lichess_games_0-100"
    # Remove _games_X-Y pattern
    stem = re.sub(r"_games_\d+-\d+$", "", stem)
    return input_dir / f"{stem}.parquet"


def combine_shards(
    input_dir: Path,
    output_path: Path | None = None,
    *,
    validate: bool = True,
    delete_shards: bool = False,
    batch_size: int = 50,
) -> None:
    """Combine multiple parquet shards into a single file.

    Args:
        input_dir: Directory containing parquet shards
        output_path: Path for the combined output file (derived from shard names if None)
        validate: If True, verify game ranges are contiguous
        delete_shards: If True, delete source shards after combining
        batch_size: Number of shards to load into memory at once (default 50)
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    all_shards, _ = _collect_shards(input_dir)

    if output_path is None:
        output_path = _derive_output_path(input_dir, all_shards)
        print(f"Output path: {output_path}")

    if validate:
        validate_shards(input_dir)

    # Estimate output size
    total_size = sum(path.stat().st_size for path, _, _ in all_shards)
    print(f"Estimated output size: {total_size / 1e9:.2f} GB")

    # Combine files in batches (balance memory vs speed)
    print(f"Combining {len(all_shards)} shards (batch_size={batch_size})...")

    total_rows = 0
    writer = None
    try:
        for batch_start in range(0, len(all_shards), batch_size):
            batch_end = min(batch_start + batch_size, len(all_shards))
            batch_shards = all_shards[batch_start:batch_end]

            # Load batch into memory
            tables = []
            for path, start, end in batch_shards:
                tables.append(pq.read_table(path))

            # Concatenate and write
            combined = pa.concat_tables(tables)
            total_rows += combined.num_rows

            if writer is None:
                writer = pq.ParquetWriter(output_path, combined.schema)

            writer.write_table(combined)
            print(f"  Wrote {batch_end}/{len(all_shards)} shards ({total_rows:,} rows)")

            # Free memory
            del tables, combined
    finally:
        if writer is not None:
            writer.close()

    print(f"Done: {output_path} ({total_rows:,} rows)")

    if delete_shards:
        print(f"Deleting {len(all_shards)} source shards...")
        for path, _, _ in all_shards:
            path.unlink()
        print("Source shards deleted")


def _read_index(path: Path) -> dict[str, int]:
    index_path = path.with_suffix(path.suffix + ".idx.json")
    if not index_path.exists():
        raise FileNotFoundError(
            f"Missing PGN index for {path}. Build it with build_pgn_index.py or the Rust indexer."
        )
    data = json.loads(index_path.read_text(encoding="utf-8"))
    stat = path.stat()
    size_in_index = data.get("size")
    mtime_in_index = data.get("mtime")
    mismatches: list[str] = []
    if size_in_index != stat.st_size:
        mismatches.append(
            f"size mismatch (index={size_in_index!r}, file={stat.st_size!r})"
        )
    if mtime_in_index != stat.st_mtime:
        mismatches.append(
            f"mtime mismatch (index={mtime_in_index!r}, file={stat.st_mtime!r})"
        )
    if mismatches:
        details = "; ".join(mismatches)
        raise ValueError(f"Index {index_path} does not match PGN file {path}: {details}")
    stride = int(data.get("stride", 1))
    if stride <= 0:
        stride = 1
    offsets = data.get("offsets") or []
    total_games = int(
        data.get("total_games", len(offsets) * stride if offsets else 0)
    )
    return {
        "total_games": total_games,
    }


def _scan_existing_shards(out_dir: Path, pgn_stem: str) -> list[tuple[int, int]]:
    """Scan output directory once and return sorted list of (start, end) ranges."""
    import re
    pattern = re.compile(rf"^{re.escape(pgn_stem)}_games_(\d+)-(\d+)\.parquet$")

    shard_ranges: list[tuple[int, int]] = []
    for p in out_dir.iterdir():
        m = pattern.match(p.name)
        if m:
            shard_ranges.append((int(m.group(1)), int(m.group(2))))

    shard_ranges.sort()
    return shard_ranges


def _is_range_covered(
    shard_ranges: list[tuple[int, int]],
    start_game: int,
    end_game: int,
) -> bool:
    """Check if a game range is fully covered by pre-scanned shard ranges."""
    if not shard_ranges:
        return False

    # Check if shards fully cover [start_game, end_game)
    # We need contiguous coverage from start_game to end_game-1
    covered_up_to = start_game - 1
    for shard_start, shard_end in shard_ranges:
        if shard_start > covered_up_to + 1:
            # Gap in coverage
            break
        if shard_start <= covered_up_to + 1 <= shard_end:
            covered_up_to = shard_end
        if covered_up_to >= end_game - 1:
            return True

    return covered_up_to >= end_game - 1


def _merge_contiguous_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge adjacent ranges like [(0,10), (10,20), (30,40)] -> [(0,20), (30,40)]."""
    if not ranges:
        return []
    merged: list[tuple[int, int]] = []
    current_start, current_end = ranges[0]
    for start, end in ranges[1:]:
        if start == current_end:  # Contiguous
            current_end = end
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    merged.append((current_start, current_end))
    return merged


def _worker(
    pgn_path: Path,
    out_dir: Path,
    assignments: list[tuple[int, int]],  # List of (start_game, end_game) tuples
    max_games_per_shard: int,
    max_plies_per_shard: int | None,
    log_every_plies: int,
    min_elo: int,
    max_elo: int,
    skip_first_n_plies: int,
    result_queue: mp.Queue,
    worker_id: int,
):
    import sys
    import traceback

    # Merge contiguous ranges to reduce overhead
    merged_assignments = _merge_contiguous_ranges(assignments)
    print(f"[Worker {worker_id}] {len(assignments)} items merged to {len(merged_assignments)} ranges")
    all_shards: list[str] = []

    for start_game, end_game in merged_assignments:
        try:
            shards = write_parquet_from_pgn(
                pgn_path=pgn_path,
                out_dir=out_dir,
                start_game=start_game,
                end_game=end_game,
                max_games_per_shard=max_games_per_shard,
                max_plies_per_shard=max_plies_per_shard,
                log_every_plies=log_every_plies,
                min_elo=min_elo,
                max_elo=max_elo,
                skip_first_n_plies=skip_first_n_plies,
                num_workers=0,
                progress_queue=result_queue,
                worker_id=worker_id,
            )
            all_shards.extend(str(p) for p in shards)
        except Exception as e:
            tb = traceback.format_exc()
            result_queue.put({
                "type": "error",
                "worker": worker_id,
                "start_game": start_game,
                "end_game": end_game,
                "error": str(e),
                "traceback": tb,
            })
            print(f"[Worker {worker_id}] FATAL ERROR processing games {start_game}-{end_game}:\n{tb}", file=sys.stderr)
            sys.exit(1)

    # Note: write_parquet_from_pgn sends "done" message, so we don't send another


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Parquet shards from a PGN file.")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Build command (default behavior)
    build_parser = subparsers.add_parser("build", help="Build parquet shards from PGN")
    build_parser.add_argument("--pgn", required=True, help="Path to PGN file")
    build_parser.add_argument("--out-dir", required=True, help="Output directory for Parquet shards")
    build_parser.add_argument("--start-game", type=int, default=0)
    build_parser.add_argument("--end-game", type=int, default=None)
    build_parser.add_argument("--max-games-per-shard", type=int, default=1000)
    build_parser.add_argument("--max-plies-per-shard", type=int, default=None)
    build_parser.add_argument("--log-every-plies", type=int, default=0)
    build_parser.add_argument("--min-elo", type=int, default=0)
    build_parser.add_argument("--max-elo", type=int, default=4000)
    build_parser.add_argument("--skip-first-n-plies", type=int, default=0)
    build_parser.add_argument("--num-workers", type=int, default=0)
    build_parser.add_argument("--batch-size", type=int, default=1)
    build_parser.add_argument("--prefetch-factor", type=int, default=2)
    build_parser.add_argument("--skip-existing", action="store_true", help="Skip shards that already exist")

    # Combine command
    combine_parser = subparsers.add_parser("combine", help="Combine parquet shards into one file")
    combine_parser.add_argument("--input-dir", required=True, help="Directory containing parquet shards")
    combine_parser.add_argument("--output", help="Output path (default: derived from shard names)")
    combine_parser.add_argument("--no-validate", action="store_true", help="Skip validation of game ranges")
    combine_parser.add_argument("--delete-shards", action="store_true", help="Delete source shards after combining")
    combine_parser.add_argument("--batch-size", type=int, default=50, help="Shards to load per batch (default: 50)")

    # Validate command (just check, don't combine)
    validate_parser = subparsers.add_parser("validate", help="Validate parquet shards without combining")
    validate_parser.add_argument("--input-dir", required=True, help="Directory containing parquet shards")

    # Index command
    index_parser = subparsers.add_parser("index", help="Build PGN index file (fast, no parsing)")
    index_parser.add_argument("--pgn", required=True, help="Path to PGN file")
    index_parser.add_argument(
        "--stride", type=int, default=10_000,
        help="Store every Nth game offset (default: 10000, matching Rust indexer)"
    )
    index_parser.add_argument(
        "--log-every", type=int, default=100_000,
        help="Print progress every N games"
    )

    args = parser.parse_args()

    # Handle combine command
    if args.command == "combine":
        combine_shards(
            input_dir=Path(args.input_dir),
            output_path=Path(args.output) if args.output else None,
            validate=not args.no_validate,
            delete_shards=args.delete_shards,
            batch_size=args.batch_size,
        )
        return 0

    # Handle validate command
    if args.command == "validate":
        ok = validate_shards(input_dir=Path(args.input_dir))
        return 0 if ok else 1

    # Handle index command
    if args.command == "index":
        build_pgn_index(
            path=Path(args.pgn),
            stride=args.stride,
            log_every=args.log_every,
        )
        return 0

    # Handle build command (or no command for backwards compatibility)
    if args.command is None:
        # For backwards compatibility, re-parse with old-style arguments
        parser = argparse.ArgumentParser(description="Build Parquet shards from a PGN file.")
        parser.add_argument("--pgn", required=True, help="Path to PGN file")
        parser.add_argument("--out-dir", required=True, help="Output directory for Parquet shards")
        parser.add_argument("--start-game", type=int, default=0)
        parser.add_argument("--end-game", type=int, default=None)
        parser.add_argument("--max-games-per-shard", type=int, default=1000)
        parser.add_argument("--max-plies-per-shard", type=int, default=None)
        parser.add_argument("--log-every-plies", type=int, default=0)
        parser.add_argument("--min-elo", type=int, default=0)
        parser.add_argument("--max-elo", type=int, default=4000)
        parser.add_argument("--skip-first-n-plies", type=int, default=0)
        parser.add_argument("--num-workers", type=int, default=0)
        parser.add_argument("--batch-size", type=int, default=1)
        parser.add_argument("--prefetch-factor", type=int, default=2)
        parser.add_argument("--skip-existing", action="store_true", help="Skip shards that already exist")
        args = parser.parse_args()

    pgn_path = Path(args.pgn)
    out_dir = Path(args.out_dir)

    if args.num_workers > 1:
        index_data = _read_index(pgn_path)
        total_games = index_data["total_games"]

        global_start = max(args.start_game, 0)
        global_end = total_games if args.end_game is None else min(args.end_game, total_games)
        if global_start >= global_end:
            return 0

        # Scan existing shards ONCE before spawning workers
        existing_shard_ranges: list[tuple[int, int]] = []
        if args.skip_existing:
            print(f"Scanning {out_dir} for existing shards (pgn_stem={pgn_path.stem})...")
            existing_shard_ranges = _scan_existing_shards(out_dir, pgn_path.stem)
            print(f"Found {len(existing_shard_ranges)} existing shards")
            if existing_shard_ranges:
                print(f"  First shard: {existing_shard_ranges[0]}")
                print(f"  Last shard: {existing_shard_ranges[-1]}")
            # DEBUG: uncomment next line to test if scan itself causes slowdown
            # existing_shard_ranges = []

        # Split into shard-sized work items and filter out covered ones
        work_items: list[tuple[int, int]] = []
        skipped_count = 0
        for start in range(global_start, global_end, args.max_games_per_shard):
            end = min(start + args.max_games_per_shard, global_end)
            if existing_shard_ranges and _is_range_covered(existing_shard_ranges, start, end):
                skipped_count += 1
            else:
                work_items.append((start, end))

        print(f"Work items: {len(work_items)} to process, {skipped_count} skipped (already covered)")

        if not work_items:
            print("All work items already covered, nothing to do")
            return 0

        # Distribute work items contiguously (not round-robin) so adjacent items
        # can be merged into single ranges, reducing per-call overhead
        num_workers = min(args.num_workers, len(work_items))
        items_per_worker = len(work_items) // num_workers
        extra = len(work_items) % num_workers

        worker_assignments: list[list[tuple[int, int]]] = []
        idx = 0
        for w in range(num_workers):
            count = items_per_worker + (1 if w < extra else 0)
            worker_assignments.append(work_items[idx:idx + count])
            idx += count

        print(f"Distributing {len(work_items)} work items to {num_workers} workers")

        ctx = mp.get_context("spawn")
        result_queue: mp.Queue = ctx.Queue()
        procs: list[mp.Process] = []
        start_time = time.perf_counter()
        worker_state: dict[int, dict[str, int]] = {}

        for worker_id, assignments in enumerate(worker_assignments):
            if not assignments:
                continue

            worker_state[worker_id] = {"plies": 0, "games": 0}
            proc = ctx.Process(
                target=_worker,
                args=(
                    pgn_path,
                    out_dir,
                    assignments,  # List of (start, end) tuples
                    args.max_games_per_shard,
                    args.max_plies_per_shard,
                    args.log_every_plies,
                    args.min_elo,
                    args.max_elo,
                    args.skip_first_n_plies,
                    result_queue,
                    worker_id,
                ),
            )
            proc.start()
            procs.append(proc)

        # Handle case where no workers were spawned (shouldn't happen given earlier check)
        if not procs:
            print("No workers spawned, nothing to do")
            return 0

        shards: list[str] = []
        finished_workers: set[int] = set()
        failed_workers: list[dict] = []
        while len(finished_workers) < len(procs):
            try:
                msg = result_queue.get(timeout=1.0)
                msg_type = msg.get("type")
                if msg_type == "progress":
                    worker_id = int(msg["worker"])
                    worker_state[worker_id]["plies"] = int(msg["plies"])
                    worker_state[worker_id]["games"] = int(msg["games"])
                    total_plies = sum(state["plies"] for state in worker_state.values())
                    total_games = sum(state["games"] for state in worker_state.values())
                    elapsed = time.perf_counter() - start_time
                    tokens_total = total_plies * SEQ_LENGTH
                    plies_per_s = total_plies / elapsed if elapsed > 0 else 0.0
                    games_per_s = total_games / elapsed if elapsed > 0 else 0.0
                    tokens_per_s = tokens_total / elapsed if elapsed > 0 else 0.0
                    print(
                        f"progress plies={total_plies} tokens={tokens_total} "
                        f"plies_per_s={plies_per_s:.2f} games_per_s={games_per_s:.4f} "
                        f"tokens_per_s={tokens_per_s:.2f}"
                    )
                elif msg_type == "error":
                    worker_id = int(msg["worker"])
                    if worker_id not in finished_workers:
                        failed_workers.append(msg)
                        finished_workers.add(worker_id)
                        print(
                            f"\n{'='*60}\n"
                            f"WORKER {worker_id} FAILED (games {msg.get('start_game')}-{msg.get('end_game')})\n"
                            f"Error: {msg.get('error')}\n"
                            f"Traceback:\n{msg.get('traceback', 'No traceback available')}"
                            f"{'='*60}\n"
                        )
                elif msg_type == "done":
                    worker_id = int(msg["worker"])
                    worker_state[worker_id]["plies"] = int(msg["plies"])
                    worker_state[worker_id]["games"] = int(msg["games"])
                    shards.extend(msg.get("shards", []))
                    # Mark worker as finished (handles multiple done messages from same worker)
                    finished_workers.add(worker_id)
            except Empty:
                for proc in procs:
                    if proc.exitcode not in (None, 0):
                        # Check if we already got an error message for this
                        if not any(fw.get("worker") == procs.index(proc) for fw in failed_workers):
                            print(
                                f"\nWorker process died with exit code {proc.exitcode} "
                                f"(no error message received - possibly OOM or signal kill)"
                            )
                        raise SystemExit(
                            f"Worker process failed with exit code {proc.exitcode}"
                        )

        for proc in procs:
            proc.join()

        if failed_workers:
            print(f"\n{'='*60}")
            print(f"SUMMARY: {len(failed_workers)} worker(s) failed")
            print("Missing game ranges that need to be re-processed:")
            for fw in failed_workers:
                print(f"  --start-game {fw.get('start_game')} --end-game {fw.get('end_game')}")
            print(f"{'='*60}\n")
            return 1

        for proc in procs:
            if proc.exitcode != 0:
                raise SystemExit(f"Worker process failed with exit code {proc.exitcode}")

        for path in shards:
            print(path)
        return 0

    # Single-worker path: check if range is already covered
    if args.skip_existing and args.end_game is not None:
        existing_shard_ranges = _scan_existing_shards(out_dir, pgn_path.stem)
        if _is_range_covered(existing_shard_ranges, args.start_game, args.end_game):
            print(f"Skipping games {args.start_game}-{args.end_game} (already covered)")
            return 0

    shards = write_parquet_from_pgn(
        pgn_path=pgn_path,
        out_dir=out_dir,
        start_game=args.start_game,
        end_game=args.end_game,
        max_games_per_shard=args.max_games_per_shard,
        max_plies_per_shard=args.max_plies_per_shard,
        log_every_plies=args.log_every_plies,
        min_elo=args.min_elo,
        max_elo=args.max_elo,
        skip_first_n_plies=args.skip_first_n_plies,
        num_workers=0,
        batch_size=args.batch_size,
        prefetch_factor=args.prefetch_factor,
    )

    for path in shards:
        print(path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
