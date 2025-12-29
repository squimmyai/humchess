#!/usr/bin/env python3
"""
Download and extract Lichess monthly standard rated PGN dumps.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from urllib.request import urlopen

import zstandard as zstd
from tqdm import tqdm


BASE_URL = "https://database.lichess.org/standard"
FILE_TEMPLATE = "lichess_db_standard_rated_{year}-{month:02d}.pgn.zst"


def parse_ym(value: str) -> tuple[int, int]:
    """Parse YYYY-MM to (year, month)."""
    try:
        year_str, month_str = value.split("-")
        year = int(year_str)
        month = int(month_str)
        if not 1 <= month <= 12:
            raise ValueError
        return year, month
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid YYYY-MM: {value}") from exc


def iter_months(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    """Inclusive list of (year, month) pairs."""
    start_date = dt.date(start[0], start[1], 1)
    end_date = dt.date(end[0], end[1], 1)
    if end_date < start_date:
        raise ValueError("End date must be >= start date.")

    months = []
    cur = start_date
    while cur <= end_date:
        months.append((cur.year, cur.month))
        if cur.month == 12:
            cur = dt.date(cur.year + 1, 1, 1)
        else:
            cur = dt.date(cur.year, cur.month + 1, 1)
    return months


def download_file(url: str, dest: Path) -> None:
    """Download a file to dest path with progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        with (
            dest.open("wb") as f_out,
            tqdm(total=total_size, unit="B", unit_scale=True, desc=dest.name) as pbar,
        ):
            while chunk := response.read(1024 * 1024):
                f_out.write(chunk)
                pbar.update(len(chunk))


def extract_zst(src: Path, dest: Path) -> None:
    """Extract a .zst file to dest path with progress bar."""
    src_size = src.stat().st_size
    with (
        src.open("rb") as f_in,
        dest.open("wb") as f_out,
        tqdm(total=src_size, unit="B", unit_scale=True, desc=f"Extracting {dest.name}") as pbar,
    ):
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f_in) as reader:
            while chunk := reader.read(1024 * 1024):
                f_out.write(chunk)
                pbar.update(f_in.tell() - pbar.n)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and extract Lichess monthly standard rated PGN dumps.",
    )
    parser.add_argument("--start", required=True, type=parse_ym, help="Start month YYYY-MM")
    parser.add_argument("--end", required=True, type=parse_ym, help="End month YYYY-MM")
    parser.add_argument("--out-dir", default="data", help="Output directory")
    parser.add_argument("--keep-zst", action="store_true", help="Keep .zst files after extract")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    months = iter_months(args.start, args.end)

    for year, month in months:
        filename = FILE_TEMPLATE.format(year=year, month=month)
        url = f"{BASE_URL}/{filename}"
        zst_path = out_dir / filename
        pgn_path = out_dir / filename.replace(".pgn.zst", ".pgn")

        if pgn_path.exists():
            print(f"Skip {pgn_path} (already extracted)")
            continue

        if not zst_path.exists():
            print(f"Downloading {url}")
            download_file(url, zst_path)
        else:
            print(f"Using existing {zst_path}")

        extract_zst(zst_path, pgn_path)

        if not args.keep_zst:
            zst_path.unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
