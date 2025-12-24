"""
Streaming PGN dataset for training.

Implements an IterableDataset that streams positions from PGN files,
applying white normalization and computing legality masks on-the-fly.
"""

import re
from pathlib import Path
from typing import Iterator, Optional

import chess
import chess.pgn
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .tokenization import (
    SEQ_LENGTH, NUM_MOVE_CLASSES,
    board_to_tokens, normalize_position, move_to_ids, is_promotion_move,
)


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
        pgn_paths: list[str | Path],
        min_elo: int = 0,
        max_elo: int = 4000,
        skip_first_n_plies: int = 0,
        include_metadata: bool = False,
    ):
        """
        Args:
            pgn_paths: List of paths to PGN files.
            min_elo: Minimum Elo to include (filters both players).
            max_elo: Maximum Elo to include.
            skip_first_n_plies: Skip opening moves (e.g., 10 to skip first 5 moves each).
        """
        self.pgn_paths = [Path(p) for p in pgn_paths]
        self.min_elo = min_elo
        self.max_elo = max_elo
        self.skip_first_n_plies = skip_first_n_plies
        self.include_metadata = include_metadata
        self._legal_mask_cache: dict[tuple[int, ...], torch.Tensor] = {}

    def __iter__(self) -> Iterator[dict]:
        worker_info = get_worker_info()

        # Shard files across workers
        if worker_info is None:
            # Single-process loading
            paths = self.pgn_paths
        else:
            # Multi-process: each worker gets a subset of files
            paths = [
                p for i, p in enumerate(self.pgn_paths)
                if i % worker_info.num_workers == worker_info.id
            ]

        for path in paths:
            yield from self._iter_pgn(path)

    def _iter_pgn(self, path: Path) -> Iterator[dict]:
        """Iterate over all positions in a single PGN file."""
        with open(path, 'r', errors='replace') as f:
            game_idx = 0
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break

                yield from self._iter_game(game, path, game_idx)
                game_idx += 1

    def _iter_game(
        self,
        game: chess.pgn.Game,
        path: Path,
        game_idx: int,
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

        ply = 0
        for node in game.mainline():
            move = node.move

            # Skip opening plies
            if ply < self.skip_first_n_plies:
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

            # Convert move to IDs
            move_id, promo_id = move_to_ids(move_uci)

            # Create legality mask (using normalized position)
            # We need to recreate the board for the normalized position
            # For simplicity, we use the original board's legal moves and transform them
            cache_key = tuple(tokens[1:66])
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
                        }
                    } if self.include_metadata else {}
                ),
            }

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


def collate_fn(batch: list[dict]) -> dict:
    """
    Collate function for DataLoader.

    Returns batched tensors.
    """
    return {
        'tokens': torch.stack([item['tokens'] for item in batch]),
        'move_id': torch.tensor([item['move_id'] for item in batch], dtype=torch.long),
        'promo_id': torch.tensor([item['promo_id'] for item in batch], dtype=torch.long),
        'legal_mask': torch.stack([item['legal_mask'] for item in batch]),
        'is_promotion': torch.tensor([item['is_promotion'] for item in batch], dtype=torch.bool),
    }
