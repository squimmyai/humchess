"""
Tokenization for HumChess.

Handles:
- Token vocabulary definitions
- FEN to token sequence conversion
- White normalization (plan.md §4)
- Move encoding/decoding (plan.md §5)

Sequence layout: [CLS, SQ_0..SQ_63, CASTLING, ELO_BUCKET, TL_BUCKET]
"""

from enum import IntEnum
from typing import Optional

import chess


# =============================================================================
# Board Piece Tokens (13 tokens)
# =============================================================================

class Piece(IntEnum):
    """Piece tokens for board squares. Order: empty, white pieces, black pieces."""
    EMPTY = 0
    WP = 1  # White Pawn
    WN = 2  # White Knight
    WB = 3  # White Bishop
    WR = 4  # White Rook
    WQ = 5  # White Queen
    WK = 6  # White King
    BP = 7  # Black Pawn
    BN = 8  # Black Knight
    BB = 9  # Black Bishop
    BR = 10 # Black Rook
    BQ = 11 # Black Queen
    BK = 12 # Black King


PIECE_CHAR_TO_TOKEN = {
    'P': Piece.WP, 'N': Piece.WN, 'B': Piece.WB,
    'R': Piece.WR, 'Q': Piece.WQ, 'K': Piece.WK,
    'p': Piece.BP, 'n': Piece.BN, 'b': Piece.BB,
    'r': Piece.BR, 'q': Piece.BQ, 'k': Piece.BK,
}

PIECE_TYPE_COLOR_TO_TOKEN = {
    (chess.PAWN, chess.WHITE): Piece.WP,
    (chess.KNIGHT, chess.WHITE): Piece.WN,
    (chess.BISHOP, chess.WHITE): Piece.WB,
    (chess.ROOK, chess.WHITE): Piece.WR,
    (chess.QUEEN, chess.WHITE): Piece.WQ,
    (chess.KING, chess.WHITE): Piece.WK,
    (chess.PAWN, chess.BLACK): Piece.BP,
    (chess.KNIGHT, chess.BLACK): Piece.BN,
    (chess.BISHOP, chess.BLACK): Piece.BB,
    (chess.ROOK, chess.BLACK): Piece.BR,
    (chess.QUEEN, chess.BLACK): Piece.BQ,
    (chess.KING, chess.BLACK): Piece.BK,
}


# =============================================================================
# Special Tokens
# =============================================================================

class Special(IntEnum):
    """Special tokens."""
    CLS = 13


# =============================================================================
# Castling Rights Tokens (16 tokens)
# =============================================================================

CASTLING_BASE = 14
NUM_CASTLING_TOKENS = 16


def castling_token(wk: bool, wq: bool, bk: bool, bq: bool) -> int:
    """Get castling rights token from individual flags. Bit order: [WK, WQ, BK, BQ]."""
    bits = (int(wk) << 3) | (int(wq) << 2) | (int(bk) << 1) | int(bq)
    return CASTLING_BASE + bits


def parse_castling_token(token: int) -> tuple[bool, bool, bool, bool]:
    """Parse castling token back to (WK, WQ, BK, BQ) flags."""
    bits = token - CASTLING_BASE
    return bool(bits & 8), bool(bits & 4), bool(bits & 2), bool(bits & 1)


# =============================================================================
# Elo Bucket Tokens (17 tokens)
# =============================================================================

ELO_BUCKET_BASE = CASTLING_BASE + NUM_CASTLING_TOKENS  # 30
ELO_BUCKET_BOUNDARIES = [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700,
                         1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500]
NUM_ELO_BUCKETS = len(ELO_BUCKET_BOUNDARIES) + 1  # 17


def elo_bucket_token(elo: int) -> int:
    """Get Elo bucket token. Buckets: <1000, 1000-1100, ..., >=2500."""
    for i, boundary in enumerate(ELO_BUCKET_BOUNDARIES):
        if elo < boundary:
            return ELO_BUCKET_BASE + i
    return ELO_BUCKET_BASE + len(ELO_BUCKET_BOUNDARIES)


# =============================================================================
# Time-Left Bucket Tokens (19 tokens)
# =============================================================================

TL_BUCKET_BASE = ELO_BUCKET_BASE + NUM_ELO_BUCKETS  # 47
TL_BUCKET_BOUNDARIES_SECONDS = [10, 30, 60] + [60 * i for i in range(2, 16)]
NUM_TL_BUCKETS = len(TL_BUCKET_BOUNDARIES_SECONDS) + 2  # 19
TL_UNKNOWN_IDX = len(TL_BUCKET_BOUNDARIES_SECONDS) + 1  # 18


def tl_bucket_token(seconds: Optional[float]) -> int:
    """Get time-left bucket token. Buckets: <10s, 10-30s, 30s-1m, 1-2m, ..., >=15m, UNKNOWN."""
    if seconds is None:
        return TL_BUCKET_BASE + TL_UNKNOWN_IDX
    for i, boundary in enumerate(TL_BUCKET_BOUNDARIES_SECONDS):
        if seconds < boundary:
            return TL_BUCKET_BASE + i
    return TL_BUCKET_BASE + len(TL_BUCKET_BOUNDARIES_SECONDS)


# =============================================================================
# Vocabulary Summary
# =============================================================================

VOCAB_SIZE = TL_BUCKET_BASE + NUM_TL_BUCKETS  # 66
SEQ_LENGTH = 1 + 64 + 1 + 1 + 1 + 6  # CLS + squares + castling + elo + tl + history = 74

# =============================================================================
# Move History
# =============================================================================

NUM_HISTORY_PLIES = 6  # 3 full moves (6 half-moves)
NO_MOVE_ID = 4096  # Padding token for positions with insufficient history


# =============================================================================
# Move Encoding
# =============================================================================

NUM_MOVE_CLASSES = 64 * 64  # 4096
NUM_PROMO_CLASSES = 4


class Promotion(IntEnum):
    """Promotion piece IDs."""
    QUEEN = 0
    ROOK = 1
    BISHOP = 2
    KNIGHT = 3


PROMO_CHAR_TO_ID = {'q': Promotion.QUEEN, 'r': Promotion.ROOK,
                    'b': Promotion.BISHOP, 'n': Promotion.KNIGHT}
PROMO_ID_TO_CHAR = {v: k for k, v in PROMO_CHAR_TO_ID.items()}


# =============================================================================
# Square Indexing
# =============================================================================

def square_name_to_idx(name: str) -> int:
    """Convert square name (e.g., 'e4') to index (0-63). A1=0, H8=63."""
    return (int(name[1]) - 1) * 8 + (ord(name[0]) - ord('a'))


def idx_to_square_name(idx: int) -> str:
    """Convert index (0-63) to square name (e.g., 'e4')."""
    return chr(ord('a') + idx % 8) + str(idx // 8 + 1)


# =============================================================================
# FEN Parsing
# =============================================================================

# =============================================================================
# Position to Tokens
# =============================================================================

def fen_to_tokens(
    fen: str,
    elo: int,
    time_left_seconds: Optional[float] = None,
) -> tuple[list[int], bool]:
    """
    Convert FEN + metadata to token sequence.

    Returns:
        (tokens, is_black_to_move) where tokens is [CLS, SQ_0..SQ_63, CASTLING, ELO, TL]
    """
    board = chess.Board(fen)
    return board_to_tokens(board, elo, time_left_seconds)


def board_to_tokens(
    board: chess.Board,
    elo: int,
    time_left_seconds: Optional[float] = None,
) -> tuple[list[int], bool]:
    """
    Convert Board + metadata to token sequence without FEN.

    Returns:
        (tokens, is_black_to_move) where tokens is [CLS, SQ_0..SQ_63, CASTLING, ELO, TL]
    """
    tokens = [Special.CLS]
    squares = [Piece.EMPTY] * 64
    for sq, piece in board.piece_map().items():
        squares[sq] = PIECE_TYPE_COLOR_TO_TOKEN[(piece.piece_type, piece.color)]
    tokens.extend(squares)
    wk = board.has_kingside_castling_rights(chess.WHITE)
    wq = board.has_queenside_castling_rights(chess.WHITE)
    bk = board.has_kingside_castling_rights(chess.BLACK)
    bq = board.has_queenside_castling_rights(chess.BLACK)
    tokens.append(castling_token(wk, wq, bk, bq))
    tokens.append(elo_bucket_token(elo))
    tokens.append(tl_bucket_token(time_left_seconds))
    return tokens, board.turn == chess.BLACK


# =============================================================================
# White Normalization
# =============================================================================

_COLOR_SWAP = {
    Piece.EMPTY: Piece.EMPTY,
    Piece.WP: Piece.BP, Piece.WN: Piece.BN, Piece.WB: Piece.BB,
    Piece.WR: Piece.BR, Piece.WQ: Piece.BQ, Piece.WK: Piece.BK,
    Piece.BP: Piece.WP, Piece.BN: Piece.WN, Piece.BB: Piece.WB,
    Piece.BR: Piece.WR, Piece.BQ: Piece.WQ, Piece.BK: Piece.WK,
}


def normalize_position(
    tokens: list[int],
    move_uci: str,
    is_black_to_move: bool,
) -> tuple[list[int], str]:
    """
    Apply white normalization if black to move.
    Rotates board 180°, swaps colors, transforms castling and move.
    """
    if not is_black_to_move:
        return tokens, move_uci

    new_tokens = tokens.copy()

    # Rotate board 180° and swap colors
    board = tokens[1:65]
    new_tokens[1:65] = [_COLOR_SWAP[board[63 - sq]] for sq in range(64)]

    # Swap castling rights (white <-> black)
    wk, wq, bk, bq = parse_castling_token(tokens[65])
    new_tokens[65] = castling_token(bk, bq, wk, wq)

    # Transform move
    return new_tokens, _normalize_move(move_uci)


def _normalize_move(move_uci: str) -> str:
    """Transform UCI move for 180° rotation."""
    from_sq = 63 - square_name_to_idx(move_uci[0:2])
    to_sq = 63 - square_name_to_idx(move_uci[2:4])
    result = idx_to_square_name(from_sq) + idx_to_square_name(to_sq)
    if len(move_uci) > 4:
        result += move_uci[4]
    return result


def denormalize_move(move_uci: str, was_black_to_move: bool) -> str:
    """Reverse move normalization. (180° rotation is self-inverse.)"""
    return _normalize_move(move_uci) if was_black_to_move else move_uci


def normalize_move_id(move_id: int, is_black_to_move: bool) -> int:
    """Apply 180° rotation to move ID if black to move."""
    # If white to move, no normalization. All history stays as is.
    if not is_black_to_move or move_id == NO_MOVE_ID:
        return move_id

    # If black to move, normalise everything so the previous move appears as black
    # (as the black to move position is rotated).
    from_sq = move_id // 64
    to_sq = move_id % 64
    return (63 - from_sq) * 64 + (63 - to_sq)


# =============================================================================
# Move ID Conversion
# =============================================================================

def move_to_ids(move_uci: str) -> tuple[int, Optional[int]]:
    """Convert UCI move to (move_id, promo_id). move_id = from*64 + to."""
    from_sq = square_name_to_idx(move_uci[0:2])
    to_sq = square_name_to_idx(move_uci[2:4])
    promo_id = PROMO_CHAR_TO_ID.get(move_uci[4].lower()) if len(move_uci) > 4 else None
    return from_sq * 64 + to_sq, promo_id


def ids_to_move(move_id: int, promo_id: Optional[int] = None) -> str:
    """Convert (move_id, promo_id) to UCI move string."""
    result = idx_to_square_name(move_id // 64) + idx_to_square_name(move_id % 64)
    if promo_id is not None:
        result += PROMO_ID_TO_CHAR[promo_id]
    return result


def is_promotion_move(move_id: int, board_tokens: list[int]) -> bool:
    """Check if move is a promotion (pawn to rank 8 after normalization)."""
    from_sq, to_sq = move_id // 64, move_id % 64
    return board_tokens[from_sq] == Piece.WP and to_sq >= 56


# =============================================================================
# Legality Mask
# =============================================================================

def get_legal_move_ids(board: chess.Board) -> list[int]:
    """Get legal move IDs for a position (normalized, white to move)."""
    return list({m.from_square * 64 + m.to_square for m in board.legal_moves})
