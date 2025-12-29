"""Tests for tokenization module."""

import pytest

import chess

from humchess.data.tokenization import (
    VOCAB_SIZE,
    SEQ_LENGTH,
    NUM_MOVE_CLASSES,
    NUM_PROMO_CLASSES,
    NUM_HISTORY_PLIES,
    NO_MOVE_ID,
    Piece,
    Special,
    fen_to_tokens,
    board_to_tokens,
    normalize_position,
    normalize_move_id,
    move_to_ids,
    ids_to_move,
    is_promotion_move,
    denormalize_move,
    elo_bucket_token,
    tl_bucket_token,
    castling_token,
    parse_castling_token,
    square_name_to_idx,
    idx_to_square_name,
)


class TestConstants:
    def test_vocab_size(self):
        assert VOCAB_SIZE == 66

    def test_seq_length(self):
        # CLS + 64 squares + castling + elo + tl + 6 history moves = 74
        assert SEQ_LENGTH == 74

    def test_move_classes(self):
        assert NUM_MOVE_CLASSES == 4096  # 64 * 64

    def test_promo_classes(self):
        assert NUM_PROMO_CLASSES == 4  # q, r, b, n


class TestSquareIndexing:
    def test_a1_is_zero(self):
        assert square_name_to_idx("a1") == 0

    def test_h1_is_seven(self):
        assert square_name_to_idx("h1") == 7

    def test_a8_is_56(self):
        assert square_name_to_idx("a8") == 56

    def test_h8_is_63(self):
        assert square_name_to_idx("h8") == 63

    def test_e4(self):
        assert square_name_to_idx("e4") == 28

    def test_round_trip(self):
        for i in range(64):
            assert square_name_to_idx(idx_to_square_name(i)) == i


class TestCastlingTokens:
    def test_no_castling(self):
        token = castling_token(False, False, False, False)
        assert parse_castling_token(token) == (False, False, False, False)

    def test_all_castling(self):
        token = castling_token(True, True, True, True)
        assert parse_castling_token(token) == (True, True, True, True)

    def test_white_only(self):
        token = castling_token(True, True, False, False)
        assert parse_castling_token(token) == (True, True, False, False)

    def test_black_only(self):
        token = castling_token(False, False, True, True)
        assert parse_castling_token(token) == (False, False, True, True)


class TestEloBuckets:
    def test_below_1000(self):
        token = elo_bucket_token(800)
        assert token == 30  # ELO_BUCKET_BASE

    def test_1000(self):
        token_999 = elo_bucket_token(999)
        token_1000 = elo_bucket_token(1000)
        assert token_999 == 30
        assert token_1000 == 31

    def test_1500(self):
        token = elo_bucket_token(1500)
        assert token == 36

    def test_above_2500(self):
        token = elo_bucket_token(2700)
        assert token == 46


class TestTimeBuckets:
    def test_under_10s(self):
        token = tl_bucket_token(5)
        assert token == 47  # TL_BUCKET_BASE

    def test_10_to_30s(self):
        token = tl_bucket_token(20)
        assert token == 48

    def test_5_minutes(self):
        # 300s is at the boundary, goes to 5-6m bucket
        token = tl_bucket_token(300)
        assert token == 54

    def test_unknown(self):
        token = tl_bucket_token(None)
        assert token == 65


class TestFenToTokens:
    def test_starting_position_black(self):
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        tokens, is_black = fen_to_tokens(fen, elo=1500, time_left_seconds=300)

        assert is_black is True
        # fen_to_tokens returns base sequence (68 tokens), history added by dataset
        assert len(tokens) == 68  # CLS + 64 squares + castling + elo + tl


class TestBoardToTokens:
    def test_starting_position_white(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        board = chess.Board(fen)
        tokens, is_black = board_to_tokens(board, elo=1500, time_left_seconds=300)

        # board_to_tokens returns base sequence (68 tokens), history added by dataset
        assert len(tokens) == 68
        assert is_black is False
        assert tokens[0] == Special.CLS
        # A1 = white rook
        assert tokens[1] == Piece.WR
        # E1 = white king
        assert tokens[5] == Piece.WK
        # A8 = black rook
        assert tokens[57] == Piece.BR
        # E8 = black king
        assert tokens[61] == Piece.BK

    def test_empty_squares(self):
        fen = "8/8/8/8/8/8/8/8 w - - 0 1"
        board = chess.Board(fen)
        tokens, _ = board_to_tokens(board, elo=1500)

        for i in range(1, 65):
            assert tokens[i] == Piece.EMPTY

    def test_matches_fen(self):
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        board = chess.Board(fen)
        tokens_fen, is_black_fen = fen_to_tokens(fen, elo=1500, time_left_seconds=300)
        tokens_board, is_black_board = board_to_tokens(board, elo=1500, time_left_seconds=300)

        assert tokens_fen == tokens_board
        assert is_black_fen == is_black_board


class TestNormalization:
    def test_white_to_move_no_change(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        tokens, is_black = fen_to_tokens(fen, elo=1500)
        norm_tokens, norm_move = normalize_position(tokens, "e2e4", is_black)

        assert norm_tokens == tokens
        assert norm_move == "e2e4"

    def test_black_to_move_rotates(self):
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        tokens, is_black = fen_to_tokens(fen, elo=1500)
        norm_tokens, norm_move = normalize_position(tokens, "e7e5", is_black)

        # After normalization, black pieces become white (side to move)
        # A1 in normalized = H8 in original = black rook -> white rook
        assert norm_tokens[1] == Piece.WR
        # Move e7e5 becomes d2d4 after 180° rotation
        assert norm_move == "d2d4"

    def test_castling_swap(self):
        # Position with only black castling rights
        fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b kq - 0 1"
        tokens, is_black = fen_to_tokens(fen, elo=1500)
        norm_tokens, _ = normalize_position(tokens, "e8g8", is_black)

        # After normalization, black castling (kq) becomes white castling (KQ)
        wk, wq, bk, bq = parse_castling_token(norm_tokens[65])
        assert wk is True
        assert wq is True
        assert bk is False
        assert bq is False


class TestMoveEncoding:
    def test_e2e4(self):
        move_id, promo_id = move_to_ids("e2e4")
        # e2 = 12, e4 = 28
        assert move_id == 12 * 64 + 28
        assert promo_id is None

    def test_promotion_queen(self):
        move_id, promo_id = move_to_ids("e7e8q")
        assert promo_id == 0

    def test_promotion_knight(self):
        move_id, promo_id = move_to_ids("a7a8n")
        assert promo_id == 3

    def test_round_trip(self):
        for move in ["e2e4", "g1f3", "e7e8q", "a2a1r"]:
            move_id, promo_id = move_to_ids(move)
            result = ids_to_move(move_id, promo_id)
            assert result == move


class TestPromotionDetection:
    def test_pawn_to_8th_rank(self):
        # Board with white pawn on e7 (square 52)
        board_tokens = [Piece.EMPTY] * 64
        board_tokens[52] = Piece.WP  # e7

        # Move e7e8 (52*64 + 60)
        move_id = 52 * 64 + 60
        assert is_promotion_move(move_id, board_tokens) is True

    def test_pawn_not_to_8th(self):
        board_tokens = [Piece.EMPTY] * 64
        board_tokens[12] = Piece.WP  # e2

        move_id = 12 * 64 + 28  # e2e4
        assert is_promotion_move(move_id, board_tokens) is False

    def test_non_pawn_to_8th(self):
        board_tokens = [Piece.EMPTY] * 64
        board_tokens[52] = Piece.WQ  # queen on e7

        move_id = 52 * 64 + 60  # e7e8
        assert is_promotion_move(move_id, board_tokens) is False


class TestDenormalization:
    def test_white_no_change(self):
        assert denormalize_move("e2e4", was_black_to_move=False) == "e2e4"

    def test_black_rotates_back(self):
        # d2d4 normalized -> e7e5 original
        assert denormalize_move("d2d4", was_black_to_move=True) == "e7e5"

    def test_promotion_preserved(self):
        assert denormalize_move("d7d8q", was_black_to_move=True) == "e2e1q"


# =============================================================================
# Integration tests using real game data
# =============================================================================

import io
import chess
import chess.pgn

from tests.test_pgn_dataset import SAMPLE_PGN


def get_first_game():
    """Get the first game from SAMPLE_PGN."""
    return chess.pgn.read_game(io.StringIO(SAMPLE_PGN))


class TestRealGameTokenization:
    """Integration tests using positions from real games."""

    @pytest.fixture
    def game(self):
        return get_first_game()

    def test_starting_position_pieces(self, game):
        """Test that starting position tokenizes correctly."""
        board = game.board()
        fen = board.fen()

        tokens, is_black = fen_to_tokens(fen, elo=2000, time_left_seconds=180)

        assert is_black is False
        # Check white pieces on rank 1
        assert tokens[1 + 0] == Piece.WR  # a1
        assert tokens[1 + 1] == Piece.WN  # b1
        assert tokens[1 + 2] == Piece.WB  # c1
        assert tokens[1 + 3] == Piece.WQ  # d1
        assert tokens[1 + 4] == Piece.WK  # e1
        assert tokens[1 + 5] == Piece.WB  # f1
        assert tokens[1 + 6] == Piece.WN  # g1
        assert tokens[1 + 7] == Piece.WR  # h1

        # Check black pieces on rank 8 (a8=56, b8=57, ..., e8=60, ..., h8=63)
        assert tokens[1 + 56] == Piece.BR  # a8
        assert tokens[1 + 57] == Piece.BN  # b8
        assert tokens[1 + 60] == Piece.BK  # e8

    def test_position_after_moves(self, game):
        """Test tokenization after several moves."""
        board = game.board()

        # Play 1. e4 e5 2. f4 exf4
        moves = list(game.mainline_moves())[:4]
        for move in moves:
            board.push(move)

        fen = board.fen()
        tokens, is_black = fen_to_tokens(fen, elo=2000, time_left_seconds=170)

        # White to move after 2...exf4
        assert is_black is False

        # e4 pawn should be on e4 (square 28)
        assert tokens[1 + 28] == Piece.WP

        # f4 pawn (captured) - f4 should have black pawn
        assert tokens[1 + 29] == Piece.BP

        # e5 should be empty (pawn took on f4)
        assert tokens[1 + 36] == Piece.EMPTY

    def test_black_to_move_normalization_early(self, game):
        """Test normalization after move 1 (black to move)."""
        board = game.board()
        moves = list(game.mainline_moves())

        # Play 1. e4 (white moved, now black to move)
        board.push(moves[0])

        fen = board.fen()
        tokens, is_black = fen_to_tokens(fen, elo=1907, time_left_seconds=180)

        assert is_black is True

        # Black's reply from the game
        black_move = moves[1].uci()
        norm_tokens, norm_move = normalize_position(tokens, black_move, is_black)

        # After normalization, the side to move's pieces should be "white" pieces
        # Black king on e8 -> after rotation at d1, should be WK
        assert norm_tokens[1 + 3] == Piece.WK  # d1 = square 3

        # Verify roundtrip
        recovered = denormalize_move(norm_move, was_black_to_move=True)
        assert recovered == black_move

    def test_black_to_move_normalization_midgame(self, game):
        """Test normalization at ply 29 (after white's 15th move)."""
        board = game.board()
        moves = list(game.mainline_moves())

        # Play first 29 plies (white just played move 15)
        for move in moves[:29]:
            board.push(move)

        assert board.turn == chess.BLACK

        fen = board.fen()
        tokens, is_black = fen_to_tokens(fen, elo=1907, time_left_seconds=90)

        assert is_black is True

        # Black's 15th move from the game
        black_move = moves[29].uci()
        norm_tokens, norm_move = normalize_position(tokens, black_move, is_black)

        # After normalization, black pieces become white (side to move)
        # Find black king and verify it's now WK in normalized tokens
        black_king_sq = board.king(chess.BLACK)
        # 180° rotation: new_sq = 63 - old_sq
        rotated_sq = 63 - black_king_sq
        assert norm_tokens[1 + rotated_sq] == Piece.WK

        # Verify roundtrip
        recovered = denormalize_move(norm_move, was_black_to_move=True)
        assert recovered == black_move

    def test_move_encoding_from_game(self, game):
        """Test move encoding with actual game moves."""
        board = game.board()

        # e2e4 from starting position
        move = list(game.mainline_moves())[0]
        uci = move.uci()

        assert uci == "e2e4"
        move_id, promo_id = move_to_ids(uci)

        # e2 = square 12, e4 = square 28
        expected_id = 12 * 64 + 28
        assert move_id == expected_id
        assert promo_id is None

    def test_legal_moves_match_chess_library(self, game):
        """Test that our legal move IDs match python-chess at multiple positions."""
        from humchess.data.tokenization import get_legal_move_ids

        board = game.board()
        moves = list(game.mainline_moves())

        # Test at plies 0, 10, 20, 30 (various game stages)
        test_plies = [0, 10, 20, 30]

        for target_ply in test_plies:
            # Reset and play to target position
            board = game.board()
            for move in moves[:target_ply]:
                board.push(move)

            # Get legal moves from python-chess
            chess_legal = {m.from_square * 64 + m.to_square for m in board.legal_moves}

            # Get legal moves from our function
            our_legal = set(get_legal_move_ids(board))

            assert chess_legal == our_legal, f"Mismatch at ply {target_ply}"

    def test_full_game_all_positions_valid(self, game):
        """Test that all positions in a game tokenize without error."""
        board = game.board()
        white_elo = 2186
        black_elo = 1907

        for ply, move in enumerate(game.mainline_moves()):
            fen = board.fen()
            is_white_turn = board.turn == chess.WHITE
            elo = white_elo if is_white_turn else black_elo

            # Should not raise
            tokens, is_black = fen_to_tokens(fen, elo=elo, time_left_seconds=180)

            # fen_to_tokens returns 68 tokens (base sequence without history)
            assert len(tokens) == 68
            assert is_black == (not is_white_turn)

            # Normalize and encode the move
            uci = move.uci()
            norm_tokens, norm_move = normalize_position(tokens, uci, is_black)
            move_id, promo_id = move_to_ids(norm_move)

            assert 0 <= move_id < 4096

            board.push(move)

    def test_denormalize_roundtrip(self, game):
        """Test that normalize -> denormalize returns original move at multiple positions."""
        board = game.board()
        moves = list(game.mainline_moves())

        # Test roundtrip at various plies (both white and black to move)
        test_plies = [0, 1, 10, 15, 20, 29]

        for ply in test_plies:
            if ply >= len(moves):
                continue

            board = game.board()
            for move in moves[:ply]:
                board.push(move)

            fen = board.fen()
            is_black = board.turn == chess.BLACK
            tokens, _ = fen_to_tokens(fen, elo=1500)

            original_move = moves[ply].uci()
            _, norm_move = normalize_position(tokens, original_move, is_black)

            # Denormalize should get back original
            recovered = denormalize_move(norm_move, was_black_to_move=is_black)
            assert recovered == original_move, f"Roundtrip failed at ply {ply}"


# =============================================================================
# Move history tests
# =============================================================================


class TestMoveHistoryConstants:
    """Test move history-related constants."""

    def test_num_history_plies(self):
        assert NUM_HISTORY_PLIES == 6

    def test_no_move_id(self):
        # Should be 4096 (outside valid move range 0-4095)
        assert NO_MOVE_ID == 4096


class TestNormalizeMoveId:
    """Test move ID normalization for history."""

    def test_white_to_move_no_change(self):
        # e2e4 move: from=12, to=28 -> move_id = 12*64 + 28 = 796
        move_id = 12 * 64 + 28
        result = normalize_move_id(move_id, is_black_to_move=False)
        assert result == move_id

    def test_black_to_move_rotates_180(self):
        # e2e4 move: from=12, to=28
        # After 180° rotation: from=63-12=51, to=63-28=35 -> 51*64 + 35 = 3299
        move_id = 12 * 64 + 28
        result = normalize_move_id(move_id, is_black_to_move=True)
        expected = (63 - 12) * 64 + (63 - 28)
        assert result == expected

    def test_no_move_id_unchanged(self):
        # Padding token should not be modified
        result = normalize_move_id(NO_MOVE_ID, is_black_to_move=True)
        assert result == NO_MOVE_ID
        result = normalize_move_id(NO_MOVE_ID, is_black_to_move=False)
        assert result == NO_MOVE_ID

    def test_double_rotation_is_identity(self):
        # Rotating twice should give back the original
        move_id = 15 * 64 + 47  # arbitrary move
        rotated = normalize_move_id(move_id, is_black_to_move=True)
        # Rotating again as black should undo
        double_rotated = normalize_move_id(rotated, is_black_to_move=True)
        assert double_rotated == move_id

    def test_corner_to_corner_rotation(self):
        # a1a8 (0 -> 56): after rotation becomes h8h1 (63 -> 7) = 63*64 + 7 = 4039
        move_id = 0 * 64 + 56
        result = normalize_move_id(move_id, is_black_to_move=True)
        assert result == 63 * 64 + 7


class TestMoveHistoryIntegration:
    """Integration tests for move history in the dataset."""

    def test_history_length(self):
        """Verify the full sequence has 74 tokens (68 base + 6 history)."""
        assert SEQ_LENGTH == 74
        # Base sequence without history
        base_length = 68  # CLS + 64 squares + castling + elo + tl
        assert SEQ_LENGTH == base_length + NUM_HISTORY_PLIES
