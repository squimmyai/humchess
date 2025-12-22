"""
Inference module for move prediction.

Provides functions to predict moves given a position, Elo, and time.
"""

from typing import Optional

import chess
import torch
import torch.nn.functional as F

from .data.tokenization import (
    NUM_MOVE_CLASSES,
    fen_to_tokens, normalize_position, move_to_ids, ids_to_move,
    is_promotion_move, denormalize_move,
)
from .model.transformer import ChessTransformer


def get_legal_move_mask(board: chess.Board, was_black: bool) -> torch.Tensor:
    """
    Create legal move mask for a position.

    Args:
        board: Chess board (in original orientation)
        was_black: Whether original position was black to move

    Returns:
        Boolean tensor of shape (4096,) - True for legal moves
    """
    mask = torch.zeros(NUM_MOVE_CLASSES, dtype=torch.bool)

    for move in board.legal_moves:
        from_sq = move.from_square
        to_sq = move.to_square

        if was_black:
            from_sq = 63 - from_sq
            to_sq = 63 - to_sq

        move_id = from_sq * 64 + to_sq
        mask[move_id] = True

    return mask


@torch.no_grad()
def predict_move(
    model: ChessTransformer,
    fen: str,
    elo: int,
    time_left_seconds: Optional[float] = None,
    device: torch.device = None,
) -> str:
    """
    Predict a move for a given position.

    Args:
        model: Trained ChessTransformer model
        fen: FEN string of the position
        elo: Elo rating to emulate
        time_left_seconds: Time remaining on clock (optional)
        device: Device to run inference on

    Returns:
        UCI move string (e.g., 'e2e4', 'e7e8q')
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Parse position
    board = chess.Board(fen)

    # Convert to tokens (returns is_black flag)
    tokens, is_black = fen_to_tokens(fen, elo, time_left_seconds)

    # Create a dummy move for normalization (we'll use any legal move)
    dummy_move = list(board.legal_moves)[0].uci()
    tokens, _ = normalize_position(tokens, dummy_move, is_black)

    # Create input tensor
    tokens_tensor = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    # Get legal move mask
    legal_mask = get_legal_move_mask(board, is_black).to(device).unsqueeze(0)

    # Forward pass
    outputs = model(tokens_tensor)
    move_logits = outputs['move_logits']  # (1, 4096)
    promo_logits = outputs['promo_logits']  # (1, 4)

    # Apply legality mask
    masked_logits = move_logits.clone()
    masked_logits[~legal_mask] = float('-inf')

    # Sample from distribution
    probs = F.softmax(masked_logits, dim=-1)
    move_id = torch.multinomial(probs, num_samples=1).item()

    # Check if promotion
    board_tokens = tokens[1:65]
    is_promo = is_promotion_move(move_id, board_tokens)

    promo_id = None
    if is_promo:
        promo_probs = F.softmax(promo_logits, dim=-1)
        promo_id = torch.multinomial(promo_probs, num_samples=1).item()

    # Convert to UCI
    move_uci = ids_to_move(move_id, promo_id)

    # Denormalize if needed
    move_uci = denormalize_move(move_uci, is_black)

    return move_uci


@torch.no_grad()
def predict_move_distribution(
    model: ChessTransformer,
    fen: str,
    elo: int,
    time_left_seconds: Optional[float] = None,
    device: torch.device = None,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """
    Get top-k moves with their probabilities.

    Args:
        model: Trained ChessTransformer model
        fen: FEN string of the position
        elo: Elo rating to emulate
        time_left_seconds: Time remaining on clock (optional)
        device: Device to run inference on
        top_k: Number of top moves to return

    Returns:
        List of (uci_move, probability) tuples, sorted by probability descending
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Parse position
    board = chess.Board(fen)

    # Convert to tokens (returns is_black flag)
    tokens, is_black = fen_to_tokens(fen, elo, time_left_seconds)

    # Dummy move for normalization
    dummy_move = list(board.legal_moves)[0].uci()
    tokens, _ = normalize_position(tokens, dummy_move, is_black)

    # Create input tensor
    tokens_tensor = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    # Get legal move mask
    legal_mask = get_legal_move_mask(board, is_black).to(device).unsqueeze(0)

    # Forward pass
    outputs = model(tokens_tensor)
    move_logits = outputs['move_logits']  # (1, 4096)

    # Apply legality mask
    masked_logits = move_logits.clone()
    masked_logits[~legal_mask] = float('-inf')

    # Get probabilities
    probs = F.softmax(masked_logits, dim=-1).squeeze(0)  # (4096,)

    # Get top-k
    top_probs, top_ids = torch.topk(probs, k=min(top_k, legal_mask.sum().item()))

    # Convert to moves
    results = []
    for move_id, prob in zip(top_ids.tolist(), top_probs.tolist()):
        # For simplicity, assume queen promotion for display
        # (full version would show all promotion options for pawn moves)
        board_tokens = tokens[1:65]
        is_promo = is_promotion_move(move_id, board_tokens)
        promo_id = 0 if is_promo else None  # Queen promotion

        move_uci = ids_to_move(move_id, promo_id)
        move_uci = denormalize_move(move_uci, is_black)
        results.append((move_uci, prob))

    return results


def load_model(checkpoint_path: str, device: torch.device = None) -> ChessTransformer:
    """
    Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model to

    Returns:
        Loaded ChessTransformer model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Infer model size from state dict
    state_dict = checkpoint['model_state_dict']
    d_model = state_dict['token_embedding.weight'].shape[1]
    n_layers = sum(1 for k in state_dict if 'self_attn.in_proj_weight' in k)

    # Create model with matching architecture
    # This is a heuristic - ideally save config in checkpoint
    if d_model == 128:
        model = ChessTransformer(d_model=128, n_heads=4, n_layers=n_layers, d_ff=512)
    elif d_model == 256:
        model = ChessTransformer(d_model=256, n_heads=8, n_layers=n_layers, d_ff=1024)
    elif d_model == 512:
        model = ChessTransformer(d_model=512, n_heads=8, n_layers=n_layers, d_ff=2048)
    else:
        model = ChessTransformer(d_model=d_model, n_layers=n_layers)

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model
