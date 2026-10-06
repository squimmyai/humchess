"""Tests for the geometry-aware square-pair move head (1b)."""

import torch

from humchess.model.transformer import SquarePairMoveHead, ChessTransformer
from humchess.data.tokenization import VOCAB_SIZE, SEQ_LENGTH, NUM_MOVE_CLASSES


def test_output_shape_matches_flat_head():
    head = SquarePairMoveHead(d_model=64, d_head=32)
    h = torch.randn(2, 64, 64)
    out = head(h)
    assert out.shape == (2, NUM_MOVE_CLASSES)


def test_indexing_is_from_times_64_plus_to():
    """move_logits[:, from*64+to] must equal q(from).k(to)."""
    d_model, d_head = 16, 8
    head = SquarePairMoveHead(d_model=d_model, d_head=d_head, use_rel_bias=False)
    h = torch.randn(1, 64, d_model)
    out = head(h)

    hh = head.norm(h)
    q = head.from_proj(hh)[0]  # (64, d_head)
    k = head.to_proj(hh)[0]
    for frm in (0, 5, 63):
        for to in (0, 17, 63):
            expected = (q[frm] @ k[to]) * head.scale
            assert torch.allclose(out[0, frm * 64 + to], expected, atol=1e-5)


def test_directed_not_symmetric():
    """e2->e4 and e4->e2 must be distinct (separate from/to projections)."""
    head = SquarePairMoveHead(d_model=16, d_head=8, use_rel_bias=False)
    h = torch.randn(1, 64, 16)
    out = head(h)
    a = out[0, 12 * 64 + 28]  # arbitrary from!=to
    b = out[0, 28 * 64 + 12]
    assert not torch.allclose(a, b)


def test_parameter_count_far_below_flat_head():
    head = SquarePairMoveHead(d_model=384, d_head=64)
    n = sum(p.numel() for p in head.parameters())
    flat = 384 * 4096 + 4096
    assert n < flat / 5


def test_rel_bias_shared_for_equal_displacement():
    head = SquarePairMoveHead(d_model=16, d_head=8, use_rel_bias=True)
    bias = head._rel_bias_matrix(torch.device("cpu"), torch.float32)  # (64, 64)
    # a1->b3 (dfile=1, drank=2) and d1->e3 share the same displacement
    assert bias.shape == (64, 64)
    assert torch.allclose(bias[0, 17], bias[3, 20])


def test_full_model_output_shape_and_grads():
    model = ChessTransformer(d_model=32, n_heads=4, n_layers=1, d_ff=64)
    tokens = torch.randint(0, VOCAB_SIZE, (2, SEQ_LENGTH))
    out = model(tokens)
    assert out["move_logits"].shape == (2, NUM_MOVE_CLASSES)
    (out["move_logits"].sum() + out["promo_logits"].sum()).backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)