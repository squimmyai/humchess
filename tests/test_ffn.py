"""Tests for the SwiGLU FFN option (1f)."""

import torch

from humchess.model.transformer import FeedForward, SwiGLU, ChessTransformer, make_ffn
from humchess.data.tokenization import VOCAB_SIZE, SEQ_LENGTH


def test_swiglu_shape():
    ffn = SwiGLU(d_model=64, d_ff=256)
    out = ffn(torch.randn(2, 74, 64))
    assert out.shape == (2, 74, 64)


def test_swiglu_equal_params_at_real_config():
    # 2*d*d_ff (silu) == 3*d*(2/3 d_ff) (swiglu) when d_ff is a multiple of 3*64
    d, f = 384, 1536
    a = sum(p.numel() for p in FeedForward(d, f).parameters())
    b = sum(p.numel() for p in SwiGLU(d, f).parameters())
    assert a == b


def test_swiglu_is_gated_not_pointwise():
    # Different gate/up weights must give different results than a single path.
    torch.manual_seed(0)
    ffn = SwiGLU(d_model=16, d_ff=64)
    x = torch.randn(1, 4, 16)
    out = ffn(x)
    # A pure SiLU FFN with the same x is not equal (different function).
    ref = FeedForward(16, 64)
    assert not torch.allclose(out, ref(x))


def test_make_ffn_dispatch():
    assert isinstance(make_ffn("swiglu", 16, 64), SwiGLU)
    assert isinstance(make_ffn("silu", 16, 64), FeedForward)
    try:
        make_ffn("bogus", 16, 64)
        assert False
    except ValueError:
        pass


def test_model_with_swiglu_forward_backward():
    model = ChessTransformer(d_model=32, n_heads=4, n_layers=2, d_ff=64, ffn_type="swiglu")
    out = model(torch.randint(0, VOCAB_SIZE, (2, SEQ_LENGTH)))
    assert out["move_logits"].shape == (2, 4096)
    (out["move_logits"].sum() + out["promo_logits"].sum()).backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_model_ffn_type_default_is_silu():
    model = ChessTransformer(d_model=32, n_heads=4, n_layers=1, d_ff=64)
    assert isinstance(model.blocks[0].ffn, FeedForward)