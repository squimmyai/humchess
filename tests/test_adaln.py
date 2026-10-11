"""Tests for AdaLN-Zero Elo/time conditioning (1d)."""

import torch
import torch.nn as nn

from humchess.model.transformer import (
    AdaLNBlock,
    ChessTransformer,
)
from humchess.data.tokenization import VOCAB_SIZE, SEQ_LENGTH, NUM_MOVE_CLASSES


def test_adaln_block_zero_init_is_identity():
    block = AdaLNBlock(d_model=32, n_heads=4, d_ff=64, d_cond=16)
    x = torch.randn(2, 74, 32)
    c = torch.randn(2, 16)
    out = block(x, c)
    # With zero-initialized modulation, gates are 0 so the block is identity.
    assert torch.allclose(out, x)


def test_adaln_block_shape():
    block = AdaLNBlock(d_model=32, n_heads=4, d_ff=64, d_cond=16)
    out = block(torch.randn(2, 74, 32), torch.randn(2, 16))
    assert out.shape == (2, 74, 32)


def test_conditioning_has_no_effect_at_init():
    model = ChessTransformer(d_model=32, n_heads=4, n_layers=2, d_ff=64, use_adaln=True)
    model.eval()
    t = torch.randint(0, VOCAB_SIZE, (2, SEQ_LENGTH))
    t2 = t.clone()
    t2[:, 66] = (t2[:, 66] + 5) % VOCAB_SIZE  # elo token
    with torch.no_grad():
        a = model(t)["move_logits"]
        b = model(t2)["move_logits"]
    # AdaLN-Zero => identity blocks at init => output independent of conditioning.
    assert torch.allclose(a, b)


def test_conditioning_learns_to_matter():
    torch.manual_seed(0)
    model = ChessTransformer(d_model=32, n_heads=4, n_layers=2, d_ff=64, use_adaln=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(20):
        t = torch.randint(0, VOCAB_SIZE, (8, SEQ_LENGTH))
        target = torch.randint(0, NUM_MOVE_CLASSES, (8,))
        mask = torch.zeros(8, NUM_MOVE_CLASSES, dtype=torch.bool)
        mask[:, :20] = True
        mask[torch.arange(8), target] = True
        out = model(t)
        loss = nn.functional.cross_entropy(
            out["move_logits"].masked_fill(~mask, float("-inf")), target
        )
        opt.zero_grad(); loss.backward(); opt.step()

    # Modulation should have moved off zero.
    assert any(b.mod.weight.abs().sum().item() > 0 for b in model.blocks)

    model.eval()
    t = torch.randint(0, VOCAB_SIZE, (2, SEQ_LENGTH))
    t2 = t.clone(); t2[:, 66] = (t2[:, 66] + 5) % VOCAB_SIZE
    with torch.no_grad():
        diff = (model(t)["move_logits"] - model(t2)["move_logits"]).abs().max()
    assert diff > 0


def test_adaln_can_be_disabled():
    model = ChessTransformer(d_model=32, n_heads=4, n_layers=2, d_ff=64, use_adaln=False)
    out = model(torch.randint(0, VOCAB_SIZE, (2, SEQ_LENGTH)))
    assert out["move_logits"].shape == (2, NUM_MOVE_CLASSES)


def test_adaln_parameter_overhead_small():
    base = ChessTransformer(d_model=128, n_heads=4, n_layers=4, d_ff=256, use_adaln=False)
    ada = ChessTransformer(d_model=128, n_heads=4, n_layers=4, d_ff=256, use_adaln=True, d_cond=64)
    overhead = (ada.count_parameters() - base.count_parameters()) / base.count_parameters()
    assert overhead < 0.25