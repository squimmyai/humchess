"""Gradient accumulation must equal a single large batch.

The whole point of grad_accum_steps is to produce the *same* gradient (and
therefore the same optimizer update) as one forward/backward over the full
effective batch, while only ever holding one micro-batch of activations. If
that equivalence breaks, the LR schedule and gradient clipping stop matching
the recipe they were tuned for.

These tests use tiny, fast CPU models and compare the accumulated gradient to
the gradient from a single concatenated batch.
"""

import torch
import torch.nn as nn

from humchess.model.transformer import ChessTransformer
from humchess.data.tokenization import SEQ_LENGTH, VOCAB_SIZE, NUM_MOVE_CLASSES


def _tiny_model():
    torch.manual_seed(0)
    return ChessTransformer(
        d_model=64, n_heads=2, n_layers=2, d_ff=128, d_head=64,
        use_adaln=True, d_cond=32, ffn_type='swiglu',
    )


def _make_batch(n, seed):
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, VOCAB_SIZE, (n, SEQ_LENGTH), generator=g)
    move_id = torch.randint(0, NUM_MOVE_CLASSES, (n,), generator=g)
    legal = torch.ones(n, NUM_MOVE_CLASSES, dtype=torch.bool)
    # Randomly mask a few moves so the masked CE is exercised.
    rows = torch.arange(n).unsqueeze(1).expand(n, 100)
    legal[rows, torch.randint(0, NUM_MOVE_CLASSES, (n, 100), generator=g)] = False
    legal[torch.arange(n), move_id] = True
    promo_id = torch.randint(0, 4, (n,), generator=g)
    is_promo = torch.zeros(n, dtype=torch.bool)
    return tokens, move_id, legal, promo_id, is_promo


def _loss(model, b):
    tokens, move_id, legal, promo_id, is_promo = b
    out = model(tokens)
    logits = out['move_logits'].masked_fill(~legal, float('-inf'))
    move_loss = nn.functional.cross_entropy(logits, move_id)
    promo_loss = nn.functional.cross_entropy(out['promo_logits'], promo_id)
    return move_loss + promo_loss


def _grads(model, loss_fn):
    model.zero_grad()
    loss_fn().backward()
    return {n: p.grad.detach().clone() for n, p in model.named_parameters()}


def test_grad_accum_matches_single_batch():
    """Mean gradient over micro-batches == gradient over the concatenated batch."""
    n, accum = 6, 3
    micro = n // accum
    # One batch of the full effective size, sliced into contiguous micro-batches
    # so both paths see exactly the same samples.
    full = _make_batch(n, seed=7)
    batches = [tuple(t[i * micro:(i + 1) * micro] for t in full) for i in range(accum)]

    # Reference: one forward/backward over the full effective batch.
    model_ref = _tiny_model()
    ref = _grads(model_ref, lambda: _loss(model_ref, full))

    # Accumulated: each micro loss divided by accum, summed.
    model_acc = _tiny_model()
    model_acc.zero_grad()
    for b in batches:
        (_loss(model_acc, b) / accum).backward()
    acc = {n: p.grad.detach().clone() for n, p in model_acc.named_parameters()}

    for name in ref:
        assert torch.allclose(ref[name], acc[name], atol=1e-6, rtol=1e-4), (
            f"gradient mismatch for {name}: max abs diff "
            f"{(ref[name] - acc[name]).abs().max().item()}"
        )


def test_grad_accum_default_equivalence():
    """accum=1 must be identical to a plain single step (no regression)."""
    n, accum = 6, 1
    batch = _make_batch(n, seed=3)

    model_ref = _tiny_model()
    ref = _grads(model_ref, lambda: _loss(model_ref, batch))

    model_acc = _tiny_model()
    model_acc.zero_grad()
    _loss(model_acc, batch).backward()
    acc = {n: p.grad.detach().clone() for n, p in model_acc.named_parameters()}

    for name in ref:
        assert torch.allclose(ref[name], acc[name], atol=1e-7, rtol=1e-5), name
