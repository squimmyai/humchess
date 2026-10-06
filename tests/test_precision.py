"""CPU tests for fp32 master weights + bf16 autocast."""

import torch

from humchess.data.tokenization import SEQ_LENGTH, NUM_MOVE_CLASSES
from humchess.model.transformer import ChessTransformer, RMSNorm
from humchess.train import train_epoch


def _make_batch(batch_size: int = 4) -> dict:
    tokens = torch.randint(0, 66, (batch_size, SEQ_LENGTH), dtype=torch.long)
    move_id = torch.randint(0, NUM_MOVE_CLASSES, (batch_size,), dtype=torch.long)
    legal_mask = torch.zeros(batch_size, NUM_MOVE_CLASSES, dtype=torch.bool)
    legal_mask[torch.arange(batch_size), move_id] = True
    legal_mask[:, :10] = True
    return {
        "tokens": tokens,
        "move_id": move_id,
        "promo_id": torch.full((batch_size,), -1, dtype=torch.long),
        "legal_mask": legal_mask,
        "is_promotion": torch.zeros(batch_size, dtype=torch.bool),
        "shard_path": "s.parquet",
        "shard_complete": True,
    }


def test_rmsnorm_preserves_input_dtype():
    norm = RMSNorm(64)
    for dtype in (torch.float32, torch.bfloat16):
        out = norm(torch.randn(2, 5, 64, dtype=dtype))
        assert out.dtype == dtype


def test_rmsnorm_computes_in_fp32_under_autocast():
    # bf16 input through autocast must not raise the dtype-mismatch fallback,
    # and output must be finite.
    norm = RMSNorm(64)
    x = torch.randn(2, 5, 64).to(torch.bfloat16)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = norm(x)
    assert torch.isfinite(out.float()).all()


def test_params_and_optimizer_state_stay_fp32():
    model = ChessTransformer(d_model=32, n_heads=4, n_layers=1, d_ff=64)
    assert all(p.dtype == torch.float32 for p in model.parameters())

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_epoch(
        model=model, dataloader=[_make_batch()], optimizer=opt,
        device=torch.device("cpu"), precision="bf16", scaler=None,
    )
    # Params unchanged dtype
    assert all(p.dtype == torch.float32 for p in model.parameters())
    # Optimizer moments fp32
    for state in opt.state.values():
        for t in state.values():
            if isinstance(t, torch.Tensor):
                assert t.dtype == torch.float32


def test_bf16_autocast_forward_runs():
    model = ChessTransformer(d_model=32, n_heads=4, n_layers=1, d_ff=64)
    tokens = torch.randint(0, 66, (2, SEQ_LENGTH))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = model(tokens)
    assert torch.isfinite(out["move_logits"].float()).all()