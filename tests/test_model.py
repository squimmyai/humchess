"""Tests for transformer model."""

import pytest
import torch

from humchess.model.transformer import (
    RMSNorm,
    SelfAttention,
    FeedForward,
    TransformerBlock,
    ChessTransformer,
)
from humchess.data.tokenization import (
    VOCAB_SIZE,
    SEQ_LENGTH,
    NUM_MOVE_CLASSES,
    NUM_PROMO_CLASSES,
)


def create_model(size: str = 'small', **kwargs) -> ChessTransformer:
    """Create model with preset sizes: tiny, small, medium, large."""
    presets = {
        'tiny': {'d_model': 128, 'n_heads': 4, 'n_layers': 4, 'd_ff': 512},
        'small': {'d_model': 256, 'n_heads': 8, 'n_layers': 6, 'd_ff': 1024},
        'medium': {'d_model': 512, 'n_heads': 8, 'n_layers': 8, 'd_ff': 2048},
        'large': {'d_model': 768, 'n_heads': 12, 'n_layers': 12, 'd_ff': 3072},
    }
    if size not in presets:
        raise ValueError(f"Unknown size '{size}'. Choose from: {list(presets.keys())}")
    return ChessTransformer(**{**presets[size], **kwargs})


class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(256)
        x = torch.randn(2, 10, 256)
        out = norm(x)

        assert out.shape == x.shape

    def test_normalized_scale(self):
        norm = RMSNorm(256)
        x = torch.randn(2, 10, 256) * 100  # large values

        out = norm(x)
        rms = out.pow(2).mean(-1).sqrt()

        # After normalization, RMS should be close to 1
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)


class TestSelfAttention:
    def test_output_shape(self):
        attn = SelfAttention(d_model=256, n_heads=8)
        x = torch.randn(2, 68, 256)
        out = attn(x)

        assert out.shape == x.shape

    def test_no_bias(self):
        attn = SelfAttention(d_model=256, n_heads=8)

        assert attn.qkv.bias is None
        assert attn.out.bias is None


class TestFeedForward:
    def test_output_shape(self):
        ffn = FeedForward(d_model=256, d_ff=1024)
        x = torch.randn(2, 68, 256)
        out = ffn(x)

        assert out.shape == x.shape

    def test_no_bias(self):
        ffn = FeedForward(d_model=256, d_ff=1024)

        assert ffn.w1.bias is None
        assert ffn.w2.bias is None


class TestTransformerBlock:
    def test_output_shape(self):
        block = TransformerBlock(d_model=256, n_heads=8, d_ff=1024)
        x = torch.randn(2, 68, 256)
        out = block(x)

        assert out.shape == x.shape

    def test_residual_connection(self):
        block = TransformerBlock(d_model=256, n_heads=8, d_ff=1024)
        x = torch.randn(2, 68, 256)

        # With residual connections, output should not be zero even if
        # submodules output zero (which they won't, but conceptually)
        out = block(x)
        assert not torch.allclose(out, torch.zeros_like(out))


class TestChessTransformer:
    def test_output_shapes(self):
        model = ChessTransformer()
        tokens = torch.randint(0, VOCAB_SIZE, (2, SEQ_LENGTH))
        outputs = model(tokens)

        assert outputs['move_logits'].shape == (2, NUM_MOVE_CLASSES)
        assert outputs['promo_logits'].shape == (2, NUM_PROMO_CLASSES)

    def test_output_dtypes(self):
        model = ChessTransformer()
        tokens = torch.randint(0, VOCAB_SIZE, (2, SEQ_LENGTH))
        outputs = model(tokens)

        assert outputs['move_logits'].dtype == torch.float32
        assert outputs['promo_logits'].dtype == torch.float32

    def test_gradients_flow(self):
        model = ChessTransformer()
        tokens = torch.randint(0, VOCAB_SIZE, (2, SEQ_LENGTH))
        outputs = model(tokens)

        loss = outputs['move_logits'].sum() + outputs['promo_logits'].sum()
        loss.backward()

        # Check that gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_custom_config(self):
        model = ChessTransformer(
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=512,
        )
        tokens = torch.randint(0, VOCAB_SIZE, (1, SEQ_LENGTH))
        outputs = model(tokens)

        assert outputs['move_logits'].shape == (1, NUM_MOVE_CLASSES)

    def test_parameter_count(self):
        model = ChessTransformer()
        count = model.count_parameters()

        assert count > 0
        assert isinstance(count, int)


class TestCreateModel:
    def test_tiny(self):
        model = create_model('tiny')
        assert model.d_model == 128

    def test_small(self):
        model = create_model('small')
        assert model.d_model == 256

    def test_medium(self):
        model = create_model('medium')
        assert model.d_model == 512

    def test_large(self):
        model = create_model('large')
        assert model.d_model == 768

    def test_invalid_size(self):
        with pytest.raises(ValueError):
            create_model('xlarge')

    def test_override_params(self):
        model = create_model('small', n_layers=2)
        # Count layers
        n_layers = len(model.blocks)
        assert n_layers == 2


class TestModelDevice:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda(self):
        model = create_model('tiny').cuda()
        tokens = torch.randint(0, VOCAB_SIZE, (1, SEQ_LENGTH)).cuda()
        outputs = model(tokens)

        assert outputs['move_logits'].device.type == 'cuda'
