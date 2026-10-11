"""Tests for head_dim alignment (2d)."""

import warnings

import torch

from humchess.model.transformer import SelfAttention, ChessTransformer


def test_aligned_head_dim_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        SelfAttention(d_model=384, n_heads=6)   # head_dim 64
        SelfAttention(d_model=512, n_heads=8)   # head_dim 64
        SelfAttention(d_model=256, n_heads=4)   # head_dim 64
        SelfAttention(d_model=1024, n_heads=8)  # head_dim 128


def test_unaligned_head_dim_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        SelfAttention(d_model=384, n_heads=8)  # head_dim 48
        assert any("head_dim" in str(x.message) for x in w)


def test_config_head_dim_is_aligned():
    # The shipped config must not trip the warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ChessTransformer(d_model=384, n_heads=6, n_layers=1, d_ff=1536)