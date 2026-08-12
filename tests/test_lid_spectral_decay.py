"""Tests for LID / spectral-decay geometry diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from src.geometry.lid_spectral_decay import spectral_decay_slope, within_class_scatter


def test_spectral_decay_slope_rank_deficient_sw() -> None:
    """High-d embeddings can yield zero trailing eigenvalues in S_w."""
    d = 32
    s_w = np.diag(np.concatenate([np.linspace(10.0, 1.0, 8), np.zeros(d - 8)]))
    slope = spectral_decay_slope(s_w)
    assert slope > 0


def test_spectral_decay_slope_requires_two_positive() -> None:
    s_w = np.diag([1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="fewer than two positive"):
        spectral_decay_slope(s_w)


def test_within_class_scatter_shape() -> None:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(80, 16))
    labels = np.repeat(np.arange(8), 10)
    s_w = within_class_scatter(features, labels, num_classes=8)
    assert s_w.shape == (16, 16)
    assert np.allclose(s_w, s_w.T)
