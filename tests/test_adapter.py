"""Tests for BottleneckAdapter dimensions and config injection."""

from __future__ import annotations

import pytest
import torch

from src.intervention.adapter import BottleneckAdapter


@pytest.mark.parametrize(
    ("dim", "r", "activation"),
    [
        (1024, 16, "gelu"),
        (768, 32, "gelu"),
        (2048, 64, "relu"),
        (1536, 128, "silu"),
    ],
)
def test_adapter_output_shape(dim: int, r: int, activation: str) -> None:
    adapter = BottleneckAdapter(dim=dim, r=r, activation=activation)
    z = torch.randn(4, dim)
    out = adapter(z)
    assert out.shape == (4, dim)


def test_adapter_identity_at_init() -> None:
    adapter = BottleneckAdapter(dim=512, r=16, activation="gelu")
    z = torch.randn(8, 512)
    delta = adapter.delta(z)
    torch.testing.assert_close(delta, torch.zeros_like(delta), atol=1e-6, rtol=0)


def test_adapter_requires_dim() -> None:
    with pytest.raises(ValueError, match="dim is required"):
        BottleneckAdapter(dim=None, r=16, activation="gelu")  # type: ignore[arg-type]
