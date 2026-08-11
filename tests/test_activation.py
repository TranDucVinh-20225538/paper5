"""Tests for per-backbone activation dispatch."""

from __future__ import annotations

import pytest
import torch.nn as nn

from src.intervention.activation import activation_class_name, resolve_activation


@pytest.mark.parametrize(
    ("name", "expected_type"),
    [
        ("gelu", nn.GELU),
        ("GELU", nn.GELU),
        ("relu", nn.ReLU),
        ("ReLU", nn.ReLU),
        ("silu", nn.SiLU),
        ("SiLU", nn.SiLU),
    ],
)
def test_resolve_activation_types(name: str, expected_type: type) -> None:
    module = resolve_activation(name)
    assert isinstance(module, expected_type)


def test_gelu_uses_exact_approximation() -> None:
    module = resolve_activation("gelu")
    assert isinstance(module, nn.GELU)
    assert module.approximate == "none"


def test_unsupported_activation_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported activation"):
        resolve_activation("tanh")


def test_activation_class_name_roundtrip() -> None:
    for name in ("gelu", "relu", "silu"):
        assert activation_class_name(resolve_activation(name)) == name
