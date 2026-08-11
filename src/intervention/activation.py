"""Resolve backbone activation functions from config (REUSE.md §4, D-031)."""

from __future__ import annotations

import torch.nn as nn

_SUPPORTED = frozenset({"gelu", "relu", "silu"})


def resolve_activation(name: str) -> nn.Module:
    """
    Map a config activation string to a fresh nn.Module instance.

    GELU uses approximate=\"none\" to match Paper 4 / PanDerm (stage4_adapter.py).
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("activation must be a non-empty string from config")

    key = name.strip().lower()
    if key not in _SUPPORTED:
        supported = ", ".join(sorted(_SUPPORTED))
        raise ValueError(f"Unsupported activation {name!r}; expected one of: {supported}")

    if key == "gelu":
        return nn.GELU(approximate="none")
    if key == "relu":
        return nn.ReLU()
    return nn.SiLU()


def activation_class_name(module: nn.Module) -> str:
    """Return a stable lowercase label for tests and manifest logging."""
    if isinstance(module, nn.GELU):
        return "gelu"
    if isinstance(module, nn.ReLU):
        return "relu"
    if isinstance(module, nn.SiLU):
        return "silu"
    raise TypeError(f"Unexpected activation module: {type(module)!r}")
