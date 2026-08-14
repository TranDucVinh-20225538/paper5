"""Tests for torch device resolution."""

from __future__ import annotations

import pytest
import torch

from src.utils.torch_device import pin_memory_for, resolve_torch_device


def test_resolve_torch_device_explicit() -> None:
    assert resolve_torch_device("cpu").type == "cpu"


def test_resolve_torch_device_env_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PAPER5_DEVICE", "cpu")
    assert resolve_torch_device().type == "cpu"


def test_pin_memory_only_cuda() -> None:
    assert pin_memory_for(torch.device("cuda")) is True
    assert pin_memory_for(torch.device("mps")) is False
    assert pin_memory_for(torch.device("cpu")) is False
