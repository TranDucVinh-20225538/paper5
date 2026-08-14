"""PyTorch device selection for extraction and training."""

from __future__ import annotations

import os

import torch


def resolve_torch_device(device: torch.device | str | None = None) -> torch.device:
    """
    Pick compute device: explicit arg > ``PAPER5_DEVICE`` env > cuda > mps > cpu.

    Set ``PAPER5_DEVICE=cpu`` to force CPU (e.g. reproducibility debugging).
    Set ``PAPER5_DEVICE=mps`` to require Apple GPU (raises if unavailable).
    """
    if device is not None:
        return torch.device(device)

    env = os.environ.get("PAPER5_DEVICE", "").strip().lower()
    if env == "cpu":
        return torch.device("cpu")
    if env == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("PAPER5_DEVICE=mps but MPS is not available on this machine.")
        return torch.device("mps")
    if env == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("PAPER5_DEVICE=cuda but CUDA is not available on this machine.")
        return torch.device("cuda")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pin_memory_for(device: torch.device) -> bool:
    """DataLoader pin_memory only helps CUDA host→device copies."""
    return device.type == "cuda"
