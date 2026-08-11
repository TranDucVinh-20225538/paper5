"""
Bottleneck adapter and losses — Paper 4 ladder, config-injected (REUSE.md §4).

z' = z + W2 · act(W1 · z),  W2 zero-init (identity at init)
L_orth(z', w) = cos(z', w)²
L_task = CrossEntropy(head(z'), y)  with y == -1 ignored (PAD rows)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.intervention.activation import resolve_activation


class BottleneckAdapter(nn.Module):
    """z' = z + W2 · act(W1 · z). No PanDerm defaults — dim and act come from config."""

    def __init__(self, dim: int, r: int, activation: str):
        if dim is None:
            raise ValueError("dim is required — do not rely on a PanDerm default")
        if r <= 0:
            raise ValueError(f"bottleneck width r must be positive, got {r}")

        super().__init__()
        self.dim = dim
        self.r = r
        self.down = nn.Linear(dim, r)
        self.act = resolve_activation(activation)
        self.up = nn.Linear(r, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def delta(self, z: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(z)))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.delta(z)


def orthogonality_loss(z_adapted: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """L_orth = mean(cos(z', w)^2)."""
    w_row = w.unsqueeze(0).expand_as(z_adapted)
    cos = F.cosine_similarity(z_adapted, w_row, dim=1)
    return (cos**2).mean()


def task_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """CE on rows with a valid label; y == -1 rows (PAD-UFES) ignored."""
    return F.cross_entropy(logits, y, ignore_index=-1)


class TaskHead(nn.Module):
    def __init__(self, dim: int, num_classes: int = 8):
        if dim is None:
            raise ValueError("dim is required for TaskHead")
        super().__init__()
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)
