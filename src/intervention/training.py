"""Adapter training on cached embeddings — config-injected (no PanDerm defaults)."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.intervention.adapter import BottleneckAdapter, TaskHead, orthogonality_loss, task_loss
from src.utils.config import BackboneConfig

DEFAULT_EPOCHS = 100
DEFAULT_LR = 1e-3
DEFAULT_BATCH_SIZE = 512
DEFAULT_SEED = 42


def train_adapter(
    z_train: np.ndarray,
    y_train: np.ndarray,
    w: np.ndarray,
    cfg: BackboneConfig,
    *,
    r: int,
    lambda_proj: float,
    seed: int = DEFAULT_SEED,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: torch.device | None = None,
) -> tuple[BottleneckAdapter, TaskHead]:
    intervention = cfg.raw.get("intervention", {})
    epochs = int(intervention.get("epochs", epochs))
    lr = float(intervention.get("lr", lr))
    batch_size = int(intervention.get("batch_size", batch_size))

    torch.manual_seed(seed)
    dev = device or torch.device("cpu")
    adapter = BottleneckAdapter(dim=cfg.embed_dim, r=r, activation=cfg.activation).to(dev)
    head = TaskHead(dim=cfg.embed_dim).to(dev)
    w_t = torch.tensor(w, dtype=torch.float32, device=dev)

    ds = TensorDataset(
        torch.tensor(z_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    opt = torch.optim.AdamW(list(adapter.parameters()) + list(head.parameters()), lr=lr)

    adapter.train()
    head.train()
    for _epoch in range(epochs):
        for zb, yb in loader:
            zb, yb = zb.to(dev), yb.to(dev)
            z_adapted = adapter(zb)
            logits = head(z_adapted)
            loss = task_loss(logits, yb) + lambda_proj * orthogonality_loss(z_adapted, w_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return adapter, head


@torch.no_grad()
def apply_adapter(adapter: BottleneckAdapter, z: np.ndarray) -> np.ndarray:
    adapter.eval()
    z_t = torch.tensor(z, dtype=torch.float32)
    return adapter(z_t).numpy()
