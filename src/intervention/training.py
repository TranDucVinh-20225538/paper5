"""Adapter training on cached embeddings — config-injected (no PanDerm defaults)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.intervention.adapter import BottleneckAdapter, TaskHead, orthogonality_loss, task_loss
from src.utils.config import BackboneConfig

DEFAULT_EPOCHS = 100
DEFAULT_LR = 1e-3
DEFAULT_BATCH_SIZE = 512
DEFAULT_SEED = 42


def resolve_training_epochs(cfg: BackboneConfig, override: int | None = None) -> int:
    """CLI override > config > protocol default."""
    if override is not None:
        return int(override)
    intervention = cfg.raw.get("intervention", {})
    val = intervention.get("epochs")
    return int(val) if val is not None else DEFAULT_EPOCHS


def _training_hparams(
    cfg: BackboneConfig,
    *,
    epochs: int | None = None,
    lr: float | None = None,
    batch_size: int | None = None,
) -> tuple[int, float, int]:
    intervention = cfg.raw.get("intervention", {})
    epoch_val = epochs if epochs is not None else intervention.get("epochs")
    lr_val = lr if lr is not None else intervention.get("lr")
    batch_val = batch_size if batch_size is not None else intervention.get("batch_size")
    return (
        int(epoch_val) if epoch_val is not None else DEFAULT_EPOCHS,
        float(lr_val) if lr_val is not None else DEFAULT_LR,
        int(batch_val) if batch_val is not None else DEFAULT_BATCH_SIZE,
    )


def train_adapter(
    z_train: np.ndarray,
    y_train: np.ndarray,
    w: np.ndarray,
    cfg: BackboneConfig,
    *,
    r: int,
    lambda_proj: float,
    seed: int = DEFAULT_SEED,
    epochs: int | None = None,
    lr: float | None = None,
    batch_size: int | None = None,
    device: torch.device | None = None,
) -> tuple[BottleneckAdapter, TaskHead]:
    epochs, lr, batch_size = _training_hparams(cfg, epochs=epochs, lr=lr, batch_size=batch_size)

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


@torch.no_grad()
def compute_delta_z(adapter: BottleneckAdapter, z: np.ndarray) -> np.ndarray:
    adapter.eval()
    z_t = torch.tensor(z, dtype=torch.float32)
    return adapter.delta(z_t).numpy()


def apply_alpha(z: np.ndarray, delta: np.ndarray, alpha: float) -> np.ndarray:
    return z + alpha * delta


def train_conventional(
    z_train: np.ndarray,
    y_train: np.ndarray,
    cfg: BackboneConfig,
    *,
    r: int,
    seed: int = DEFAULT_SEED,
    epochs: int | None = None,
    lr: float | None = None,
    batch_size: int | None = None,
    device: torch.device | None = None,
) -> tuple[BottleneckAdapter, TaskHead]:
    """Conventional arm — L_task only, orthogonality term structurally absent."""
    epochs, lr, batch_size = _training_hparams(cfg, epochs=epochs, lr=lr, batch_size=batch_size)
    torch.manual_seed(seed)
    dev = device or torch.device("cpu")
    adapter = BottleneckAdapter(dim=cfg.embed_dim, r=r, activation=cfg.activation).to(dev)
    head = TaskHead(dim=cfg.embed_dim).to(dev)
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
            logits = head(adapter(zb))
            loss = task_loss(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return adapter, head


def train_linear_probe(
    z_train: np.ndarray,
    y_train: np.ndarray,
    cfg: BackboneConfig,
    *,
    seed: int = DEFAULT_SEED,
    epochs: int | None = None,
    lr: float | None = None,
    batch_size: int | None = None,
    device: torch.device | None = None,
) -> TaskHead:
    """Adaptation rung 1 — linear head only, z unchanged."""
    epochs, lr, batch_size = _training_hparams(cfg, epochs=epochs, lr=lr, batch_size=batch_size)
    torch.manual_seed(seed)
    dev = device or torch.device("cpu")
    head = TaskHead(dim=cfg.embed_dim).to(dev)
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
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    head.train()
    for _epoch in range(epochs):
        for zb, yb in loader:
            yb = yb.to(dev)
            loss = task_loss(head(zb.to(dev)), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return head


def train_adapter_task_only(
    z_train: np.ndarray,
    y_train: np.ndarray,
    cfg: BackboneConfig,
    *,
    r: int,
    seed: int = DEFAULT_SEED,
    epochs: int | None = None,
    lr: float | None = None,
    batch_size: int | None = None,
    device: torch.device | None = None,
) -> tuple[BottleneckAdapter, TaskHead]:
    """Adaptation partial-FT rung — adapter + task loss only."""
    return train_conventional(
        z_train,
        y_train,
        cfg,
        r=r,
        seed=seed,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        device=device,
    )


def save_adapter_checkpoint(
    path: Path,
    adapter: BottleneckAdapter,
    head: TaskHead | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"adapter": adapter.state_dict()}
    if head is not None:
        payload["head"] = head.state_dict()
    torch.save(payload, path)


def load_adapter_checkpoint(
    path: Path,
    cfg: BackboneConfig,
    *,
    r: int,
) -> BottleneckAdapter:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    adapter = BottleneckAdapter(dim=cfg.embed_dim, r=r, activation=cfg.activation)
    adapter.load_state_dict(ckpt["adapter"])
    return adapter
