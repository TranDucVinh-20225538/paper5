#!/usr/bin/env python3
"""One-image MedSAM forward smoke test on the resolved torch device (MPS on Mac)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

from src.backbone.loaders.medsam_loader import (
    create_medsam_model,
    forward_medsam_pre_neck,
    verify_medsam_forward,
)
from src.utils.config import load_backbone_config
from src.utils.torch_device import resolve_torch_device


def main() -> int:
    dev = resolve_torch_device()
    print(f"device: {dev}")
    cfg = load_backbone_config(REPO_ROOT / "configs" / "medsam.yaml")
    model = create_medsam_model(cfg)
    verify_medsam_forward(cfg, model)
    encoder = model.vision_encoder.to(dev)
    x = torch.randn(1, 3, 1024, 1024, device=dev)
    with torch.no_grad():
        t0 = time.perf_counter()
        out = forward_medsam_pre_neck(encoder, x)
        if dev.type == "mps":
            torch.mps.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"forward ok shape={tuple(out.shape)} time={elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
