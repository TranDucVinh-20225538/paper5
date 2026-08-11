"""Build torchvision transforms from frozen preprocessing JSON assets."""

from __future__ import annotations

import json
from pathlib import Path

import torchvision.transforms as T


def load_preprocessing_pipeline(asset_path: Path | str) -> T.Compose:
    spec = json.loads(Path(asset_path).read_text(encoding="utf-8"))
    ops = spec.get("pipeline")
    if not isinstance(ops, list) or not ops:
        raise ValueError(f"{asset_path}: preprocessing.pipeline must be a non-empty list")

    steps: list = []
    for step in ops:
        op = step["op"]
        if op == "Resize":
            size = step.get("size")
            if isinstance(size, list):
                steps.append(T.Resize(tuple(size)))
            else:
                steps.append(T.Resize(int(size)))
        elif op == "CenterCrop":
            size = step["size"]
            steps.append(T.CenterCrop(tuple(size) if isinstance(size, list) else int(size)))
        elif op == "ToTensor":
            steps.append(T.ToTensor())
        elif op == "Normalize":
            steps.append(
                T.Normalize(
                    mean=step["mean"],
                    std=step["std"],
                )
            )
        else:
            raise ValueError(f"Unsupported preprocessing op {op!r} in {asset_path}")
    return T.Compose(steps)
