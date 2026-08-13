"""Build torchvision transforms from frozen preprocessing JSON assets."""

from __future__ import annotations

import json
from pathlib import Path

import torchvision.transforms as T
from PIL import Image

from src.backbone.pooling import sam_resize_dimensions


class ResizeLongestSide:
    """Resize so the longest side equals ``size`` (SAM/MedSAM step 1)."""

    def __init__(self, size: int) -> None:
        self.size = int(size)

    def __call__(self, img: Image.Image) -> Image.Image:
        width, height = img.size
        new_h, new_w = sam_resize_dimensions(height, width, target_long_side=self.size)
        if (new_w, new_h) == (width, height):
            return img
        return img.resize((new_w, new_h), Image.BILINEAR)


class PadSquare:
    """Pad bottom/right with zeros to a square canvas (SAM/MedSAM step 3)."""

    def __init__(self, size: int, *, fill: int = 0) -> None:
        self.size = int(size)
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        width, height = img.size
        if width > self.size or height > self.size:
            raise ValueError(
                f"PadSquare: image {width}x{height} exceeds canvas {self.size}x{self.size}"
            )
        if width == self.size and height == self.size:
            return img
        canvas = Image.new(img.mode, (self.size, self.size), self.fill)
        canvas.paste(img, (0, 0))
        return canvas


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
        elif op == "ResizeLongestSide":
            steps.append(ResizeLongestSide(step["size"]))
        elif op == "PadSquare":
            steps.append(PadSquare(step["size"], fill=int(step.get("fill", 0))))
        else:
            raise ValueError(f"Unsupported preprocessing op {op!r} in {asset_path}")
    return T.Compose(steps)
