"""Lesion label ordering — vendored from CSG-SKin/src/datasets/constants.py (Paper 4 split)."""

LABELS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(LABELS)}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
