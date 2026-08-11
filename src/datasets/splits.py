"""Preregistered ISIC/PAD splits from master_metadata.csv — Paper 4 / CSG-SKin logic."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.datasets.constants import LABEL_TO_INDEX
from src.utils.paths import DatasetPaths, load_split_checksum_spec, resolve_image_path


@dataclass(frozen=True)
class SplitConfig:
    random_state: int = 42
    isic_test_fraction: float = 0.2
    val_fraction: float = 0.2


def _optional_dataset_paths() -> DatasetPaths | None:
    try:
        from src.utils.paths import load_dataset_paths

        return load_dataset_paths()
    except FileNotFoundError:
        return None


def load_filtered_master(
    metadata_csv: Path | str,
    *,
    dataset_paths: DatasetPaths | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(metadata_csv)
    required = {"path", "label", "domain"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required metadata columns: {sorted(missing)}")

    df = df[df["label"].isin(LABEL_TO_INDEX)].copy()
    if df.empty:
        raise ValueError("No valid rows after filtering to expected lesion labels.")

    df["label_idx"] = df["label"].map(LABEL_TO_INDEX).astype(int)

    paths = dataset_paths if dataset_paths is not None else _optional_dataset_paths()

    def _resolve_row(path_val: str) -> str | None:
        if paths is not None:
            resolved = resolve_image_path(path_val, paths)
            return str(resolved) if resolved is not None else None
        return path_val if os.path.isfile(str(path_val)) else None

    df["path"] = df["path"].map(_resolve_row)
    df = df[df["path"].notna()].reset_index(drop=True)
    if df.empty:
        raise ValueError("No rows with existing image paths.")
    return df


def build_isic_train_df(
    metadata_csv: Path | str,
    *,
    split: SplitConfig | None = None,
) -> pd.DataFrame:
    """ISIC train only (n=16211 on Papers 1–4 artifact) — fitting population for extraction."""
    split = split or SplitConfig()
    df = load_filtered_master(metadata_csv)
    isic_df = df[df["domain"] == "isic"].copy()
    isic_train_val, _isic_test = train_test_split(
        isic_df,
        test_size=split.isic_test_fraction,
        stratify=isic_df["label_idx"],
        random_state=split.random_state,
    )
    isic_train, _isic_val = train_test_split(
        isic_train_val,
        test_size=split.val_fraction,
        stratify=isic_train_val["label_idx"],
        random_state=split.random_state,
    )
    return isic_train.reset_index(drop=True)


def build_eval_pool_df(
    metadata_csv: Path | str,
    *,
    split: SplitConfig | None = None,
) -> pd.DataFrame:
    """ISIC test + full PAD-UFES (n=7365) — eval pool for extraction."""
    split = split or SplitConfig()
    df = load_filtered_master(metadata_csv)
    isic_df = df[df["domain"] == "isic"].copy()
    pad_df = df[df["domain"] == "pad_ufes"].copy()
    if pad_df.empty:
        raise ValueError("Eval pool requires pad_ufes rows in metadata.")

    _isic_train_val, isic_test = train_test_split(
        isic_df,
        test_size=split.isic_test_fraction,
        stratify=isic_df["label_idx"],
        random_state=split.random_state,
    )
    return pd.concat([isic_test, pad_df], ignore_index=True)


def assert_split_counts(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    repo_root: Path | None = None,
) -> None:
    """Verify row counts match pinned split spec."""
    spec = load_split_checksum_spec(repo_root)
    expected_train = int(spec["partition_counts"]["isic_train"])
    expected_eval = int(spec["partition_counts"]["eval_pool"])
    if len(train_df) != expected_train:
        raise ValueError(f"ISIC train n={len(train_df)} != expected {expected_train}")
    if len(eval_df) != expected_eval:
        raise ValueError(f"Eval pool n={len(eval_df)} != expected {expected_eval}")
