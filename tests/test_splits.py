"""Tests for preregistered split construction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.datasets.constants import LABELS
from src.datasets.splits import SplitConfig, build_eval_pool_df, build_isic_train_df


def _synthetic_metadata(n_isic: int = 400, n_pad: int = 80) -> pd.DataFrame:
    rows = []
    for i in range(n_isic):
        label = LABELS[i % len(LABELS)]
        rows.append({"path": f"/tmp/isic_{i}.jpg", "label": label, "domain": "isic"})
    for i in range(n_pad):
        rows.append({"path": f"/tmp/pad_{i}.jpg", "label": "NV", "domain": "pad_ufes"})
    return pd.DataFrame(rows)


def test_build_splits_counts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    meta = tmp_path / "master_metadata.csv"
    _synthetic_metadata().to_csv(meta, index=False)

    def _fake_isfile(path: str) -> bool:
        return str(path).startswith("/tmp/")

    monkeypatch.setattr("src.datasets.splits.os.path.isfile", _fake_isfile)

    train = build_isic_train_df(meta, split=SplitConfig())
    eval_df = build_eval_pool_df(meta, split=SplitConfig())
    assert len(train) + len(eval_df) > 0
    assert set(train["domain"]) == {"isic"}
    assert set(eval_df["domain"]) == {"isic", "pad_ufes"}
