"""Tests for nuisance direction (protocol Step 4)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.intervention.embeddings import EmbeddingArtifacts
from src.intervention.nuisance import compute_nuisance_direction


def _artifacts() -> EmbeddingArtifacts:
    train_z = np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (5, 1))
    eval_z = np.vstack(
        [
            np.tile([1.0, 0.0, 0.0], (3, 1)),
            np.tile([0.0, 1.0, 0.0], (2, 1)),
        ]
    )
    train_meta = pd.DataFrame({"domain": ["isic"] * 5, "label_idx": [0] * 5})
    eval_meta = pd.DataFrame(
        {
            "domain": ["isic"] * 3 + ["pad_ufes"] * 2,
            "label_idx": [0, 1, 2, 0, 0],
        }
    )
    from pathlib import Path

    return EmbeddingArtifacts(
        train_embeddings=train_z,
        train_metadata=train_meta,
        eval_embeddings=eval_z,
        eval_metadata=eval_meta,
        train_dir=Path("/tmp/train"),
        eval_dir=Path("/tmp/eval"),
    )


def test_nuisance_direction_unit_norm() -> None:
    result = compute_nuisance_direction(_artifacts())
    assert result.w.shape == (3,)
    assert result.w_raw_norm > 0
    np.testing.assert_allclose(np.linalg.norm(result.w), 1.0, rtol=1e-6)


def test_nuisance_direction_formula() -> None:
    result = compute_nuisance_direction(_artifacts())
    # mu_isic = [1,0,0], mu_pad = [0,1,0] -> w points along [1,-1,0]
    assert result.w[0] > 0
    assert result.w[1] < 0
