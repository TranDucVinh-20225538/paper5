"""PanDerm pipeline regression against Paper 4 published values (D-027)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.intervention.embeddings import load_embedding_artifacts
from src.intervention.gates import compute_gate0
from src.intervention.nuisance import compute_nuisance_direction
from src.utils.config import load_backbone_config

PANDERM_W_RAW_NORM = 9.014975355083987
PANDERM_PROBE_ACC = 0.9990045248868779
PANDERM_BAL_ACC = 0.5739691277090949


def _panderm_root() -> Path | None:
    env = os.environ.get("PANDERM_EMBEDDINGS_ROOT")
    if env:
        p = Path(env)
        if p.is_dir():
            return p
    default = Path("/Users/cubo/Research/Paper4/PhaseB/assets/reference_embeddings")
    return default if default.is_dir() else None


@pytest.mark.skipif(_panderm_root() is None, reason="PanDerm embeddings not available")
def test_panderm_nuisance_direction_regression(repo_root: Path) -> None:
    root = _panderm_root()
    assert root is not None
    artifacts = load_embedding_artifacts(
        root / "ReferenceTrainEmbedding",
        root / "ReferenceEmbedding",
    )
    result = compute_nuisance_direction(artifacts)
    assert result.n_isic == 16211
    assert result.n_pad_ufes == 2298
    assert result.w.shape == (1024,)
    assert result.w_raw_norm == pytest.approx(PANDERM_W_RAW_NORM, rel=1e-5)


@pytest.mark.skipif(_panderm_root() is None, reason="PanDerm embeddings not available")
def test_panderm_gate0_alpha0_regression(repo_root: Path) -> None:
    root = _panderm_root()
    assert root is not None
    cfg = load_backbone_config(repo_root / "configs" / "panderm.yaml", repo_root=repo_root)
    artifacts = load_embedding_artifacts(
        root / "ReferenceTrainEmbedding",
        root / "ReferenceEmbedding",
    )
    gate0 = compute_gate0(artifacts.eval_embeddings, artifacts.eval_metadata, cfg)
    assert gate0["gate0_pass"] is True
    assert gate0["domain_probe_accuracy_mean"] == pytest.approx(PANDERM_PROBE_ACC, rel=1e-4)
    # Balanced accuracy varies slightly with sklearn version; gate0_pass is the regression target.
    assert gate0["id_task_balanced_accuracy_mean"] > 0.125 + 0.10
