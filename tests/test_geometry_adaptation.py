"""Adaptation-arm geometry rows for D-034 (no adapter on linear-probe)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.geometry.metrics import run_geometry_completion
from src.intervention.arms import ArmCheckpoints
from src.intervention.embeddings import EmbeddingArtifacts
from src.utils.config import load_backbone_config

FIXTURE_CFG = (
    Path(__file__).resolve().parent / "fixtures" / "configs" / "fixture_backbone.yaml"
)


def _artifacts(n: int = 80, d: int = 16, num_classes: int = 8) -> EmbeddingArtifacts:
    rng = np.random.default_rng(0)
    z = rng.normal(size=(n, d)).astype(np.float32)
    labels = np.repeat(np.arange(num_classes), n // num_classes)
    meta = pd.DataFrame({"label_idx": labels, "domain": ["isic"] * n})
    dummy = Path("/tmp")
    return EmbeddingArtifacts(
        train_embeddings=z,
        train_metadata=meta,
        eval_embeddings=z,
        eval_metadata=meta,
        train_dir=dummy,
        eval_dir=dummy,
    )


def test_geometry_adaptation_linear_probe_has_no_alpha(tmp_path: Path) -> None:
    cfg = load_backbone_config(FIXTURE_CFG)
    checkpoints = ArmCheckpoints(backbone=cfg.name, root=tmp_path)
    adapt_dir = checkpoints.arm_dir("adaptation")
    adapt_dir.mkdir(parents=True)
    (adapt_dir / "manifest.json").write_text(
        json.dumps(
            {
                "rungs": {
                    "linear-probe": [{"seed": 42}, {"seed": 52}],
                    "partial-FT": [],
                    "full-adapter-FT": [{"seed": 42, "reused_from": "conventional"}],
                }
            }
        ),
        encoding="utf-8",
    )
    w = np.ones(16, dtype=np.float32)
    w /= np.linalg.norm(w)
    out = run_geometry_completion(
        cfg,
        _artifacts(),
        checkpoints,
        w,
        r=16,
        alphas=[0.0, 0.25, 1.0],
        output_dir=tmp_path,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    adapt = payload["results"]["adaptation"]
    assert "canonical" not in payload["results"]
    assert [row["rung"] for row in adapt] == ["linear-probe", "linear-probe"]
    assert [row["seed"] for row in adapt] == [42, 52]
    for row in adapt:
        assert "alpha" not in row
        assert "condition_number" in row
        assert "lid_mean" in row
        assert "spectral_decay_slope" in row
        assert row["rung"] != "full-adapter-FT"
