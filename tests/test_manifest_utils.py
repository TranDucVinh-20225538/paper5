"""Tests for manifest provenance helpers."""

from __future__ import annotations

import json
from pathlib import Path

from src.utils.manifest import latest_manifest_record


def test_latest_manifest_record_filters_backbone_and_step(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {"backbone": "a", "step": "3_extract_embeddings", "n": 1},
        {"backbone": "b", "step": "6_gate0", "n": 2},
        {"backbone": "a", "step": "6_gate0", "r": 16},
        {"backbone": "a", "step": "7_train_arms", "n": 3},
    ]
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    rec = latest_manifest_record(manifest, "a", step="6_gate0")
    assert rec is not None
    assert rec["r"] == 16

    rec7 = latest_manifest_record(manifest, "a", step="7_train_arms")
    assert rec7 is not None
    assert rec7["n"] == 3

    assert latest_manifest_record(manifest, "missing") is None
