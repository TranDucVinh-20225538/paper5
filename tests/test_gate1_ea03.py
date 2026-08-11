"""EA-03 Gate 1 manipulation check — regression against Paper 4 rescored JSON."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.intervention.gates import gate1_manipulation_pass, score_gate1_ea03

PAPER4_ROOT = Path("/Users/cubo/Research/Paper4/PhaseB/AlphaLadder")
LADDER_JSON = PAPER4_ROOT / "alpha_ladder_results.json"
EA03_JSON = PAPER4_ROOT / "gate1_ea03_rescored.json"

ARM_MAP = {"Intervention": "canonical", "Conventional": "conventional"}


@pytest.mark.skipif(not LADDER_JSON.is_file(), reason="Paper 4 alpha ladder JSON not available")
def test_ea03_rescore_matches_paper4() -> None:
    ladder = json.loads(LADDER_JSON.read_text(encoding="utf-8"))
    expected = json.loads(EA03_JSON.read_text(encoding="utf-8"))
    mapped = {
        ARM_MAP[k]: v for k, v in ladder["results"].items() if k in ARM_MAP
    }
    scored = score_gate1_ea03(mapped, ladder["alphas"])
    for paper_arm, our_arm in ARM_MAP.items():
        exp_rows = {r["alpha"]: r for r in expected["results"][paper_arm]}
        for row in scored[our_arm]:
            exp = exp_rows[row["alpha"]]
            assert row["lid_reproducible"] == exp["lid_reproducible"]
            assert row["slope_reproducible"] == exp["slope_reproducible"]
            assert row["gate1_pass"] == exp["gate1_pass"]
    assert gate1_manipulation_pass(scored) is True
