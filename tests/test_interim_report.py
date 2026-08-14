"""Tests for interim exploratory report (no confirmatory stats)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.reporting.interim_report import (
    WATERMARK,
    build_interim_report,
    build_summary_rows,
    step12_completed_backbones,
)


def _write_manifest(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )


def _minimal_csv_tree(repo_root: Path, backbone: str) -> None:
    csv_dir = repo_root / "results" / "csv" / backbone
    csv_dir.mkdir(parents=True, exist_ok=True)
    (csv_dir / "reliability_scorers.json").write_text(
        json.dumps(
            {
                "results": {
                    "canonical": [
                        {"seed": 42, "alpha": 1.0, "maha_auroc": 0.9, "cosine_auroc": 0.8}
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    (csv_dir / "alpha_ladder_results.json").write_text(
        json.dumps(
            {
                "results": {
                    "canonical": [
                        {
                            "seed": 42,
                            "alpha": 0.0,
                            "gate0": {
                                "domain_probe_accuracy_mean": 0.99,
                                "id_task_balanced_accuracy_mean": 0.5,
                            },
                            "gate1_measurement": {
                                "lid_mean": 20.0,
                                "spectral_decay_slope": 0.003,
                            },
                        },
                        {
                            "seed": 42,
                            "alpha": 1.0,
                            "gate1_measurement": {
                                "lid_mean": 19.0,
                                "spectral_decay_slope": 0.004,
                            },
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    (csv_dir / "gate1_ea03.json").write_text(
        json.dumps(
            {
                "results": {
                    "canonical": [{"alpha": 0.25, "gate1_pass": True}],
                }
            }
        ),
        encoding="utf-8",
    )


def test_step12_completed_excludes_fixture(repo_root: Path, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(
        manifest,
        [
            {"backbone": "fixture_test", "step": "12_record", "gate1_pass": True},
            {"backbone": "panderm", "step": "12_record", "gate1_pass": True},
        ],
    )
    done = step12_completed_backbones(manifest)
    assert set(done) == {"panderm"}


def test_build_summary_rows_reads_csv(repo_root: Path, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(
        manifest,
        [
            {"backbone": "panderm", "step": "12_record", "gate1_pass": True, "selected_r": 16},
            {"backbone": "panderm", "step": "6_gate0", "w_raw_norm": 9.0},
        ],
    )
    mini_root = tmp_path / "repo"
    mini_root.mkdir()
    (mini_root / "configs").mkdir()
    (mini_root / "results" / "csv").mkdir(parents=True)
    src_cfg = repo_root / "configs" / "panderm.yaml"
    if src_cfg.is_file():
        (mini_root / "configs" / "panderm.yaml").write_text(
            src_cfg.read_text(encoding="utf-8"), encoding="utf-8"
        )
    _minimal_csv_tree(mini_root, "panderm")
    rows = build_summary_rows(mini_root, manifest)
    assert len(rows) == 1
    assert rows[0]["backbone"] == "panderm"
    assert rows[0]["maha_auroc_alpha1"] == pytest.approx(0.9)


def test_build_interim_report_writes_watermarked_outputs(repo_root: Path, tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(
        manifest,
        [
            {"backbone": "panderm", "step": "12_record", "gate1_pass": True, "selected_r": 16},
        ],
    )
    mini_root = tmp_path / "repo"
    (mini_root / "configs").mkdir(parents=True)
    src_cfg = repo_root / "configs" / "panderm.yaml"
    if src_cfg.is_file():
        (mini_root / "configs" / "panderm.yaml").write_text(
            src_cfg.read_text(encoding="utf-8"), encoding="utf-8"
        )
    _minimal_csv_tree(mini_root, "panderm")
    interim = tmp_path / "interim"
    figures = tmp_path / "figures"

    outputs = build_interim_report(
        repo_root=mini_root,
        manifest_path=manifest,
        interim_dir=interim,
        figures_dir=figures,
    )

    csv_text = outputs.summary_csv.read_text(encoding="utf-8")
    md_text = outputs.summary_md.read_text(encoding="utf-8")
    assert WATERMARK in csv_text
    assert WATERMARK in md_text
    assert outputs.summary_csv.is_file()
    assert len(outputs.figure_paths) >= 1
