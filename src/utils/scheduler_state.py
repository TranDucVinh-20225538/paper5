"""Backbone phase detection and scheduler queue helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from src.utils.config import find_repo_root, load_backbone_config
from src.utils.manifest import latest_manifest_record

Phase = Literal["pending", "gpu_done", "done", "failed"]


def backbone_name_from_config(config_path: Path | str) -> str:
    return load_backbone_config(config_path).name


def _arms_manifest(root: Path, backbone: str) -> Path:
    return root / "experiments" / backbone / "arms" / "conventional" / "manifest.json"


def backbone_phase(
    config_path: Path | str,
    *,
    repo_root: Path | None = None,
    manifest_path: Path | None = None,
) -> Phase:
    """Return pipeline phase for resume: pending → gpu_done → done."""
    cfg = load_backbone_config(config_path, repo_root=repo_root)
    root = repo_root or find_repo_root(Path(config_path))
    manifest = manifest_path or (root / "results" / "manifest.jsonl")

    rec12 = latest_manifest_record(manifest, cfg.name, step="12_record")
    if rec12 is not None:
        gate1 = rec12.get("gate1") or rec12.get("gate1_pass")
        if gate1 in ("pass", True):
            return "done"
        if gate1 in ("fail", "not_testable", False):
            return "failed"

    rec7 = latest_manifest_record(manifest, cfg.name, step="7_train_arms")
    if rec7 is not None and _arms_manifest(root, cfg.name).is_file():
        return "gpu_done"

    return "pending"


def filter_queue_by_phase(
    config_paths: list[Path | str],
    *,
    repo_root: Path | None = None,
    manifest_path: Path | None = None,
    skip_done: bool = True,
) -> dict[str, list[str]]:
    """Partition configs into pending (GPU), gpu_done (CPU-only), done, failed."""
    out: dict[str, list[str]] = {
        "pending": [],
        "gpu_done": [],
        "done": [],
        "failed": [],
    }
    for cfg in config_paths:
        phase = backbone_phase(cfg, repo_root=repo_root, manifest_path=manifest_path)
        key = phase if not (skip_done and phase == "done") else "done"
        if skip_done and phase == "done":
            out["done"].append(str(cfg))
        elif phase == "failed":
            out["failed"].append(str(cfg))
        else:
            out[phase].append(str(cfg))
    return out


def scheduler_status_json(
    config_paths: list[Path | str],
    *,
    repo_root: Path | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    parts = filter_queue_by_phase(
        config_paths, repo_root=repo_root, manifest_path=manifest_path, skip_done=False
    )
    return {
        "queue": [str(p) for p in config_paths],
        "pending": parts["pending"],
        "gpu_done": parts["gpu_done"],
        "done": parts["done"],
        "failed": parts["failed"],
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Scheduler backbone phase/status")
    parser.add_argument("configs", nargs="+", type=Path, help="configs/*.yaml paths")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--json", action="store_true", help="Emit JSON status")
    args = parser.parse_args(argv)

    if args.json:
        print(json.dumps(scheduler_status_json(args.configs, manifest_path=args.manifest)))
        return 0

    for cfg in args.configs:
        phase = backbone_phase(cfg, manifest_path=args.manifest)
        name = backbone_name_from_config(cfg)
        print(f"{name}\t{phase}\t{cfg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
