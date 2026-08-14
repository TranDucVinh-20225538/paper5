#!/usr/bin/env python3
"""
Exploratory interim report for Step-12 backbones.

    python scripts/interim_report.py

Reads ``results/manifest.jsonl`` (step ``12_record`` only) and exported CSV
artifacts under ``results/csv/<backbone>/``. Writes:

  - results/interim/summary.csv
  - results/interim/summary.md
  - results/figures/interim/scatter_*.png

No confirmatory statistics, multiplicity correction, or inferential claims.
Every output is watermarked: INTERIM / EXPLORATORY — DO NOT USE FOR INFERENCE.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.reporting.interim_report import WATERMARK, build_interim_report  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Interim exploratory backbone report")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "results" / "manifest.jsonl",
        help="Pipeline manifest (default: results/manifest.jsonl)",
    )
    parser.add_argument(
        "--interim-dir",
        type=Path,
        default=REPO_ROOT / "results" / "interim",
        help="Directory for summary.csv / summary.md",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=REPO_ROOT / "results" / "figures" / "interim",
        help="Directory for scatter plots",
    )
    args = parser.parse_args(argv)

    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:
        print(
            "matplotlib is required for scatter plots. "
            "Install with: pip install matplotlib",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    outputs = build_interim_report(
        repo_root=REPO_ROOT,
        manifest_path=args.manifest,
        interim_dir=args.interim_dir,
        figures_dir=args.figures_dir,
    )

    print(WATERMARK)
    print(f"summary:  {outputs.summary_csv}")
    print(f"markdown: {outputs.summary_md}")
    for path in outputs.figure_paths:
        print(f"figure:   {path}")
    if not outputs.figure_paths:
        print("(no scatter plots — need ≥1 backbone with numeric CSV metrics)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
