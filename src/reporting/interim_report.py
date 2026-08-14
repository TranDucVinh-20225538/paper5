"""Interim exploratory summary from Step-12 backbones — no confirmatory statistics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from src.utils.config import find_repo_root, load_backbone_config
from src.utils.manifest import iter_manifest_records, latest_manifest_record

WATERMARK = "INTERIM / EXPLORATORY — DO NOT USE FOR INFERENCE"

_EXCLUDE_BACKBONES = frozenset({"fixture_test"})

_FAMILY_LABELS = {
    "cnn": "CNN",
    "medical_ssl": "Medical SSL",
    "medical_vlm": "Medical VLM",
    "general_ssl": "General SSL",
    "general_vlm": "General VLM",
    "probe": "Probe",
}


@dataclass(frozen=True)
class InterimReportOutputs:
    summary_csv: Path
    summary_md: Path
    figure_paths: tuple[Path, ...]


def step12_completed_backbones(manifest_path: Path) -> dict[str, dict[str, Any]]:
    """Latest ``12_record`` line per backbone (excluding test fixtures)."""
    completed: dict[str, dict[str, Any]] = {}
    for record in iter_manifest_records(manifest_path):
        if record.get("step") != "12_record":
            continue
        name = record.get("backbone")
        if not name or name in _EXCLUDE_BACKBONES:
            continue
        completed[str(name)] = record
    return completed


def _csv_dir(repo_root: Path, backbone: str) -> Path:
    return repo_root / "results" / "csv" / backbone


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_canonical(
    payload: dict[str, Any] | None,
    *,
    alpha: float,
    key: str,
    nested: str | None = None,
) -> float | None:
    if payload is None:
        return None
    rows = payload.get("results", {}).get("canonical", [])
    vals: list[float] = []
    for row in rows:
        if row.get("alpha") != alpha:
            continue
        if nested:
            bucket = row.get(nested)
            if not isinstance(bucket, dict) or key not in bucket:
                continue
            vals.append(float(bucket[key]))
        elif key in row:
            vals.append(float(row[key]))
    return float(mean(vals)) if vals else None


def _backbone_family(repo_root: Path, backbone: str) -> str:
    cfg_path = repo_root / "configs" / f"{backbone}.yaml"
    if not cfg_path.is_file():
        return "unknown"
    try:
        cfg = load_backbone_config(cfg_path)
        return cfg.family
    except (FileNotFoundError, ValueError, KeyError):
        return "unknown"


def _embed_dim(repo_root: Path, backbone: str) -> int | None:
    cfg_path = repo_root / "configs" / f"{backbone}.yaml"
    if not cfg_path.is_file():
        return None
    try:
        return int(load_backbone_config(cfg_path).embed_dim)
    except (FileNotFoundError, ValueError, KeyError, TypeError):
        return None


def build_summary_rows(
    repo_root: Path,
    manifest_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    completed = step12_completed_backbones(manifest_path)

    for backbone in sorted(completed):
        rec12 = completed[backbone]
        rec6 = latest_manifest_record(manifest_path, backbone, step="6_gate0")
        csv_dir = _csv_dir(repo_root, backbone)

        reliability = _load_json(csv_dir / "reliability_scorers.json")
        alpha_ladder = _load_json(csv_dir / "alpha_ladder_results.json")
        gate1 = _load_json(csv_dir / "gate1_ea03.json")

        family = _backbone_family(repo_root, backbone)
        rows.append(
            {
                "backbone": backbone,
                "family": family,
                "family_label": _FAMILY_LABELS.get(family, family),
                "gate1_pass": bool(rec12.get("gate1_pass", rec12.get("gate1") == "pass")),
                "selected_r": rec12.get("selected_r"),
                "selected_lambda_proj": rec12.get("selected_lambda_proj"),
                "w_raw_norm": rec6.get("w_raw_norm") if rec6 else None,
                "embed_dim": _embed_dim(repo_root, backbone),
                "domain_probe_alpha0": _mean_canonical(
                    alpha_ladder, alpha=0.0, key="domain_probe_accuracy_mean", nested="gate0"
                ),
                "id_bal_acc_alpha0": _mean_canonical(
                    alpha_ladder, alpha=0.0, key="id_task_balanced_accuracy_mean", nested="gate0"
                ),
                "lid_mean_alpha0": _mean_canonical(
                    alpha_ladder, alpha=0.0, key="lid_mean", nested="gate1_measurement"
                ),
                "lid_mean_alpha1": _mean_canonical(
                    alpha_ladder, alpha=1.0, key="lid_mean", nested="gate1_measurement"
                ),
                "spectral_slope_alpha0": _mean_canonical(
                    alpha_ladder, alpha=0.0, key="spectral_decay_slope", nested="gate1_measurement"
                ),
                "spectral_slope_alpha1": _mean_canonical(
                    alpha_ladder, alpha=1.0, key="spectral_decay_slope", nested="gate1_measurement"
                ),
                "maha_auroc_alpha1": _mean_canonical(reliability, alpha=1.0, key="maha_auroc"),
                "cosine_auroc_alpha1": _mean_canonical(reliability, alpha=1.0, key="cosine_auroc"),
                "ea03_pass_alpha025": _ea03_pass_at_alpha(gate1, 0.25),
                "step12_timestamp": rec12.get("timestamp"),
                "commit": rec12.get("commit"),
            }
        )
    return rows


def _ea03_pass_at_alpha(gate1: dict[str, Any] | None, alpha: float) -> bool | None:
    if gate1 is None:
        return None
    for arm in ("canonical", "conventional"):
        for row in gate1.get("results", {}).get(arm, []):
            if row.get("alpha") == alpha:
                return bool(row.get("gate1_pass"))
    return None


def _write_summary_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"# {WATERMARK}\n")
        df.to_csv(fh, index=False)


def _write_summary_md(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Interim backbone summary",
        "",
        f"> **{WATERMARK}**",
        "",
        f"Backbones with manifest `step=12_record`: **{len(df)}**",
        "",
    ]
    if df.empty:
        lines.append("_No Step-12 backbones found._")
    else:
        cols = list(df.columns)
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join("---" for _ in cols) + " |")
        for _, row in df.iterrows():
            cells = []
            for col in cols:
                val = row[col]
                if isinstance(val, float):
                    cells.append(f"{val:.4f}")
                elif val is None or (isinstance(val, float) and pd.isna(val)):
                    cells.append("")
                else:
                    cells.append(str(val))
            lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _add_watermark(fig) -> None:
    fig.text(
        0.5,
        0.02,
        WATERMARK,
        ha="center",
        va="bottom",
        fontsize=9,
        color="crimson",
        alpha=0.85,
        wrap=True,
    )


def _scatter_by_family(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str,
    out_path: Path,
) -> Path | None:
    import matplotlib.pyplot as plt

    plot_df = df.dropna(subset=[x, y])
    if plot_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(7, 5))
    families = sorted(plot_df["family_label"].unique())
    cmap = plt.get_cmap("tab10")

    for i, fam in enumerate(families):
        sub = plot_df[plot_df["family_label"] == fam]
        ax.scatter(
            sub[x],
            sub[y],
            label=fam,
            s=70,
            alpha=0.85,
            color=cmap(i % 10),
            edgecolors="white",
            linewidths=0.5,
        )
        for _, row in sub.iterrows():
            ax.annotate(
                row["backbone"],
                (row[x], row[y]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )

    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel(y.replace("_", " "))
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    _add_watermark(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _write_scatter_plots(df: pd.DataFrame, figures_dir: Path) -> tuple[Path, ...]:
    specs = [
        ("w_raw_norm", "maha_auroc_alpha1", "Nuisance norm vs Mahalanobis AUROC (α=1)"),
        ("lid_mean_alpha0", "maha_auroc_alpha1", "LID @ α=0 vs Mahalanobis AUROC @ α=1"),
        ("spectral_slope_alpha0", "lid_mean_alpha1", "Spectral slope @ α=0 vs LID @ α=1"),
        ("embed_dim", "maha_auroc_alpha1", "Embedding dim vs Mahalanobis AUROC (α=1)"),
    ]
    paths: list[Path] = []
    for x, y, title in specs:
        fname = f"scatter_{x}_vs_{y}.png"
        out = _scatter_by_family(
            df,
            x=x,
            y=y,
            title=title,
            out_path=figures_dir / fname,
        )
        if out is not None:
            paths.append(out)
    return tuple(paths)


def build_interim_report(
    *,
    repo_root: Path | None = None,
    manifest_path: Path | None = None,
    interim_dir: Path | None = None,
    figures_dir: Path | None = None,
) -> InterimReportOutputs:
    """
    Build interim summary table and exploratory scatter plots.

    Reads only Step-12-completed backbones and exported CSV artifacts.
    Does not run hypothesis tests or multiplicity correction.
    """
    root = repo_root or find_repo_root()
    manifest = manifest_path or (root / "results" / "manifest.jsonl")
    interim = interim_dir or (root / "results" / "interim")
    figures = figures_dir or (root / "results" / "figures" / "interim")

    rows = build_summary_rows(root, manifest)
    df = pd.DataFrame(rows)
    summary_csv = interim / "summary.csv"
    summary_md = interim / "summary.md"
    _write_summary_csv(summary_csv, df)
    _write_summary_md(summary_md, df)
    figure_paths = _write_scatter_plots(df, figures)
    return InterimReportOutputs(
        summary_csv=summary_csv,
        summary_md=summary_md,
        figure_paths=figure_paths,
    )
