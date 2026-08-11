"""Load and validate per-backbone YAML configs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# D-021: extend r grid for high-dimensional backbones
_HIGH_DIM_R_RUNG = 256
_HIGH_DIM_THRESHOLD = 1536


@dataclass(frozen=True)
class BackboneConfig:
    """Parsed backbone config with fields the pipeline injects into modules."""

    raw: dict[str, Any]
    path: Path
    name: str
    family: str
    embed_dim: int
    activation: str
    representation_status: str
    pooling: str | None
    preprocessing_asset: Path
    preprocessing_sha256: str | None
    intervention_r: int | None
    intervention_lambda_proj: float | None
    grid_r: list[int]
    grid_lambda_proj: list[float]
    seeds: list[int]

    @property
    def is_representation_resolved(self) -> bool:
        return self.representation_status.upper() == "RESOLVED"

    @property
    def is_probe(self) -> bool:
        return self.family.upper() == "PROBE"


def _repo_root(start: Path | None = None) -> Path:
    """Find repo root by walking up from start until ONE_PAGE_SUMMARY.md exists."""
    here = (start or Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        if (parent / "ONE_PAGE_SUMMARY.md").is_file():
            return parent
    raise FileNotFoundError("Could not locate repo root (ONE_PAGE_SUMMARY.md)")


def _require_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"config missing required mapping: {key}")
    return value


def _require_int(data: dict[str, Any], key: str, *, section: str) -> int:
    value = data.get(key)
    if value is None:
        raise ValueError(f"{section}.{key} is null — pipeline must refuse to run")
    if not isinstance(value, int):
        raise TypeError(f"{section}.{key} must be int, got {type(value).__name__}")
    return value


def _resolve_preprocessing_asset(asset: str, repo_root: Path) -> Path:
    path = Path(asset)
    if not path.is_absolute():
        path = repo_root / path
    return path


def effective_r_grid(embed_dim: int, grid_r: list[int]) -> list[int]:
    """
    Apply D-021: include r=256 when embed_dim >= 1536.

    Base grid from config is preserved; 256 is appended if absent.
    """
    rungs = list(grid_r)
    if embed_dim >= _HIGH_DIM_THRESHOLD and _HIGH_DIM_R_RUNG not in rungs:
        rungs.append(_HIGH_DIM_R_RUNG)
    return sorted(rungs)


def load_backbone_config(
    config_path: Path | str,
    *,
    repo_root: Path | None = None,
) -> BackboneConfig:
    """Load configs/<backbone>.yaml and validate required fields."""
    path = Path(config_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")

    root = repo_root or _repo_root(path)
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping: {path}")

    bb = _require_mapping(raw, "backbone")
    prep = _require_mapping(raw, "preprocessing")
    intervention = _require_mapping(raw, "intervention")
    grid = _require_mapping(intervention, "grid")

    name = bb.get("name")
    family = bb.get("family")
    activation = bb.get("activation")
    if not name or not family or not activation:
        raise ValueError(f"backbone.name, family, and activation are required in {path}")

    embed_dim = _require_int(bb, "embed_dim", section="backbone")
    rep = _require_mapping(bb, "representation")
    rep_status = rep.get("status")
    if not rep_status:
        raise ValueError(f"backbone.representation.status is required in {path}")

    asset = prep.get("asset")
    if not asset:
        raise ValueError(f"preprocessing.asset is required in {path}")

    grid_r = grid.get("r")
    grid_lambda = grid.get("lambda_proj")
    if not isinstance(grid_r, list) or not grid_r:
        raise ValueError(f"intervention.grid.r must be a non-empty list in {path}")
    if not isinstance(grid_lambda, list) or not grid_lambda:
        raise ValueError(f"intervention.grid.lambda_proj must be a non-empty list in {path}")

    seeds = raw.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError(f"seeds must be a non-empty list in {path}")

    r_val = intervention.get("r")
    lambda_val = intervention.get("lambda_proj")
    if r_val is not None and not isinstance(r_val, int):
        raise TypeError("intervention.r must be int or null")
    if lambda_val is not None and not isinstance(lambda_val, (int, float)):
        raise TypeError("intervention.lambda_proj must be number or null")

    return BackboneConfig(
        raw=raw,
        path=path,
        name=str(name),
        family=str(family),
        embed_dim=embed_dim,
        activation=str(activation),
        representation_status=str(rep_status),
        pooling=rep.get("pooling"),
        preprocessing_asset=_resolve_preprocessing_asset(str(asset), root),
        preprocessing_sha256=prep.get("sha256"),
        intervention_r=r_val,
        intervention_lambda_proj=float(lambda_val) if lambda_val is not None else None,
        grid_r=effective_r_grid(embed_dim, [int(x) for x in grid_r]),
        grid_lambda_proj=[float(x) for x in grid_lambda],
        seeds=[int(s) for s in seeds],
    )


def validate_for_pipeline_run(cfg: BackboneConfig) -> None:
    """
    Hard stops from protocol Step 0 — raise before any compute.

    Does not verify preprocessing hash (see verify_preprocessing_hash).
    """
    if not cfg.is_representation_resolved:
        raise ValueError(
            f"{cfg.name}: representation.status is {cfg.representation_status!r}; "
            "extraction and training refuse to run on UNRESOLVED"
        )

    if cfg.pooling == "gap_masked" and cfg.name != "medsam":
        raise ValueError(
            f"{cfg.name}: gap_masked pooling is reserved for MedSAM (D-026)"
        )
