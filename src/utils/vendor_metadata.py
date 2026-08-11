"""Pinned provenance for vendored functions (REUSE.md §2, decision D-028)."""

from __future__ import annotations

VENDOR_RECORDS: dict[str, dict[str, str]] = {
    "condition_number": {
        "source_repo": "paper-3",
        "source_path": "scripts/geometry_diagnostics.py",
        "source_commit": "5fedcb3870b1eb17d15c36f79cc11421e7067522",
        "target_module": "src.geometry.condition_number",
    },
    "compute_mahalanobis_params_from_arrays": {
        "source_repo": "CSG-SKin",
        "source_path": "src/utils/ood_metrics.py",
        "source_commit": "12338983d87a35b3ad040687ad09f04908fb0c55",
        "target_module": "src.estimators.mahalanobis",
    },
}
