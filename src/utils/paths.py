"""Dataset path resolution — env or gitignored local file only.

CSG_DATA_ROOT / CSG_ROOT are for **dataset files only** (images, master_metadata.csv).
They must not be used to import CSG-SKin Python code — vendor into src/ instead (REUSE.md).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from src.utils.config import find_repo_root
from src.utils.preprocessing import sha256_file


@dataclass(frozen=True)
class DatasetPaths:
    """Resolved on-disk locations for Papers 1–4 dataset artifacts."""

    csg_data_root: Path
    master_metadata: Path
    isic2019_root: Path
    pad_ufes_root: Path

    def verify_layout(self) -> None:
        """Raise if expected dataset files are missing."""
        if not self.master_metadata.is_file():
            raise FileNotFoundError(f"master_metadata.csv not found: {self.master_metadata}")
        if not self.isic2019_root.is_dir():
            raise FileNotFoundError(f"ISIC 2019 root not found: {self.isic2019_root}")
        if not self.pad_ufes_root.is_dir():
            raise FileNotFoundError(f"PAD-UFES-20 root not found: {self.pad_ufes_root}")


def _parse_paths_local(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _resolve_csg_data_root(repo_root: Path) -> Path | None:
    local = _parse_paths_local(repo_root / "datasets" / "paths.local")
    for key in ("CSG_DATA_ROOT", "CSG_ROOT"):
        if key in local:
            return Path(local[key]).expanduser().resolve()
        env_val = os.environ.get(key)
        if env_val:
            return Path(env_val).expanduser().resolve()

    research = os.environ.get("RESEARCH_ROOT")
    if research:
        candidate = Path(research).expanduser().resolve() / "CSG-SKin" / "data"
        if candidate.is_dir():
            return candidate
    return None


def resolve_image_path(raw: str | Path, paths: DatasetPaths) -> Path | None:
    """
    Map metadata ``path`` values to on-disk files under CSG_DATA_ROOT.

    master_metadata.csv may store absolute paths from another machine (e.g. Paper4
    archive on a Mac). When the literal path is missing, fall back to basename
    lookup under the configured ISIC / PAD image roots.
    """
    p = Path(raw).expanduser()
    if p.is_file():
        return p.resolve()

    name = p.name
    candidates = (
        paths.isic2019_root / name,
        paths.pad_ufes_root / "images" / name,
        paths.pad_ufes_root / name,
        paths.csg_data_root / name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def load_dataset_paths(*, repo_root: Path | None = None) -> DatasetPaths:
    """
    Resolve dataset paths from datasets/paths.local or environment variables.

    Precedence: datasets/paths.local > CSG_DATA_ROOT > CSG_ROOT > $RESEARCH_ROOT/CSG-SKin/data
    """
    root = repo_root or find_repo_root()
    csg_root = _resolve_csg_data_root(root)
    if csg_root is None:
        raise FileNotFoundError(
            "Dataset paths not configured. Set CSG_DATA_ROOT (or CSG_ROOT) to CSG-SKin/data, "
            "or copy datasets/paths.local.example to datasets/paths.local"
        )
    return DatasetPaths(
        csg_data_root=csg_root,
        master_metadata=csg_root / "master_metadata.csv",
        isic2019_root=csg_root / "ISIC_2019_Training_Input",
        pad_ufes_root=csg_root / "pad_ufes20",
    )


def load_panderm_embeddings_root(*, repo_root: Path | None = None) -> Path:
    """Paper 4 frozen PanDerm embeddings (separate from CSG dataset root)."""
    root = repo_root or find_repo_root()
    local = _parse_paths_local(root / "datasets" / "paths.local")
    for key in ("PANDERM_EMBEDDINGS_ROOT",):
        if key in local:
            return Path(local[key]).expanduser().resolve()
        env_val = os.environ.get(key)
        if env_val:
            return Path(env_val).expanduser().resolve()

    research = os.environ.get("RESEARCH_ROOT")
    if research:
        candidate = (
            Path(research).expanduser().resolve()
            / "Paper4"
            / "PhaseB"
            / "assets"
            / "reference_embeddings"
        )
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        "PANDERM_EMBEDDINGS_ROOT not set. Point at Paper4/PhaseB/assets/reference_embeddings/"
    )


def load_split_checksum_spec(repo_root: Path | None = None) -> dict:
    root = repo_root or find_repo_root()
    spec_path = root / "datasets" / "checksums" / "split_seed42.sha256"
    if not spec_path.is_file():
        raise FileNotFoundError(f"Split checksum spec missing: {spec_path}")
    return json.loads(spec_path.read_text(encoding="utf-8"))


def verify_master_metadata_checksum(
    paths: DatasetPaths,
    *,
    repo_root: Path | None = None,
) -> str:
    """Verify live master_metadata.csv matches the pinned Papers 1–4 split."""
    spec = load_split_checksum_spec(repo_root)
    expected = spec["sha256"]
    actual = sha256_file(paths.master_metadata)
    if actual != expected:
        raise ValueError(
            f"master_metadata.csv sha256 mismatch: expected {expected}, got {actual}. "
            "Do not regenerate — use the Papers 1–4 artifact."
        )
    return actual
