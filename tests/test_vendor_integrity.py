"""Verify vendored modules carry pinned commit metadata (REUSE.md §2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.vendor_metadata import VENDOR_RECORDS

REPO_ROOT = Path(__file__).resolve().parents[1]

VENDORED_FILES = {
    "condition_number": REPO_ROOT / "src" / "geometry" / "condition_number.py",
    "compute_mahalanobis_params_from_arrays": REPO_ROOT / "src" / "estimators" / "mahalanobis.py",
}


@pytest.mark.parametrize("symbol", VENDOR_RECORDS.keys())
def test_vendor_metadata_has_commit(symbol: str) -> None:
    record = VENDOR_RECORDS[symbol]
    assert len(record["source_commit"]) == 40
    assert record["source_repo"] in {"paper-3", "CSG-SKin"}


@pytest.mark.parametrize("symbol,path", VENDORED_FILES.items())
def test_vendored_file_contains_commit_sha(symbol: str, path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    commit = VENDOR_RECORDS[symbol]["source_commit"]
    assert commit in text, f"{path} must pin source commit {commit}"


@pytest.mark.parametrize("symbol,path", VENDORED_FILES.items())
def test_vendored_file_documents_source_path(symbol: str, path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    source_path = VENDOR_RECORDS[symbol]["source_path"]
    assert source_path in text


def test_no_syspath_hacks_in_vendored_modules() -> None:
    for path in VENDORED_FILES.values():
        text = path.read_text(encoding="utf-8")
        assert "sys.path" not in text
