"""Tests for dataset path resolution and split checksum verification."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.utils.paths import (
    load_dataset_paths,
    load_split_checksum_spec,
    verify_master_metadata_checksum,
)


def test_split_checksum_spec_loads(repo_root: Path) -> None:
    spec = load_split_checksum_spec(repo_root)
    assert spec["sha256"] == "7841f7324129881a9b648051fe110f14768f7708c37c602b7f378fe34a3289ce"
    assert spec["hashed_columns_in_order"] == ["image_id", "label_idx", "domain", "partition"]
    assert spec["partition_counts"]["isic_train"] == 16211
    assert spec["row_count"] == 23576


def test_load_dataset_paths_from_env(repo_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csg = Path("/tmp/csg-data-test")
    monkeypatch.setenv("CSG_DATA_ROOT", str(csg))
    paths = load_dataset_paths(repo_root=repo_root)
    assert paths.csg_data_root == csg.resolve()
    assert paths.master_metadata == csg.resolve() / "master_metadata.csv"


def test_load_dataset_paths_csg_root_alias(repo_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CSG_DATA_ROOT", raising=False)
    monkeypatch.setenv("CSG_ROOT", "/tmp/csg-root-alias")
    paths = load_dataset_paths(repo_root=repo_root)
    assert paths.csg_data_root == Path("/tmp/csg-root-alias").resolve()


@pytest.mark.skipif(
    not Path(os.environ.get("CSG_DATA_ROOT", "/Users/cubo/Research/CSG-SKin/data")).joinpath(
        "master_metadata.csv"
    ).is_file(),
    reason="CSG dataset not available on this machine",
)
def test_verify_master_metadata_matches_paper4(repo_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csg = os.environ.get("CSG_DATA_ROOT", "/Users/cubo/Research/CSG-SKin/data")
    monkeypatch.setenv("CSG_DATA_ROOT", csg)
    paths = load_dataset_paths(repo_root=repo_root)
    digest = verify_master_metadata_checksum(paths, repo_root=repo_root)
    assert digest == load_split_checksum_spec(repo_root)["sha256"]
