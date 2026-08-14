"""D-050: split pin hashes assignments, not raw CSV bytes."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from src.datasets.split_pin import (
    assignments_from_train_eval,
    canonical_payload,
    compute_split_assignment_digest_from_embeddings,
    compute_split_assignment_digest_from_master,
    digest_assignments,
)
from src.utils.paths import (
    DatasetPaths,
    load_split_checksum_spec,
    verify_master_metadata_checksum,
)

_PANDERM_TRAIN_META = (
    Path(__file__).resolve().parents[1]
    / "reference_embeddings"
    / "ReferenceTrainEmbedding"
    / "metadata.csv"
)


def _tiny_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(
        {
            "path": ["/x/ISIC_aaa.jpg", "/x/ISIC_bbb.jpg"],
            "label_idx": [0, 1],
            "domain": ["isic", "isic"],
        }
    )
    eval_df = pd.DataFrame(
        {
            "path": ["/x/ISIC_ccc.jpg", "/x/PAT_1.png"],
            "label_idx": [2, 3],
            "domain": ["isic", "pad_ufes"],
        }
    )
    return train, eval_df


def test_digest_independent_of_row_order() -> None:
    train, eval_df = _tiny_frames()
    d0 = digest_assignments(assignments_from_train_eval(train, eval_df))
    d1 = digest_assignments(
        assignments_from_train_eval(train.iloc[::-1].reset_index(drop=True), eval_df)
    )
    assert d0 == d1


def test_digest_independent_of_path_prefix() -> None:
    train, eval_df = _tiny_frames()
    d0 = digest_assignments(assignments_from_train_eval(train, eval_df))
    train2 = train.copy()
    train2["path"] = ["/other/machine/ISIC_aaa.jpg", "/other/machine/ISIC_bbb.jpg"]
    d1 = digest_assignments(assignments_from_train_eval(train2, eval_df))
    assert d0 == d1


def test_digest_changes_if_domain_flipped() -> None:
    train, eval_df = _tiny_frames()
    d0 = digest_assignments(assignments_from_train_eval(train, eval_df))
    flipped = eval_df.copy()
    flipped.loc[0, "domain"] = "pad_ufes"
    d1 = digest_assignments(assignments_from_train_eval(train, flipped))
    assert d0 != d1


def test_digest_changes_if_id_moves_between_train_and_eval() -> None:
    train, eval_df = _tiny_frames()
    d0 = digest_assignments(assignments_from_train_eval(train, eval_df))
    moved = train.iloc[[0]].copy()
    rest = train.iloc[1:].reset_index(drop=True)
    eval2 = pd.concat([eval_df, moved], ignore_index=True)
    d1 = digest_assignments(assignments_from_train_eval(rest, eval2))
    assert d0 != d1


def test_payload_is_stable_ascii_tsv() -> None:
    train, eval_df = _tiny_frames()
    payload = canonical_payload(assignments_from_train_eval(train, eval_df))
    text = payload.decode("utf-8")
    assert "\r" not in text
    assert text.endswith("\n")
    lines = text.strip().split("\n")
    assert lines == sorted(lines)
    assert lines[0] == "ISIC_aaa.jpg\t0\tisic\tisic_train"


def test_verify_rejects_a_different_assignment(
    repo_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.datasets.constants import LABELS

    rows = []
    for i in range(400):
        rows.append({"path": f"/tmp/isic_{i}.jpg", "label": LABELS[i % 8], "domain": "isic"})
    for i in range(80):
        rows.append({"path": f"/tmp/pad_{i}.jpg", "label": "NV", "domain": "pad_ufes"})
    meta = tmp_path / "master_metadata.csv"
    pd.DataFrame(rows).to_csv(meta, index=False)

    monkeypatch.setattr("src.datasets.splits._optional_dataset_paths", lambda: None)
    monkeypatch.setattr("src.datasets.splits.os.path.isfile", lambda p: True)

    paths = DatasetPaths(
        csg_data_root=tmp_path,
        master_metadata=meta,
        isic2019_root=tmp_path,
        pad_ufes_root=tmp_path,
    )
    with pytest.raises(ValueError, match="split assignment digest mismatch"):
        verify_master_metadata_checksum(paths, repo_root=repo_root)


@pytest.mark.skipif(
    not Path(os.environ.get("CSG_DATA_ROOT", "")).joinpath("master_metadata.csv").is_file()
    or not _PANDERM_TRAIN_META.is_file(),
    reason="Live CSG master_metadata or PanDerm frozen embeddings not available",
)
def test_live_master_and_panderm_embeddings_share_digest(repo_root: Path) -> None:
    spec = load_split_checksum_spec(repo_root)
    live = Path(os.environ["CSG_DATA_ROOT"]) / "master_metadata.csv"
    d_live = compute_split_assignment_digest_from_master(live)
    d_p = compute_split_assignment_digest_from_embeddings(
        repo_root / "reference_embeddings" / "ReferenceTrainEmbedding" / "metadata.csv",
        repo_root / "reference_embeddings" / "ReferenceEmbedding" / "metadata.csv",
    )
    assert d_live == d_p == spec["sha256"]
