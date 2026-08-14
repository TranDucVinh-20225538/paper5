"""D-050: pin the split by hashing assignments, not the raw CSV.

The Papers 1–4 ``master_metadata.csv`` carries a ``path`` column of absolute
machine-local paths, so a whole-file SHA-256 cannot pass on a second machine
and gets bypassed. This module hashes only split-relevant fields.

Payload (this is the contract; also recorded in datasets/checksums/split_seed42.sha256):

  columns, in order: image_id, label_idx, domain, partition
  sorted by:         image_id (Unicode code-point / Python str order)
  encoding:          UTF-8
  field separator:   U+0009 TAB
  row separator:     U+000A LF
  trailing newline:  yes (after the last row)
  label_idx:         decimal integer, no sign, no padding
  partition:         isic_train | isic_test | pad_ufes
  image_id:          basename of the image path (filename only)

isic_val is unused by Paper 5 extraction and is absent from the PanDerm frozen
artifact, so it is not in the digest. The digest is the SHA-256 of the payload
bytes, lowercase hex.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from src.datasets.splits import build_eval_pool_df, build_isic_train_df

HASHED_COLUMNS = ("image_id", "label_idx", "domain", "partition")
FIELD_SEP = "\t"
ROW_SEP = "\n"
PARTITIONS = ("isic_train", "isic_test", "pad_ufes")


def image_id_from_path(path: str | Path) -> str:
    return Path(str(path)).name


def assignments_from_train_eval(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the four-column assignment table from train + eval frames."""
    rows: list[dict[str, Any]] = []
    for rec in train_df.itertuples(index=False):
        domain = str(rec.domain)
        if domain != "isic":
            raise ValueError(f"ISIC train row has domain={domain!r}")
        rows.append(
            {
                "image_id": image_id_from_path(rec.path),
                "label_idx": int(rec.label_idx),
                "domain": domain,
                "partition": "isic_train",
            }
        )
    for rec in eval_df.itertuples(index=False):
        domain = str(rec.domain)
        if domain == "isic":
            partition = "isic_test"
        elif domain == "pad_ufes":
            partition = "pad_ufes"
        else:
            raise ValueError(f"Eval row has unexpected domain={domain!r}")
        rows.append(
            {
                "image_id": image_id_from_path(rec.path),
                "label_idx": int(rec.label_idx),
                "domain": domain,
                "partition": partition,
            }
        )
    table = pd.DataFrame(rows, columns=list(HASHED_COLUMNS))
    if table["image_id"].duplicated().any():
        dup = table.loc[table["image_id"].duplicated(), "image_id"].tolist()[:5]
        raise ValueError(f"Duplicate image_id in split assignments: {dup}")
    return table


def assignments_from_master_metadata(metadata_csv: Path | str) -> pd.DataFrame:
    """Assignments via the preregistered split (does not change the split)."""
    train_df = build_isic_train_df(metadata_csv)
    eval_df = build_eval_pool_df(metadata_csv)
    return assignments_from_train_eval(train_df, eval_df)


def assignments_from_embedding_metadata(
    train_metadata_csv: Path | str,
    eval_metadata_csv: Path | str,
) -> pd.DataFrame:
    """Assignments recorded on a frozen embedding artifact (e.g. Paper 4 PanDerm)."""
    train_df = pd.read_csv(train_metadata_csv)
    eval_df = pd.read_csv(eval_metadata_csv)
    for frame, name in ((train_df, "train"), (eval_df, "eval")):
        missing = {"path", "label_idx", "domain"}.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} embedding metadata missing columns: {sorted(missing)}")
    return assignments_from_train_eval(train_df, eval_df)


def canonical_payload(table: pd.DataFrame) -> bytes:
    """Stable UTF-8 serialisation of the assignment table. No paths, no floats."""
    if list(table.columns) != list(HASHED_COLUMNS):
        raise ValueError(f"columns must be {HASHED_COLUMNS}, got {list(table.columns)}")
    ordered = table.sort_values("image_id", kind="mergesort").reset_index(drop=True)
    lines: list[str] = []
    for rec in ordered.itertuples(index=False):
        image_id = str(rec.image_id)
        label_idx = int(rec.label_idx)
        domain = str(rec.domain)
        partition = str(rec.partition)
        if FIELD_SEP in image_id or ROW_SEP in image_id:
            raise ValueError(f"image_id contains a separator: {image_id!r}")
        if partition not in PARTITIONS:
            raise ValueError(f"unexpected partition {partition!r}")
        lines.append(
            FIELD_SEP.join((image_id, str(label_idx), domain, partition))
        )
    return (ROW_SEP.join(lines) + ROW_SEP).encode("utf-8")


def digest_assignments(table: pd.DataFrame) -> str:
    return hashlib.sha256(canonical_payload(table)).hexdigest()


def compute_split_assignment_digest_from_master(metadata_csv: Path | str) -> str:
    return digest_assignments(assignments_from_master_metadata(metadata_csv))


def compute_split_assignment_digest_from_embeddings(
    train_metadata_csv: Path | str,
    eval_metadata_csv: Path | str,
) -> str:
    return digest_assignments(
        assignments_from_embedding_metadata(train_metadata_csv, eval_metadata_csv)
    )
