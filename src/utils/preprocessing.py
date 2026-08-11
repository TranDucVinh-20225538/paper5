"""SHA-256 verification for frozen preprocessing assets (protocol Step 0/1)."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path | str) -> str:
    """Return lowercase hex SHA-256 of file contents."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Preprocessing asset not found: {p}")
    digest = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_preprocessing_hash(
    asset_path: Path | str,
    expected_sha256: str | None,
    *,
    require_hash: bool = True,
) -> str:
    """
    Verify config sha256 matches the asset on disk.

    Returns the computed hash. Raises if require_hash and expected is missing/mismatch.
    """
    computed = sha256_file(asset_path)
    if expected_sha256 is None:
        if require_hash:
            raise ValueError(
                f"preprocessing.sha256 is null for {asset_path}; "
                "freeze the asset and record the hash before running"
            )
        return computed

    expected = expected_sha256.strip().lower()
    if computed != expected:
        raise ValueError(
            f"Preprocessing hash mismatch for {asset_path}: "
            f"expected {expected}, got {computed}"
        )
    return computed
