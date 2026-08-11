"""Tests for preprocessing asset hashing (protocol Step 0)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.preprocessing import sha256_file, verify_preprocessing_hash


def test_sha256_file_deterministic(test_preprocessing_asset: Path) -> None:
    h1 = sha256_file(test_preprocessing_asset)
    h2 = sha256_file(test_preprocessing_asset)
    assert h1 == h2
    assert len(h1) == 64


def test_verify_preprocessing_hash_match(test_preprocessing_asset: Path) -> None:
    expected = sha256_file(test_preprocessing_asset)
    assert verify_preprocessing_hash(test_preprocessing_asset, expected) == expected


def test_verify_preprocessing_hash_mismatch(test_preprocessing_asset: Path) -> None:
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_preprocessing_hash(test_preprocessing_asset, "0" * 64)


def test_verify_preprocessing_hash_null_required(test_preprocessing_asset: Path) -> None:
    with pytest.raises(ValueError, match="sha256 is null"):
        verify_preprocessing_hash(test_preprocessing_asset, None, require_hash=True)


def test_verify_preprocessing_hash_null_allowed(test_preprocessing_asset: Path) -> None:
    computed = verify_preprocessing_hash(
        test_preprocessing_asset, None, require_hash=False
    )
    assert len(computed) == 64
