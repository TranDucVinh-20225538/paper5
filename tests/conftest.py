"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def panderm_config_path() -> Path:
    return CONFIGS_DIR / "panderm.yaml"


@pytest.fixture
def resnet50_config_path() -> Path:
    return CONFIGS_DIR / "resnet50.yaml"


@pytest.fixture
def medsam_config_path() -> Path:
    return CONFIGS_DIR / "medsam.yaml"


@pytest.fixture
def test_preprocessing_asset() -> Path:
    return FIXTURES_DIR / "preprocessing" / "test_backbone.json"


@pytest.fixture
def fixture_config() -> Path:
    return FIXTURES_DIR / "configs" / "fixture_backbone.yaml"
