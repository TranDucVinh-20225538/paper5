"""Tests for cross-machine image path resolution."""

from __future__ import annotations

from pathlib import Path

from src.utils.paths import DatasetPaths, resolve_image_path


def test_resolve_image_path_by_basename(tmp_path: Path) -> None:
    isic_root = tmp_path / "ISIC_2019_Training_Input"
    pad_root = tmp_path / "pad_ufes20" / "images"
    isic_root.mkdir(parents=True)
    pad_root.mkdir(parents=True)
    isic_file = isic_root / "ISIC_0000000.jpg"
    pad_file = pad_root / "PAT_1.png"
    isic_file.write_bytes(b"jpg")
    pad_file.write_bytes(b"png")

    paths = DatasetPaths(
        csg_data_root=tmp_path,
        master_metadata=tmp_path / "master_metadata.csv",
        isic2019_root=isic_root,
        pad_ufes_root=tmp_path / "pad_ufes20",
    )

    assert resolve_image_path("/other/machine/Paper4/archive/NV/ISIC_0000000.jpg", paths) == isic_file.resolve()

    nested = isic_root / "ISIC_2019_Training_Input" / "ISIC_0000001.jpg"
    nested.parent.mkdir(parents=True, exist_ok=True)
    nested.write_bytes(b"jpg2")
    assert (
        resolve_image_path("/missing/ISIC_0000001.jpg", paths) == nested.resolve()
    )

    lesion = tmp_path / "lesion_only_images" / "ISIC_2019_Training_Input" / "ISIC_0000002.jpg"
    lesion.parent.mkdir(parents=True, exist_ok=True)
    lesion.write_bytes(b"jpg3")
    assert resolve_image_path("/x/ISIC_0000002.jpg", paths) == lesion.resolve()

    assert resolve_image_path("/other/pad_ufes20/images/PAT_1.png", paths) == pad_file.resolve()
    assert resolve_image_path(isic_file, paths) == isic_file.resolve()
    assert resolve_image_path("/missing/foo.jpg", paths) is None
