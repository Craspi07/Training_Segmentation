"""Tests for cellpose_converter.py — no detectron2 dependency required."""

import tempfile
from pathlib import Path

import numpy as np
import pytest


def _make_seg_npy(path: Path, h: int = 64, w: int = 64, n_cells: int = 3) -> Path:
    """Write a minimal *_seg.npy file to *path*."""
    masks = np.zeros((h, w), dtype=np.int32)
    rng = np.random.default_rng(0)
    for i in range(1, n_cells + 1):
        cy, cx = rng.integers(10, h - 10), rng.integers(10, w - 10)
        r = rng.integers(4, 8)
        yy, xx = np.ogrid[:h, :w]
        masks[(yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2] = i

    seg = {"masks": masks}
    np.save(str(path), seg)
    return path


class TestConvertSegNpy:
    def test_basic(self, tmp_path):
        seg_file = _make_seg_npy(tmp_path / "test_seg.npy")
        from cellseg_trainer.cellpose_converter import convert_seg_npy

        anns, next_id = convert_seg_npy(seg_file, image_id=1, start_annotation_id=1)
        assert len(anns) >= 1
        assert next_id > 1
        for ann in anns:
            assert ann["image_id"] == 1
            assert ann["category_id"] == 1
            assert ann["area"] > 0
            assert len(ann["bbox"]) == 4
            assert ann["iscrowd"] == 0

    def test_missing_file(self):
        from cellseg_trainer.cellpose_converter import convert_seg_npy

        with pytest.raises(FileNotFoundError):
            convert_seg_npy("/nonexistent/file_seg.npy")

    def test_empty_mask(self, tmp_path):
        seg_file = tmp_path / "empty_seg.npy"
        np.save(str(seg_file), {"masks": np.zeros((32, 32), dtype=np.int32)})
        from cellseg_trainer.cellpose_converter import convert_seg_npy

        anns, next_id = convert_seg_npy(seg_file)
        assert anns == []
        assert next_id == 1

    def test_convert_mask_array(self):
        from cellseg_trainer.cellpose_converter import convert_mask_array

        mask = np.zeros((64, 64), dtype=np.int32)
        mask[10:20, 10:20] = 1
        mask[30:45, 30:45] = 2
        anns, _ = convert_mask_array(mask, image_id=5)
        assert len(anns) == 2
        assert all(a["image_id"] == 5 for a in anns)
