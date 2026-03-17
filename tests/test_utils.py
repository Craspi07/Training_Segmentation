"""Tests for utils.py."""

import numpy as np
import pytest


class TestNormalizeUint8:
    def test_float_range(self):
        from cellseg_trainer.utils import normalize_uint8

        arr = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        out = normalize_uint8(arr)
        assert out.dtype == np.uint8
        assert out.min() == 0
        assert out.max() == 255

    def test_constant_array(self):
        from cellseg_trainer.utils import normalize_uint8

        # Constant input: hi == lo so no scaling, value cast to uint8 as-is
        arr = np.ones((4, 4), dtype=np.float32) * 42
        out = normalize_uint8(arr)
        assert out.dtype == np.uint8
        assert out.min() == out.max()


class TestInstanceToBinary:
    def test_conversion(self):
        from cellseg_trainer.utils import instance_to_binary

        mask = np.array([[0, 1, 2], [0, 3, 0]], dtype=np.int32)
        binary = instance_to_binary(mask)
        expected = np.array([[0, 1, 1], [0, 1, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(binary, expected)


class TestCollectImages:
    def test_empty_dir(self, tmp_path):
        from cellseg_trainer.utils import collect_images

        result = collect_images(tmp_path)
        assert result == []

    def test_finds_tif(self, tmp_path):
        from cellseg_trainer.utils import collect_images

        (tmp_path / "a.tif").touch()
        (tmp_path / "b.png").touch()
        (tmp_path / "ignore.txt").touch()
        result = collect_images(tmp_path)
        names = {p.name for p in result}
        assert "a.tif" in names
        assert "b.png" in names
        assert "ignore.txt" not in names


class TestEnsureDir:
    def test_creates_nested(self, tmp_path):
        from cellseg_trainer.utils import ensure_dir

        target = tmp_path / "a" / "b" / "c"
        result = ensure_dir(target)
        assert result.is_dir()
