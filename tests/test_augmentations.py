"""Tests for augmentations.py."""

import numpy as np
import pytest


def _pair(h=64, w=64):
    img = np.random.rand(h, w).astype(np.float32)
    mask = np.random.randint(0, 4, (h, w), dtype=np.int32)
    return img, mask


class TestRandomFlip:
    def test_shapes_preserved(self):
        from cellseg_trainer.augmentations import random_flip

        img, mask = _pair()
        ai, am = random_flip(img, mask)
        assert ai.shape == img.shape
        assert am.shape == mask.shape

    def test_horizontal_only(self):
        from cellseg_trainer.augmentations import random_flip
        import random

        random.seed(0)
        img, mask = _pair()
        ai, am = random_flip(img, mask, horizontal=True, vertical=False)
        assert ai.shape == img.shape


class TestIntensityScale:
    def test_range(self):
        from cellseg_trainer.augmentations import intensity_scale

        img, mask = _pair()
        ai, am = intensity_scale(img, mask, scale_range=(0.5, 0.5))
        np.testing.assert_allclose(ai, img * 0.5, rtol=1e-5)
        np.testing.assert_array_equal(am, mask)


class TestGaussianNoise:
    def test_adds_noise(self):
        from cellseg_trainer.augmentations import gaussian_noise

        img = np.zeros((64, 64), dtype=np.float32)
        ai, _ = gaussian_noise(img, img.copy(), std=0.1)
        assert ai.std() > 0


class TestAugmentPair:
    def test_disabled(self):
        from cellseg_trainer.augmentations import augment_pair

        img, mask = _pair()
        ai, am = augment_pair(img, mask, {"ENABLE": False})
        np.testing.assert_array_equal(ai, img)
        np.testing.assert_array_equal(am, mask)

    def test_enabled_shapes(self):
        from cellseg_trainer.augmentations import augment_pair

        img, mask = _pair()
        cfg = {
            "ENABLE": True,
            "FLIP_HORIZONTAL": True,
            "FLIP_VERTICAL": True,
            "ROTATION": True,
            "ROTATION_RANGE": 90,
            "INTENSITY_SCALE": True,
            "INTENSITY_SCALE_RANGE": [0.9, 1.1],
            "GAUSSIAN_NOISE": True,
            "GAUSSIAN_NOISE_STD": 0.01,
            "ELASTIC_DEFORMATION": False,
        }
        ai, am = augment_pair(img, mask, cfg)
        assert ai.shape == img.shape
        assert am.shape == mask.shape
