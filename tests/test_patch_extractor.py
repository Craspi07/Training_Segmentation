"""Tests for patch_extractor.py."""

import numpy as np
import pytest


class TestExtractPatches:
    def test_single_patch_small_image(self):
        from cellseg_trainer.patch_extractor import extract_patches

        img = np.ones((64, 64), dtype=np.float32)
        patches = list(extract_patches(img, patch_size=128, overlap=0))
        assert len(patches) == 1
        patch, bbox = patches[0]
        assert patch.shape == (64, 64)

    def test_multiple_patches(self):
        from cellseg_trainer.patch_extractor import extract_patches

        img = np.ones((256, 256), dtype=np.float32)
        patches = list(extract_patches(img, patch_size=128, overlap=0))
        assert len(patches) == 4  # 2×2

    def test_overlap(self):
        from cellseg_trainer.patch_extractor import extract_patches

        img = np.ones((256, 256), dtype=np.float32)
        patches_no_overlap = list(extract_patches(img, patch_size=128, overlap=0))
        patches_with_overlap = list(extract_patches(img, patch_size=128, overlap=32))
        assert len(patches_with_overlap) > len(patches_no_overlap)

    def test_invalid_overlap(self):
        from cellseg_trainer.patch_extractor import extract_patches

        with pytest.raises(ValueError):
            list(extract_patches(np.ones((64, 64)), patch_size=32, overlap=32))

    def test_3d_image(self):
        from cellseg_trainer.patch_extractor import extract_patches

        img = np.ones((3, 256, 256), dtype=np.float32)
        patches = list(extract_patches(img, patch_size=128, overlap=0))
        assert len(patches) == 4
        for patch, _ in patches:
            assert patch.shape[0] == 3


class TestReconstructFromPatches:
    def test_round_trip(self):
        from cellseg_trainer.patch_extractor import extract_patches, reconstruct_from_patches

        img = np.random.rand(128, 128).astype(np.float32)
        patches_and_bboxes = list(extract_patches(img, patch_size=64, overlap=0))
        patches = [p for p, _ in patches_and_bboxes]
        bboxes = [b for _, b in patches_and_bboxes]
        recon = reconstruct_from_patches(patches, bboxes, (128, 128))
        np.testing.assert_allclose(recon, img, atol=1e-5)
