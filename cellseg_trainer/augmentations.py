"""
augmentations.py — Data augmentation for microscopy images.

Supports: rotation, flips, intensity scaling, Gaussian noise, elastic deformation.
All transforms operate on (image, mask) pairs maintaining spatial consistency.
"""

from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual transforms
# ---------------------------------------------------------------------------


def random_flip(
    image: np.ndarray,
    mask: np.ndarray,
    horizontal: bool = True,
    vertical: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly flip image and mask.

    Args:
        image: 2-D or 3-D float array.
        mask: 2-D integer or float array.
        horizontal: Enable horizontal flip.
        vertical: Enable vertical flip.

    Returns:
        Augmented (image, mask) pair.
    """
    if horizontal and random.random() > 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)
    if vertical and random.random() > 0.5:
        image = np.flipud(image)
        mask = np.flipud(mask)
    return image.copy(), mask.copy()


def random_rotation(
    image: np.ndarray,
    mask: np.ndarray,
    max_angle: float = 360.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate image and mask by a random angle using scipy.

    Args:
        image: 2-D float array (H, W).
        mask: 2-D integer array (H, W).
        max_angle: Maximum rotation angle in degrees.

    Returns:
        Rotated (image, mask) pair.
    """
    try:
        from scipy.ndimage import rotate  # type: ignore
    except ImportError as exc:
        raise ImportError("scipy is required: pip install scipy") from exc

    angle = random.uniform(-max_angle / 2, max_angle / 2)
    rot_img = rotate(image, angle, reshape=False, order=1, mode="reflect")
    rot_mask = rotate(mask.astype(np.float32), angle, reshape=False, order=0, mode="constant")
    return rot_img.astype(image.dtype), rot_mask.astype(mask.dtype)


def intensity_scale(
    image: np.ndarray,
    mask: np.ndarray,
    scale_range: tuple[float, float] = (0.8, 1.2),
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly scale image intensity.

    Args:
        image: Float or integer image array.
        mask: Mask (unchanged).
        scale_range: (min_scale, max_scale) for the multiplier.

    Returns:
        Scaled (image, mask) pair.
    """
    scale = random.uniform(*scale_range)
    result = image.astype(np.float32) * scale
    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        result = np.clip(result, info.min, info.max)
    return result.astype(image.dtype), mask


def gaussian_noise(
    image: np.ndarray,
    mask: np.ndarray,
    std: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Add Gaussian noise to the image.

    Args:
        image: Float or integer image array.
        mask: Mask (unchanged).
        std: Standard deviation of the noise as a fraction of the image value range.

    Returns:
        Noisy (image, mask) pair.
    """
    img_f32 = image.astype(np.float32)
    value_range = float(img_f32.max() - img_f32.min())
    # Scale std relative to the actual data range so it has visible effect
    noise_std = std * value_range if value_range > 0 else std
    noise = np.random.normal(0.0, noise_std, image.shape).astype(np.float32)
    result = img_f32 + noise
    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        result = np.clip(result, info.min, info.max)
    return result.astype(image.dtype), mask


def elastic_deformation(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 50.0,
    sigma: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply elastic deformation using random displacement fields.

    Args:
        image: 2-D float array (H, W).
        mask: 2-D integer array (H, W).
        alpha: Displacement field scaling factor (controls deformation magnitude).
        sigma: Gaussian smoothing sigma (controls deformation smoothness).

    Returns:
        Deformed (image, mask) pair.
    """
    try:
        from scipy.ndimage import gaussian_filter, map_coordinates  # type: ignore
    except ImportError as exc:
        raise ImportError("scipy is required: pip install scipy") from exc

    h, w = image.shape[:2]
    dx = gaussian_filter(np.random.randn(h, w), sigma) * alpha
    dy = gaussian_filter(np.random.randn(h, w), sigma) * alpha

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coords = [yy + dy, xx + dx]

    deformed_img = map_coordinates(image, coords, order=1, mode="reflect").astype(image.dtype)
    deformed_mask = map_coordinates(
        mask.astype(np.float32), coords, order=0, mode="constant"
    ).astype(mask.dtype)
    return deformed_img, deformed_mask


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def augment_pair(
    image: np.ndarray,
    mask: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a configurable augmentation pipeline to an image/mask pair.

    Args:
        image: Float image array (H, W).
        mask: Integer mask array (H, W).
        config: Augmentation config dict (``AUGMENTATION`` section from YAML).

    Returns:
        Augmented (image, mask) pair.
    """
    if not config.get("ENABLE", True):
        return image, mask

    if config.get("FLIP_HORIZONTAL", True) or config.get("FLIP_VERTICAL", True):
        image, mask = random_flip(
            image, mask,
            horizontal=config.get("FLIP_HORIZONTAL", True),
            vertical=config.get("FLIP_VERTICAL", True),
        )

    if config.get("ROTATION", True):
        max_angle = config.get("ROTATION_RANGE", 360)
        image, mask = random_rotation(image, mask, max_angle=max_angle)

    if config.get("INTENSITY_SCALE", True):
        scale_range = tuple(config.get("INTENSITY_SCALE_RANGE", [0.8, 1.2]))
        image, mask = intensity_scale(image, mask, scale_range=scale_range)

    if config.get("GAUSSIAN_NOISE", True):
        std = config.get("GAUSSIAN_NOISE_STD", 0.05)
        image, mask = gaussian_noise(image, mask, std=std)

    if config.get("ELASTIC_DEFORMATION", False):
        alpha = config.get("ELASTIC_ALPHA", 50.0)
        sigma = config.get("ELASTIC_SIGMA", 5.0)
        try:
            image, mask = elastic_deformation(image, mask, alpha=alpha, sigma=sigma)
        except Exception as exc:
            logger.warning("Elastic deformation failed: %s", exc)

    return image, mask


def create_augmented_dataset(
    images: list[np.ndarray],
    masks: list[np.ndarray],
    config: dict[str, Any],
    copies_per_image: int = 4,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Generate augmented copies of each image/mask pair.

    Args:
        images: List of float image arrays.
        masks: List of integer mask arrays.
        config: Augmentation config dict.
        copies_per_image: Number of augmented copies per original.

    Returns:
        Combined list of (original + augmented) images and masks.
    """
    aug_images = list(images)
    aug_masks = list(masks)

    for img, msk in zip(images, masks):
        for _ in range(copies_per_image):
            a_img, a_msk = augment_pair(img, msk, config)
            aug_images.append(a_img)
            aug_masks.append(a_msk)

    logger.info(
        "Augmented dataset: %d originals → %d total",
        len(images),
        len(aug_images),
    )
    return aug_images, aug_masks
