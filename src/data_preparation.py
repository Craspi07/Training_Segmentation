"""
Data preparation utilities for the segmentation training pipeline.

Handles loading images/masks, applying augmentations defined in the
BiaPy-style YAML config, and organizing data for Cellpose training.
"""

import os
import glob
import logging
from pathlib import Path

import numpy as np
import tifffile
from skimage import io as skio
from skimage.transform import resize
from scipy.ndimage import gaussian_filter, map_coordinates

logger = logging.getLogger(__name__)


def load_image(path: str) -> np.ndarray:
    """Load an image from disk (supports TIFF, PNG, JPG, ND2).

    For .nd2 files with multiple positions, returns only the first position.
    Use ``expand_nd2_positions`` to split multi-position files first.
    """
    ext = Path(path).suffix.lower()
    if ext in (".tif", ".tiff"):
        return tifffile.imread(path)
    if ext == ".nd2":
        import nd2
        with nd2.ND2File(path) as f:
            sizes = f.sizes
            arr = f.asarray()
            # Collapse extra dimensions (P, T, Z) — take first index
            axis_order = list(sizes.keys())
            for dim in ("P", "T", "Z"):
                if dim in sizes:
                    ax = axis_order.index(dim)
                    arr = arr.take(0, axis=ax)
                    axis_order.pop(ax)
        return arr
    return skio.imread(path)


def expand_nd2_positions(
    nd2_path: str,
    output_dir: str,
    position_indices: list[int] | None = None,
) -> list[str]:
    """Split a multi-position .nd2 file into individual TIFF files.

    Each position is saved as ``<stem>_pos<NNN>.tif`` inside *output_dir*.
    Only the specified *position_indices* are extracted; ``None`` means all.

    Returns:
        List of paths to the saved TIFF files.
    """
    import nd2

    os.makedirs(output_dir, exist_ok=True)
    stem = Path(nd2_path).stem
    saved: list[str] = []

    with nd2.ND2File(nd2_path) as f:
        sizes = f.sizes
        arr = f.asarray()

        # Collapse T and Z axes first (take first frame/slice)
        axis_order = list(sizes.keys())
        for dim in ("T", "Z"):
            if dim in sizes:
                ax = axis_order.index(dim)
                arr = arr.take(0, axis=ax)
                axis_order.pop(ax)

        if "P" not in axis_order:
            # Single-position file — save as-is
            out_path = os.path.join(output_dir, f"{stem}.tif")
            tifffile.imwrite(out_path, arr)
            saved.append(out_path)
            logger.info(f"Single-position ND2 saved to {out_path}")
            return saved

        n_positions = sizes["P"]
        p_axis = axis_order.index("P")

        indices = position_indices if position_indices is not None else list(range(n_positions))
        for idx in indices:
            if idx < 0 or idx >= n_positions:
                logger.warning(f"Position index {idx} out of range (0-{n_positions-1}), skipping")
                continue
            pos_data = np.take(arr, idx, axis=p_axis)
            out_path = os.path.join(output_dir, f"{stem}_pos{idx:03d}.tif")
            tifffile.imwrite(out_path, pos_data)
            saved.append(out_path)

        logger.info(
            f"Expanded {n_positions}-position ND2 -> {len(saved)} TIFFs in {output_dir}"
        )

    return saved


def get_nd2_info(nd2_path: str) -> dict:
    """Return metadata for an .nd2 file without loading the full array.

    Returns:
        Dict with keys: path, sizes, n_positions, shape, dtype.
    """
    import nd2

    with nd2.ND2File(nd2_path) as f:
        return {
            "path": nd2_path,
            "sizes": dict(f.sizes),
            "n_positions": f.sizes.get("P", 1),
            "shape": f.shape,
            "dtype": str(f.dtype),
        }


def save_image(path: str, image: np.ndarray) -> None:
    """Save an image to disk."""
    ext = Path(path).suffix.lower()
    if ext in (".tif", ".tiff"):
        tifffile.imwrite(path, image)
    else:
        skio.imsave(path, image, check_contrast=False)


def find_image_pairs(
    image_dir: str,
    mask_dir: str,
    image_filter: str = "_img",
    mask_filter: str = "_masks",
) -> list[tuple[str, str]]:
    """
    Find matching image/mask pairs by filename convention.

    Images are matched to masks by replacing image_filter with mask_filter
    in the filename stem. Supports common image formats.
    """
    extensions = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg", "*.nd2")
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))

    pairs = []
    for img_path in sorted(image_files):
        stem = Path(img_path).stem
        if image_filter and image_filter in stem:
            mask_stem = stem.replace(image_filter, mask_filter)
        else:
            mask_stem = stem + mask_filter

        # Search for mask with any supported extension
        mask_path = None
        for ext in (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".nd2"):
            candidate = os.path.join(mask_dir, mask_stem + ext)
            if os.path.exists(candidate):
                mask_path = candidate
                break

        if mask_path is not None:
            pairs.append((img_path, mask_path))
        else:
            logger.warning(f"No mask found for image: {img_path}")

    logger.info(f"Found {len(pairs)} image/mask pairs in {image_dir}")
    return pairs


def load_dataset(
    image_dir: str,
    mask_dir: str,
    image_filter: str = "_img",
    mask_filter: str = "_masks",
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """
    Load all image/mask pairs from directories.

    Returns:
        images: List of image arrays
        labels: List of mask arrays (instance labels: 0=background, 1,2,...=objects)
        names: List of image filenames
    """
    pairs = find_image_pairs(image_dir, mask_dir, image_filter, mask_filter)
    images, labels, names = [], [], []

    for img_path, mask_path in pairs:
        img = load_image(img_path).astype(np.float32)
        mask = load_image(mask_path)

        # Ensure mask is integer-valued instance labels
        if mask.dtype in (np.float32, np.float64):
            mask = mask.astype(np.int32)

        images.append(img)
        labels.append(mask)
        names.append(Path(img_path).name)

    return images, labels, names


# ---------------------------------------------------------------------------
# Augmentation functions (BiaPy-style, applied before Cellpose training)
# ---------------------------------------------------------------------------

def random_flip(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Random horizontal and vertical flips."""
    if np.random.rand() > 0.5:
        image = np.flip(image, axis=-1).copy()
        mask = np.flip(mask, axis=-1).copy()
    if np.random.rand() > 0.5:
        image = np.flip(image, axis=-2).copy()
        mask = np.flip(mask, axis=-2).copy()
    return image, mask


def random_rotation(
    image: np.ndarray, mask: np.ndarray, max_degrees: int = 180
) -> tuple[np.ndarray, np.ndarray]:
    """Random rotation by a multiple of 90 degrees up to max_degrees."""
    k = np.random.randint(0, 4)  # 0, 90, 180, 270
    axes = (-2, -1)
    image = np.rot90(image, k=k, axes=axes).copy()
    mask = np.rot90(mask, k=k, axes=axes).copy()
    return image, mask


def elastic_deformation(
    image: np.ndarray,
    mask: np.ndarray,
    alpha_range: tuple = (20, 40),
    sigma_range: tuple = (5, 7),
) -> tuple[np.ndarray, np.ndarray]:
    """Apply elastic deformation to image and mask."""
    alpha = np.random.uniform(*alpha_range)
    sigma = np.random.uniform(*sigma_range)
    shape = image.shape[-2:]

    dx = gaussian_filter(np.random.randn(*shape), sigma) * alpha
    dy = gaussian_filter(np.random.randn(*shape), sigma) * alpha

    y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    indices = [y + dy, x + dx]

    # Handle multi-channel images
    if image.ndim == 3:
        warped_img = np.stack(
            [map_coordinates(image[c], indices, order=1, mode="reflect") for c in range(image.shape[0])],
            axis=0,
        )
    else:
        warped_img = map_coordinates(image, indices, order=1, mode="reflect")

    warped_mask = map_coordinates(mask, indices, order=0, mode="reflect")
    return warped_img, warped_mask


def gaussian_noise(
    image: np.ndarray, mean: float = 0.0, std: float = 0.05
) -> np.ndarray:
    """Add Gaussian noise to image."""
    noise = np.random.normal(mean, std, image.shape).astype(image.dtype)
    return image + noise


def brightness_contrast(
    image: np.ndarray,
    brightness_range: tuple = (-0.1, 0.1),
    contrast_range: tuple = (0.9, 1.1),
) -> np.ndarray:
    """Randomly adjust brightness and contrast."""
    brightness = np.random.uniform(*brightness_range)
    contrast = np.random.uniform(*contrast_range)
    return image * contrast + brightness


def augment_pair(
    image: np.ndarray, mask: np.ndarray, aug_config: dict
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply augmentations specified in the config to an image/mask pair.

    Args:
        image: Input image array
        mask: Label mask array
        aug_config: AUGMENTATION section from YAML config
    """
    if not aug_config.get("ENABLE", False):
        return image, mask

    if aug_config.get("RANDOM_FLIP", False):
        image, mask = random_flip(image, mask)

    rot_cfg = aug_config.get("RANDOM_ROTATION", {})
    if rot_cfg.get("ENABLE", False):
        image, mask = random_rotation(image, mask, rot_cfg.get("DEGREES", 180))

    elastic_cfg = aug_config.get("ELASTIC_DEFORM", {})
    if elastic_cfg.get("ENABLE", False) and np.random.rand() > 0.5:
        image, mask = elastic_deformation(
            image, mask,
            alpha_range=tuple(elastic_cfg.get("ALPHA", [20, 40])),
            sigma_range=tuple(elastic_cfg.get("SIGMA", [5, 7])),
        )

    noise_cfg = aug_config.get("GAUSSIAN_NOISE", {})
    if noise_cfg.get("ENABLE", False) and np.random.rand() > 0.5:
        image = gaussian_noise(
            image, noise_cfg.get("MEAN", 0.0), noise_cfg.get("STD", 0.05)
        )

    bc_cfg = aug_config.get("BRIGHTNESS_CONTRAST", {})
    if bc_cfg.get("ENABLE", False) and np.random.rand() > 0.5:
        image = brightness_contrast(
            image,
            tuple(bc_cfg.get("BRIGHTNESS_RANGE", [-0.1, 0.1])),
            tuple(bc_cfg.get("CONTRAST_RANGE", [0.9, 1.1])),
        )

    return image, mask


def prepare_augmented_dataset(
    images: list[np.ndarray],
    labels: list[np.ndarray],
    aug_config: dict,
    num_augmented_per_image: int = 4,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Create an augmented dataset from original images.

    Returns the original images plus augmented copies.
    """
    aug_images = list(images)
    aug_labels = list(labels)

    for img, lbl in zip(images, labels):
        for _ in range(num_augmented_per_image):
            aug_img, aug_lbl = augment_pair(img.copy(), lbl.copy(), aug_config)
            aug_images.append(aug_img)
            aug_labels.append(aug_lbl)

    logger.info(
        f"Augmented dataset: {len(images)} original -> {len(aug_images)} total"
    )
    return aug_images, aug_labels
