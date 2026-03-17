"""
patch_extractor.py — Split large microscopy images into overlapping patches.

Patches are written as individual TIFF files alongside corresponding mask patches.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator, Optional

import numpy as np

from cellseg_trainer.ome_loader import load_image_array
from cellseg_trainer.utils import ensure_dir, tqdm_or_plain

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------


def extract_patches(
    image: np.ndarray,
    patch_size: int = 512,
    overlap: int = 64,
) -> Generator[tuple[np.ndarray, tuple[int, int, int, int]], None, None]:
    """Yield (patch, bbox) pairs from an image array.

    Args:
        image: 2-D (H, W) or 3-D (C, H, W) float32 array.
        patch_size: Patch height and width in pixels.
        overlap: Overlap between adjacent patches in pixels.

    Yields:
        Tuple of ``(patch_array, (y_start, x_start, y_end, x_end))``.
    """
    if image.ndim == 2:
        h, w = image.shape
    else:
        _, h, w = image.shape

    stride = patch_size - overlap
    if stride <= 0:
        raise ValueError(f"overlap ({overlap}) must be less than patch_size ({patch_size})")

    y = 0
    while y < h:
        y_end = min(y + patch_size, h)
        y_start = max(0, y_end - patch_size)
        x = 0
        while x < w:
            x_end = min(x + patch_size, w)
            x_start = max(0, x_end - patch_size)

            if image.ndim == 2:
                patch = image[y_start:y_end, x_start:x_end]
            else:
                patch = image[:, y_start:y_end, x_start:x_end]

            yield patch, (y_start, x_start, y_end, x_end)

            if x_end == w:
                break
            x += stride
        if y_end == h:
            break
        y += stride


def extract_and_save_patches(
    image_dir: str | Path,
    mask_dir: str | Path,
    output_dir: str | Path,
    patch_size: int = 512,
    overlap: int = 64,
    mask_suffix: str = "_seg.npy",
    min_cell_fraction: float = 0.0,
) -> int:
    """Extract patches from all images in a directory and save as TIFF files.

    Patches are saved under:
      ``<output_dir>/images/<stem>_y<Y>_x<X>.tif``
      ``<output_dir>/masks/<stem>_y<Y>_x<X>_mask.tif``

    Args:
        image_dir: Directory with source images.
        mask_dir: Directory with mask files.
        output_dir: Root output directory.
        patch_size: Patch size (square).
        overlap: Overlap in pixels.
        mask_suffix: Suffix to locate mask for each image.
        min_cell_fraction: Minimum fraction of patch pixels that must be
            foreground to include the patch (0.0 = include all).

    Returns:
        Total number of patches saved.
    """
    try:
        import tifffile  # type: ignore
    except ImportError as exc:
        raise ImportError("tifffile is required: pip install tifffile") from exc

    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    out_img_dir = ensure_dir(Path(output_dir) / "images")
    out_mask_dir = ensure_dir(Path(output_dir) / "masks")

    from cellseg_trainer.utils import collect_images

    image_paths = collect_images(image_dir)
    total = 0

    for img_path in tqdm_or_plain(image_paths, desc="Extracting patches"):
        # Locate mask
        is_npy = mask_suffix.endswith(".npy")
        mask_path = mask_dir / (img_path.stem + mask_suffix)
        if not mask_path.exists():
            logger.warning("Mask not found for %s — skipping", img_path.name)
            continue

        image = load_image_array(img_path)

        if is_npy:
            data = np.load(str(mask_path), allow_pickle=True).item()
            mask = data.get("masks", np.zeros(image.shape[:2], dtype=np.int32))
        else:
            mask = load_image_array(mask_path).astype(np.int32)

        for patch, (ys, xs, ye, xe) in extract_patches(image, patch_size, overlap):
            mask_patch = mask[ys:ye, xs:xe]

            # Filter by cell fraction
            if min_cell_fraction > 0:
                frac = (mask_patch > 0).mean()
                if frac < min_cell_fraction:
                    continue

            stem = f"{img_path.stem}_y{ys:05d}_x{xs:05d}"
            tifffile.imwrite(str(out_img_dir / f"{stem}.tif"), patch.astype(np.float32))
            tifffile.imwrite(str(out_mask_dir / f"{stem}_mask.tif"), mask_patch.astype(np.int32))
            total += 1

    logger.info("Saved %d patches to %s", total, output_dir)
    return total


def reconstruct_from_patches(
    patches: list[np.ndarray],
    bboxes: list[tuple[int, int, int, int]],
    output_shape: tuple[int, int],
    blend: bool = True,
) -> np.ndarray:
    """Reconstruct a full image from patches using averaging in overlap regions.

    Args:
        patches: List of 2-D patch arrays.
        bboxes: Corresponding bounding boxes ``(y_start, x_start, y_end, x_end)``.
        output_shape: ``(H, W)`` of the reconstructed image.
        blend: If ``True``, average overlapping regions; else use last-write.

    Returns:
        Reconstructed float32 array of shape *output_shape*.
    """
    canvas = np.zeros(output_shape, dtype=np.float64)
    weight = np.zeros(output_shape, dtype=np.float64)

    for patch, (ys, xs, ye, xe) in zip(patches, bboxes):
        ph, pw = patch.shape[:2]
        canvas[ys:ye, xs:xe] += patch[:ph, :pw]
        weight[ys:ye, xs:xe] += 1.0

    if blend:
        weight = np.maximum(weight, 1.0)
        return (canvas / weight).astype(np.float32)
    return canvas.astype(np.float32)
