"""
visualization.py — Overlay segmentation masks on microscopy images.

Produces PNG and TIFF outputs with bounding boxes, colored instance masks,
and optional confidence score labels.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from cellseg_trainer.utils import normalize_uint8

logger = logging.getLogger(__name__)


def overlay_masks(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "tab20",
) -> np.ndarray:
    """Blend an instance mask on top of an image.

    Args:
        image: 2-D float or uint8 array (H, W).
        mask: 2-D integer instance label array (H, W).
        alpha: Transparency of the mask overlay (0 = invisible, 1 = opaque).
        colormap: Matplotlib colormap name for instance colours.

    Returns:
        uint8 RGB array (H, W, 3).
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise ImportError("matplotlib is required") from exc

    u8 = normalize_uint8(image)
    rgb = np.stack([u8, u8, u8], axis=-1)

    if mask.max() == 0:
        return rgb

    cmap = plt.get_cmap(colormap)
    for inst_id in np.unique(mask):
        if inst_id == 0:
            continue
        color = np.array(cmap(inst_id % cmap.N)[:3]) * 255
        region = mask == inst_id
        rgb[region] = (rgb[region] * (1 - alpha) + color * alpha).astype(np.uint8)

    return rgb


def draw_bboxes(
    image_rgb: np.ndarray,
    bboxes: Sequence[tuple[int, int, int, int]],
    scores: Optional[Sequence[float]] = None,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes (and optional scores) on an RGB image.

    Uses cv2 when available for text labels; falls back to pure numpy
    rectangle drawing so the function always works.

    Args:
        image_rgb: uint8 RGB array (H, W, 3).
        bboxes: List of ``(x, y, w, h)`` bounding boxes.
        scores: Optional list of confidence scores.
        color: RGB colour tuple.
        thickness: Line thickness.

    Returns:
        Annotated uint8 RGB array.
    """
    out = image_rgb.copy()
    t = max(1, int(thickness))

    try:
        import cv2  # type: ignore

        for i, (x, y, w, h) in enumerate(bboxes):
            cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)), color, t)
            if scores is not None:
                label = f"{scores[i]:.2f}"
                cv2.putText(out, label, (int(x), max(int(y) - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    except ImportError:
        # Fallback: draw rectangle edges with numpy slice assignment
        H, W = out.shape[:2]
        for x, y, w, h in bboxes:
            x0, y0, x1, y1 = int(x), int(y), int(x + w), int(y + h)
            # Clamp to image bounds
            x0c, y0c = max(x0, 0), max(y0, 0)
            x1c, y1c = min(x1, W), min(y1, H)
            # Top / bottom edges
            out[y0c:min(y0c + t, H), x0c:x1c] = color
            out[max(y1c - t, 0):y1c, x0c:x1c] = color
            # Left / right edges
            out[y0c:y1c, x0c:min(x0c + t, W)] = color
            out[y0c:y1c, max(x1c - t, 0):x1c] = color

    return out


def save_visualization(
    image: np.ndarray,
    mask: np.ndarray,
    output_path: str | Path,
    bboxes: Optional[Sequence] = None,
    scores: Optional[Sequence[float]] = None,
    alpha: float = 0.5,
    save_tiff: bool = False,
) -> None:
    """Save a 3-panel visualization: raw | mask | overlay.

    Args:
        image: 2-D float or uint8 image (H, W).
        mask: 2-D integer instance mask (H, W).
        output_path: Destination ``.png`` (or ``.tif``) path.
        bboxes: Optional bounding boxes for overlay panel.
        scores: Optional scores matching *bboxes*.
        alpha: Mask overlay transparency.
        save_tiff: Also save as 16-bit TIFF (for downstream analysis).
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise ImportError("matplotlib is required") from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    overlay = overlay_masks(image, mask, alpha=alpha)
    if bboxes:
        overlay = draw_bboxes(overlay, bboxes, scores)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(normalize_uint8(image), cmap="gray")
    axes[0].set_title("Raw Image")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="nipy_spectral")
    axes[1].set_title(f"Mask ({mask.max()} cells)")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    fig.tight_layout()
    png_path = output_path.with_suffix(".png")
    fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.debug("Saved visualization → %s", png_path)

    if save_tiff:
        try:
            import tifffile  # type: ignore

            tiff_path = output_path.with_suffix(".tif")
            tifffile.imwrite(str(tiff_path), mask.astype(np.uint16))
            logger.debug("Saved mask TIFF → %s", tiff_path)
        except ImportError:
            pass


def visualize_batch(
    image_dir: str | Path,
    mask_dir: str | Path,
    output_dir: str | Path,
    max_images: int = 20,
    alpha: float = 0.5,
    save_tiff: bool = False,
) -> None:
    """Visualize all image/mask pairs in a directory.

    Args:
        image_dir: Source image directory.
        mask_dir: Mask directory (TIFF or npy).
        output_dir: Destination for PNG outputs.
        max_images: Maximum number of images to process.
        alpha: Mask transparency.
        save_tiff: Also save TIFF overlays.
    """
    from cellseg_trainer.utils import collect_images
    from cellseg_trainer.ome_loader import load_image_array

    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(image_dir)[:max_images]

    for img_path in images:
        # Try TIFF mask first, then _seg.npy
        mask_tif = mask_dir / f"{img_path.stem}_mask.tif"
        mask_npy = mask_dir / f"{img_path.stem}_seg.npy"

        if mask_tif.exists():
            mask = load_image_array(mask_tif).astype(np.int32)
        elif mask_npy.exists():
            data = np.load(str(mask_npy), allow_pickle=True).item()
            mask = data.get("masks", np.zeros((1, 1), dtype=np.int32))
        else:
            logger.warning("No mask found for %s", img_path.name)
            continue

        image = load_image_array(img_path)
        save_visualization(
            image, mask,
            output_dir / f"{img_path.stem}_vis.png",
            alpha=alpha,
            save_tiff=save_tiff,
        )
