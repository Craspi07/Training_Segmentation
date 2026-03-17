"""
ome_loader.py — Load OME-TIFF and standard TIFF microscopy images.

Handles single-channel, multi-channel, and Z-stack images.
Supports max-projection and channel selection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def load_ome_tif(
    path: str | Path,
    channel: Optional[int] = None,
    z_projection: str = "max",
    z_slice: Optional[int] = None,
) -> np.ndarray:
    """Load an OME-TIFF (or standard TIFF) and return a 2-D or 3-D array.

    Dimension order is inferred from array shape.  Supported axes: TCZYX,
    CZYX, ZYX, CYX, YX (any subset in canonical order).

    Args:
        path: Path to the ``.ome.tif`` or ``.tif`` file.
        channel: Zero-based channel index to extract.  If *None* and the image
            is multi-channel the first channel is returned.
        z_projection: How to collapse a Z-stack: ``"max"`` (maximum
            projection) or ``"mean"``.  Ignored when *z_slice* is set.
        z_slice: If set, extract this specific Z-plane instead of projecting.

    Returns:
        2-D float32 array (H, W) after channel/Z selection, or 3-D (C, H, W)
        when *channel* is ``None`` and image has C > 1.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If requested channel or Z-slice is out of range.
    """
    try:
        import tifffile  # type: ignore
    except ImportError as exc:
        raise ImportError("tifffile is required: pip install tifffile") from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with tifffile.TiffFile(str(path)) as tif:
        # Use OME metadata axes when available
        axes = tif.series[0].axes.upper() if tif.series else ""
        data = tif.asarray()

    data = data.astype(np.float32)
    logger.debug("Loaded %s — shape %s axes '%s'", path.name, data.shape, axes)

    # Squeeze trivial leading dimensions (T, batch)
    while data.ndim > 2 and data.shape[0] == 1:
        data = data.squeeze(0)

    # At this point expect: CZYX | ZYX | CYX | YX
    if data.ndim == 2:
        return data

    if data.ndim == 3:
        # Heuristic: if first dim is large it's probably Z or C
        # Distinguish Z-stack from multi-channel via OME axes when available
        if "Z" in axes and "C" not in axes:
            # Shape is (Z, H, W)
            data = _project_z(data, z_projection, z_slice)
            return data
        if "C" in axes:
            # Shape is (C, H, W)
            return _select_channel(data, channel)
        # Ambiguous: treat as (Z, H, W) if first dim < 64, else (C, H, W)
        if data.shape[0] < 64:
            data = _project_z(data, z_projection, z_slice)
            return data
        return _select_channel(data, channel)

    if data.ndim == 4:
        # (C, Z, H, W) or (Z, C, H, W)
        if "C" in axes and axes.index("C") < axes.index("Z"):
            c, z_stack = data.shape[0], data.shape[1]
            ch = channel if channel is not None else 0
            if ch >= c:
                raise ValueError(f"Channel {ch} out of range (max {c - 1})")
            data = _project_z(data[ch], z_projection, z_slice)
        else:
            z_stack, c = data.shape[0], data.shape[1]
            ch = channel if channel is not None else 0
            if ch >= c:
                raise ValueError(f"Channel {ch} out of range (max {c - 1})")
            data = _project_z(data[:, ch], z_projection, z_slice)
        return data

    raise ValueError(f"Unsupported image dimensionality: {data.ndim}D shape {data.shape}")


def load_image_array(
    path: str | Path,
    channel: Optional[int] = None,
    z_projection: str = "max",
    z_slice: Optional[int] = None,
) -> np.ndarray:
    """Load any supported image format to a float32 2-D array.

    Thin wrapper that dispatches ``.ome.tif`` / ``.tif`` to :func:`load_ome_tif`
    and PNG/JPG to OpenCV.

    Args:
        path: Image file path.
        channel: Channel index (for multi-channel images).
        z_projection: Z-projection method (``"max"`` or ``"mean"``).
        z_slice: Z-plane index (overrides projection).

    Returns:
        2-D float32 array (H, W).
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        return load_ome_tif(path, channel=channel, z_projection=z_projection, z_slice=z_slice)

    try:
        import cv2  # type: ignore

        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise OSError(f"cv2 could not read {path}")
        if img.ndim == 3:
            img = img[:, :, 0] if channel is None else img[:, :, channel]
        return img.astype(np.float32)
    except ImportError:
        pass

    # Fallback: matplotlib / PIL
    try:
        import matplotlib.image as mpimg  # type: ignore

        img = mpimg.imread(str(path)).astype(np.float32)
        if img.ndim == 3:
            img = img[:, :, 0] if channel is None else img[:, :, channel]
        return img
    except Exception as exc:
        raise OSError(f"Could not read image {path}: {exc}") from exc


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _project_z(z_stack: np.ndarray, method: str, z_slice: Optional[int]) -> np.ndarray:
    """Project or slice along the first (Z) axis.

    Args:
        z_stack: Array of shape (Z, H, W).
        method: ``"max"`` or ``"mean"``.
        z_slice: If set, extract this Z-plane.

    Returns:
        2-D array (H, W).
    """
    n_z = z_stack.shape[0]
    if z_slice is not None:
        if z_slice >= n_z:
            raise ValueError(f"z_slice {z_slice} out of range (max {n_z - 1})")
        return z_stack[z_slice]
    if method == "max":
        return z_stack.max(axis=0)
    if method == "mean":
        return z_stack.mean(axis=0)
    raise ValueError(f"Unknown z_projection method: '{method}'")


def _select_channel(data: np.ndarray, channel: Optional[int]) -> np.ndarray:
    """Extract a channel from (C, H, W) array.

    Returns (H, W) when channel is given, (C, H, W) when channel is None.
    """
    if channel is None:
        return data
    if channel >= data.shape[0]:
        raise ValueError(f"Channel {channel} out of range (max {data.shape[0] - 1})")
    return data[channel]
