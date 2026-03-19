"""
utils.py — Shared utilities for cellseg_trainer (logging, I/O, multiprocessing helpers).
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import Callable, Iterable, TypeVar

import numpy as np

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging(level: int = logging.INFO, log_file: str | Path | None = None) -> None:
    """Configure root logger for the package.

    Args:
        level: Logging level (e.g. logging.DEBUG).
        log_file: Optional path to write log output.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(level=level, format=LOG_FORMAT, handlers=handlers)


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".ome.tif"}


def collect_images(directory: str | Path, extensions: set[str] | None = None) -> list[Path]:
    """Collect image paths from a directory (non-recursive).

    Args:
        directory: Directory to search.
        extensions: Set of extensions to include (with dot). Defaults to IMAGE_EXTENSIONS.

    Returns:
        Sorted list of matching Path objects.
    """
    exts = extensions or IMAGE_EXTENSIONS
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in exts)


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it does not exist.

    Args:
        path: Directory path.

    Returns:
        Path object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def file_md5(path: str | Path, chunk_size: int = 65536) -> str:
    """Compute MD5 checksum of a file.

    Args:
        path: File path.
        chunk_size: Read chunk size in bytes.

    Returns:
        Hex MD5 string.
    """
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Array / mask helpers
# ---------------------------------------------------------------------------


def normalize_uint8(image: np.ndarray) -> np.ndarray:
    """Normalize an array to uint8 [0, 255].

    Args:
        image: Input array.

    Returns:
        uint8 array.
    """
    img = image.astype(np.float32)
    lo, hi = img.min(), img.max()
    if hi > lo:
        img = (img - lo) / (hi - lo) * 255.0
    return img.astype(np.uint8)


def instance_to_binary(mask: np.ndarray) -> np.ndarray:
    """Convert instance-label mask to binary mask.

    Args:
        mask: Integer-labeled instance mask (H, W).

    Returns:
        Binary uint8 mask.
    """
    return (mask > 0).astype(np.uint8)


def rle_encode(binary_mask: np.ndarray) -> dict:
    """Encode a binary mask as COCO RLE.

    Requires pycocotools.

    Args:
        binary_mask: Fortran-order uint8 mask (H, W).

    Returns:
        COCO RLE dict with keys ``counts`` and ``size``.
    """
    from pycocotools import mask as mask_util  # type: ignore

    return mask_util.encode(np.asfortranarray(binary_mask.astype(np.uint8)))


# ---------------------------------------------------------------------------
# Progress / iteration helpers
# ---------------------------------------------------------------------------


def tqdm_or_plain(iterable: Iterable[T], desc: str = "", **kwargs) -> Iterable[T]:
    """Wrap *iterable* with tqdm if available, else return as-is.

    Args:
        iterable: Any iterable.
        desc: Progress bar description.

    Returns:
        Wrapped iterable.
    """
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(iterable, desc=desc, **kwargs)
    except ImportError:
        return iterable


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------


def available_gpus() -> list[int]:
    """Return list of available CUDA device indices.

    Returns:
        List of integer device indices, or empty list if CUDA unavailable.
    """
    try:
        import torch

        return list(range(torch.cuda.device_count()))
    except ImportError:
        return []


def prepare_weights_for_detectron2(weights_path: str | Path, cache_dir: str | Path | None = None) -> str:
    """Ensure a weights file is in Detectron2-compatible format.

    Detectron2 expects a checkpoint dict with a ``"model"`` key::

        {"model": {<layer>: <tensor>, ...}}

    Plain ``.pt`` files saved with ``torch.save(model.state_dict(), ...)``
    lack this wrapper.  This function detects the format and writes a
    converted ``.pth`` file when necessary.

    Supported input formats:
      - ``.pt``  — raw ``state_dict`` or full ``nn.Module``
      - ``.pth`` — Detectron2 checkpoint (``{"model": ...}``) or raw state_dict
      - ``.pkl`` — Detectron2 / Caffe2 pickle checkpoint

    Args:
        weights_path: Path to the source weights file (``.pt``, ``.pth``, ``.pkl``).
        cache_dir: Directory to write the converted file.  Defaults to the
            same directory as *weights_path*.

    Returns:
        Path string to a Detectron2-compatible weights file (may be the
        original path if no conversion was needed).

    Raises:
        FileNotFoundError: If *weights_path* does not exist.
        RuntimeError: If the file cannot be parsed as a PyTorch checkpoint.
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    import torch

    suffix = weights_path.suffix.lower()

    # .pkl files are loaded natively by Detectron2 — no conversion needed
    if suffix == ".pkl":
        logger.info("Weights are .pkl (Detectron2/Caffe2 format) — no conversion needed")
        return str(weights_path)

    # Load the checkpoint
    try:
        ckpt = torch.load(str(weights_path), map_location="cpu")
    except Exception as exc:
        raise RuntimeError(f"Cannot load weights from {weights_path}: {exc}") from exc

    # Detect format
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            # Already Detectron2-compatible (has "model" key)
            logger.info("Weights already in Detectron2 format — no conversion needed")
            return str(weights_path)
        # Raw state_dict dict (keys are layer names)
        state_dict = ckpt
    else:
        # Full nn.Module object saved with torch.save(model, ...)
        try:
            state_dict = ckpt.state_dict()
        except AttributeError as exc:
            raise RuntimeError(
                f"Cannot extract state_dict from {weights_path}. "
                "Expected a state_dict dict or nn.Module."
            ) from exc

    # Strip common prefixes added by DataParallel / DistributedDataParallel
    cleaned: dict = {}
    for k, v in state_dict.items():
        new_k = k
        for prefix in ("module.", "model."):
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]
        cleaned[new_k] = v

    # Wrap in Detectron2 format
    d2_ckpt = {"model": cleaned}

    out_dir = Path(cache_dir) if cache_dir else weights_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (weights_path.stem + "_d2converted.pth")
    torch.save(d2_ckpt, str(out_path))
    logger.info(
        "Converted %s → Detectron2 format → %s (%d layers)",
        weights_path.name,
        out_path.name,
        len(cleaned),
    )
    return str(out_path)


def gpu_memory_gb(device: int = 0) -> float:
    """Return total GPU memory in GB for a given device.

    Args:
        device: CUDA device index.

    Returns:
        Total memory in GB, or 0.0 if unavailable.
    """
    try:
        import torch

        props = torch.cuda.get_device_properties(device)
        return props.total_memory / (1024 ** 3)
    except Exception:
        return 0.0
