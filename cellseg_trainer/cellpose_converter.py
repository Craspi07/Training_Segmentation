"""
cellpose_converter.py — Convert Cellpose *_seg.npy masks to COCO-ready format.

Each ``_seg.npy`` file contains a dict with at least:
  - ``masks``: (H, W) int32 instance label array
  - (optionally) ``outlines``, ``flows``, etc.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# COCO polygon format requires at least 3 points (6 coordinates)
_MIN_POLYGON_POINTS = 3
_MIN_AREA = 4  # pixels² — discard tiny artefacts


def convert_seg_npy(
    seg_file: str | Path,
    image_id: int = 0,
    start_annotation_id: int = 1,
) -> tuple[list[dict[str, Any]], int]:
    """Convert a single Cellpose ``_seg.npy`` file to COCO annotation records.

    Args:
        seg_file: Path to the ``*_seg.npy`` file.
        image_id: COCO ``image_id`` to assign to all annotations.
        start_annotation_id: First annotation id to use (incremented per instance).

    Returns:
        Tuple of:
          - List of COCO annotation dicts (``id``, ``image_id``, ``category_id``,
            ``segmentation``, ``bbox``, ``area``, ``iscrowd``).
          - Next available annotation id (for chaining calls).

    Raises:
        FileNotFoundError: If ``seg_file`` does not exist.
        KeyError: If the ``.npy`` file does not contain a ``masks`` key.
    """
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError("opencv-python is required: pip install opencv-python") from exc

    seg_file = Path(seg_file)
    if not seg_file.exists():
        raise FileNotFoundError(f"Seg file not found: {seg_file}")

    data = np.load(str(seg_file), allow_pickle=True).item()
    if "masks" not in data:
        raise KeyError(f"'masks' key not found in {seg_file}. Keys: {list(data.keys())}")

    masks: np.ndarray = data["masks"]  # (H, W) int32
    annotations: list[dict[str, Any]] = []
    ann_id = start_annotation_id

    instance_ids = np.unique(masks)
    instance_ids = instance_ids[instance_ids > 0]  # exclude background

    for inst_id in instance_ids:
        binary = (masks == inst_id).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        polygons: list[list[float]] = []
        for cnt in contours:
            if cnt.shape[0] < _MIN_POLYGON_POINTS:
                continue
            poly = cnt.flatten().tolist()
            if len(poly) >= _MIN_POLYGON_POINTS * 2:
                polygons.append([float(v) for v in poly])

        if not polygons:
            logger.debug("Instance %d in %s produced no valid polygon — skipped", inst_id, seg_file.name)
            continue

        # Bounding box from binary mask
        ys, xs = np.where(binary)
        if len(xs) == 0:
            continue
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        bbox_w = x_max - x_min + 1
        bbox_h = y_max - y_min + 1
        area = int(binary.sum())

        if area < _MIN_AREA:
            continue

        annotations.append(
            {
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": polygons,
                "bbox": [x_min, y_min, bbox_w, bbox_h],
                "area": area,
                "iscrowd": 0,
            }
        )
        ann_id += 1

    logger.debug(
        "%s → %d instances, %d valid annotations",
        seg_file.name,
        len(instance_ids),
        len(annotations),
    )
    return annotations, ann_id


def convert_mask_array(
    mask: np.ndarray,
    image_id: int = 0,
    start_annotation_id: int = 1,
) -> tuple[list[dict[str, Any]], int]:
    """Convert an in-memory instance-label array to COCO annotations.

    Args:
        mask: (H, W) integer-labeled instance mask.
        image_id: COCO image id.
        start_annotation_id: Starting annotation id.

    Returns:
        Same as :func:`convert_seg_npy`.
    """
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError("opencv-python is required: pip install opencv-python") from exc

    annotations: list[dict[str, Any]] = []
    ann_id = start_annotation_id

    instance_ids = np.unique(mask)
    instance_ids = instance_ids[instance_ids > 0]

    for inst_id in instance_ids:
        binary = (mask == inst_id).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        polygons: list[list[float]] = []
        for cnt in contours:
            if cnt.shape[0] < _MIN_POLYGON_POINTS:
                continue
            poly = cnt.flatten().tolist()
            if len(poly) >= _MIN_POLYGON_POINTS * 2:
                polygons.append([float(v) for v in poly])

        if not polygons:
            continue

        ys, xs = np.where(binary)
        if len(xs) == 0:
            continue
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        area = int(binary.sum())

        if area < _MIN_AREA:
            continue

        annotations.append(
            {
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": polygons,
                "bbox": [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1],
                "area": area,
                "iscrowd": 0,
            }
        )
        ann_id += 1

    return annotations, ann_id


def batch_convert(
    seg_files: list[Path],
    n_workers: int = 4,
) -> list[tuple[list[dict], int]]:
    """Convert a list of ``_seg.npy`` files using a process pool.

    Args:
        seg_files: List of seg file paths.
        n_workers: Number of parallel workers.

    Returns:
        List of ``(annotations, next_ann_id)`` tuples in input order.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    results: dict[int, tuple] = {}

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(convert_seg_npy, f, idx, 1): idx
            for idx, f in enumerate(seg_files)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                logger.error("Failed to convert %s: %s", seg_files[idx], exc)
                results[idx] = ([], 1)

    return [results[i] for i in range(len(seg_files))]
