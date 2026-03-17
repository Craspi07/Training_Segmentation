"""
coco_builder.py — Build COCO-format JSON datasets from image/mask pairs.

Outputs:
  - ``instances_train.json``
  - ``instances_val.json``
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np

from cellseg_trainer.cellpose_converter import convert_seg_npy, convert_mask_array
from cellseg_trainer.ome_loader import load_image_array
from cellseg_trainer.utils import collect_images, ensure_dir

logger = logging.getLogger(__name__)

_CATEGORIES = [{"id": 1, "name": "cell", "supercategory": "cell"}]


def build_coco_dataset(
    image_dir: str | Path,
    mask_dir: str | Path,
    output_dir: str | Path,
    train_split: float = 0.85,
    seed: int = 42,
    mask_suffix: str = "_seg.npy",
    n_workers: int = 4,
    image_extensions: Optional[set[str]] = None,
) -> tuple[Path, Path]:
    """Convert image/mask pairs to COCO JSON files.

    For each image the function looks for a corresponding mask:
      - If ``mask_suffix`` ends with ``.npy`` → Cellpose ``_seg.npy`` format.
      - Otherwise → integer-labeled TIFF mask loaded via :func:`load_image_array`.

    Args:
        image_dir: Directory containing source images.
        mask_dir: Directory containing mask files.
        output_dir: Where to write ``instances_train.json`` and ``instances_val.json``.
        train_split: Fraction of data used for training (rest → validation).
        seed: Random seed for reproducible train/val split.
        mask_suffix: Suffix appended to the image stem to locate its mask.
        n_workers: Number of parallel workers for mask conversion.
        image_extensions: Override default image extensions.

    Returns:
        Tuple of ``(train_json_path, val_json_path)``.
    """
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    output_dir = ensure_dir(output_dir)

    images = collect_images(image_dir, image_extensions)
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")

    # Pair images with masks
    pairs: list[tuple[Path, Path]] = []
    for img_path in images:
        mask_path = mask_dir / (img_path.stem + mask_suffix)
        if not mask_path.exists():
            logger.warning("Mask not found for %s — skipping", img_path.name)
            continue
        pairs.append((img_path, mask_path))

    if not pairs:
        raise FileNotFoundError(f"No valid image/mask pairs found (mask_suffix='{mask_suffix}')")

    logger.info("Found %d valid image/mask pairs", len(pairs))

    # Split
    random.seed(seed)
    shuffled = pairs[:]
    random.shuffle(shuffled)
    n_train = max(1, int(len(shuffled) * train_split))
    train_pairs = shuffled[:n_train]
    val_pairs = shuffled[n_train:]

    train_json = _build_split(train_pairs, mask_suffix, n_workers)
    val_json = _build_split(val_pairs, mask_suffix, n_workers)

    train_path = output_dir / "instances_train.json"
    val_path = output_dir / "instances_val.json"

    with open(train_path, "w") as f:
        json.dump(train_json, f, indent=2)
    with open(val_path, "w") as f:
        json.dump(val_json, f, indent=2)

    logger.info(
        "COCO dataset written → train: %d images / %d anns, val: %d images / %d anns",
        len(train_json["images"]),
        len(train_json["annotations"]),
        len(val_json["images"]),
        len(val_json["annotations"]),
    )
    return train_path, val_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_split(
    pairs: list[tuple[Path, Path]],
    mask_suffix: str,
    n_workers: int,
) -> dict[str, Any]:
    """Build a single COCO split dict from image/mask pairs."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    images_info: list[dict] = []
    all_annotations: list[dict] = []
    ann_id_counter = [1]

    def _process(idx: int, img_path: Path, mask_path: Path) -> tuple[int, dict, list[dict]]:
        # Image dimensions
        arr = load_image_array(img_path)
        h, w = arr.shape[:2]
        img_info = {
            "id": idx + 1,
            "file_name": img_path.name,
            "height": h,
            "width": w,
        }
        # Annotations
        is_npy = mask_path.suffix.lower() == ".npy"
        if is_npy:
            anns, _ = convert_seg_npy(mask_path, image_id=idx + 1, start_annotation_id=1)
        else:
            mask = load_image_array(mask_path).astype(np.int32)
            anns, _ = convert_mask_array(mask, image_id=idx + 1, start_annotation_id=1)
        return idx, img_info, anns

    results: dict[int, tuple[dict, list[dict]]] = {}
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_process, i, ip, mp): i for i, (ip, mp) in enumerate(pairs)}
        for fut in as_completed(futs):
            try:
                idx, img_info, anns = fut.result()
                results[idx] = (img_info, anns)
            except Exception as exc:
                logger.error("Error processing pair index %d: %s", futs[fut], exc)

    # Reassemble in order and renumber annotation ids
    for idx in sorted(results.keys()):
        img_info, anns = results[idx]
        images_info.append(img_info)
        for ann in anns:
            ann = dict(ann)
            ann["id"] = ann_id_counter[0]
            ann["image_id"] = img_info["id"]
            ann_id_counter[0] += 1
            all_annotations.append(ann)

    return {
        "info": {"description": "cellseg_trainer COCO dataset"},
        "licenses": [],
        "categories": _CATEGORIES,
        "images": images_info,
        "annotations": all_annotations,
    }
