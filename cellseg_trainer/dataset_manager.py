"""
dataset_manager.py — Register COCO datasets with Detectron2 DatasetCatalog.

Handles registration of train/val splits and provides the Detectron2
dataset_dicts format required by DefaultTrainer.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Dataset name prefix used in Detectron2 catalog
_CATALOG_PREFIX = "cellseg"


def register_coco_datasets(
    dataset_dir: str | Path,
    train_json: str | Path | None = None,
    val_json: str | Path | None = None,
    image_root: str | Path | None = None,
    dataset_name: str = "cellseg",
) -> tuple[str, str]:
    """Register train and val COCO JSON datasets with Detectron2.

    Args:
        dataset_dir: Directory containing ``instances_train.json`` and
            ``instances_val.json`` (used when explicit paths are not given).
        train_json: Explicit path to train JSON (overrides auto-detection).
        val_json: Explicit path to val JSON (overrides auto-detection).
        image_root: Directory containing the actual image files referenced by
            the JSON.  Defaults to ``dataset_dir``.
        dataset_name: Base name prefix for catalog entries.

    Returns:
        Tuple of ``(train_dataset_name, val_dataset_name)`` strings.

    Raises:
        ImportError: If detectron2 is not installed.
        FileNotFoundError: If JSON files cannot be located.
    """
    try:
        from detectron2.data.datasets import register_coco_instances  # type: ignore
        from detectron2.data import MetadataCatalog, DatasetCatalog  # type: ignore
    except ImportError as exc:
        raise ImportError("detectron2 is required for dataset registration") from exc

    dataset_dir = Path(dataset_dir)
    img_root = Path(image_root) if image_root else dataset_dir

    train_json_path = Path(train_json) if train_json else dataset_dir / "instances_train.json"
    val_json_path = Path(val_json) if val_json else dataset_dir / "instances_val.json"

    for p in (train_json_path, val_json_path):
        if not p.exists():
            raise FileNotFoundError(f"Dataset JSON not found: {p}")

    train_name = f"{dataset_name}_train"
    val_name = f"{dataset_name}_val"

    # Only register if not already in catalog
    if train_name not in DatasetCatalog.list():
        register_coco_instances(train_name, {}, str(train_json_path), str(img_root))
        logger.info("Registered dataset '%s' from %s", train_name, train_json_path)
    else:
        logger.debug("Dataset '%s' already registered", train_name)

    if val_name not in DatasetCatalog.list():
        register_coco_instances(val_name, {}, str(val_json_path), str(img_root))
        logger.info("Registered dataset '%s' from %s", val_name, val_json_path)
    else:
        logger.debug("Dataset '%s' already registered", val_name)

    # Set thing_classes metadata
    for name in (train_name, val_name):
        MetadataCatalog.get(name).set(thing_classes=["cell"])

    return train_name, val_name


def get_dataset_stats(json_path: str | Path) -> dict[str, Any]:
    """Return basic statistics for a COCO JSON file.

    Args:
        json_path: Path to COCO JSON.

    Returns:
        Dict with keys: ``n_images``, ``n_annotations``, ``avg_annotations_per_image``.
    """
    with open(json_path) as f:
        data = json.load(f)

    n_imgs = len(data.get("images", []))
    n_anns = len(data.get("annotations", []))
    return {
        "n_images": n_imgs,
        "n_annotations": n_anns,
        "avg_annotations_per_image": round(n_anns / max(n_imgs, 1), 2),
    }
