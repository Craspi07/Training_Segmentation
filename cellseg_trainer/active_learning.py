"""
active_learning.py — Self-training / active learning loop.

Detects low-confidence predictions, suggests images for manual annotation,
and optionally retrains the model on newly labelled data.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from cellseg_trainer.utils import ensure_dir, tqdm_or_plain

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Confidence analysis
# ---------------------------------------------------------------------------


def find_low_confidence_images(
    predictions_json: str | Path,
    confidence_threshold: float = 0.6,
    max_suggestions: int = 50,
) -> list[dict[str, Any]]:
    """Return images where the average detection confidence is below a threshold.

    Args:
        predictions_json: Path to ``predictions.json`` written by the inference module.
        confidence_threshold: Mean score below which an image is flagged.
        max_suggestions: Maximum number of suggestions to return.

    Returns:
        List of dicts with keys: ``file_name``, ``mean_score``, ``n_instances``.
    """
    with open(predictions_json) as f:
        preds = json.load(f)

    # Group by file_name
    by_file: dict[str, list[float]] = {}
    for p in preds:
        fname = p.get("file_name", "unknown")
        by_file.setdefault(fname, []).append(p.get("score", 0.0))

    suggestions = []
    for fname, scores in by_file.items():
        mean_score = float(np.mean(scores)) if scores else 0.0
        if mean_score < confidence_threshold:
            suggestions.append(
                {
                    "file_name": fname,
                    "mean_score": round(mean_score, 4),
                    "n_instances": len(scores),
                }
            )

    suggestions.sort(key=lambda x: x["mean_score"])
    return suggestions[:max_suggestions]


def find_uncertain_patches(
    predictions_json: str | Path,
    image_dir: str | Path,
    output_dir: str | Path,
    confidence_threshold: float = 0.6,
    max_suggestions: int = 50,
) -> list[Path]:
    """Copy low-confidence images to a review directory.

    Args:
        predictions_json: Path to inference predictions JSON.
        image_dir: Source image directory.
        output_dir: Where to copy uncertain images for review.
        confidence_threshold: Score threshold.
        max_suggestions: Maximum number of images to copy.

    Returns:
        List of copied image paths.
    """
    image_dir = Path(image_dir)
    output_dir = ensure_dir(output_dir)

    suggestions = find_low_confidence_images(predictions_json, confidence_threshold, max_suggestions)
    copied: list[Path] = []

    for s in suggestions:
        src = image_dir / s["file_name"]
        if src.exists():
            dst = output_dir / src.name
            shutil.copy2(src, dst)
            copied.append(dst)
            logger.info("Flagged for review: %s (mean_score=%.3f)", src.name, s["mean_score"])
        else:
            logger.warning("Image not found: %s", src)

    logger.info("Copied %d images for manual review → %s", len(copied), output_dir)
    return copied


# ---------------------------------------------------------------------------
# Self-training loop
# ---------------------------------------------------------------------------


def self_train(
    d2_config_path: str | Path,
    cellseg_config: dict[str, Any],
    initial_model: str | Path,
    unlabelled_dir: str | Path,
    dataset_dir: str | Path,
    output_dir: str | Path,
    n_iterations: int = 3,
    pseudo_label_threshold: float = 0.7,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Path:
    """Self-training loop: predict pseudo-labels → add to dataset → retrain.

    Each iteration:
      1. Run inference on *unlabelled_dir* using the current model.
      2. Keep high-confidence predictions as pseudo-labels.
      3. Convert pseudo-labels to COCO format and merge with *dataset_dir*.
      4. Retrain the model on the expanded dataset.

    Args:
        d2_config_path: Path to Detectron2 YAML config.
        cellseg_config: Merged cellseg_trainer config.
        initial_model: Starting model weights path.
        unlabelled_dir: Directory of unlabelled images.
        dataset_dir: Existing labelled COCO dataset directory.
        output_dir: Root output directory for all iterations.
        n_iterations: Number of self-training rounds.
        pseudo_label_threshold: Minimum confidence score for pseudo-labels.
        progress_callback: Optional callable(message) for progress reporting.

    Returns:
        Path to the final trained model weights.
    """
    from cellseg_trainer import inference as inf_module
    from cellseg_trainer import training as train_module
    from cellseg_trainer.coco_builder import build_coco_dataset

    output_dir = ensure_dir(output_dir)
    current_model = Path(initial_model)

    def _log(msg: str) -> None:
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    for iteration in range(1, n_iterations + 1):
        _log(f"Self-training iteration {iteration}/{n_iterations}")

        iter_dir = ensure_dir(output_dir / f"iteration_{iteration:02d}")
        pred_dir = ensure_dir(iter_dir / "predictions")

        # Step 1: Infer on unlabelled images
        _log("  Running inference on unlabelled images …")
        results = inf_module.run_inference(
            model_path=current_model,
            d2_config_path=d2_config_path,
            image_dir=unlabelled_dir,
            output_dir=pred_dir,
            score_thresh=pseudo_label_threshold,
            output_masks=True,
            output_overlay=False,
            output_json=True,
        )
        _log(f"  Generated {results['n_total_instances']} pseudo-label instances")

        # Step 2: Merge pseudo-labels with existing dataset
        merged_dir = ensure_dir(iter_dir / "merged_dataset")
        _merge_coco_datasets(
            base_dir=dataset_dir,
            pseudo_mask_dir=pred_dir / "masks",
            image_dir=unlabelled_dir,
            output_dir=merged_dir,
        )

        # Step 3: Retrain
        _log("  Retraining model on merged dataset …")
        train_out = ensure_dir(iter_dir / "model")

        # Update model weights to current checkpoint for fine-tuning
        override = dict(cellseg_config)
        override.setdefault("MODEL", {})["WEIGHTS"] = str(current_model)
        override.setdefault("TRAIN", {})["FAST_FINETUNE"] = True

        current_model = train_module.train(
            d2_config_path=d2_config_path,
            cellseg_config=override,
            dataset_dir=merged_dir,
            output_dir=train_out,
            multi_gpu=cellseg_config.get("TRAIN", {}).get("MULTI_GPU", True),
            amp=cellseg_config.get("TRAIN", {}).get("AMP", True),
        )
        _log(f"  Iteration {iteration} model saved → {current_model}")

    _log(f"Self-training complete. Final model: {current_model}")
    return current_model


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _merge_coco_datasets(
    base_dir: Path,
    pseudo_mask_dir: Path,
    image_dir: Path,
    output_dir: Path,
) -> None:
    """Merge base COCO JSON with pseudo-labelled data.

    Loads ``instances_train.json`` from *base_dir*, appends pseudo-label
    records from *pseudo_mask_dir*, and writes merged JSONs to *output_dir*.
    """
    import copy

    # Copy base dataset files
    for split in ("train", "val"):
        src = base_dir / f"instances_{split}.json"
        dst = output_dir / f"instances_{split}.json"
        if src.exists():
            shutil.copy2(src, dst)

    # Append pseudo-labels to train split only
    train_json = output_dir / "instances_train.json"
    if not train_json.exists():
        logger.warning("Base train JSON not found — cannot merge pseudo-labels")
        return

    with open(train_json) as f:
        coco = json.load(f)

    max_img_id = max((img["id"] for img in coco["images"]), default=0)
    max_ann_id = max((ann["id"] for ann in coco["annotations"]), default=0)

    from cellseg_trainer.cellpose_converter import convert_mask_array
    from cellseg_trainer.ome_loader import load_image_array
    from cellseg_trainer.utils import collect_images

    mask_paths = list(pseudo_mask_dir.glob("*_mask.tif"))
    for mask_path in mask_paths:
        stem = mask_path.stem.replace("_mask", "")
        # Find matching image
        img_candidates = list(image_dir.glob(f"{stem}.*"))
        if not img_candidates:
            continue
        img_path = img_candidates[0]
        arr = load_image_array(img_path)
        h, w = arr.shape[:2]
        mask = load_image_array(mask_path).astype(np.int32)

        max_img_id += 1
        coco["images"].append(
            {"id": max_img_id, "file_name": img_path.name, "height": h, "width": w}
        )
        anns, next_id = convert_mask_array(mask, image_id=max_img_id, start_annotation_id=max_ann_id + 1)
        coco["annotations"].extend(anns)
        max_ann_id = next_id - 1

    with open(train_json, "w") as f:
        json.dump(coco, f, indent=2)

    logger.info(
        "Merged dataset: %d images, %d annotations",
        len(coco["images"]),
        len(coco["annotations"]),
    )
