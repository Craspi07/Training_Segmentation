"""
Evaluation and inference module for trained Cellpose segmentation models.

Provides:
  - Running inference on new images with a trained model
  - Computing segmentation metrics (IoU, Dice, AP)
  - Saving segmentation results as labeled masks and overlays
"""

import os
import logging
from pathlib import Path

import numpy as np
import tifffile
from skimage import io as skio
from cellpose import models, utils, metrics

from data_preparation import load_dataset, load_image, save_image

logger = logging.getLogger(__name__)


def run_inference(
    model_path: str,
    image_dir: str,
    output_dir: str,
    channels: list[int] = [0, 0],
    diameter: float | None = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    normalize: dict | bool = True,
    use_gpu: bool = True,
) -> tuple[list[np.ndarray], list[str]]:
    """
    Run Cellpose inference on a directory of images.

    Args:
        model_path: Path to trained Cellpose model.
        image_dir: Directory containing images to segment.
        output_dir: Directory to save predicted masks.
        channels: Cellpose channel configuration [chan1, chan2].
        diameter: Expected object diameter (None for auto-detect).
        flow_threshold: Flow error threshold for mask filtering.
        cellprob_threshold: Cell probability threshold.
        normalize: Normalization settings.
        use_gpu: Whether to use GPU.

    Returns:
        masks: List of predicted mask arrays.
        names: List of corresponding filenames.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load model (supports BioImage.io identifiers and local paths)
    from train_cellpose import resolve_pretrained_model
    resolved = resolve_pretrained_model(model_path)
    logger.info(f"Loading model from {resolved or 'default cpsam'}")
    if resolved:
        model = models.CellposeModel(gpu=use_gpu, pretrained_model=resolved)
    else:
        model = models.CellposeModel(gpu=use_gpu)

    # Find images
    extensions = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg", "*.nd2")
    import glob
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
    image_files = sorted(image_files)

    if not image_files:
        logger.warning(f"No images found in {image_dir}")
        return [], []

    logger.info(f"Running inference on {len(image_files)} images...")

    # Load images
    images = [load_image(f) for f in image_files]
    names = [Path(f).name for f in image_files]

    # Run evaluation
    masks, flows, styles = model.eval(
        images,
        diameter=diameter,
        channels=channels,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        normalize=normalize,
    )

    # Save masks
    for mask, name in zip(masks, names):
        stem = Path(name).stem
        mask_path = os.path.join(output_dir, f"{stem}_masks.tif")
        tifffile.imwrite(mask_path, mask.astype(np.uint16))

    logger.info(f"Saved {len(masks)} predicted masks to {output_dir}")
    return masks, names


def compute_metrics(
    predicted_masks: list[np.ndarray],
    ground_truth_masks: list[np.ndarray],
    thresholds: list[float] | None = None,
) -> dict:
    """
    Compute segmentation quality metrics.

    Uses Cellpose's built-in metrics: average precision (AP) at multiple
    IoU thresholds, plus per-image IoU and Dice scores.

    Args:
        predicted_masks: List of predicted instance masks.
        ground_truth_masks: List of ground truth instance masks.
        thresholds: IoU thresholds for AP calculation.

    Returns:
        Dictionary with metric arrays and summary statistics.
    """
    if thresholds is None:
        thresholds = [0.5, 0.75, 0.9]

    # Cellpose average precision
    ap, tp, fp, fn = metrics.average_precision(
        ground_truth_masks, predicted_masks, threshold=thresholds
    )

    # Per-image IoU and Dice
    ious = []
    dices = []
    for pred, gt in zip(predicted_masks, ground_truth_masks):
        # Binary masks for overall IoU/Dice
        pred_bin = (pred > 0).astype(np.float32)
        gt_bin = (gt > 0).astype(np.float32)

        intersection = np.sum(pred_bin * gt_bin)
        union = np.sum(pred_bin) + np.sum(gt_bin) - intersection

        iou = intersection / union if union > 0 else 0.0
        dice = (2 * intersection) / (np.sum(pred_bin) + np.sum(gt_bin)) if (np.sum(pred_bin) + np.sum(gt_bin)) > 0 else 0.0

        ious.append(iou)
        dices.append(dice)

    results = {
        "average_precision": ap,
        "thresholds": thresholds,
        "mean_ap_50": float(np.mean(ap[:, 0])) if ap.size > 0 else 0.0,
        "mean_ap_75": float(np.mean(ap[:, 1])) if ap.shape[1] > 1 else 0.0,
        "mean_iou": float(np.mean(ious)),
        "mean_dice": float(np.mean(dices)),
        "per_image_iou": ious,
        "per_image_dice": dices,
    }

    logger.info(
        f"Metrics: mAP@0.5={results['mean_ap_50']:.4f}, "
        f"mAP@0.75={results['mean_ap_75']:.4f}, "
        f"mIoU={results['mean_iou']:.4f}, "
        f"mDice={results['mean_dice']:.4f}"
    )
    return results


def save_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    output_path: str,
    alpha: float = 0.4,
) -> None:
    """Save an overlay visualization of mask on image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original image
    if image.ndim == 3 and image.shape[0] in (1, 2, 3):
        display_img = np.moveaxis(image, 0, -1)
    else:
        display_img = image
    axes[0].imshow(display_img, cmap="gray" if display_img.ndim == 2 else None)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # Mask
    axes[1].imshow(mask, cmap="nipy_spectral", interpolation="nearest")
    axes[1].set_title(f"Segmentation ({mask.max()} objects)")
    axes[1].axis("off")

    # Overlay
    axes[2].imshow(display_img, cmap="gray" if display_img.ndim == 2 else None)
    mask_overlay = np.ma.masked_where(mask == 0, mask)
    axes[2].imshow(mask_overlay, cmap="nipy_spectral", alpha=alpha, interpolation="nearest")
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluate_model(config: dict, model_path: str) -> dict:
    """
    Full evaluation pipeline: run inference on test set and compute metrics.

    Args:
        config: Parsed YAML configuration.
        model_path: Path to trained model.

    Returns:
        Metrics dictionary.
    """
    data_cfg = config["DATA"]
    inf_cfg = config.get("INFERENCE", {})
    paths_cfg = config["PATHS"]
    system_cfg = config.get("SYSTEM", {})

    results_dir = paths_cfg.get("RESULT_DIR", "results")
    os.makedirs(results_dir, exist_ok=True)

    # Build normalize dict
    norm_cfg = inf_cfg.get("NORMALIZE", {})
    if norm_cfg:
        normalize = {"normalize": True}
        if "TILE_NORM_BLOCKSIZE" in norm_cfg:
            normalize["tile_norm_blocksize"] = norm_cfg["TILE_NORM_BLOCKSIZE"]
    else:
        normalize = True

    # Run inference
    pred_masks, pred_names = run_inference(
        model_path=model_path,
        image_dir=data_cfg["TEST"]["PATH"],
        output_dir=os.path.join(results_dir, "predictions"),
        channels=data_cfg.get("CHANNELS", [0, 0]),
        diameter=inf_cfg.get("DIAMETER"),
        flow_threshold=inf_cfg.get("FLOW_THRESHOLD", 0.4),
        cellprob_threshold=inf_cfg.get("CELLPROB_THRESHOLD", 0.0),
        normalize=normalize,
        use_gpu=system_cfg.get("NUM_GPUS", 0) > 0,
    )

    if not pred_masks:
        logger.warning("No predictions generated.")
        return {}

    # Load ground truth
    gt_images, gt_labels, gt_names = load_dataset(
        image_dir=data_cfg["TEST"]["PATH"],
        mask_dir=data_cfg["TEST"]["MASK_PATH"],
        image_filter=data_cfg["TEST"].get("IMAGE_FILTER", "_img"),
        mask_filter=data_cfg["TEST"].get("MASK_FILTER", "_masks"),
    )

    # Compute metrics
    result_metrics = {}
    if gt_labels:
        result_metrics = compute_metrics(pred_masks, gt_labels)

        # Save metrics
        import json
        metrics_path = os.path.join(results_dir, "metrics.json")
        serializable = {
            k: v if not isinstance(v, np.ndarray) else v.tolist()
            for k, v in result_metrics.items()
        }
        with open(metrics_path, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")

    # Save overlays for a sample of images
    overlay_dir = os.path.join(results_dir, "overlays")
    os.makedirs(overlay_dir, exist_ok=True)
    num_overlays = min(10, len(pred_masks))
    for i in range(num_overlays):
        img = load_image(
            os.path.join(data_cfg["TEST"]["PATH"], pred_names[i])
        )
        save_overlay(
            img, pred_masks[i],
            os.path.join(overlay_dir, f"{Path(pred_names[i]).stem}_overlay.png"),
        )

    return result_metrics


if __name__ == "__main__":
    import argparse
    import yaml

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Evaluate Cellpose segmentation model")
    parser.add_argument("--config", type=str, default=None, help="YAML config path (required for full evaluation)")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--image-dir", type=str, default=None, help="Image directory (inference-only mode)")
    parser.add_argument("--output-dir", type=str, default="results/inference", help="Output directory (inference-only mode)")
    args = parser.parse_args()

    if args.image_dir:
        masks, names = run_inference(
            model_path=args.model,
            image_dir=args.image_dir,
            output_dir=args.output_dir,
        )
        print(f"Inference complete: {len(masks)} images processed")
    elif args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        evaluate_model(config, args.model)
    else:
        parser.error("Either --config or --image-dir must be provided")
