"""
inference.py — Run trained Detectron2 model on new microscopy images.

Outputs: segmentation masks (TIFF), overlay PNGs, COCO-style JSON.
Supports batch processing and multi-GPU inference.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from cellseg_trainer.ome_loader import load_image_array
from cellseg_trainer.utils import collect_images, ensure_dir, normalize_uint8, tqdm_or_plain

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core predictor
# ---------------------------------------------------------------------------


class CellSegPredictor:
    """Detectron2 predictor wrapper for cell segmentation.

    Args:
        model_path: Path to ``model_final.pth`` or a checkpoint.
        d2_config_path: Path to the Detectron2 YAML config used during training.
        score_thresh: Instance confidence threshold.
        device: Torch device string (``"cuda:0"`` etc.).
    """

    def __init__(
        self,
        model_path: str | Path,
        d2_config_path: str | Path,
        score_thresh: float = 0.5,
        device: str = "cuda:0",
    ) -> None:
        self.model_path = Path(model_path)
        self.d2_config_path = Path(d2_config_path)
        self.score_thresh = score_thresh
        self.device = device
        self._predictor = None

    def _build(self) -> None:
        """Lazy-initialize the Detectron2 predictor."""
        try:
            from detectron2.config import get_cfg  # type: ignore
            from detectron2.engine.defaults import DefaultPredictor  # type: ignore
        except ImportError as exc:
            raise ImportError("detectron2 is required") from exc

        cfg = get_cfg()
        cfg.merge_from_file(str(self.d2_config_path))
        cfg.MODEL.WEIGHTS = str(self.model_path)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.score_thresh
        cfg.MODEL.DEVICE = self.device
        cfg.freeze()
        self._predictor = DefaultPredictor(cfg)
        logger.info("Predictor built on %s", self.device)

    def predict(self, image: np.ndarray) -> dict[str, Any]:
        """Run inference on a single image.

        Args:
            image: uint8 RGB or grayscale array (H, W) or (H, W, 3).

        Returns:
            Detectron2 ``Instances`` wrapped in a dict with key ``"instances"``.
        """
        if self._predictor is None:
            self._build()

        import torch

        # Convert grayscale to BGR for Detectron2
        if image.ndim == 2:
            import cv2  # type: ignore

            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 1:
            import cv2  # type: ignore

            image = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)

        with torch.no_grad():
            return self._predictor(image)


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------


def run_inference(
    model_path: str | Path,
    d2_config_path: str | Path,
    image_dir: str | Path,
    output_dir: str | Path,
    score_thresh: float = 0.5,
    device: str = "cuda:0",
    output_masks: bool = True,
    output_overlay: bool = True,
    output_json: bool = True,
    channel: Optional[int] = None,
) -> dict[str, Any]:
    """Run inference on all images in a directory.

    Args:
        model_path: Trained model weights.
        d2_config_path: Detectron2 YAML config path.
        image_dir: Directory of input images.
        output_dir: Where to save results.
        score_thresh: Confidence threshold for detections.
        device: CUDA or CPU device string.
        output_masks: Save instance mask TIFFs.
        output_overlay: Save overlay PNG visualizations.
        output_json: Save COCO-style predictions JSON.
        channel: Image channel to use (for multi-channel inputs).

    Returns:
        Dict with ``"predictions"`` list and summary stats.
    """
    try:
        import tifffile  # type: ignore
    except ImportError as exc:
        raise ImportError("tifffile is required: pip install tifffile") from exc

    output_dir = ensure_dir(output_dir)
    masks_dir = ensure_dir(output_dir / "masks") if output_masks else None
    overlays_dir = ensure_dir(output_dir / "overlays") if output_overlay else None

    predictor = CellSegPredictor(model_path, d2_config_path, score_thresh, device)

    images = collect_images(Path(image_dir))
    if not images:
        logger.warning("No images found in %s", image_dir)
        return {"predictions": [], "n_images": 0}

    coco_results: list[dict] = []
    predictions_summary: list[dict] = []

    for img_path in tqdm_or_plain(images, desc="Running inference"):
        try:
            raw = load_image_array(img_path, channel=channel)
            u8 = normalize_uint8(raw)
            outputs = predictor.predict(u8)
            instances = outputs["instances"].to("cpu")

            n_instances = len(instances)
            scores = instances.scores.numpy().tolist() if instances.has("scores") else []

            # Build instance mask
            h, w = raw.shape[:2]
            inst_mask = np.zeros((h, w), dtype=np.int32)
            if instances.has("pred_masks"):
                for i, m in enumerate(instances.pred_masks.numpy(), start=1):
                    inst_mask[m > 0] = i

            if output_masks and masks_dir is not None:
                tifffile.imwrite(str(masks_dir / f"{img_path.stem}_mask.tif"), inst_mask)

            if output_overlay and overlays_dir is not None:
                _save_overlay(u8, inst_mask, overlays_dir / f"{img_path.stem}_overlay.png")

            # COCO predictions
            if output_json and instances.has("pred_boxes"):
                for i, (box, score) in enumerate(
                    zip(instances.pred_boxes.tensor.numpy(), scores)
                ):
                    x1, y1, x2, y2 = box
                    coco_results.append(
                        {
                            "file_name": img_path.name,
                            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                            "score": float(score),
                            "category_id": 1,
                        }
                    )

            predictions_summary.append(
                {"file": img_path.name, "n_instances": n_instances, "scores": scores}
            )
        except Exception as exc:
            logger.error("Error on %s: %s", img_path.name, exc)

    if output_json:
        json_path = output_dir / "predictions.json"
        with open(json_path, "w") as f:
            json.dump(coco_results, f, indent=2)
        logger.info("Saved predictions JSON → %s", json_path)

    return {
        "predictions": predictions_summary,
        "n_images": len(images),
        "n_total_instances": sum(p["n_instances"] for p in predictions_summary),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _save_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    path: Path,
    alpha: float = 0.5,
) -> None:
    """Save a simple color overlay of the mask on the image."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import matplotlib.colors as mcolors  # type: ignore
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap="gray")
    if mask.max() > 0:
        colored = plt.cm.tab20(mask % 20) if mask.max() > 0 else None
        if colored is not None:
            colored[..., 3] = (mask > 0).astype(float) * alpha
            ax.imshow(colored)
    ax.axis("off")
    fig.savefig(str(path), bbox_inches="tight", pad_inches=0, dpi=150)
    plt.close(fig)
