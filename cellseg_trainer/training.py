"""
training.py — Detectron2-based training with multi-GPU and AMP support.

Loads an existing Detectron2 YAML config, overrides dataset paths and
training hyper-parameters, then runs distributed training via
``detectron2.engine.launch``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trainer class
# ---------------------------------------------------------------------------


class CellSegTrainer:
    """Thin wrapper around Detectron2's DefaultTrainer for cell segmentation.

    Args:
        cellseg_config: cellseg_trainer merged config dict (from config_loader).
        d2_config_path: Path to the existing Detectron2 YAML file.
    """

    def __init__(
        self,
        cellseg_config: dict[str, Any],
        d2_config_path: str | Path,
    ) -> None:
        self.cfg = cellseg_config
        self.d2_config_path = Path(d2_config_path)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def train(
        self,
        dataset_dir: str | Path,
        output_dir: str | Path,
        multi_gpu: bool = True,
        amp: bool = True,
        num_gpus: Optional[int] = None,
        resume: bool = False,
    ) -> Path:
        """Launch training.

        Args:
            dataset_dir: Directory with ``instances_train.json`` / ``instances_val.json``.
            output_dir: Where to save checkpoints and logs.
            multi_gpu: Enable multi-GPU training via ``launch``.
            amp: Enable automatic mixed precision.
            num_gpus: Number of GPUs to use (defaults to all available).
            resume: Resume from last checkpoint if available.

        Returns:
            Path to the final model weights (``model_final.pth``).
        """
        from detectron2.engine import launch  # type: ignore

        dataset_dir = Path(dataset_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        import torch

        gpus = num_gpus or (torch.cuda.device_count() if multi_gpu else 1)
        gpus = max(gpus, 1)

        logger.info("Starting training — GPUs: %d, AMP: %s", gpus, amp)

        if gpus > 1 and multi_gpu:
            launch(
                _train_worker,
                num_gpus_per_machine=gpus,
                args=(self.d2_config_path, self.cfg, dataset_dir, output_dir, amp, resume),
            )
        else:
            _train_worker(
                self.d2_config_path,
                self.cfg,
                dataset_dir,
                output_dir,
                amp,
                resume,
            )

        # Detectron2 saves checkpoints as .pth
        final_model = output_dir / "model_final.pth"
        if final_model.exists():
            logger.info("Training complete → %s", final_model)
        else:
            logger.warning("model_final.pth not found in %s — check output dir", output_dir)
        return final_model


# ---------------------------------------------------------------------------
# Worker (runs inside each GPU process)
# ---------------------------------------------------------------------------


def _train_worker(
    d2_config_path: Path,
    cellseg_cfg: dict[str, Any],
    dataset_dir: Path,
    output_dir: Path,
    amp: bool,
    resume: bool,
) -> None:
    """Training function executed inside each distributed process."""
    try:
        from detectron2.config import get_cfg  # type: ignore
        from detectron2.engine import DefaultTrainer  # type: ignore
        from detectron2.utils.logger import setup_logger  # type: ignore
    except ImportError as exc:
        raise ImportError("detectron2 is required for training") from exc

    from cellseg_trainer.dataset_manager import register_coco_datasets

    setup_logger()

    # --- Build Detectron2 config ---
    cfg = get_cfg()
    cfg.merge_from_file(str(d2_config_path))

    # Register and override datasets
    train_name, val_name = register_coco_datasets(dataset_dir)
    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATASETS.TEST = (val_name,)

    # Apply cellseg overrides
    train_cfg = cellseg_cfg.get("TRAIN", {})
    model_cfg = cellseg_cfg.get("MODEL", {})
    dataset_cfg = cellseg_cfg.get("DATASET", {})

    cfg.OUTPUT_DIR = str(output_dir)
    cfg.SOLVER.MAX_ITER = train_cfg.get("MAX_ITER", cfg.SOLVER.MAX_ITER)
    cfg.SOLVER.BASE_LR = train_cfg.get("BASE_LR", cfg.SOLVER.BASE_LR)
    cfg.SOLVER.IMS_PER_BATCH = train_cfg.get("IMS_PER_BATCH", cfg.SOLVER.IMS_PER_BATCH)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = train_cfg.get("ROI_BATCH_SIZE", 128)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = model_cfg.get("NUM_CLASSES", 1)
    cfg.DATALOADER.NUM_WORKERS = train_cfg.get("NUM_WORKERS", 4)
    cfg.TEST.EVAL_PERIOD = train_cfg.get("EVAL_PERIOD", 500)
    cfg.SOLVER.CHECKPOINT_PERIOD = train_cfg.get("CHECKPOINT_PERIOD", 500)

    # Weights override (fine-tuning) — supports .pt, .pth, .pkl
    weights = model_cfg.get("WEIGHTS", "")
    if weights and Path(weights).exists():
        from cellseg_trainer.utils import prepare_weights_for_detectron2
        weights = prepare_weights_for_detectron2(weights, cache_dir=output_dir)
        cfg.MODEL.WEIGHTS = weights
        logger.info("Fine-tuning from weights: %s", weights)

    # Fast fine-tune mode
    if train_cfg.get("FAST_FINETUNE", False):
        cfg.SOLVER.MAX_ITER = train_cfg.get("FAST_FINETUNE_ITER", 1000)
        logger.info("Fast fine-tune mode: %d iterations", cfg.SOLVER.MAX_ITER)

    # Patch size for input
    patch_size = dataset_cfg.get("PATCH_SIZE", 512)
    cfg.INPUT.MIN_SIZE_TRAIN = (patch_size,)
    cfg.INPUT.MAX_SIZE_TRAIN = patch_size * 2
    cfg.INPUT.MIN_SIZE_TEST = patch_size
    cfg.INPUT.MAX_SIZE_TEST = patch_size * 2

    # AMP
    cfg.SOLVER.AMP.ENABLED = amp

    # Freeze backbone
    if model_cfg.get("FREEZE_BACKBONE", False):
        cfg.MODEL.BACKBONE.FREEZE_AT = 4
        logger.info("Backbone frozen up to stage 4")

    cfg.freeze()

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=resume)
    trainer.train()


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def train(
    d2_config_path: str | Path,
    cellseg_config: dict[str, Any],
    dataset_dir: str | Path,
    output_dir: str | Path,
    multi_gpu: bool = True,
    amp: bool = True,
    num_gpus: Optional[int] = None,
    resume: bool = False,
) -> Path:
    """Shortcut to build a :class:`CellSegTrainer` and run training.

    Args:
        d2_config_path: Path to existing Detectron2 YAML.
        cellseg_config: Merged cellseg_trainer config dict.
        dataset_dir: COCO dataset directory.
        output_dir: Training output directory.
        multi_gpu: Enable multi-GPU.
        amp: Enable AMP.
        num_gpus: GPU count override.
        resume: Resume training.

    Returns:
        Path to ``model_final.pth`` (Detectron2 output format).
    """
    trainer = CellSegTrainer(cellseg_config, d2_config_path)
    return trainer.train(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        multi_gpu=multi_gpu,
        amp=amp,
        num_gpus=num_gpus,
        resume=resume,
    )
