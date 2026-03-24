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

# Keys that only appear in BioImage Model Zoo RDF files (never in Detectron2 configs)
_BIOIMAGEIO_KEYS = {"format_version", "attachments", "rdf_source", "covers", "cite", "tags", "authors"}
# At least one of these top-level keys must exist in a valid Detectron2 config
_D2_REQUIRED_KEYS = {"MODEL", "SOLVER", "DATASETS", "DATALOADER", "INPUT", "TEST"}


def _is_valid_d2_yaml(path: Path) -> bool:
    """Return True if *path* is a readable Detectron2 YAML config."""
    import yaml
    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            return False
        keys = set(data.keys())
        return bool(keys & _D2_REQUIRED_KEYS) and not bool(keys & _BIOIMAGEIO_KEYS)
    except Exception:
        return False


def resolve_d2_config(path: str | Path) -> Path:
    """Resolve a Detectron2 config path, auto-detecting it from a BioImage RDF if needed.

    If *path* is already a valid Detectron2 YAML, returns it unchanged.
    If *path* is a BioImage Model Zoo ``rdf.yaml``, searches the ``attachments``
    section and the same directory for a Detectron2 config and returns it.

    Raises:
        ValueError: If no valid Detectron2 config can be found.
    """
    import yaml
    path = Path(path)

    if _is_valid_d2_yaml(path):
        return path

    # Check if it is a BioImage RDF
    try:
        with open(path) as f:
            rdf = yaml.safe_load(f) or {}
    except Exception as exc:
        raise ValueError(f"Cannot read '{path}': {exc}") from exc

    keys = set(rdf.keys()) if isinstance(rdf, dict) else set()
    is_rdf = bool(keys & _BIOIMAGEIO_KEYS)

    candidates: list[Path] = []

    if is_rdf:
        # 1. Collect YAML files listed in attachments
        attachments = rdf.get("attachments", {})
        # BioImage Zoo format_version <0.5: attachments is a dict with 'files' list
        # format_version >=0.5: attachments is a list of dicts with 'source'
        if isinstance(attachments, dict):
            for item in attachments.get("files", []):
                p = path.parent / str(item)
                if p.suffix in {".yaml", ".yml"} and p != path:
                    candidates.append(p)
        elif isinstance(attachments, list):
            for item in attachments:
                src = item.get("source", "") if isinstance(item, dict) else str(item)
                p = path.parent / str(src)
                if p.suffix in {".yaml", ".yml"} and p != path:
                    candidates.append(p)

    # 2. Fall back: scan the same directory for any valid D2 config
    for p in sorted(path.parent.glob("*.yaml")) + sorted(path.parent.glob("*.yml")):
        if p != path and p not in candidates:
            candidates.append(p)

    for candidate in candidates:
        if candidate.exists() and _is_valid_d2_yaml(candidate):
            logger.info("Auto-resolved Detectron2 config from '%s' → '%s'", path.name, candidate.name)
            return candidate

    if is_rdf:
        raise ValueError(
            f"'{path}' is a BioImage Model Zoo RDF file, and no Detectron2 config "
            f"was found alongside it in '{path.parent}'.\n"
            "Expected a 'config.yaml' file with MODEL/SOLVER keys in the same folder."
        )
    raise ValueError(
        f"'{path}' does not appear to be a Detectron2 config "
        f"(none of {sorted(_D2_REQUIRED_KEYS)} found at top level).\n"
        "Please select a valid Detectron2 YAML config file."
    )


def _assert_detectron2_yaml(path: Path) -> None:
    """Raise a descriptive ValueError if *path* is not a Detectron2 config (no auto-resolve)."""
    resolve_d2_config(path)  # raises on failure, result discarded — caller uses original path


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
                dist_url="auto",
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
    d2_config_path = resolve_d2_config(d2_config_path)
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
