"""
config_loader.py — Load, validate, and merge YAML configurations for cellseg_trainer.
Supports Detectron2-style YAML configs alongside cellseg_trainer training overrides.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULTS: dict[str, Any] = {
    "DATASET": {
        "TRAIN_IMAGE_DIR": "",
        "TRAIN_MASK_DIR": "",
        "VAL_IMAGE_DIR": "",
        "VAL_MASK_DIR": "",
        "OUTPUT_DIR": "dataset",
        "PATCH_SIZE": 512,
        "PATCH_OVERLAP": 64,
        "TRAIN_SPLIT": 0.85,
    },
    "MODEL": {
        "DETECTRON2_CONFIG": "",
        "WEIGHTS": "",
        "NUM_CLASSES": 1,
        "FREEZE_BACKBONE": False,
    },
    "TRAIN": {
        "OUTPUT_DIR": "output",
        "MAX_ITER": 5000,
        "BASE_LR": 0.00025,
        "BATCH_SIZE": 2,
        "NUM_WORKERS": 4,
        "MULTI_GPU": True,
        "AMP": True,
        "CHECKPOINT_PERIOD": 500,
        "EVAL_PERIOD": 500,
        "FAST_FINETUNE": False,
        "FAST_FINETUNE_ITER": 1000,
        "IMS_PER_BATCH": 2,
        "ROI_BATCH_SIZE": 128,
    },
    "AUGMENTATION": {
        "ENABLE": True,
        "FLIP_HORIZONTAL": True,
        "FLIP_VERTICAL": True,
        "ROTATION": True,
        "ROTATION_RANGE": 360,
        "INTENSITY_SCALE": True,
        "INTENSITY_SCALE_RANGE": [0.8, 1.2],
        "GAUSSIAN_NOISE": True,
        "GAUSSIAN_NOISE_STD": 0.05,
        "ELASTIC_DEFORMATION": False,
        "ELASTIC_ALPHA": 50.0,
        "ELASTIC_SIGMA": 5.0,
    },
    "INFERENCE": {
        "SCORE_THRESH": 0.5,
        "NMS_THRESH": 0.5,
        "BATCH_SIZE": 1,
        "OUTPUT_MASKS": True,
        "OUTPUT_OVERLAY": True,
        "OUTPUT_JSON": True,
    },
    "ACTIVE_LEARNING": {
        "CONFIDENCE_THRESHOLD": 0.6,
        "MAX_SUGGESTIONS": 50,
        "RETRAIN_THRESHOLD": 20,
        "SELF_TRAIN_ITERATIONS": 3,
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a cellseg_trainer YAML config, merging with defaults.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Merged configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the YAML is malformed.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        user_cfg = yaml.safe_load(f) or {}

    merged = _deep_merge(DEFAULTS, user_cfg)
    logger.info("Loaded config from %s", config_path)
    return merged


def save_config(config: dict[str, Any], output_path: str | Path) -> None:
    """Save a configuration dictionary to YAML.

    Args:
        config: Configuration dictionary.
        output_path: Destination YAML path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info("Saved config to %s", output_path)


def override_from_args(config: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Apply CLI argument overrides to a config dict.

    Dot-notation keys (e.g. ``TRAIN.BASE_LR``) are resolved into nested keys.

    Args:
        config: Base configuration dictionary.
        **kwargs: Key-value overrides (None values are skipped).

    Returns:
        Updated configuration dictionary.
    """
    for key, value in kwargs.items():
        if value is None:
            continue
        parts = key.split(".")
        d = config
        for part in parts[:-1]:
            existing = d.get(part)
            if not isinstance(existing, dict):
                # Replace a missing or non-dict intermediate node with a dict
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return config


def load_detectron2_config(d2_config_path: str | Path) -> Any:
    """Load a Detectron2 CfgNode from a YAML file.

    Args:
        d2_config_path: Path to a Detectron2-compatible YAML config.

    Returns:
        Detectron2 CfgNode with values loaded.

    Raises:
        ImportError: If detectron2 is not installed.
    """
    try:
        from detectron2.config import get_cfg
    except ImportError as exc:
        raise ImportError(
            "detectron2 is required for this feature. "
            "Install it from https://github.com/facebookresearch/detectron2"
        ) from exc

    cfg = get_cfg()
    cfg.merge_from_file(str(d2_config_path))
    return cfg


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
