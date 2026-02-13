"""
Cellpose training module for both DIC whole-cell and fluorescence nucleus
segmentation.

Follows a BiaPy-inspired workflow:
  1. Load configuration from YAML
  2. Prepare training/test data
  3. Optionally augment training data
  4. Train a Cellpose model (fine-tuning cpsam)
  5. Save model and training metrics
"""

import os
import logging
from pathlib import Path

import numpy as np
import yaml
from cellpose import io as cpio
from cellpose import models, train

from data_preparation import (
    load_dataset,
    prepare_augmented_dataset,
)

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load a BiaPy-style YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_pretrained_model(model_spec: str | None) -> str | None:
    """Resolve a pretrained model specification to a local file path.

    Supported formats for *model_spec*:

    * ``None`` / ``"null"`` — use the Cellpose default (cpsam).
    * A local file path (``/path/to/model``) — used directly.
    * A BioImage.io identifier or path:
      - ``"bioimage.io:<resource-id>"`` e.g. ``"bioimage.io:affable-shark"``
      - A path to a ``rdf.yaml`` / ``bioimageio.yaml`` file
      - A path to a BioImage.io ``.zip`` package

    For BioImage.io sources the function loads the model description,
    locates the ``pytorch_state_dict`` weights, and returns their local
    path so that Cellpose can load them with ``pretrained_model=``.

    Returns:
        Local path to model weights, or *None* for the Cellpose default.
    """
    if model_spec is None or str(model_spec).lower() in ("null", "none", ""):
        return None

    model_spec = str(model_spec).strip()

    # -- BioImage.io identifier (prefix) --
    if model_spec.startswith("bioimage.io:"):
        resource_id = model_spec[len("bioimage.io:"):]
        return _load_bioimageio_weights(resource_id)

    # -- BioImage.io YAML or ZIP file --
    spec_path = Path(model_spec)
    if spec_path.is_file() and spec_path.suffix.lower() in (".yaml", ".yml", ".zip"):
        name = spec_path.name.lower()
        if name in ("rdf.yaml", "bioimageio.yaml") or name.endswith(".zip"):
            return _load_bioimageio_weights(str(spec_path))
        # Check if it looks like a bioimageio yaml (contains 'format_version')
        if spec_path.suffix.lower() in (".yaml", ".yml"):
            try:
                with open(spec_path) as fh:
                    header = fh.read(512)
                if "format_version" in header and "weights" in header:
                    return _load_bioimageio_weights(str(spec_path))
            except Exception:
                pass

    # -- Plain local model path --
    return model_spec


def _load_bioimageio_weights(source: str) -> str:
    """Download / locate PyTorch weights from a BioImage.io model.

    Args:
        source: A bioimage.io resource identifier (e.g. ``"affable-shark"``)
                or a local path to ``rdf.yaml`` / ``.zip``.

    Returns:
        Path to the PyTorch state-dict weights file on disk.
    """
    try:
        from bioimageio.spec import load_description
    except ImportError:
        raise ImportError(
            "The 'bioimageio.spec' package is required to load BioImage.io "
            "models.  Install it with:  pip install 'bioimageio.core'"
        )

    logger.info(f"Loading BioImage.io model description: {source}")
    model_descr = load_description(source)

    # Locate pytorch_state_dict weights
    weights = getattr(model_descr, "weights", None)
    if weights is None:
        raise ValueError(f"BioImage.io model '{source}' has no weights field.")

    pt_weights = getattr(weights, "pytorch_state_dict", None)
    if pt_weights is None:
        available = getattr(weights, "available_formats", [])
        raise ValueError(
            f"BioImage.io model '{source}' has no pytorch_state_dict weights. "
            f"Available weight formats: {available}"
        )

    weights_path = str(pt_weights.source)
    logger.info(f"BioImage.io PyTorch weights resolved to: {weights_path}")
    return weights_path


def setup_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_normalize_dict(config: dict) -> dict | bool:
    """
    Build the Cellpose normalize parameter from config.

    Cellpose accepts either True (default percentile normalization) or a dict
    with keys like tile_norm_blocksize, percentile, invert, etc.
    """
    inf_cfg = config.get("INFERENCE", {})
    norm_cfg = inf_cfg.get("NORMALIZE", {})

    if not norm_cfg:
        return config.get("DATA", {}).get("NORMALIZE", True)

    cellpose_norm = {"normalize": True}
    if "TILE_NORM_BLOCKSIZE" in norm_cfg:
        cellpose_norm["tile_norm_blocksize"] = norm_cfg["TILE_NORM_BLOCKSIZE"]
    if "PERCENTILE" in norm_cfg:
        cellpose_norm["percentile"] = norm_cfg["PERCENTILE"]
    if "INVERT" in norm_cfg:
        cellpose_norm["invert"] = norm_cfg["INVERT"]

    return cellpose_norm


def train_cellpose_model(config: dict) -> str:
    """
    Train a Cellpose model based on the provided configuration.

    Args:
        config: Parsed YAML configuration dict.

    Returns:
        Path to the saved trained model.
    """
    # ---- Setup ----
    system_cfg = config.get("SYSTEM", {})
    setup_seed(system_cfg.get("SEED", 42))
    use_gpu = system_cfg.get("NUM_GPUS", 0) > 0

    data_cfg = config["DATA"]
    train_cfg = config["TRAIN"]
    paths_cfg = config["PATHS"]
    aug_cfg = config.get("AUGMENTATION", {})

    if not train_cfg.get("ENABLE", True):
        logger.info("Training disabled in config, skipping.")
        return ""

    # ---- Load training data ----
    logger.info("Loading training data...")
    train_images, train_labels, train_names = load_dataset(
        image_dir=data_cfg["TRAIN"]["PATH"],
        mask_dir=data_cfg["TRAIN"]["MASK_PATH"],
        image_filter=data_cfg["TRAIN"].get("IMAGE_FILTER", "_img"),
        mask_filter=data_cfg["TRAIN"].get("MASK_FILTER", "_masks"),
    )

    if len(train_images) == 0:
        raise ValueError(
            f"No training images found in {data_cfg['TRAIN']['PATH']}. "
            "Ensure images match the IMAGE_FILTER/MASK_FILTER naming convention."
        )

    # ---- Load test data ----
    logger.info("Loading test data...")
    test_images, test_labels, test_names = load_dataset(
        image_dir=data_cfg["TEST"]["PATH"],
        mask_dir=data_cfg["TEST"]["MASK_PATH"],
        image_filter=data_cfg["TEST"].get("IMAGE_FILTER", "_img"),
        mask_filter=data_cfg["TEST"].get("MASK_FILTER", "_masks"),
    )

    # ---- Apply augmentation ----
    if aug_cfg.get("ENABLE", False):
        logger.info("Applying data augmentation to training set...")
        train_images, train_labels = prepare_augmented_dataset(
            train_images, train_labels, aug_cfg, num_augmented_per_image=4
        )

    # ---- Initialize Cellpose model ----
    model_cfg = config.get("MODEL", {})
    pretrained_spec = model_cfg.get("PRETRAINED_MODEL", None)
    pretrained = resolve_pretrained_model(pretrained_spec)

    logger.info(f"Initializing CellposeModel (gpu={use_gpu})")
    if pretrained:
        logger.info(f"Pretrained model: {pretrained}")
        model = models.CellposeModel(gpu=use_gpu, pretrained_model=pretrained)
    else:
        # Default: start from the built-in cpsam model
        model = models.CellposeModel(gpu=use_gpu)

    # ---- Prepare save directory ----
    model_dir = paths_cfg.get("MODEL_DIR", "models")
    os.makedirs(model_dir, exist_ok=True)
    model_name = paths_cfg.get("MODEL_NAME", "cellpose_model")

    # ---- Train ----
    logger.info(
        f"Starting training: {train_cfg['EPOCHS']} epochs, "
        f"lr={train_cfg['LEARNING_RATE']}, batch_size={train_cfg['BATCH_SIZE']}"
    )

    normalize = build_normalize_dict(config)

    model_path, train_losses, test_losses = train.train_seg(
        net=model.net,
        train_data=train_images,
        train_labels=train_labels,
        test_data=test_images if test_images else None,
        test_labels=test_labels if test_labels else None,
        learning_rate=train_cfg.get("LEARNING_RATE", 1e-5),
        weight_decay=train_cfg.get("WEIGHT_DECAY", 0.1),
        n_epochs=train_cfg.get("EPOCHS", 100),
        batch_size=train_cfg.get("BATCH_SIZE", 1),
        normalize=normalize,
        save_path=model_dir,
        save_every=train_cfg.get("SAVE_EVERY", 25),
        min_train_masks=train_cfg.get("MIN_TRAIN_MASKS", 5),
        model_name=model_name,
    )

    logger.info(f"Training complete. Model saved to: {model_path}")
    logger.info(f"Final train loss: {train_losses[-1]:.4f}")
    if test_losses:
        logger.info(f"Final test loss: {test_losses[-1]:.4f}")

    # ---- Save loss curves ----
    results_dir = paths_cfg.get("RESULT_DIR", "results")
    os.makedirs(results_dir, exist_ok=True)
    _save_loss_curves(train_losses, test_losses, results_dir, model_name)

    return model_path


def _save_loss_curves(
    train_losses: list,
    test_losses: list,
    results_dir: str,
    model_name: str,
) -> None:
    """Save training/test loss curves as a plot and CSV."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_losses, label="Train Loss")
    if test_losses:
        ax.plot(test_losses, label="Test Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training Curves - {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plot_path = os.path.join(results_dir, f"{model_name}_loss.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Loss curves saved to {plot_path}")

    # Save raw loss values
    csv_path = os.path.join(results_dir, f"{model_name}_loss.csv")
    with open(csv_path, "w") as f:
        f.write("epoch,train_loss,test_loss\n")
        for i, tl in enumerate(train_losses):
            test_l = test_losses[i] if test_losses and i < len(test_losses) else ""
            f.write(f"{i},{tl},{test_l}\n")


def train_dic_wholecell(config_path: str = "configs/dic_wholecell.yaml") -> str:
    """Train a Cellpose model for DIC brightfield whole-cell segmentation."""
    logger.info("=" * 60)
    logger.info("DIC Brightfield Whole-Cell Segmentation Training")
    logger.info("=" * 60)
    config = load_config(config_path)
    return train_cellpose_model(config)


def train_fluor_nucleus(config_path: str = "configs/fluor_nucleus.yaml") -> str:
    """Train a Cellpose model for fluorescence nucleus segmentation."""
    logger.info("=" * 60)
    logger.info("Fluorescence Nucleus Segmentation Training")
    logger.info("=" * 60)
    config = load_config(config_path)
    return train_cellpose_model(config)


if __name__ == "__main__":
    cpio.logger_setup()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    import argparse

    parser = argparse.ArgumentParser(description="Train Cellpose segmentation models")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train_cellpose_model(config)
