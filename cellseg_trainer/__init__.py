"""
cellseg_trainer — Production-quality cell segmentation package.

Supports Cellpose mask conversion, Detectron2-based training,
multi-GPU inference, active learning, and visualization.
"""

__version__ = "0.1.0"
__author__ = "cellseg_trainer contributors"

from cellseg_trainer import (
    cellpose_converter,
    coco_builder,
    ome_loader,
    patch_extractor,
    augmentations,
    training,
    inference,
    visualization,
    active_learning,
    utils,
    config_loader,
)

__all__ = [
    "cellpose_converter",
    "coco_builder",
    "ome_loader",
    "patch_extractor",
    "augmentations",
    "training",
    "inference",
    "visualization",
    "active_learning",
    "utils",
    "config_loader",
]
