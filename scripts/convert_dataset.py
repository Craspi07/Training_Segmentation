#!/usr/bin/env python3
"""
scripts/convert_dataset.py
Convert Cellpose *_seg.npy masks and OME-TIF images to a COCO JSON dataset.

Usage:
    python scripts/convert_dataset.py \
        --images data/images \
        --masks  data/masks \
        --output dataset \
        --patch-size 512
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow running from repository root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from cellseg_trainer.coco_builder import build_coco_dataset
from cellseg_trainer.patch_extractor import extract_and_save_patches
from cellseg_trainer.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images",       required=True, help="Source image directory")
    parser.add_argument("--masks",        required=True, help="Cellpose mask directory")
    parser.add_argument("--output",       required=True, help="Output dataset directory")
    parser.add_argument("--mask-suffix",  default="_seg.npy", help="Mask file suffix")
    parser.add_argument("--train-split",  type=float, default=0.85)
    parser.add_argument("--workers",      type=int, default=4)
    parser.add_argument("--patch-size",   type=int, default=0,  help="0 = no patching")
    parser.add_argument("--overlap",      type=int, default=64)
    parser.add_argument("--log-level",    default="INFO")
    args = parser.parse_args()

    setup_logging(getattr(logging, args.log_level))
    log = logging.getLogger("convert_dataset")

    image_dir = Path(args.images)
    mask_dir  = Path(args.masks)
    output_dir = Path(args.output)
    mask_suffix = args.mask_suffix

    if args.patch_size > 0:
        log.info("Patch extraction: size=%d overlap=%d", args.patch_size, args.overlap)
        patch_dir = output_dir / "patches"
        n = extract_and_save_patches(
            image_dir=image_dir,
            mask_dir=mask_dir,
            output_dir=patch_dir,
            patch_size=args.patch_size,
            overlap=args.overlap,
            mask_suffix=mask_suffix,
        )
        log.info("Extracted %d patches", n)
        image_dir  = patch_dir / "images"
        mask_dir   = patch_dir / "masks"
        mask_suffix = "_mask.tif"

    train_json, val_json = build_coco_dataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        output_dir=output_dir,
        train_split=args.train_split,
        mask_suffix=mask_suffix,
        n_workers=args.workers,
    )
    log.info("Done.\n  Train: %s\n  Val:   %s", train_json, val_json)


if __name__ == "__main__":
    main()
