#!/usr/bin/env python3
"""
scripts/run_inference.py
Run a trained Detectron2 model on a directory of microscopy images.

Usage:
    python scripts/run_inference.py \
        --model   output/model_final.pth \
        --d2-config configs_detectron2/mask_rcnn_R_50_FPN.yaml \
        --images  new_images/ \
        --output  predictions/
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cellseg_trainer.inference import run_inference
from cellseg_trainer.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model",         required=True, help="Trained model weights (.pth)")
    parser.add_argument("--d2-config",     required=True, help="Detectron2 YAML config")
    parser.add_argument("--images",        required=True, help="Input image directory")
    parser.add_argument("--output",        required=True, help="Output directory")
    parser.add_argument("--score-thresh",  type=float, default=0.5)
    parser.add_argument("--device",        default="cuda:0")
    parser.add_argument("--channel",       type=int, default=None)
    parser.add_argument("--no-masks",      dest="output_masks",   action="store_false", default=True)
    parser.add_argument("--no-overlay",    dest="output_overlay", action="store_false", default=True)
    parser.add_argument("--no-json",       dest="output_json",    action="store_false", default=True)
    parser.add_argument("--log-level",     default="INFO")
    args = parser.parse_args()

    setup_logging(getattr(logging, args.log_level))
    log = logging.getLogger("run_inference")

    results = run_inference(
        model_path=args.model,
        d2_config_path=args.d2_config,
        image_dir=args.images,
        output_dir=args.output,
        score_thresh=args.score_thresh,
        device=args.device,
        output_masks=args.output_masks,
        output_overlay=args.output_overlay,
        output_json=args.output_json,
        channel=args.channel,
    )
    log.info(
        "Done: %d images processed, %d total instances detected",
        results["n_images"],
        results["n_total_instances"],
    )


if __name__ == "__main__":
    main()
