#!/usr/bin/env python3
"""
scripts/train_model.py
Train or fine-tune a Detectron2 model using a cellseg_trainer config.

Usage:
    python scripts/train_model.py \
        --dataset   dataset/ \
        --d2-config configs_detectron2/mask_rcnn_R_50_FPN.yaml \
        --config    configs_detectron2/example_config.yaml \
        --output    output/ \
        --multi-gpu \
        --amp \
        --patch-size 512
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cellseg_trainer.config_loader import load_config, DEFAULTS, override_from_args
from cellseg_trainer.training import train
from cellseg_trainer.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset",        required=True, help="COCO dataset directory")
    parser.add_argument("--d2-config",      required=True, help="Detectron2 YAML config path")
    parser.add_argument("--config",         default=None,  help="cellseg_trainer YAML config")
    parser.add_argument("--output",         required=True, help="Output directory")
    parser.add_argument("--multi-gpu",      action="store_true", default=True)
    parser.add_argument("--no-multi-gpu",   dest="multi_gpu", action="store_false")
    parser.add_argument("--amp",            action="store_true", default=True)
    parser.add_argument("--no-amp",         dest="amp", action="store_false")
    parser.add_argument("--num-gpus",       type=int, default=None)
    parser.add_argument("--max-iter",       type=int, default=None)
    parser.add_argument("--lr",             type=float, default=None)
    parser.add_argument("--patch-size",     type=int, default=512)
    parser.add_argument("--freeze-backbone",action="store_true", default=False)
    parser.add_argument("--fast-finetune",  action="store_true", default=False)
    parser.add_argument("--weights",        default=None, help="Pre-trained weights for fine-tuning")
    parser.add_argument("--resume",         action="store_true", default=False)
    parser.add_argument("--num-classes",    type=int, default=1)
    parser.add_argument("--log-level",      default="INFO")
    args = parser.parse_args()

    setup_logging(getattr(logging, args.log_level))
    log = logging.getLogger("train_model")

    cfg = load_config(args.config) if args.config else dict(DEFAULTS)
    cfg = override_from_args(
        cfg,
        **{
            "TRAIN.MAX_ITER":       args.max_iter,
            "TRAIN.BASE_LR":        args.lr,
            "TRAIN.MULTI_GPU":      args.multi_gpu,
            "TRAIN.AMP":            args.amp,
            "TRAIN.FAST_FINETUNE":  args.fast_finetune,
            "MODEL.FREEZE_BACKBONE":args.freeze_backbone,
            "MODEL.WEIGHTS":        args.weights,
            "MODEL.NUM_CLASSES":    args.num_classes,
            "DATASET.PATCH_SIZE":   args.patch_size,
        },
    )

    model_path = train(
        d2_config_path=args.d2_config,
        cellseg_config=cfg,
        dataset_dir=args.dataset,
        output_dir=args.output,
        multi_gpu=args.multi_gpu,
        amp=args.amp,
        num_gpus=args.num_gpus,
        resume=args.resume,
    )
    log.info("Training complete → %s", model_path)


if __name__ == "__main__":
    main()
