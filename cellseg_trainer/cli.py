"""
cli.py — Command-line interface for cellseg_trainer.

Entry point: ``cellseg_trainer`` (defined in pyproject.toml).

Sub-commands:
  convert      Convert Cellpose masks to COCO dataset
  train        Train / fine-tune a Detectron2 model
  predict      Run inference on new images
  self-train   Active learning / self-training loop
  visualize    Visualize image/mask pairs
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cellseg_trainer",
        description="Cell segmentation training and inference toolkit.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        metavar="PATH",
        help="Optional path to write log output",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ---- convert ----
    p_conv = sub.add_parser("convert", help="Convert Cellpose masks to COCO JSON dataset")
    p_conv.add_argument("--images", required=True, metavar="DIR", help="Source image directory")
    p_conv.add_argument("--masks", required=True, metavar="DIR", help="Cellpose mask directory")
    p_conv.add_argument("--output", required=True, metavar="DIR", help="Output dataset directory")
    p_conv.add_argument("--mask-suffix", default="_seg.npy", help="Mask file suffix (default: _seg.npy)")
    p_conv.add_argument("--train-split", type=float, default=0.85, help="Train fraction (default: 0.85)")
    p_conv.add_argument("--workers", type=int, default=4, help="Parallel workers (default: 4)")
    p_conv.add_argument("--patch-size", type=int, default=0, help="Extract patches of this size before conversion (0=disabled)")
    p_conv.add_argument("--overlap", type=int, default=64, help="Patch overlap pixels (default: 64)")

    # ---- train ----
    p_train = sub.add_parser("train", help="Train Detectron2 model")
    p_train.add_argument("--dataset", required=True, metavar="DIR", help="COCO dataset directory")
    p_train.add_argument("--d2-config", required=True, metavar="YAML", help="Detectron2 YAML config path")
    p_train.add_argument("--config", default=None, metavar="YAML", help="cellseg_trainer YAML config (optional)")
    p_train.add_argument("--output", required=True, metavar="DIR", help="Output directory")
    p_train.add_argument("--multi-gpu", action="store_true", default=True, help="Enable multi-GPU training")
    p_train.add_argument("--no-multi-gpu", dest="multi_gpu", action="store_false")
    p_train.add_argument("--amp", action="store_true", default=True, help="Enable AMP (default: on)")
    p_train.add_argument("--no-amp", dest="amp", action="store_false")
    p_train.add_argument("--num-gpus", type=int, default=None, help="Number of GPUs (default: all available)")
    p_train.add_argument("--max-iter", type=int, default=None, help="Override max training iterations")
    p_train.add_argument("--lr", type=float, default=None, help="Override base learning rate")
    p_train.add_argument("--patch-size", type=int, default=512, help="Input patch size (default: 512)")
    p_train.add_argument("--freeze-backbone", action="store_true", default=False, help="Freeze backbone layers")
    p_train.add_argument("--fast-finetune", action="store_true", default=False, help="Fast fine-tuning mode")
    p_train.add_argument("--weights", default=None, metavar="PATH", help="Pre-trained weights to fine-tune from (.pt, .pth, or .pkl)")
    p_train.add_argument("--resume", action="store_true", default=False, help="Resume from last checkpoint")
    p_train.add_argument("--num-classes", type=int, default=1, help="Number of object classes (default: 1)")

    # ---- predict ----
    p_pred = sub.add_parser("predict", help="Run inference on images")
    p_pred.add_argument("--model", required=True, metavar="PATH", help="Trained model weights (.pt, .pth, or .pkl)")
    p_pred.add_argument("--d2-config", required=True, metavar="YAML", help="Detectron2 YAML config path")
    p_pred.add_argument("--images", required=True, metavar="DIR", help="Input image directory")
    p_pred.add_argument("--output", required=True, metavar="DIR", help="Output directory")
    p_pred.add_argument("--score-thresh", type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    p_pred.add_argument("--device", default="cuda:0", help="Device string (default: cuda:0)")
    p_pred.add_argument("--channel", type=int, default=None, help="Image channel to use")
    p_pred.add_argument("--no-masks", dest="output_masks", action="store_false", default=True)
    p_pred.add_argument("--no-overlay", dest="output_overlay", action="store_false", default=True)
    p_pred.add_argument("--no-json", dest="output_json", action="store_false", default=True)

    # ---- self-train ----
    p_st = sub.add_parser("self-train", help="Self-training / active learning loop")
    p_st.add_argument("--dataset", required=True, metavar="DIR", help="Labelled COCO dataset directory")
    p_st.add_argument("--model", required=True, metavar="PATH", help="Initial model weights (.pt, .pth, or .pkl)")
    p_st.add_argument("--d2-config", required=True, metavar="YAML", help="Detectron2 YAML config")
    p_st.add_argument("--unlabelled", required=True, metavar="DIR", help="Unlabelled image directory")
    p_st.add_argument("--output", required=True, metavar="DIR", help="Output directory")
    p_st.add_argument("--config", default=None, metavar="YAML", help="cellseg_trainer YAML config")
    p_st.add_argument("--iterations", type=int, default=3, help="Self-training rounds (default: 3)")
    p_st.add_argument("--pseudo-thresh", type=float, default=0.7, help="Pseudo-label confidence threshold")
    p_st.add_argument("--suggest-only", action="store_true", help="Only suggest uncertain images, do not retrain")

    # ---- visualize ----
    p_vis = sub.add_parser("visualize", help="Visualize image/mask pairs")
    p_vis.add_argument("--images", required=True, metavar="DIR")
    p_vis.add_argument("--masks", required=True, metavar="DIR")
    p_vis.add_argument("--output", required=True, metavar="DIR")
    p_vis.add_argument("--max-images", type=int, default=20, help="Max images to visualize (default: 20)")
    p_vis.add_argument("--alpha", type=float, default=0.5)
    p_vis.add_argument("--save-tiff", action="store_true")

    return parser


def main() -> None:
    """Entry point for the ``cellseg_trainer`` CLI."""
    from cellseg_trainer.utils import setup_logging

    parser = _build_parser()
    args = parser.parse_args()
    setup_logging(getattr(logging, args.log_level), args.log_file)
    logger = logging.getLogger("cellseg_trainer.cli")

    if args.command == "convert":
        _cmd_convert(args, logger)
    elif args.command == "train":
        _cmd_train(args, logger)
    elif args.command == "predict":
        _cmd_predict(args, logger)
    elif args.command == "self-train":
        _cmd_self_train(args, logger)
    elif args.command == "visualize":
        _cmd_visualize(args, logger)


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------


def _cmd_convert(args: argparse.Namespace, logger: logging.Logger) -> None:
    from cellseg_trainer.coco_builder import build_coco_dataset
    from cellseg_trainer.patch_extractor import extract_and_save_patches

    image_dir = Path(args.images)
    mask_dir = Path(args.masks)
    output_dir = Path(args.output)

    if args.patch_size > 0:
        logger.info("Extracting patches (size=%d, overlap=%d) …", args.patch_size, args.overlap)
        patch_dir = output_dir / "patches"
        extract_and_save_patches(
            image_dir=image_dir,
            mask_dir=mask_dir,
            output_dir=patch_dir,
            patch_size=args.patch_size,
            overlap=args.overlap,
            mask_suffix=args.mask_suffix,
        )
        image_dir = patch_dir / "images"
        mask_dir = patch_dir / "masks"
        mask_suffix = "_mask.tif"
    else:
        mask_suffix = args.mask_suffix

    train_path, val_path = build_coco_dataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        output_dir=output_dir,
        train_split=args.train_split,
        mask_suffix=mask_suffix,
        n_workers=args.workers,
    )
    logger.info("Dataset created:\n  Train: %s\n  Val:   %s", train_path, val_path)


def _cmd_train(args: argparse.Namespace, logger: logging.Logger) -> None:
    from cellseg_trainer.config_loader import load_config, DEFAULTS, override_from_args
    from cellseg_trainer.training import train

    cfg = load_config(args.config) if args.config else dict(DEFAULTS)
    cfg = override_from_args(
        cfg,
        **{
            "TRAIN.MAX_ITER": args.max_iter,
            "TRAIN.BASE_LR": args.lr,
            "TRAIN.MULTI_GPU": args.multi_gpu,
            "TRAIN.AMP": args.amp,
            "TRAIN.FAST_FINETUNE": args.fast_finetune,
            "MODEL.FREEZE_BACKBONE": args.freeze_backbone,
            "MODEL.WEIGHTS": args.weights,
            "MODEL.NUM_CLASSES": args.num_classes,
            "DATASET.PATCH_SIZE": args.patch_size,
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
    logger.info("Training finished → %s", model_path)


def _cmd_predict(args: argparse.Namespace, logger: logging.Logger) -> None:
    from cellseg_trainer.inference import run_inference

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
    logger.info(
        "Inference complete: %d images, %d total instances",
        results["n_images"],
        results["n_total_instances"],
    )


def _cmd_self_train(args: argparse.Namespace, logger: logging.Logger) -> None:
    from cellseg_trainer.config_loader import load_config, DEFAULTS
    from cellseg_trainer.active_learning import self_train, find_uncertain_patches

    cfg = load_config(args.config) if args.config else dict(DEFAULTS)

    if args.suggest_only:
        # Only flag uncertain images for manual review
        pred_json = Path(args.output) / "predictions.json"
        if not pred_json.exists():
            logger.error("--suggest-only requires a predictions.json in --output dir")
            sys.exit(1)
        copied = find_uncertain_patches(
            predictions_json=pred_json,
            image_dir=args.unlabelled,
            output_dir=Path(args.output) / "for_review",
            confidence_threshold=args.pseudo_thresh,
        )
        logger.info("Flagged %d images for manual review", len(copied))
        return

    final_model = self_train(
        d2_config_path=args.d2_config,
        cellseg_config=cfg,
        initial_model=args.model,
        unlabelled_dir=args.unlabelled,
        dataset_dir=args.dataset,
        output_dir=args.output,
        n_iterations=args.iterations,
        pseudo_label_threshold=args.pseudo_thresh,
    )
    logger.info("Self-training complete → %s", final_model)


def _cmd_visualize(args: argparse.Namespace, logger: logging.Logger) -> None:
    from cellseg_trainer.visualization import visualize_batch

    visualize_batch(
        image_dir=args.images,
        mask_dir=args.masks,
        output_dir=args.output,
        max_images=args.max_images,
        alpha=args.alpha,
        save_tiff=args.save_tiff,
    )
    logger.info("Visualizations saved to %s", args.output)


if __name__ == "__main__":
    main()
