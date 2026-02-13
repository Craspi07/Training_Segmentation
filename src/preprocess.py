"""
Preprocessing module for 16-bit multi-channel microscopy images.

Handles the full preprocessing workflow:
  1. Load multi-channel 16-bit TIFFs (DIC + fluorescence channels)
  2. Split and normalize each channel independently
  3. Run Cellpose cpsam (or custom model) to generate draft masks
  4. Save masks to a separate folder for manual curation

Channel mapping for this dataset:
  0 = DIC (brightfield)
  1 = mEGFP
  2 = mScarlet
  3 = miRFPnano3 (not always present)

16-bit normalization strategy:
  - Percentile-based (1st-99th by default) per-channel to float32
  - Tile normalization for DIC/brightfield (uneven illumination)
  - Adjustable percentile range for dim fluorescence signals
  - Channels with no dynamic range are zeroed out
"""

import os
import glob
import logging
from pathlib import Path

import numpy as np
import tifffile
from skimage import io as skio

logger = logging.getLogger(__name__)

CHANNEL_NAMES = {
    0: "DIC",
    1: "mEGFP",
    2: "mScarlet",
    3: "miRFPnano3",
}


def load_multichannel_image(path: str) -> np.ndarray:
    """
    Load a multi-channel image.

    Returns array with shape (C, H, W) for multi-channel or (H, W) for
    single-channel. Handles 8-bit, 16-bit, and float inputs.
    Supports TIFF, PNG, JPG, and ND2 (Nikon NIS-Elements) formats.
    """
    ext = Path(path).suffix.lower()
    if ext == ".nd2":
        import nd2
        img = nd2.imread(path)
    elif ext in (".tif", ".tiff"):
        img = tifffile.imread(path)
    else:
        img = skio.imread(path)

    # Normalize axis order to (C, H, W)
    if img.ndim == 2:
        return img  # Single channel
    elif img.ndim == 3:
        # Could be (C, H, W) or (H, W, C)
        # Heuristic: if last dim is small (<=4), it's likely channels-last
        if img.shape[-1] <= 4 and img.shape[0] > 4:
            img = np.moveaxis(img, -1, 0)  # (H, W, C) -> (C, H, W)
        return img
    else:
        return img


def get_image_info(path: str) -> dict:
    """
    Get metadata about a multi-channel image without fully loading it.

    Returns dict with shape, dtype, num_channels, and channel availability.
    """
    img = load_multichannel_image(path)
    n_channels = img.shape[0] if img.ndim == 3 else 1

    info = {
        "path": path,
        "shape": img.shape,
        "dtype": str(img.dtype),
        "num_channels": n_channels,
        "is_16bit": img.dtype in (np.uint16, np.int16),
        "channels": {},
    }

    for ch_idx in range(n_channels):
        ch_name = CHANNEL_NAMES.get(ch_idx, f"Channel {ch_idx}")
        ch_data = img[ch_idx] if img.ndim == 3 else img
        info["channels"][ch_idx] = {
            "name": ch_name,
            "min": int(ch_data.min()),
            "max": int(ch_data.max()),
            "mean": float(ch_data.mean()),
            "has_signal": int(ch_data.max()) > int(ch_data.min()),
        }

    return info


def normalize_channel(
    channel: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    tile_blocksize: int = 0,
) -> np.ndarray:
    """
    Normalize a single 16-bit (or any bit-depth) channel to float32.

    Uses percentile-based normalization matching Cellpose's normalize99
    approach: the lower percentile maps to 0.0 and upper percentile maps
    to 1.0. Values outside this range are NOT clipped (Cellpose expects this).

    Args:
        channel: 2D array (H, W), any dtype.
        lower_percentile: Lower percentile for normalization floor.
        upper_percentile: Upper percentile for normalization ceiling.
        tile_blocksize: If > 0, normalize in tiles of this size.
                        Useful for DIC/brightfield with uneven illumination.

    Returns:
        Normalized float32 array.
    """
    channel = channel.astype(np.float32)

    if tile_blocksize > 0:
        return _tile_normalize(channel, lower_percentile, upper_percentile, tile_blocksize)

    p_low = np.percentile(channel, lower_percentile)
    p_high = np.percentile(channel, upper_percentile)

    if p_high - p_low < 1e-3:
        logger.warning("Channel has no dynamic range, zeroing out")
        return np.zeros_like(channel)

    return (channel - p_low) / (p_high - p_low)


def _tile_normalize(
    channel: np.ndarray,
    lower_percentile: float,
    upper_percentile: float,
    blocksize: int,
) -> np.ndarray:
    """
    Tile-based normalization for uneven illumination (e.g., DIC).

    Computes percentiles in local tiles and interpolates for smooth
    normalization across the image.
    """
    h, w = channel.shape
    result = np.zeros_like(channel)

    # Compute per-tile normalization factors
    n_tiles_y = max(1, h // blocksize)
    n_tiles_x = max(1, w // blocksize)

    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            y0 = ty * blocksize
            y1 = min((ty + 1) * blocksize, h)
            x0 = tx * blocksize
            x1 = min((tx + 1) * blocksize, w)

            tile = channel[y0:y1, x0:x1]
            p_low = np.percentile(tile, lower_percentile)
            p_high = np.percentile(tile, upper_percentile)

            if p_high - p_low < 1e-3:
                result[y0:y1, x0:x1] = 0.0
            else:
                result[y0:y1, x0:x1] = (tile - p_low) / (p_high - p_low)

    return result


def preprocess_image(
    image: np.ndarray,
    segment_channel: int = 0,
    nuclear_channel: int = 0,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    tile_blocksize_dic: int = 128,
    invert_dic: bool = False,
) -> np.ndarray:
    """
    Preprocess a multi-channel 16-bit image for Cellpose.

    Extracts the requested channels, normalizes each independently with
    settings appropriate for its modality, and returns a Cellpose-ready array.

    Args:
        image: Multi-channel image (C, H, W) or single-channel (H, W).
        segment_channel: Channel index for the structure to segment.
        nuclear_channel: Optional nuclear channel (0 = none/grayscale).
        lower_percentile: Lower percentile for normalization.
        upper_percentile: Upper percentile for normalization.
        tile_blocksize_dic: Tile size for DIC normalization (0 = no tiling).
        invert_dic: Invert DIC channel (cells dark on light background).

    Returns:
        Preprocessed image ready for Cellpose model.eval().
        Shape: (H, W) for grayscale or (H, W, 2) for two-channel.
    """
    if image.ndim == 2:
        # Single channel
        ch = normalize_channel(image, lower_percentile, upper_percentile)
        if invert_dic and segment_channel == 0:
            ch = -ch
        return ch

    n_channels = image.shape[0]

    # Validate channel indices
    if segment_channel >= n_channels:
        raise ValueError(
            f"Segment channel {segment_channel} ({CHANNEL_NAMES.get(segment_channel, '?')}) "
            f"not available. Image has {n_channels} channels."
        )

    # Determine if DIC channel needs tile normalization
    is_dic = (segment_channel == 0)
    tile_bs = tile_blocksize_dic if is_dic else 0

    # Determine percentile range per channel type
    if is_dic:
        # DIC: standard percentiles with tile normalization
        seg_ch = normalize_channel(
            image[segment_channel], lower_percentile, upper_percentile, tile_bs
        )
        if invert_dic:
            seg_ch = -seg_ch
    else:
        # Fluorescence: wider percentile range for dim signals
        fluor_low = max(0.1, lower_percentile * 0.5)
        fluor_high = min(99.99, upper_percentile + (100 - upper_percentile) * 0.9)
        seg_ch = normalize_channel(image[segment_channel], fluor_low, fluor_high)

    # Nuclear channel
    if nuclear_channel > 0 and nuclear_channel < n_channels:
        nuc_ch = normalize_channel(
            image[nuclear_channel],
            max(0.1, lower_percentile * 0.5),
            min(99.99, upper_percentile + (100 - upper_percentile) * 0.9),
        )
        # Return (H, W, 2) — Cellpose two-channel format
        return np.stack([seg_ch, nuc_ch], axis=-1)

    return seg_ch


def generate_masks(
    image_dir: str,
    output_dir: str,
    segment_channel: int = 0,
    nuclear_channel: int = 0,
    model_path: str | None = None,
    diameter: float | None = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    tile_blocksize_dic: int = 128,
    invert_dic: bool = False,
    use_gpu: bool = True,
    progress_callback=None,
) -> list[tuple[str, int]]:
    """
    Generate draft segmentation masks for all images in a directory.

    This is the main preprocessing function. It:
      1. Loads each multi-channel image
      2. Extracts and normalizes the requested channels
      3. Runs Cellpose (cpsam by default) to predict masks
      4. Saves masks as 16-bit TIFFs for manual curation

    Args:
        image_dir: Directory with input images.
        output_dir: Directory to save generated masks.
        segment_channel: Which channel to segment.
        nuclear_channel: Optional nuclear helper channel (0 = none).
        model_path: Path to custom model (None = default cpsam).
        diameter: Object diameter in pixels (None = auto-detect).
        flow_threshold: Cellpose flow threshold.
        cellprob_threshold: Cellpose cell probability threshold.
        lower_percentile: Lower normalization percentile.
        upper_percentile: Upper normalization percentile.
        tile_blocksize_dic: Tile normalization block size for DIC.
        invert_dic: Whether to invert DIC brightness.
        use_gpu: Use GPU for inference.
        progress_callback: Optional callable(current, total, filename) for GUI.

    Returns:
        List of (filename, num_objects) tuples.
    """
    from cellpose import models

    os.makedirs(output_dir, exist_ok=True)

    # Find images
    extensions = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg", "*.nd2")
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
    image_files = sorted(image_files)

    if not image_files:
        logger.warning(f"No images found in {image_dir}")
        return []

    logger.info(f"Found {len(image_files)} images in {image_dir}")

    # Initialize model
    if model_path:
        logger.info(f"Loading custom model: {model_path}")
        model = models.CellposeModel(gpu=use_gpu, pretrained_model=model_path)
    else:
        logger.info("Using default cpsam model")
        model = models.CellposeModel(gpu=use_gpu)

    # Determine Cellpose channels parameter
    # For cpsam (v4), channels are not used the same way — it uses first 3 channels.
    # We preprocess and feed the right channels, so use [0, 0] for grayscale input.
    if nuclear_channel > 0:
        cp_channels = [1, 2]  # two-channel input
    else:
        cp_channels = [0, 0]  # grayscale

    results = []
    for i, img_path in enumerate(image_files):
        filename = Path(img_path).name
        logger.info(f"[{i+1}/{len(image_files)}] Processing: {filename}")

        if progress_callback:
            progress_callback(i, len(image_files), filename)

        try:
            # Load and preprocess
            raw = load_multichannel_image(img_path)
            img_info = get_image_info(img_path)

            # Check if requested channel has signal
            if segment_channel in img_info["channels"]:
                ch_info = img_info["channels"][segment_channel]
                if not ch_info["has_signal"]:
                    logger.warning(
                        f"  Channel {segment_channel} ({ch_info['name']}) has no signal, skipping"
                    )
                    results.append((filename, 0))
                    continue

            preprocessed = preprocess_image(
                raw,
                segment_channel=segment_channel,
                nuclear_channel=nuclear_channel,
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile,
                tile_blocksize_dic=tile_blocksize_dic,
                invert_dic=invert_dic,
            )

            # Run Cellpose
            mask, flow, style = model.eval(
                preprocessed,
                diameter=diameter,
                channels=cp_channels,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
            )

            num_objects = int(mask.max())
            logger.info(f"  Found {num_objects} objects")

            # Save mask
            stem = Path(filename).stem
            mask_path = os.path.join(output_dir, f"{stem}_masks.tif")
            tifffile.imwrite(mask_path, mask.astype(np.uint16))

            results.append((filename, num_objects))

        except Exception as e:
            logger.error(f"  Failed to process {filename}: {e}")
            results.append((filename, -1))

    if progress_callback:
        progress_callback(len(image_files), len(image_files), "Done")

    total_objects = sum(n for _, n in results if n > 0)
    successful = sum(1 for _, n in results if n >= 0)
    logger.info(
        f"Preprocessing complete: {successful}/{len(image_files)} images, "
        f"{total_objects} total objects detected"
    )

    return results


def save_channel_preview(
    image_path: str,
    output_dir: str,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    tile_blocksize_dic: int = 128,
) -> str:
    """
    Save a side-by-side preview of all channels in an image.

    Useful for inspecting multi-channel data before running preprocessing.
    Returns path to the saved preview image.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = load_multichannel_image(image_path)
    info = get_image_info(image_path)
    n_channels = info["num_channels"]

    fig, axes = plt.subplots(1, n_channels, figsize=(5 * n_channels, 5))
    if n_channels == 1:
        axes = [axes]

    for ch_idx in range(n_channels):
        ax = axes[ch_idx]
        ch_data = raw[ch_idx] if raw.ndim == 3 else raw
        ch_info = info["channels"][ch_idx]

        # Normalize for display
        is_dic = (ch_idx == 0)
        tile_bs = tile_blocksize_dic if is_dic else 0
        normalized = normalize_channel(ch_data, lower_percentile, upper_percentile, tile_bs)
        display = np.clip(normalized, 0, 1)

        ax.imshow(display, cmap="gray")
        ax.set_title(
            f"Ch {ch_idx}: {ch_info['name']}\n"
            f"[{ch_info['min']}-{ch_info['max']}] "
            f"{'16-bit' if info['is_16bit'] else '8-bit'}"
        )
        ax.axis("off")

    fig.suptitle(Path(image_path).name, fontsize=12)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    stem = Path(image_path).stem
    preview_path = os.path.join(output_dir, f"{stem}_channels.png")
    fig.savefig(preview_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return preview_path


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Preprocess multi-channel images and generate draft masks"
    )
    parser.add_argument("--image-dir", required=True, help="Input image directory")
    parser.add_argument("--output-dir", required=True, help="Output mask directory")
    parser.add_argument("--segment-channel", type=int, default=0, help="Channel to segment (0=DIC, 1=mEGFP, 2=mScarlet, 3=miRFPnano3)")
    parser.add_argument("--nuclear-channel", type=int, default=0, help="Nuclear helper channel (0=none)")
    parser.add_argument("--model", default=None, help="Custom model path (default: cpsam)")
    parser.add_argument("--diameter", type=float, default=None, help="Object diameter (None=auto)")
    parser.add_argument("--lower-pct", type=float, default=1.0, help="Lower normalization percentile")
    parser.add_argument("--upper-pct", type=float, default=99.0, help="Upper normalization percentile")
    parser.add_argument("--tile-blocksize", type=int, default=128, help="Tile norm blocksize for DIC")
    parser.add_argument("--invert-dic", action="store_true", help="Invert DIC channel")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU")

    args = parser.parse_args()

    generate_masks(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        segment_channel=args.segment_channel,
        nuclear_channel=args.nuclear_channel,
        model_path=args.model,
        diameter=args.diameter,
        lower_percentile=args.lower_pct,
        upper_percentile=args.upper_pct,
        tile_blocksize_dic=args.tile_blocksize,
        invert_dic=args.invert_dic,
        use_gpu=not args.no_gpu,
    )
