"""
File renaming utility for the Cellpose segmentation pipeline.

Renames image and mask files in a folder to match the naming convention
required by the pipeline:
  - Images: <prefix><NNN>_img.<ext>
  - Masks:  <prefix><NNN>_masks.<ext>

Supports batch renaming, dry-run preview, and automatic pairing of
images with their corresponding masks.
"""

import os
import re
import shutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def list_image_files(directory: str) -> list[str]:
    """List all supported image files in a directory, sorted."""
    files = []
    for f in sorted(os.listdir(directory)):
        if Path(f).suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(f)
    return files


def generate_new_name(
    index: int,
    prefix: str,
    suffix: str,
    extension: str,
    zero_pad: int = 3,
) -> str:
    """
    Generate a pipeline-compatible filename.

    Args:
        index: Sequential number for this file.
        prefix: Name prefix (e.g., "dic_", "fluor_", "cell_").
        suffix: Either "_img" or "_masks".
        extension: File extension including dot (e.g., ".tif").
        zero_pad: Number of digits for zero-padding the index.

    Returns:
        New filename string.
    """
    return f"{prefix}{str(index).zfill(zero_pad)}{suffix}{extension}"


def rename_files(
    directory: str,
    suffix: str = "_img",
    prefix: str = "",
    target_extension: str | None = None,
    dry_run: bool = False,
    copy_mode: bool = False,
    output_dir: str | None = None,
) -> list[tuple[str, str]]:
    """
    Rename (or copy) all image files in a directory to the pipeline convention.

    Args:
        directory: Source directory containing files to rename.
        suffix: The suffix to apply ("_img" for images, "_masks" for masks).
        prefix: Optional prefix for filenames (e.g., "dic_", "fluor_").
        target_extension: Convert all files to this extension (e.g., ".tif").
                          If None, keep original extensions.
        dry_run: If True, return the rename plan without modifying files.
        copy_mode: If True, copy files to output_dir instead of renaming in place.
        output_dir: Destination directory when copy_mode is True.

    Returns:
        List of (old_name, new_name) tuples.
    """
    files = list_image_files(directory)
    if not files:
        logger.warning(f"No image files found in {directory}")
        return []

    renames = []
    for i, filename in enumerate(files, start=1):
        old_path = os.path.join(directory, filename)
        ext = target_extension or Path(filename).suffix.lower()
        new_name = generate_new_name(i, prefix, suffix, ext)

        if copy_mode and output_dir:
            new_path = os.path.join(output_dir, new_name)
        else:
            new_path = os.path.join(directory, new_name)

        renames.append((old_path, new_path))

    if dry_run:
        logger.info(f"Dry run — {len(renames)} files would be renamed:")
        for old, new in renames:
            logger.info(f"  {Path(old).name}  ->  {Path(new).name}")
        return [(Path(o).name, Path(n).name) for o, n in renames]

    # Execute renames
    if copy_mode and output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for old_path, new_path in renames:
        if old_path == new_path:
            continue
        if copy_mode:
            shutil.copy2(old_path, new_path)
            logger.info(f"Copied: {Path(old_path).name} -> {Path(new_path).name}")
        else:
            # Rename in place — use a temp name to avoid collisions
            tmp_path = old_path + ".tmp_rename"
            os.rename(old_path, tmp_path)
            os.rename(tmp_path, new_path)
            logger.info(f"Renamed: {Path(old_path).name} -> {Path(new_path).name}")

    return [(Path(o).name, Path(n).name) for o, n in renames]


def rename_image_mask_pair(
    image_dir: str,
    mask_dir: str,
    prefix: str = "",
    target_extension: str | None = None,
    dry_run: bool = False,
    copy_mode: bool = False,
    image_output_dir: str | None = None,
    mask_output_dir: str | None = None,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """
    Rename both image and mask directories together, ensuring paired indices.

    Files are sorted alphabetically in each directory and assigned matching
    indices (image 001 pairs with mask 001, etc.).

    Args:
        image_dir: Directory with raw images.
        mask_dir: Directory with mask/label images.
        prefix: Filename prefix.
        target_extension: Convert to this extension (or None to keep original).
        dry_run: Preview only.
        copy_mode: Copy to output dirs instead of renaming in place.
        image_output_dir: Destination for images (copy_mode).
        mask_output_dir: Destination for masks (copy_mode).

    Returns:
        Tuple of (image_renames, mask_renames).
    """
    img_files = list_image_files(image_dir)
    mask_files = list_image_files(mask_dir)

    if len(img_files) != len(mask_files):
        logger.warning(
            f"Mismatch: {len(img_files)} images vs {len(mask_files)} masks. "
            "Files will be renamed independently — verify pairing manually."
        )

    img_renames = rename_files(
        directory=image_dir,
        suffix="_img",
        prefix=prefix,
        target_extension=target_extension,
        dry_run=dry_run,
        copy_mode=copy_mode,
        output_dir=image_output_dir,
    )

    mask_renames = rename_files(
        directory=mask_dir,
        suffix="_masks",
        prefix=prefix,
        target_extension=target_extension,
        dry_run=dry_run,
        copy_mode=copy_mode,
        output_dir=mask_output_dir,
    )

    return img_renames, mask_renames


def auto_sort_and_rename(
    source_dir: str,
    task: str = "dic",
    split: str = "train",
    project_root: str = ".",
    dry_run: bool = False,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """
    Convenience function: sort files from a flat source directory into
    the pipeline's data structure and rename them.

    Expects the source directory to have two subdirectories:
      - images/ (or raw/)
      - masks/ (or labels/)

    Args:
        source_dir: Root of the source data.
        task: "dic" or "fluor".
        split: "train" or "test".
        project_root: Project root directory.
        dry_run: Preview only.

    Returns:
        Tuple of (image_renames, mask_renames).
    """
    # Try common subdirectory names
    img_candidates = ["images", "raw", "imgs", "input"]
    mask_candidates = ["masks", "labels", "annotations", "gt", "ground_truth"]

    img_subdir = None
    for name in img_candidates:
        candidate = os.path.join(source_dir, name)
        if os.path.isdir(candidate):
            img_subdir = candidate
            break

    mask_subdir = None
    for name in mask_candidates:
        candidate = os.path.join(source_dir, name)
        if os.path.isdir(candidate):
            mask_subdir = candidate
            break

    # If no subdirs found, check if source_dir itself has images
    if img_subdir is None:
        files = list_image_files(source_dir)
        if files:
            logger.info(
                f"No images/ subdirectory found. "
                f"Please separate images and masks into subdirectories."
            )
            return [], []

    if img_subdir is None or mask_subdir is None:
        logger.error(
            f"Could not find image/mask subdirectories in {source_dir}. "
            f"Expected subdirectories like images/ and masks/"
        )
        return [], []

    modality = "dic" if task == "dic" else "fluor"
    img_output = os.path.join(project_root, "data", split, f"{modality}_raw")
    mask_output = os.path.join(project_root, "data", split, f"{modality}_labels")

    prefix = f"{modality}_"

    return rename_image_mask_pair(
        image_dir=img_subdir,
        mask_dir=mask_subdir,
        prefix=prefix,
        target_extension=".tif",
        dry_run=dry_run,
        copy_mode=True,
        image_output_dir=img_output,
        mask_output_dir=mask_output,
    )


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Rename files to match pipeline naming conventions"
    )
    parser.add_argument("--image-dir", required=True, help="Directory with images")
    parser.add_argument("--mask-dir", required=True, help="Directory with masks")
    parser.add_argument("--prefix", default="", help="Filename prefix (e.g., 'dic_')")
    parser.add_argument("--ext", default=None, help="Target extension (e.g., '.tif')")
    parser.add_argument("--dry-run", action="store_true", help="Preview without renaming")
    parser.add_argument("--copy", action="store_true", help="Copy instead of rename")
    parser.add_argument("--image-output", default=None, help="Output dir for images (copy mode)")
    parser.add_argument("--mask-output", default=None, help="Output dir for masks (copy mode)")

    args = parser.parse_args()

    rename_image_mask_pair(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        prefix=args.prefix,
        target_extension=args.ext,
        dry_run=args.dry_run,
        copy_mode=args.copy,
        image_output_dir=args.image_output,
        mask_output_dir=args.mask_output,
    )
