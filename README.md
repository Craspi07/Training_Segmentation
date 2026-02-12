# Cellpose Segmentation Training Pipeline

A BiaPy-inspired training pipeline for bioimage segmentation using [Cellpose](https://github.com/MouseLand/cellpose). Supports two modalities:

- **DIC Brightfield** — whole-cell segmentation
- **Fluorescence** — nucleus segmentation

## Project Structure

```
Training_Segmentation/
├── configs/
│   ├── dic_wholecell.yaml      # Config for DIC whole-cell segmentation
│   └── fluor_nucleus.yaml      # Config for fluorescence nucleus segmentation
├── data/
│   ├── train/
│   │   ├── dic_raw/            # DIC training images (*_img.tif)
│   │   ├── dic_labels/         # DIC training masks  (*_masks.tif)
│   │   ├── fluor_raw/          # Fluorescence training images
│   │   └── fluor_labels/       # Fluorescence training masks
│   └── test/
│       ├── dic_raw/
│       ├── dic_labels/
│       ├── fluor_raw/
│       └── fluor_labels/
├── src/
│   ├── gui.py                  # Tkinter GUI application
│   ├── pipeline.py             # Main orchestration script (CLI)
│   ├── train_cellpose.py       # Cellpose training module
│   ├── evaluate.py             # Evaluation and inference
│   ├── rename_files.py         # File renaming utility
│   └── data_preparation.py     # Data loading and augmentation
├── models/                     # Saved trained models
├── results/                    # Outputs, metrics, overlays
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## GUI

Launch the graphical interface:

```bash
cd src/
python gui.py
```

The GUI has four tabs:

| Tab | Function |
|-----|----------|
| **File Renaming** | Batch rename images/masks to the pipeline naming convention (`<prefix>NNN_img.tif` / `<prefix>NNN_masks.tif`). Supports preview, in-place rename, and copy-to-output-dir modes. |
| **Configuration** | Load/edit/save YAML configs with preset buttons for DIC and fluorescence tasks. |
| **Training** | Launch and monitor Cellpose model training with live log output. |
| **Evaluation** | Run inference on new images and compute metrics (mAP, IoU, Dice). |

## File Renaming

The pipeline requires files named `<prefix><NNN>_img.<ext>` for images and `<prefix><NNN>_masks.<ext>` for masks. Use either the GUI or the CLI tool to rename your files:

```bash
cd src/

# Preview renames (dry run)
python rename_files.py --image-dir /path/to/images --mask-dir /path/to/masks --prefix dic_ --dry-run

# Rename in place
python rename_files.py --image-dir /path/to/images --mask-dir /path/to/masks --prefix dic_ --ext .tif

# Copy to pipeline data directories instead of renaming
python rename_files.py --image-dir /path/to/images --mask-dir /path/to/masks \
    --prefix dic_ --ext .tif --copy \
    --image-output ../data/train/dic_raw --mask-output ../data/train/dic_labels
```

## Data Preparation

Place your images and masks in the `data/` directories following the naming convention:

- Images: `<name>_img.tif` (or `.png`, `.jpg`)
- Masks: `<name>_masks.tif` — instance-labeled (0 = background, 1, 2, ... = individual objects)

The naming filters are configurable in the YAML configs (`IMAGE_FILTER`, `MASK_FILTER`).

## CLI Usage

```bash
cd src/

# Train DIC whole-cell segmentation model
python pipeline.py --task dic

# Train fluorescence nucleus segmentation model
python pipeline.py --task fluor

# Train both models sequentially
python pipeline.py --task both

# Use a custom configuration
python pipeline.py --config ../configs/dic_wholecell.yaml

# Inference only with a pre-trained model
python pipeline.py --task dic --inference-only --model ../models/cellpose_dic_wholecell
```

## Configuration

Each YAML config follows BiaPy conventions and controls:

| Section | Key Parameters |
|---------|---------------|
| `DATA` | Train/test paths, image/mask filters, channels, normalization |
| `AUGMENTATION` | Flips, rotation, elastic deformation, noise, brightness/contrast |
| `MODEL` | Backend (cellpose), pretrained model, architecture |
| `TRAIN` | Epochs, learning rate, weight decay, batch size |
| `INFERENCE` | Diameter, flow threshold, cell probability threshold |
| `PATHS` | Model save directory, results directory, model name |

### Channel Reference

| Channel | Source |
|---------|--------|
| 0 | DIC (brightfield) |
| 1 | mEGFP |
| 2 | mScarlet |
| 3 | miRFPnano3 (not present in every image) |

`CHANNELS: [segment_channel, nuclear_channel]` — first value is the channel to segment, second is an optional nuclear helper (0 = none).

- **DIC whole-cell segmentation:** `CHANNELS: [0, 0]` — segment DIC, no nuclear channel
- **Nucleus from mEGFP:** `CHANNELS: [1, 0]` — segment mEGFP channel
- **Nucleus from mScarlet:** `CHANNELS: [2, 0]` — segment mScarlet channel
- **Nucleus from miRFPnano3:** `CHANNELS: [3, 0]` — segment miRFPnano3 channel

## Pipeline Stages

1. **Data Loading** — Reads image/mask pairs from directories
2. **Augmentation** — Applies configured transforms (flips, rotation, elastic deform, noise)
3. **Training** — Fine-tunes the Cellpose cpsam model with `cellpose.train.train_seg`
4. **Evaluation** — Runs inference on test set, computes mAP, IoU, Dice
5. **Reporting** — Saves loss curves, metrics JSON, overlay visualizations

## References

- [Cellpose Documentation](https://cellpose.readthedocs.io/)
- [Cellpose Training Guide](https://cellpose.readthedocs.io/en/latest/train.html)
- [BiaPy Documentation](https://biapy.readthedocs.io/en/latest/)
- [BiaPy Semantic Segmentation](https://biapy.readthedocs.io/en/latest/workflows/semantic_segmentation.html)
