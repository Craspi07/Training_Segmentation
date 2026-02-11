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
│   ├── pipeline.py             # Main orchestration script
│   ├── train_cellpose.py       # Cellpose training module
│   ├── evaluate.py             # Evaluation and inference
│   └── data_preparation.py     # Data loading and augmentation
├── models/                     # Saved trained models
├── results/                    # Outputs, metrics, overlays
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Data Preparation

Place your images and masks in the `data/` directories following the naming convention:

- Images: `<name>_img.tif` (or `.png`, `.jpg`)
- Masks: `<name>_masks.tif` — instance-labeled (0 = background, 1, 2, ... = individual objects)

The naming filters are configurable in the YAML configs (`IMAGE_FILTER`, `MASK_FILTER`).

## Usage

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

### Channel Configuration

- **DIC/Brightfield (grayscale):** `CHANNELS: [0, 0]`
- **Fluorescence (single channel):** `CHANNELS: [0, 0]`
- **Multi-channel (e.g., green cyto + blue nuclei):** `CHANNELS: [2, 3]`

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
