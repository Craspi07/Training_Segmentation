#!/usr/bin/env bash
# setup_wsl.sh — Install all ML dependencies inside WSL 2 (Ubuntu)
#
# Run this script from INSIDE WSL, from the project directory:
#   cd /mnt/c/path/to/Training_Segmentation
#   bash setup_wsl.sh
#
# What it does:
#   1. Installs system packages (build tools, libGL, …)
#   2. Installs PyTorch with CUDA (auto-detects CUDA version)
#   3. Installs detectron2 and all other ML dependencies
#   4. Installs cellseg_trainer in editable mode
#   5. Verifies the installation
#
# GPU drivers must already be installed on the Windows HOST — do NOT install
# CUDA inside WSL.  Only the CUDA toolkit headers/libraries are installed here.
#
# Tested on: Ubuntu 22.04 / WSL 2 / NVIDIA driver ≥ 470

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

###############################################################################
# Helpers
###############################################################################
info()  { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()    { echo -e "\033[1;32m[ OK ]\033[0m  $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
die()   { echo -e "\033[1;31m[FAIL]\033[0m  $*" >&2; exit 1; }

###############################################################################
# 0. Sanity checks
###############################################################################
info "Checking WSL 2 environment …"

# Must be running inside WSL
if [[ -z "${WSL_DISTRO_NAME:-}" && ! -f /proc/sys/fs/binfmt_misc/WSLInterop ]]; then
    warn "WSL_DISTRO_NAME not set and WSLInterop not found."
    warn "Make sure you are running this script inside WSL 2, not on a native Linux host."
fi

# Check nvidia-smi is reachable (GPU passthrough)
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true)
    ok "GPU(s) detected:\n${GPU_INFO}"
    CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9.]+")
    info "Host CUDA Version: ${CUDA_VERSION}"
else
    warn "nvidia-smi not found or failed.  GPU acceleration will not be available."
    warn "Update NVIDIA drivers on the Windows host (≥ 470.x) and run again."
    CUDA_VERSION=""
fi

###############################################################################
# 1. System packages
###############################################################################
info "Installing system packages …"
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    build-essential python3-dev \
    git curl wget \
    libgl1-mesa-glx libglib2.0-0 \
    libsm6 libxext6 libxrender-dev \
    ffmpeg

ok "System packages installed."

###############################################################################
# 2. Upgrade pip / setuptools
###############################################################################
info "Upgrading pip …"
python3 -m pip install --upgrade pip setuptools wheel

###############################################################################
# 3. PyTorch with CUDA
###############################################################################
info "Installing PyTorch …"

# Map CUDA version string to PyTorch wheel tag (e.g. 12.1 → cu121)
if [[ -n "${CUDA_VERSION}" ]]; then
    MAJOR=$(echo "${CUDA_VERSION}" | cut -d. -f1)
    MINOR=$(echo "${CUDA_VERSION}" | cut -d. -f2)
    CU_TAG="cu${MAJOR}${MINOR}"
else
    warn "No CUDA detected — installing CPU-only PyTorch."
    CU_TAG="cpu"
fi

TORCH_INDEX="https://download.pytorch.org/whl/${CU_TAG}"
info "PyTorch wheel index: ${TORCH_INDEX}"
python3 -m pip install torch torchvision --index-url "${TORCH_INDEX}"
ok "PyTorch installed."

# Verify
python3 -c "
import torch
print(f'  torch:          {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

###############################################################################
# 4. Common ML / image-processing packages
###############################################################################
info "Installing core ML packages …"
python3 -m pip install \
    numpy"<2.0" \
    scipy \
    scikit-image \
    tifffile \
    Pillow \
    matplotlib \
    tqdm \
    pyyaml \
    pycocotools \
    opencv-python-headless \
    cellpose">=4.0" \
    nd2 \
    aicsimageio

ok "Core ML packages installed."

###############################################################################
# 5. detectron2
###############################################################################
info "Installing detectron2 …"

# Try pre-built wheel first (fast); fall back to building from source
TORCH_VER=$(python3 -c "import torch; v=torch.__version__.split('+')[0]; print(v)")
D2_WHEEL_INDEX="https://dl.fbaipublicfiles.com/detectron2/wheels/${CU_TAG}/torch${TORCH_VER}/index.html"

if python3 -m pip install detectron2 -f "${D2_WHEEL_INDEX}" 2>/dev/null; then
    ok "detectron2 installed from pre-built wheel."
else
    warn "Pre-built wheel not found for ${CU_TAG}/torch${TORCH_VER} — building from source (this takes a few minutes) …"
    python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
    ok "detectron2 built and installed from source."
fi

###############################################################################
# 6. cellseg_trainer (this project)
###############################################################################
info "Installing cellseg_trainer in editable mode …"
cd "${SCRIPT_DIR}"
python3 -m pip install -e .
ok "cellseg_trainer installed."

###############################################################################
# 7. Verification
###############################################################################
info "Running import checks …"
python3 - <<'EOF'
failures = []
checks = [
    ("torch",          "import torch"),
    ("torchvision",    "import torchvision"),
    ("detectron2",     "import detectron2; print('  detectron2', detectron2.__version__)"),
    ("cv2",            "import cv2; print('  cv2', cv2.__version__)"),
    ("pycocotools",    "import pycocotools"),
    ("cellpose",       "import cellpose; print('  cellpose', cellpose.__version__)"),
    ("tifffile",       "import tifffile"),
    ("cellseg_trainer","from cellseg_trainer.cli import main"),
]
for name, stmt in checks:
    try:
        exec(stmt)
        print(f"  ✓ {name}")
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        failures.append(name)
if failures:
    raise SystemExit(f"\nFailed: {', '.join(failures)}")
EOF

echo ""
ok "All checks passed.  WSL environment is ready."
echo ""
echo "  You can now launch the GUI from Windows:"
echo "    run_gui.bat"
echo ""
echo "  Or run training directly from WSL:"
echo "    python3 src/pipeline.py --config configs/dic_wholecell.yaml"
echo "    cellseg_trainer train --help"
