#!/bin/bash
#SBATCH --job-name=fidip-inspect
#SBATCH --partition=short
#SBATCH --constraint=ib
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=slurm-logs/inspect-%j.out

# Read-only quota/environment inspection. No sync, downloads, builds, or GPU.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
printf 'Job: %s\nHost: %s\n' "$SLURM_JOB_ID" "$(hostname)"
printf '\n--- Home quota ---\n'
check-quota "$HOME"
printf '\n--- Existing uv ---\n'
"$HOME/.local/bin/uv" --version
"$HOME/.local/bin/uv" cache dir
printf '\n--- Existing environment ---\n'
.venv/bin/python -B - <<'PY'
import sys
from importlib.metadata import version
from pathlib import Path

print("Python:", sys.version, flush=True)
print("Executable:", sys.executable, flush=True)
print("Resolved executable:", Path(sys.executable).resolve(), flush=True)
for package in (
    "torch", "torchvision", "numpy", "opencv-python-headless", "scipy", "pillow",
    "pycocotools", "json-tricks", "yacs", "dsntnn", "tensorboard", "tensorboardx",
    "cython", "setuptools", "wheel",
):
    print(f"{package}: {version(package)}", flush=True)
import torch
import torchvision
import numpy
print("PyTorch CUDA build:", torch.version.cuda, flush=True)
x = torch.ones((2, 2), device="cpu")
assert (x @ x).sum().item() == 8
print("PASS: torch/torchvision/numpy imports and tiny CPU matmul", flush=True)
PY
printf '\n--- Installed dependency consistency ---\n'
"$HOME/.local/bin/uv" pip check --python .venv/bin/python
printf '\nPASS: quota and existing CPU environment inspection; GPU/native extensions not tested\n'
