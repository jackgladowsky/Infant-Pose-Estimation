#!/bin/bash
#SBATCH --job-name=fidip-env
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-logs/env-%j.out

# Submit from repository root. No data or pretrained weights required.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
module load cuda/12.1.1
export OMP_NUM_THREADS=2
source .venv/bin/activate
# Build directly rather than using Makefile, which can mask compiler failures.
(cd lib/nms && python setup_linux.py build_ext --inplace)
python tools/train_adaptive_model_hrnet.py --help
python tools/test_adaptive_model.py --help
python tools/check_environment.py
