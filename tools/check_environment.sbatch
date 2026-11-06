#!/bin/bash
#SBATCH --job-name=fidip-env
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=slurm-logs/env-%j.out

# Run only after tools/build_environment.sbatch succeeds; no compilation on GPU time.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source tools/environment_check_common.sh
stage gpu_inventory 15s nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
stage source_hashes 15s sha256sum tools/check_environment.sbatch \
    tools/environment_check_common.sh tools/check_environment.py \
    lib/nms/setup_linux.py lib/nms/cpu_nms.pyx lib/nms/gpu_nms.pyx
stage gpu_and_project_checks 150s python -u tools/check_environment.py --phase all
stage train_cli 45s python -u tools/train_adaptive_model_hrnet.py --help
stage eval_cli 45s python -u tools/test_adaptive_model.py --help
check_log 'PASS: CUDA/NMS checks and training/evaluation CLI imports; no training performed'
