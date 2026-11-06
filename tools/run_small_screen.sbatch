#!/bin/bash
#SBATCH --job-name=fidip-small-screen
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-logs/small-screen-%j.out
# Three independent tests run sequentially on one device for controlled comparisons.
# Checkpoint at steps 300/600 and persist each result immediately; no best-AP selection.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
: "${FIDIP_SCREEN_PROFILE_JOB:?Export successful CPU preparation job ID}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
source tools/environment_check_common.sh
stage gpu_inventory 15s nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
stage source_hashes 15s sha256sum tools/run_small_screen.py tools/run_small_screen.sbatch \
    tools/pilot_finetune.py "output/small-screen-profile-${FIDIP_SCREEN_PROFILE_JOB}.json"
stage three_prespecified_tests 780s python -u tools/run_small_screen.py \
    --profile "output/small-screen-profile-${FIDIP_SCREEN_PROFILE_JOB}.json"
check_log 'PASS: three tests complete; no data/label repair or Test100-based checkpoint selection'
