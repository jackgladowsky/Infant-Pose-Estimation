#!/bin/bash
#SBATCH --job-name=fidip-two-diagnostics
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-logs/two-diagnostics-%j.out
# One capped allocation: matched controls + BN restoration + fixed 3-crop TTA, then STOP.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
: "${FIDIP_DIAG_ANALYSIS_JOB:?Export successful CPU diagnostic-analysis job ID}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
source tools/environment_check_common.sh
stage gpu_inventory 15s nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
stage source_hashes 15s sha256sum tools/run_bounded_diagnostics.py tools/bounded_diagnostics.py \
    tools/run_bounded_diagnostics.sbatch "output/bounded-analysis-${FIDIP_DIAG_ANALYSIS_JOB}.json"
stage two_fixed_inference_diagnostics 480s python -u tools/run_bounded_diagnostics.py \
    --analysis "output/bounded-analysis-${FIDIP_DIAG_ANALYSIS_JOB}.json"
check_log 'PASS: bounded diagnostics complete; no further training or sweep authorized'
