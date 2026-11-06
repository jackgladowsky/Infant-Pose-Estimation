#!/bin/bash
#SBATCH --job-name=fidip-train-pilot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:05:00
#SBATCH --output=slurm-logs/train-pilot-%j.out
# Typed H200 override permitted only after checking readily available capacity.
# Use --deadline to bound the latest start rather than leaving a long Hopper queue.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
: "${FIDIP_AUDIT_JOB:?Export the successful training-data admission job ID}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
source tools/environment_check_common.sh
stage gpu_inventory 15s nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
stage source_hashes 15s sha256sum tools/pilot_finetune.py tools/pilot_finetune.sbatch \
    lib/models/adaptive_pose_hrnet.py lib/dataset/JointsDataset.py lib/dataset/syrip.py \
    lib/core/loss.py "output/finetune-readiness-${FIDIP_AUDIT_JOB}.json"
stage training_and_resume_pilot 240s python -u tools/pilot_finetune.py \
    --admission "output/finetune-readiness-${FIDIP_AUDIT_JOB}.json" --steps 50 --batch-size 8
check_log 'PASS: diagnostic pilot only; no Test100 scoring or comparison experiment launched'
