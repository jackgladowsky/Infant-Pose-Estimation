#!/bin/bash
#SBATCH --job-name=fidip-train-audit
#SBATCH --partition=short
#SBATCH --constraint=ib
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=slurm-logs/train-audit-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
source tools/environment_check_common.sh
stage home_quota 30s check-quota "$HOME"
stage source_hashes 15s sha256sum tools/audit_finetune_readiness.py tools/pilot_finetune.py \
    lib/utils/sampler.py tools/train_adaptive_model_hrnet.py lib/core/function.py \
    tests/test_finetune_pilot.py
stage resume_unit_tests 60s python -B -m unittest discover -s tests -p test_finetune_pilot.py -v
stage training_data_admission 90s python -u tools/audit_finetune_readiness.py
check_log 'PASS: pilot-only admission; full training profile still requires explicit approval'
