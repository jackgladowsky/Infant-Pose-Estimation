#!/bin/bash
#SBATCH --job-name=fidip-data-check
#SBATCH --partition=short
#SBATCH --constraint=ib
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=slurm-logs/data-%j.out

# CPU-only data-transfer validation. No installations, training, or GPU requests.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source tools/environment_check_common.sh
stage source_hashes 15s sha256sum tools/validate_dataset.sbatch \
    tools/validate_transferred_dataset.py data/validate_datasets.py \
    data/archive-manifest.json data/image-manifest.local.json data/validation-report.local.json
stage home_quota 30s check-quota "$HOME"
stage dataset_transfer_validation 240s python -u tools/validate_transferred_dataset.py
check_log 'PASS: transferred Test100 assets validated; known training caveats are not waived'
