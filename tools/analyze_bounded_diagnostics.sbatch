#!/bin/bash
#SBATCH --job-name=fidip-diag-analysis
#SBATCH --partition=short
#SBATCH --constraint=ib
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=slurm-logs/diag-analysis-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
export CUDA_VISIBLE_DEVICES=""
source tools/environment_check_common.sh
stage home_quota 30s check-quota "$HOME"
stage source_hashes 15s sha256sum tools/bounded_diagnostics.py tools/analyze_bounded_diagnostics.py \
    tools/run_bounded_diagnostics.py tests/test_bounded_diagnostics.py
stage geometry_and_BN_tests 90s python -B -m unittest discover -s tests -p test_bounded_diagnostics.py -v
stage existing_artifact_analysis 150s python -u tools/analyze_bounded_diagnostics.py \
    --screen-dir "${FIDIP_DIAG_SCREEN_DIR:-output/small-screen-11553889}" \
    --trained-sha256 "${FIDIP_DIAG_TRAINED_SHA:-b146f8c25d5c45f882603f858d1c532d054e1caedc687fedb41e154fd3be3f6e}" \
    --expected-trained-ap "${FIDIP_DIAG_TRAINED_AP:-93.198637527681}"
check_log 'PASS: CPU diagnostics and two fixed inference-only variants admitted; no training'
