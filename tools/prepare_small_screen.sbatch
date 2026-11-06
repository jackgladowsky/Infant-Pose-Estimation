#!/bin/bash
#SBATCH --job-name=fidip-screen-prep
#SBATCH --partition=short
#SBATCH --constraint=ib
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=slurm-logs/screen-prep-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
export CUDA_VISIBLE_DEVICES=""
source tools/environment_check_common.sh
stage home_quota 30s check-quota "$HOME"
stage source_hashes 15s sha256sum tools/prepare_small_screen.py tools/run_small_screen.py \
    tools/prepare_small_screen.sbatch tools/run_small_screen.sbatch tests/test_small_screen.py
stage screen_unit_tests 90s python -B -m unittest discover -s tests -p test_small_screen.py -v
stage resume_unit_tests 60s python -B -m unittest discover -s tests -p test_finetune_pilot.py -v
stage approved_selection_manifest 90s python -u tools/prepare_small_screen.py --approve-exclusions
check_log 'PASS: approved exclusion manifest and screen/unit guards; original datasets untouched'
