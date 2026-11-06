#!/bin/bash
#SBATCH --job-name=fidip-build
#SBATCH --partition=short
#SBATCH --constraint=ib
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=slurm-logs/build-%j.out

# Compile CUDA code without reserving a GPU; regress CPU NMS before a GPU job.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source tools/environment_check_common.sh
export NMS_CUDA_ARCH="${NMS_CUDA_ARCH:-70}"
stage compiler 15s nvcc --version
stage source_hashes 15s sha256sum tools/build_environment.sbatch \
    tools/environment_check_common.sh tools/check_environment.py \
    lib/nms/setup_linux.py lib/nms/cpu_nms.pyx lib/nms/gpu_nms.pyx \
    tests/test_environment_checks.py
stage harness_regression 15s python -u -m unittest discover -s tests -v
stage native_build 180s bash -c 'cd lib/nms && exec python -u setup_linux.py build_ext --inplace --force --cython-c-in-temp'
stage cpu_nms_regression 60s python -u tools/check_environment.py --phase cpu
check_log 'PASS: native compilation and CPU NMS regression; no GPU requested'
