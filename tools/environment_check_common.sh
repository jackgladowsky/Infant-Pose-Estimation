#!/bin/bash
# Shared logging for bounded smoke-test jobs. No installation or downloads.
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
CHECK_STARTED=$SECONDS
CHECK_STAGE=bootstrap

check_log() {
    printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

check_finish() {
    local rc=$?
    check_log "JOB_END job=${SLURM_JOB_ID:-local} stage=$CHECK_STAGE rc=$rc elapsed_s=$((SECONDS - CHECK_STARTED))"
}
trap check_finish EXIT
trap 'check_log "JOB_SIGNAL signal=TERM stage=$CHECK_STAGE"; exit 143' TERM

stage() {
    local label=$1 limit=$2 started=$SECONDS rc
    shift 2
    CHECK_STAGE=$label
    check_log "STAGE_START name=$label limit=$limit"
    if timeout --signal=TERM --kill-after=5s "$limit" "$@"; then
        rc=0
    else
        rc=$?
    fi
    check_log "STAGE_END name=$label rc=$rc elapsed_s=$((SECONDS - started))"
    return "$rc"
}

check_log "JOB_START job=${SLURM_JOB_ID:-local} host=$(hostname) cwd=$PWD"
check_log "STAGE_START name=bootstrap"
module load cuda/12.1.1
source .venv/bin/activate
check_log "STAGE_END name=bootstrap rc=0 elapsed_s=$((SECONDS - CHECK_STARTED))"
