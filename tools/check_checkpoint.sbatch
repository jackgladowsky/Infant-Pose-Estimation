#!/bin/bash
#SBATCH --job-name=fidip-checkpoint
#SBATCH --partition=short
#SBATCH --constraint=ib
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=slurm-logs/checkpoint-%j.out

# CPU-only admission check before using GPU time. No training/install/download.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
source tools/environment_check_common.sh
stage home_quota 30s check-quota "$HOME"
stage source_hashes 15s sha256sum tools/check_checkpoint.sbatch \
    tools/test_adaptive_model.py lib/models/adaptive_pose_hrnet.py \
    experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml
stage checkpoint_check 180s python -u - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys
import time

started = time.perf_counter()
sys.path.insert(0, str(Path.cwd() / 'lib'))
import torch
from config import cfg
from models.adaptive_pose_hrnet import PoseHighResolutionNet

model_path = Path('models/hrnet_fidip.pth')
config_path = Path('experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml')
expected_sha = 'bfbf9df225674614652d5dbdd066daf71c9b9d7f3c31976b7ed33bb96af55d07'
with model_path.open('rb') as stream:
    actual_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
if actual_sha != expected_sha:
    raise RuntimeError('Transferred checkpoint SHA-256 mismatch')
print('CHECKPOINT_SHA256:', actual_sha, flush=True)
state = torch.load(model_path, map_location='cpu', weights_only=True)
if not isinstance(state, dict) or not state or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
    raise RuntimeError('Expected a nonempty tensor state_dict')
if not all(torch.isfinite(value).all().item() for value in state.values()):
    raise RuntimeError('Non-finite checkpoint tensor')
config = cfg.clone()
config.defrost()
config.merge_from_file(str(config_path))
config.freeze()
if config.MODEL.NAME != 'adaptive_pose_hrnet':
    raise RuntimeError('Unexpected baseline architecture')
model = PoseHighResolutionNet(config)
model.load_state_dict(state, strict=True)
model.eval()
with torch.inference_mode():
    features, heatmaps = model(torch.zeros((1, 3, config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0])))
expected_shape = (1, config.MODEL.NUM_JOINTS, config.MODEL.HEATMAP_SIZE[1], config.MODEL.HEATMAP_SIZE[0])
if tuple(heatmaps.shape) != expected_shape or not torch.isfinite(heatmaps).all().item():
    raise RuntimeError(f'Invalid CPU forward result: {tuple(heatmaps.shape)}')
report = {
    'job_id': os.environ['SLURM_JOB_ID'], 'model_file': str(model_path),
    'sha256': actual_sha, 'bytes': model_path.stat().st_size, 'state_dict_keys': len(state),
    'config': str(config_path), 'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
    'strict_load_passed': True, 'finite_tensors': True, 'cpu_forward_passed': True,
    'heatmap_shape': list(heatmaps.shape), 'feature_shape': list(features.shape),
    'torch': torch.__version__, 'elapsed_s': round(time.perf_counter() - started, 3),
}
report_path = Path('log') / f'checkpoint-{os.environ["SLURM_JOB_ID"]}.json'
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2), flush=True)
print('PASS: exact checkpoint identity, tensor-only strict load, and CPU forward', flush=True)
PY
check_log 'PASS: checkpoint admitted for GPU baseline evaluation'
