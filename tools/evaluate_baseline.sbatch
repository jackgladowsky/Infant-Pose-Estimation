#!/bin/bash
#SBATCH --job-name=fidip-baseline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=slurm-logs/baseline-%j.out

# Original Test100/pretrained checkpoint only. No training or dataset repair.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
: "${FIDIP_CHECKPOINT_JOB:?Supply the successful CPU checkpoint-check job ID}"
source tools/environment_check_common.sh
export BASELINE_OUTPUT="output/baseline-test100-${SLURM_JOB_ID}"
export BASELINE_LOGROOT="log/baseline-test100-${SLURM_JOB_ID}"
mkdir -p "$BASELINE_OUTPUT" "$BASELINE_LOGROOT"
stage gpu_inventory 15s nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
stage source_hashes 15s sha256sum tools/evaluate_baseline.sbatch tools/test_adaptive_model.py \
    experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml \
    data/syrip/annotations/person_keypoints_validate_infant.json
stage admission 15s python -u - <<'PY'
import json, os
from pathlib import Path
checkpoint = json.loads((Path('log') / f'checkpoint-{os.environ["FIDIP_CHECKPOINT_JOB"]}.json').read_text())
data = json.loads(Path('data/explorer-transfer-validation.json').read_text())
if not all(checkpoint.get(k) for k in ('strict_load_passed', 'finite_tensors', 'cpu_forward_passed')):
    raise RuntimeError('CPU checkpoint admission did not pass')
if checkpoint['sha256'] != 'bfbf9df225674614652d5dbdd066daf71c9b9d7f3c31976b7ed33bb96af55d07':
    raise RuntimeError('Unexpected checkpoint identity')
if not data['baseline_test100_ready']:
    raise RuntimeError('Original Test100 data is not validated')
print('PASS: CPU checkpoint and Test100 data admission; training issues remain separate', flush=True)
PY
stage checkpoint_identity 15s bash -c 'printf "%s\n" "bfbf9df225674614652d5dbdd066daf71c9b9d7f3c31976b7ed33bb96af55d07  models/hrnet_fidip.pth" | sha256sum -c -'
stage baseline_evaluation 180s python -u tools/test_adaptive_model.py \
    --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml \
    TEST.MODEL_FILE models/hrnet_fidip.pth TEST.USE_GT_BBOX True \
    DATASET.ROOT data/syrip/ OUTPUT_DIR "$BASELINE_OUTPUT" LOG_DIR "$BASELINE_LOGROOT" \
    DEBUG.DEBUG False PRINT_FREQ 1
stage full_precision_metrics 30s python -u - <<'PY'
import hashlib
import json
import os
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

root = Path(os.environ['BASELINE_OUTPUT'])
annotations = Path('data/syrip/annotations/person_keypoints_validate_infant.json')
files = list(root.rglob('keypoints_validate_infant_results_0.json'))
if len(files) != 1:
    raise RuntimeError(f'Expected one prediction file, found {files}')
gt = COCO(str(annotations))
if len(gt.getImgIds()) != 100:
    raise RuntimeError('Refusing evaluation on a split other than original Test100')
results = gt.loadRes(str(files[0]))
evaluator = COCOeval(gt, results, 'keypoints')
evaluator.params.useSegm = None
evaluator.evaluate()
evaluator.accumulate()
evaluator.summarize()
names = ['AP', 'AP50', 'AP75', 'AP_medium', 'AP_large', 'AR', 'AR50', 'AR75', 'AR_medium', 'AR_large']
metrics = {name: float(value) for name, value in zip(names, evaluator.stats)}
if not 0 <= metrics['AP'] <= 1:
    raise RuntimeError('Invalid AP result')
report = {
    'job_id': os.environ['SLURM_JOB_ID'], 'checkpoint_check_job': os.environ['FIDIP_CHECKPOINT_JOB'],
    'checkpoint': 'models/hrnet_fidip.pth',
    'checkpoint_sha256': 'bfbf9df225674614652d5dbdd066daf71c9b9d7f3c31976b7ed33bb96af55d07',
    'annotation': str(annotations), 'annotation_sha256': hashlib.sha256(annotations.read_bytes()).hexdigest(),
    'test_images': 100, 'predictions': str(files[0]), 'metrics': metrics,
    'AP_percent': metrics['AP'] * 100, 'published_reference_AP_percent': 93.6,
    'config': 'experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml',
    'inference_changes': 'none; GT boxes/flip/postprocess preserved; only debug rendering/log frequency/output paths overridden',
}
(root / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2), flush=True)
PY
check_log 'PASS: pretrained FiDIP evaluation on original Test100; full-precision metrics saved'
