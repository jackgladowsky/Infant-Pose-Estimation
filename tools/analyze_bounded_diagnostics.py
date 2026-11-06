"""CPU-only analysis of existing predictions and BN buffers; no new pose inference/training."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

from bounded_diagnostics import bn_buffer_names, pose_error, restore_bn_buffers, SCALES
from pilot_finetune import tree_digest
from run_small_screen import sha, MODEL_SHA

ROOT = Path(__file__).resolve().parents[1]
TRAINED = 'output/small-screen-11553889/stage4_head/final_pose.pth'
TRAINED_SHA = 'b146f8c25d5c45f882603f858d1c532d054e1caedc687fedb41e154fd3be3f6e'
GT = 'data/syrip/annotations/person_keypoints_validate_infant.json'
GT_SHA = '8b299a879a95913483498e3f2115a826ac23ebc5262c3efb8d65f63718e3194c'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screen-dir', type=Path, default=Path('output/small-screen-11553889'))
    parser.add_argument('--trained-sha256', default=TRAINED_SHA)
    parser.add_argument('--expected-trained-ap', type=float, default=93.198637527681)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run CPU analysis in a compute allocation')
    screen_dir = (ROOT / args.screen_dir).resolve().relative_to(ROOT)
    trained_path = str(screen_dir / 'stage4_head/final_pose.pth')
    fixed = {'models/hrnet_fidip.pth': MODEL_SHA, trained_path: args.trained_sha256, GT: GT_SHA}
    for path, expected in fixed.items():
        assert sha(ROOT / path) == expected, f'Input changed: {path}'
    data = json.loads((ROOT / GT).read_text())
    truth = {row['image_id']: row for row in data['annotations']}
    assert len(truth) == len(data['annotations']) == 100
    from pycocotools.cocoeval import COCOeval
    sigmas = COCOeval(iouType='keypoints').params.kpt_oks_sigmas
    scores, joint_errors, visible = {}, {}, None
    for mode in ('original_flip_on', 'head', 'stage4_head'):
        path = ROOT / screen_dir / mode / 'results/keypoints_validate_infant_results_0.json'
        fixed[str(path.relative_to(ROOT))] = sha(path)
        records = json.loads(path.read_text())
        predictions = {row['image_id']: row for row in records}
        assert len(records) == 100 and set(predictions) == set(truth)
        rows = [pose_error(truth[key], predictions[key], sigmas) for key in sorted(truth)]
        scores[mode] = np.array([row[0] for row in rows])
        joint_errors[mode] = np.stack([row[1] for row in rows])
        visible = np.stack([row[2] for row in rows])
    comparisons = {}
    baseline = scores['original_flip_on']
    for mode in ('head', 'stage4_head'):
        delta = scores[mode] - baseline
        comparisons[mode] = {
            'mean_per_person_OKS': float(scores[mode].mean()),
            'mean_OKS_delta': float(delta.mean()),
            'improved_by_at_least_0.01_OKS': int((delta >= .01).sum()),
            'worsened_by_at_least_0.01_OKS': int((delta <= -.01).sum()),
            'within_0.01_OKS': int((np.abs(delta) < .01).sum()),
            'per_person_OKS_correlation_to_original': float(np.corrcoef(baseline, scores[mode])[0, 1]),
        }
    labels = data['categories'][0]['keypoints']
    joint_summary = []
    for joint, label in enumerate(labels):
        mask = visible[:, joint]
        joint_summary.append({'joint': label, 'labeled_examples': int(mask.sum()),
                              **{mode: float(error[mask, joint].mean()) if mask.any() else None
                                 for mode, error in joint_errors.items()}})
    ids = sorted(truth)
    worst = [{'image_id': ids[i], 'original_OKS': float(baseline[i]),
              'head_OKS': float(scores['head'][i]), 'stage4_OKS': float(scores['stage4_head'][i])}
             for i in np.argsort(baseline)[:10]]

    sys.path.insert(0, str(ROOT / 'lib'))
    from config import cfg
    from models.adaptive_pose_hrnet import PoseHighResolutionNet
    config_path = ROOT / 'experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml'
    config = cfg.clone()
    config.defrost()
    config.merge_from_file(str(config_path))
    config.freeze()
    model = PoseHighResolutionNet(config)
    original = torch.load(ROOT / 'models/hrnet_fidip.pth', map_location='cpu', weights_only=True)
    trained = torch.load(ROOT / trained_path, map_location='cpu', weights_only=True)
    model.load_state_dict(original, strict=True)
    restored, names = restore_bn_buffers(model, original, trained)
    assert tree_digest({key: value for key, value in restored.items() if key not in names}) == \
        tree_digest({key: value for key, value in trained.items() if key not in names})
    changed, drift = [], []
    for key in bn_buffer_names(model):
        if not torch.equal(original[key], trained[key]):
            changed.append(key)
        if key.endswith('running_mean') and (not torch.equal(original[key], trained[key]) or
                                            not torch.equal(original[key.replace('running_mean', 'running_var')],
                                                            trained[key.replace('running_mean', 'running_var')])):
            variance_key = key.replace('running_mean', 'running_var')
            shift = ((trained[key] - original[key]).abs() / (original[variance_key] + 1e-5).sqrt()).numpy()
            ratios = (trained[variance_key] / original[variance_key].clamp_min(1e-12)).numpy()
            drift.append({'module': key.rsplit('.', 1)[0], 'mean_shift_original_std_median': float(np.median(shift)),
                          'mean_shift_original_std_max': float(shift.max()),
                          'variance_ratio_median': float(np.median(ratios)),
                          'variance_ratio_min': float(ratios.min()), 'variance_ratio_max': float(ratios.max())})
    for path in ('tools/bounded_diagnostics.py', 'tools/run_bounded_diagnostics.py',
                 'tools/analyze_bounded_diagnostics.py', 'lib/utils/transforms.py',
                 'lib/core/inference.py', 'lib/dataset/JointsDataset.py', 'lib/dataset/syrip.py',
                 'lib/models/adaptive_pose_hrnet.py', str(config_path.relative_to(ROOT))):
        fixed[path] = sha(ROOT / path)
    report = {
        'cpu_job': os.environ['SLURM_JOB_ID'], 'input_hashes': fixed,
        'trained_checkpoint': trained_path, 'expected_trained_AP': args.expected_trained_ap,
        'analysis': 'Existing Test100 outputs only; per-person OKS/error is NOT confidence-ranked AP. Exploratory, not new validation.',
        'original_mean_per_person_OKS': float(baseline.mean()), 'prediction_comparisons': comparisons,
        'joint_mean_distance_over_sqrt_GT_area': joint_summary, 'worst_original_examples': worst,
        'BN_buffer_count': len(names), 'BN_buffers_changed_after_training': len(changed),
        'changed_BN_buffer_names': changed, 'BN_drift_changed_modules': drift,
        'non_BN_weights_identical_under_restoration': True,
        'prespecified_GPU_tests': {
            'BN': 'stage4 trained parameters/affine weights unchanged; restore ONLY original running_mean/running_var/num_batches_tracked',
            'TTA': {'source': 'original released checkpoint', 'crop_factors': list(SCALES),
                    'flip': True, 'fusion': 'repo-affine aligned heatmaps, uniform available-view mean, then existing DarkPose decoder'},
        },
        'stop_rule': 'Two diagnostic variants only; no fine-tuning or subsequent hyperparameter sweep; one GPU allocation capped at 10min',
    }
    target = ROOT / 'output' / f'bounded-analysis-{os.environ["SLURM_JOB_ID"]}.json'
    target.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key not in
                     ('input_hashes', 'joint_mean_distance_over_sqrt_GT_area', 'changed_BN_buffer_names', 'BN_drift_changed_modules')}, indent=2), flush=True)
    print(f'BN changed modules: {len(drift)}, buffers: {len(changed)}; report={target}', flush=True)
    print('PASS: CPU analysis, safe checkpoint loading, exact BN-only restoration and pinned inputs', flush=True)


if __name__ == '__main__':
    main()
