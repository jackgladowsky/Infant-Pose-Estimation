"""Build an explicitly approved selection manifest, without editing any dataset payload/label."""
import argparse
import json
import os
from pathlib import Path

from audit_finetune_readiness import main as audit
from run_small_screen import sha, MODEL_SHA

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--approve-exclusions', action='store_true', required=True)
    parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('CPU compute allocation required')
    # Fresh CPU audit verifies every training/Test100 payload against the transferred source audit.
    audit()
    admission_path = ROOT / 'output' / f'finetune-readiness-{os.environ["SLURM_JOB_ID"]}.json'
    admission = json.loads(admission_path.read_text())
    annotation = ROOT / 'data/syrip/annotations/person_keypoints_train_infant.json'
    train = json.loads(annotation.read_text())
    excluded = {name: 'JPEG/annotation dimension mismatch' for name in admission['dimension_mismatches']}
    for name in admission['train_test_byte_overlaps']:
        assert name not in excluded
        excluded[name] = 'exact-byte overlap with original Test100'
    source = json.loads((ROOT / 'data/image-manifest.local.json').read_text())
    included = []
    for image in train['images']:
        name = f'train{image["id"]:05d}.jpg'
        if name in excluded:
            continue
        path = ROOT / 'data/syrip/images/train_infant' / name
        key = str(path.resolve().relative_to((ROOT / 'data').resolve()))
        included.append({'id': image['id'], 'file': name, 'synthetic': bool(image['is_synthetic']),
                         'sha256': source[key]['sha256']})
    assert len(included) == 1150 and len(excluded) == 50
    assert sum(not row['synthetic'] for row in included) == 196
    inputs = [
        'models/hrnet_fidip.pth',
        'data/syrip/annotations/person_keypoints_train_infant.json',
        'data/syrip/annotations/person_keypoints_validate_infant.json',
        'lib/models/adaptive_pose_hrnet.py', 'lib/dataset/JointsDataset.py',
        'lib/dataset/syrip.py', 'lib/core/function.py', 'lib/core/inference.py',
        'lib/core/loss.py', 'lib/utils/transforms.py', 'tools/run_small_screen.py',
        'tools/pilot_finetune.py',
    ]
    profile = {
        'approval': 'user approved three-test screen with separate flagged-example exclusion list',
        'approval_context': '2026-11-01: user chose head-only, final-stage+head and flip-off tests after the automatic exclusion list was proposed',
        'cpu_job': os.environ['SLURM_JOB_ID'], 'admission_sha256': sha(admission_path),
        'steps': 600, 'batch_size': 8, 'learning_rate': 1e-5, 'seed': 7373,
        'included': included, 'excluded': excluded,
        'input_hashes': {name: sha(ROOT / name) for name in inputs},
        'evaluation': 'fixed-budget final checkpoints; one original Test100 score per trained model, no early stopping or adaptive sweep',
        'limitations': admission['limitations'],
        'originals_modified': False,
    }
    assert profile['input_hashes']['models/hrnet_fidip.pth'] == MODEL_SHA
    target = ROOT / 'output' / f'small-screen-profile-{os.environ["SLURM_JOB_ID"]}.json'
    if target.exists():
        raise RuntimeError(f'Refusing overwrite: {target}')
    target.write_text(json.dumps(profile, indent=2) + '\n')
    print(f'PASS: approved 1,150-example selection manifest: {target}; original images/labels untouched', flush=True)


if __name__ == '__main__':
    main()
