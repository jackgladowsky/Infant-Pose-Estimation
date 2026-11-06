"""CPU-only, read-only admission for a diagnostic pilot, not a repaired training split."""
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA = 'bfbf9df225674614652d5dbdd066daf71c9b9d7f3c31976b7ed33bb96af55d07'
TRAIN_SHA = '003aaccac1698e48251f0b76d335c2369eefb40610def9c5cc359e661b8e0505'
TEST_SHA = '8b299a879a95913483498e3f2115a826ac23ebc5262c3efb8d65f63718e3194c'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run this audit in a CPU compute allocation')
    train_path = ROOT / 'data/syrip/annotations/person_keypoints_train_infant.json'
    test_path = ROOT / 'data/syrip/annotations/person_keypoints_validate_infant.json'
    assert sha(train_path) == TRAIN_SHA and sha(test_path) == TEST_SHA
    assert sha(ROOT / 'models/hrnet_fidip.pth') == MODEL_SHA
    train = json.loads(train_path.read_text())
    test = json.loads(test_path.read_text())
    source_manifest = json.loads((ROOT / 'data/image-manifest.local.json').read_text())
    def verified_image_sha(path):
        actual = sha(path)
        key = str(path.resolve().relative_to((ROOT / 'data').resolve()))
        assert actual == source_manifest[key]['sha256'], f'Image changed after transfer: {path}'
        return actual
    test_hashes = {
        verified_image_sha(ROOT / 'data/syrip/images/validate_infant' / f'test{image["id"]}.jpg')
        for image in test['images']
    }
    eligible, mismatches, overlaps = [], [], []
    for image in sorted(train['images'], key=lambda row: row['id']):
        path = ROOT / 'data/syrip/images/train_infant' / f'train{image["id"]:05d}.jpg'
        with Image.open(path) as decoded:
            dimensions = decoded.size
        mismatch = dimensions != (image['width'], image['height'])
        overlap = verified_image_sha(path) in test_hashes
        if mismatch:
            mismatches.append(path.name)
        if overlap:
            overlaps.append(path.name)
        if not mismatch and not overlap:
            eligible.append({'id': image['id'], 'file': path.name,
                             'synthetic': bool(image['is_synthetic'])})
    assert len(train['images']) == 1200 and len(test['images']) == 100
    assert len(mismatches) == 46 and len(overlaps) == 4
    # Pilot selection only: fixed small coverage of both domains, not an experiment split.
    selected = [row for domain in (False, True)
                for row in [r for r in eligible if r['synthetic'] == domain][:16]]
    assert len(selected) == 32
    for row in selected:
        row['sha256'] = sha(ROOT / 'data/syrip/images/train_infant' / row['file'])
    report = {
        'job_id': os.environ['SLURM_JOB_ID'],
        'purpose': 'diagnostic pilot admission only; original datasets/labels unchanged',
        'model_sha256': MODEL_SHA, 'train_annotation_sha256': TRAIN_SHA,
        'test_annotation_sha256': TEST_SHA,
        'training_records': len(train['images']),
        'dimension_mismatches': mismatches, 'train_test_byte_overlaps': overlaps,
        'candidate_eligible_count': len(eligible),
        'candidate_domain_counts': dict(Counter('synthetic' if row['synthetic'] else 'real'
                                               for row in eligible)),
        'pilot_samples': selected,
        'author_training_sampler': 'BalancedBatchSampler alternates real/synthetic: already 1:1',
        'author_freezing': 'freezes first 109 parameter tensors by index; BatchNorm stats not frozen',
        'author_resume_limitations': 'omits domain optimizer/scheduler/RNG/sampler state',
        'proposed_comparison_profile': 'exclude flagged 46 dimension mismatches and 4 exact Test100 overlaps; requires explicit approval',
        'experiment_profile_approved': False,
        'limitations': [
            'An exclusion in future fine-tuning cannot undo possible prior exposure in released weights.',
            'Existing checkpoint may already have seen training-derived development samples.',
            'No independent development-validation protocol established yet.',
        ],
    }
    out = ROOT / 'output' / f'finetune-readiness-{os.environ["SLURM_JOB_ID"]}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'pilot_samples'}, indent=2), flush=True)
    print(f'PASS: diagnostic-only admission: {out}', flush=True)


if __name__ == '__main__':
    main()
