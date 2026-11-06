"""Bounded pose-only diagnostic. No Test100 evaluation or scientific experiment selection.

Uses untouched, verified training examples admitted by audit_finetune_readiness.py.
The released pose checkpoint is immutable. This is not adversarial FiDIP retraining.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def rng_state():
    state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [state[0], state[1].tolist(), *state[2:]],
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def tree_digest(value):
    """Canonical state fingerprint independent of device/storage/pickle identifiers."""
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str((str(tensor.dtype), list(tensor.shape))).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=repr):
                digest.update(repr(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for part in item:
                visit(part)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def save_state(path, model, optimizer, step, protocol):
    if path.exists():
        raise RuntimeError(f'Refusing to overwrite checkpoint: {path}')
    temporary = path.with_suffix('.tmp')
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'step': step, 'rng': rng_state(), 'protocol': protocol}, temporary)
    os.replace(temporary, path)


def load_state(path, model, optimizer, protocol):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['protocol'] != protocol:
        raise RuntimeError('Resume protocol mismatch')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    restore_rng(state['rng'])
    return state['step']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--admission', required=True, type=Path)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Requires a GPU compute allocation')
    if args.steps < 10 or args.batch_size < 2 or args.batch_size % 2:
        raise ValueError('At least 10 steps and an even batch size >=2 required')
    started = time.perf_counter()
    gpu = torch.cuda.get_device_name()
    if not any(name in gpu.upper() for name in ('A100', 'V100', 'H100', 'H200')):
        raise RuntimeError(f'Unapproved GPU: {gpu}')
    admission = json.loads(args.admission.read_text())
    out = ROOT / 'output' / f'finetune-pilot-{os.environ["SLURM_JOB_ID"]}'
    out.mkdir(parents=True, exist_ok=False)
    weight_path = ROOT / 'models/hrnet_fidip.pth'
    with weight_path.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == admission['model_sha256']
    sys.path.insert(0, str(ROOT / 'lib'))
    from config import cfg
    from core.loss import JointsMSELoss
    from dataset.syrip import SYRIPDataset
    from models.adaptive_pose_hrnet import PoseHighResolutionNet
    from torchvision import transforms
    import cv2

    seed = 7373
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    config_path = ROOT / 'experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml'
    config = cfg.clone()
    config.defrost()
    config.merge_from_file(str(config_path))
    config.DEBUG.DEBUG = False
    config.freeze()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    data = SYRIPDataset(config, str(ROOT / 'data/syrip'), 'train_infant', True, transform)
    samples = admission['pilot_samples']
    selected = {row['file']: row for row in samples}
    data.db = [record for record in data.db if Path(record['image']).name in selected]
    assert len(data) == len(samples) == 32
    domains = {False: [], True: []}
    for index, record in enumerate(data.db):
        row = selected[Path(record['image']).name]
        with Path(record['image']).open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == row['sha256']
        assert bool(record['synthetic']) == row['synthetic']
        domains[row['synthetic']].append(index)
    assert len(domains[False]) == len(domains[True]) == 16

    model = PoseHighResolutionNet(config)
    model.load_state_dict(torch.load(weight_path, map_location='cpu', weights_only=True), strict=True)
    initial_head = model.final_layer.weight.detach().clone()
    model.cuda().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=config.TRAIN.WD)
    criterion = JointsMSELoss(use_target_weight=config.LOSS.USE_TARGET_WEIGHT).cuda()
    protocol = {
        'purpose': 'diagnostic only; not a scored comparison trial', 'seed': seed,
        'admission_sha256': hashlib.sha256(args.admission.read_bytes()).hexdigest(),
        'checkpoint_sha256': admission['model_sha256'],
        'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'steps': args.steps, 'batch_size': args.batch_size, 'learning_rate': 1e-5,
        'sampling': 'balanced with replacement on 16 real + 16 synthetic verified samples',
        'loss': 'supervised JointsMSELoss only; no adversarial loss or domain classifier',
        'trainable': 'all pose parameters; BatchNorm statistics updated',
        'precision': 'FP32; TF32 off; deterministic algorithms',
    }
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    torch.cuda.reset_peak_memory_stats()
    timings, losses = [], []
    def train_step(step, measured=True):
        torch.cuda.synchronize()
        then = time.perf_counter()
        sampling = random.Random(seed + step)
        indices = [sampling.choice(domains[domain]) for domain in (False, True)
                   for _ in range(args.batch_size // 2)]
        sampling.shuffle(indices)
        items = [data[index] for index in indices]  # workers=0: resume owns all augmentation RNG
        inputs = torch.stack([item[0] for item in items]).cuda()
        targets = torch.stack([item[1] for item in items]).cuda()
        weights = torch.stack([item[2] for item in items]).cuda()
        optimizer.zero_grad(set_to_none=True)
        _, heatmaps = model(inputs)
        loss = criterion(heatmaps, targets, weights)
        if not torch.isfinite(loss).item():
            raise RuntimeError('Non-finite training loss')
        loss.backward()
        if not torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all().item():
            raise RuntimeError('Non-finite gradient')
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - then
        if measured:
            timings.append(elapsed)
            losses.append(float(loss.item()))
        if step % 10 == 0 or not measured:
            print(f'STEP {step + 1}: loss={loss.item():.7f} elapsed_s={elapsed:.3f} measured={measured}', flush=True)

    print(f'GPU={gpu}; diagnostic protocol={json.dumps(protocol)}', flush=True)
    for step in range(args.steps):
        train_step(step)
    assert not torch.equal(initial_head, model.final_layer.weight.detach().cpu())
    checkpoint = out / 'resume.pth'
    save_state(checkpoint, model, optimizer, args.steps, protocol)
    # A/B branch comparison: uninterrupted continuation versus disk reload + RNG restoration.
    for step in range(args.steps, args.steps + 3):
        train_step(step, measured=False)
    expected = tree_digest({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'rng': rng_state()})
    restored_step = load_state(checkpoint, model, optimizer, protocol)
    assert restored_step == args.steps
    for step in range(restored_step, restored_step + 3):
        train_step(step, measured=False)
    actual = tree_digest({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'rng': rng_state()})
    assert actual == expected, 'Resume did not exactly reproduce uninterrupted training'
    with weight_path.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == admission['model_sha256']
    steady = timings[5:]
    report = {
        'job_id': os.environ['SLURM_JOB_ID'], 'gpu': gpu, 'torch': str(torch.__version__),
        'protocol': protocol, 'completed_training_steps': args.steps,
        'resume_probe_extra_steps': 6, 'resume_exact_state_match': True,
        'original_checkpoint_unchanged': True, 'finite_loss_and_gradients': True,
        'head_weights_changed': True, 'losses': losses, 'step_seconds': timings,
        'median_steady_step_s': float(np.median(steady)),
        'estimated_500_step_compute_minutes': float(np.median(steady) * 500 / 60),
        'estimated_1000_step_compute_minutes': float(np.median(steady) * 1000 / 60),
        'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3,
        'peak_reserved_gib': torch.cuda.max_memory_reserved() / 1024**3,
        'checkpoint_bytes': checkpoint.stat().st_size,
        'elapsed_s': time.perf_counter() - started,
        'scientific_metrics': None, 'Test100_evaluated': False,
        'caveat': 'Tiny-subset pilot timings include exhaustive gradient checks but exclude full-dataset I/O, validation, queueing and repeat seeds.',
    }
    (out / 'pilot-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    print('PASS: finite training, weight updates, exact model/Adam/RNG resume, immutable source checkpoint', flush=True)


if __name__ == '__main__':
    main()
