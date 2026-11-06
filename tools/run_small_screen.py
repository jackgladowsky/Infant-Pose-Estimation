"""Three prespecified exploratory tests; no Test100-based checkpoint selection.

Reuses the repository's dataset, supervised loss, decoder, flip-back and COCO evaluation.
Runtime helpers are intentionally unpublished. Original annotations/weights are immutable.
"""
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

from pilot_finetune import save_state, tree_digest

ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA = 'bfbf9df225674614652d5dbdd066daf71c9b9d7f3c31976b7ed33bb96af55d07'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def configure_layers(model, mode):
    if mode not in ('head', 'stage4_head'):
        raise ValueError(mode)
    prefixes = ('final_layer.',) if mode == 'head' else ('stage4.', 'final_layer.')
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(prefixes))
    # Do NOT call model.train() afterward: frozen BN/dropout must remain in eval mode.
    model.eval()
    model.final_layer.train()
    if mode == 'stage4_head':
        model.stage4.train()
    frozen = [name for name in model.state_dict() if not name.startswith(prefixes)]
    return frozen


def balanced_indices(domains, seed, step, batch_size):
    if batch_size <= 0 or batch_size % 2:
        raise ValueError('Balanced batches require a positive even size')
    generator = random.Random(seed + step)
    indices = [generator.choice(domains[domain]) for domain in (False, True)
               for _ in range(batch_size // 2)]
    generator.shuffle(indices)
    return indices


def set_augmentation_seed(seed, step):
    random.seed(seed + 100000 + step)
    np.random.seed(seed + 100000 + step)


def configure_runtime(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def benchmark_forward(model, inputs, repeats=9):
    """Paired, warmed, device-resident CNN timing; excludes I/O, decoding and flip-back."""
    inputs = inputs.cuda()
    flipped = torch.flip(inputs, dims=[3])
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            model(inputs)
            model(flipped)
        torch.cuda.synchronize()
        results = {False: [], True: []}
        for repeat in range(repeats):
            for flip in ((False, True) if repeat % 2 == 0 else (True, False)):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                model(inputs)
                if flip:
                    model(flipped)
                end.record()
                end.synchronize()
                results[flip].append(start.elapsed_time(end) / len(inputs))
    return {
        'scope': 'CNN forward only, device-resident inputs, warmed; excludes preprocessing, transfer, flip-back and decoding',
        'batch_size': len(inputs), 'paired_repeats': repeats,
        'flip_on_ms_per_original_image': float(np.median(results[True])),
        'flip_off_ms_per_original_image': float(np.median(results[False])),
        'flip_off_forward_speedup': float(np.median(results[True]) / np.median(results[False])),
        'raw_ms_per_image': {str(key): value for key, value in results.items()},
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('GPU compute allocation required')
    profile = json.loads(args.profile.read_text())
    if profile.get('approval') != 'user approved three-test screen with separate flagged-example exclusion list':
        raise RuntimeError('Filtered profile lacks explicit approval')
    if profile['steps'] != 600 or profile['batch_size'] != 8 or profile['learning_rate'] != 1e-5:
        raise RuntimeError('Unapproved screen budget')
    gpu = torch.cuda.get_device_name()
    if not any(kind in gpu.upper() for kind in ('V100', 'A100', 'H100', 'H200')):
        raise RuntimeError(f'Unapproved GPU: {gpu}')
    for name, expected in profile['input_hashes'].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError(f'Input changed after CPU admission: {name}')
    out = ROOT / 'output' / f'small-screen-{os.environ["SLURM_JOB_ID"]}'
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(ROOT / 'lib'))
    from config import cfg
    from core.function import validate
    from core.loss import JointsMSELoss
    from dataset.syrip import SYRIPDataset
    from models.adaptive_pose_hrnet import PoseHighResolutionNet
    from torchvision import transforms
    import cv2
    cv2.setNumThreads(1)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    seed = profile['seed']
    configure_runtime(seed)
    config_path = ROOT / 'experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml'
    config = cfg.clone()
    config.defrost()
    config.merge_from_file(str(config_path))
    config.DATASET.ROOT = str(ROOT / 'data/syrip')
    config.OUTPUT_DIR = str(out)
    config.DEBUG.DEBUG = False
    config.CUDNN.BENCHMARK = False
    config.CUDNN.DETERMINISTIC = True
    config.PRINT_FREQ = 1
    config.WORKERS = 0
    config.freeze()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_data = SYRIPDataset(config, config.DATASET.ROOT, 'train_infant', True, transform)
    allowed = {row['file']: row for row in profile['included']}
    train_data.db = [record for record in train_data.db if Path(record['image']).name in allowed]
    assert len(train_data) == len(allowed) == 1150
    domains = {False: [], True: []}
    for index, record in enumerate(train_data.db):
        label = bool(record['synthetic'])
        assert label == allowed[Path(record['image']).name]['synthetic']
        domains[label].append(index)
    assert (len(domains[False]), len(domains[True])) == (196, 954)

    class RecordingDataset(SYRIPDataset):
        def evaluate(self, *values, **keywords):
            result = super().evaluate(*values, **keywords)
            self.last_metrics = {str(key): float(value) for key, value in result[0].items()}
            return result
    val_data = RecordingDataset(config, config.DATASET.ROOT, 'validate_infant', False, transform)
    assert len(val_data) == 100
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=16, shuffle=False,
                                             num_workers=0, pin_memory=True)
    criterion = JointsMSELoss(use_target_weight=config.LOSS.USE_TARGET_WEIGHT).cuda()
    original = torch.load(ROOT / 'models/hrnet_fidip.pth', map_location='cpu', weights_only=True)
    def new_model():
        model = PoseHighResolutionNet(config)
        model.load_state_dict(original, strict=True)
        return model.cuda()

    protocol = {
        'job_id': os.environ['SLURM_JOB_ID'], 'gpu': gpu, 'torch': str(torch.__version__),
        'profile_sha256': sha(args.profile), 'config_sha256': sha(config_path),
        'initial_checkpoint_sha256': MODEL_SHA, 'seed': seed,
        'steps': 600, 'batch_size': 8, 'learning_rate': 1e-5,
        'sampling': '1:1 real/synthetic with replacement; same step-indexed batches and augmentation seed',
        'optimizer': f'Adam, constant LR, weight_decay={config.TRAIN.WD}',
        'objective': 'supervised pose MSE only; no new domain classifier/adversarial training',
        'precision': 'FP32, TF32 off, deterministic; same device for all tests',
        'evaluation': 'original Test100 GT boxes, DarkPose decoder, batch16; flip on except noflip test',
        'selection': 'fixed 600-step final checkpoints only; no Test100 early stopping or adaptive search',
        'limitation': 'exploratory single-seed comparisons; released checkpoint may have prior training/Test100 overlap',
    }
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    results = {}
    def save_results():
        (out / 'results.json').write_text(json.dumps({'protocol': protocol, 'results': results}, indent=2) + '\n')
    def evaluate(model, name, flip):
        target = out / name
        target.mkdir(parents=True, exist_ok=True)
        current = config.clone()
        current.defrost()
        current.TEST.FLIP_TEST = flip
        current.freeze()
        model.eval()
        # Warm the two evaluation batch shapes without looking at labels or scores.
        with torch.no_grad():
            for size in (16, 4):
                model(torch.zeros(size, 3, 384, 288, device='cuda'))
        wrapped = torch.nn.DataParallel(model, device_ids=[0])
        torch.cuda.synchronize()
        start = time.perf_counter()
        ap = validate(current, val_loader, val_data, wrapped, criterion, str(target), str(target / 'tb'))
        torch.cuda.synchronize()
        metrics = dict(val_data.last_metrics)
        assert abs(metrics['AP'] - ap) < 1e-12
        return {'metrics_fraction': metrics, 'AP': ap * 100,
                'evaluation_wall_s': time.perf_counter() - start, 'flip_test': flip,
                'predictions': str(target / 'results/keypoints_validate_infant_results_0.json')}

    model = new_model()
    results['original_flip_on'] = evaluate(model, 'original_flip_on', True)
    if abs(results['original_flip_on']['AP'] - 93.6071706887983) > .001:
        raise RuntimeError('Original-checkpoint control does not reproduce known baseline; stop screen')
    results['original_flip_off'] = evaluate(model, 'original_flip_off', False)
    results['original_flip_off']['paired_forward_benchmark'] = benchmark_forward(model, next(iter(val_loader))[0])
    save_results()
    del model
    torch.cuda.empty_cache()

    for mode in ('head', 'stage4_head'):
        configure_runtime(seed)
        model = new_model()
        frozen_names = configure_layers(model, mode)
        state = model.state_dict()
        frozen_before = tree_digest({name: state[name] for name in frozen_names})
        del state
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=1e-5, weight_decay=config.TRAIN.WD)
        variant_dir = out / mode
        variant_dir.mkdir()
        variant_protocol = {**protocol, 'mode': mode, 'trainable_parameters': sum(p.numel() for p in parameters)}
        (variant_dir / 'protocol.json').write_text(json.dumps(variant_protocol, indent=2) + '\n')
        timings, losses = [], []
        torch.cuda.reset_peak_memory_stats()
        for step in range(600):
            torch.cuda.synchronize()
            start = time.perf_counter()
            indices = balanced_indices(domains, seed, step, 8)
            set_augmentation_seed(seed, step)
            items = [train_data[index] for index in indices]
            inputs = torch.stack([item[0] for item in items]).cuda()
            targets = torch.stack([item[1] for item in items]).cuda()
            weights = torch.stack([item[2] for item in items]).cuda()
            optimizer.zero_grad(set_to_none=True)
            _, outputs = model(inputs)
            loss = criterion(outputs, targets, weights)
            if not torch.isfinite(loss).item():
                raise RuntimeError(f'Non-finite loss: {mode}/{step}')
            loss.backward()
            if not torch.stack([torch.isfinite(p.grad).all() for p in parameters if p.grad is not None]).all().item():
                raise RuntimeError(f'Non-finite gradient: {mode}/{step}')
            optimizer.step()
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)
            losses.append(float(loss.item()))
            if step % 50 == 0:
                print(f'{mode} step {step+1}/600 loss={loss.item():.8f} step_s={timings[-1]:.3f}', flush=True)
            if step + 1 in (300, 600):
                save_state(variant_dir / f'resume-{step+1}.pth', model, optimizer, step+1, variant_protocol)
        state = model.state_dict()
        frozen_after = tree_digest({name: state[name] for name in frozen_names})
        del state
        assert frozen_after == frozen_before, 'Frozen parameters or BatchNorm statistics changed'
        assert not torch.equal(model.final_layer.weight.detach().cpu(), original['final_layer.weight'])
        pose_path = variant_dir / 'final_pose.pth'
        torch.save(model.state_dict(), pose_path)
        # Evaluation only after the predefined final training step; never choose an intermediate AP.
        results[mode] = evaluate(model, mode, True)
        results[mode].update({
            'delta_AP': results[mode]['AP'] - results['original_flip_on']['AP'],
            'trainable_parameters': sum(p.numel() for p in parameters),
            'frozen_parameters_and_buffers_unchanged': True, 'training_steps': 600,
            'losses': losses, 'step_seconds': timings,
            'median_step_s': float(np.median(timings[5:])),
            'training_step_sum_s': float(sum(timings)),
            'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3,
            'pose_checkpoint': str(pose_path), 'pose_checkpoint_sha256': sha(pose_path),
        })
        save_results()
        print(f'COMPLETED {mode}: AP={results[mode]["AP"]:.8f} delta={results[mode]["delta_AP"]:+.8f}', flush=True)
        del model, optimizer, parameters
        torch.cuda.empty_cache()
    assert sha(ROOT / 'models/hrnet_fidip.pth') == MODEL_SHA
    results['original_flip_off']['delta_AP'] = results['original_flip_off']['AP'] - results['original_flip_on']['AP']
    save_results()
    summary = {key: {name: value for name, value in row.items() if name not in ('losses', 'step_seconds')}
               for key, row in results.items()}
    print(json.dumps(summary, indent=2), flush=True)
    print('PASS: three prespecified tests complete; originals unchanged; all outcomes saved', flush=True)


if __name__ == '__main__':
    main()
