"""Two inference-only hypotheses plus matched controls. No training or parameter sweep."""
import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from bounded_diagnostics import SCALES, merge_heatmaps, restore_bn_buffers, scaled_copy
from pilot_finetune import tree_digest
from run_small_screen import configure_runtime, sha

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', required=True, type=Path)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('GPU compute allocation required')
    gpu = torch.cuda.get_device_name()
    if not any(kind in gpu.upper() for kind in ('V100', 'A100')):
        raise RuntimeError(f'Use economical approved inference hardware, got {gpu}')
    analysis = json.loads(args.analysis.read_text())
    assert analysis['prespecified_GPU_tests']['TTA']['crop_factors'] == list(SCALES)
    for path, expected in analysis['input_hashes'].items():
        assert sha(ROOT / path) == expected, f'Pinned input changed: {path}'
    out = ROOT / 'output' / f'bounded-diagnostics-{os.environ["SLURM_JOB_ID"]}'
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(ROOT / 'lib'))
    from config import cfg
    from core.inference import get_final_preds
    from dataset.syrip import SYRIPDataset
    from models.adaptive_pose_hrnet import PoseHighResolutionNet
    from torchvision import transforms
    from utils.transforms import flip_back
    import cv2
    cv2.setNumThreads(1)
    configure_runtime(7373)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    config = cfg.clone()
    config.defrost()
    config.merge_from_file(str(ROOT / 'experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml'))
    config.DATASET.ROOT = str(ROOT / 'data/syrip')
    config.TEST.USE_GT_BBOX = True
    config.TEST.FLIP_TEST = True
    config.DEBUG.DEBUG = False
    config.CUDNN.BENCHMARK = False
    config.CUDNN.DETERMINISTIC = True
    config.freeze()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        mean=[0.485, .456, .406], std=[.229, .224, .225])])
    data = SYRIPDataset(config, config.DATASET.ROOT, 'validate_infant', False, transform)
    assert len(data) == 100
    original_path = ROOT / 'models/hrnet_fidip.pth'
    trained_path = ROOT / analysis.get('trained_checkpoint', 'output/small-screen-11553889/stage4_head/final_pose.pth')
    original = torch.load(original_path, map_location='cpu', weights_only=True)
    trained = torch.load(trained_path, map_location='cpu', weights_only=True)
    model = PoseHighResolutionNet(config)
    restored, buffer_names = restore_bn_buffers(model, original, trained)
    non_buffer = [name for name in trained if name not in buffer_names]
    assert tree_digest({key: trained[key] for key in non_buffer}) == tree_digest({key: restored[key] for key in non_buffer})
    model.cuda().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    torch.cuda.reset_peak_memory_stats()
    protocol = {
        'job_id': os.environ['SLURM_JOB_ID'], 'gpu': gpu, 'torch': str(torch.__version__),
        'analysis_sha256': sha(args.analysis), 'crop_factors': list(SCALES), 'flip_test': True,
        'weights': {str(path.relative_to(ROOT)): sha(path) for path in (original_path, trained_path)},
        'BN_restoration': 'Only original running_mean/running_var/num_batches_tracked; trained parameters and BN affine weights retained',
        'BN_buffer_keys_restored': buffer_names,
        'TTA': 'Original weights; crop extents0.9/1/1.1 at fixed288x384; repo-affine warps to reference heatmap; uniform coverage-normalized mean; then existing decoder',
        'precision': 'FP32/TF32 off/deterministic; one device, batch16, no training',
        'stop_rule': 'two variants only; no adaptive choice of scales or subsequent fine-tuning',
        'interpretation': 'Exploratory Test100 diagnostics, not independent confirmation; confidence-ranked AP differs from per-person OKS',
    }
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    results = {}
    def save_results():
        (out / 'results.json').write_text(json.dumps({'protocol': protocol, 'results': results}, indent=2) + '\n')

    def infer_view(inputs):
        # Exactly the repo's flip-back and GPU heatmap averaging convention.
        inputs_gpu = inputs.cuda()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        _, base = model(inputs_gpu)
        end.record()
        end.synchronize()
        elapsed = start.elapsed_time(end) / 1000
        flipped = torch.from_numpy(np.flip(inputs.numpy(), 3).copy()).cuda()
        start.record()
        _, mirror = model(flipped)
        end.record()
        end.synchronize()
        elapsed += start.elapsed_time(end) / 1000
        corrected = torch.from_numpy(flip_back(mirror.cpu().numpy(), data.flip_pairs).copy()).cuda()
        mean = (base + corrected) * .5
        return mean.cpu().numpy(), elapsed

    def evaluate(name, state, factors):
        model.load_state_dict(state, strict=True)
        model.eval()
        target = out / name
        target.mkdir()
        with torch.no_grad():
            for count in (16, 4):
                model(torch.zeros(count, 3, 384, 288, device='cuda'))
        loaders = [torch.utils.data.DataLoader(scaled_copy(data, factor), batch_size=16,
                   shuffle=False, num_workers=0, pin_memory=True) for factor in factors]
        reference_index = factors.index(1.0)
        predictions = np.zeros((100, 17, 3), np.float32)
        boxes = np.zeros((100, 6))
        paths, gpu_seconds, seen = [], 0.0, 0
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            for batch_index, batches in enumerate(zip(*loaders)):
                meta = batches[reference_index][3]
                centers, scales = meta['center'].numpy(), meta['scale'].numpy()
                maps = []
                for inputs, _, _, current_meta in batches:
                    assert current_meta['image'] == meta['image']
                    heatmaps, elapsed = infer_view(inputs)
                    maps.append(heatmaps)
                    gpu_seconds += elapsed
                combined = merge_heatmaps(maps, centers, scales, factors)
                coords, confidence = get_final_preds(config, combined.copy(), centers, scales)
                count = len(coords)
                predictions[seen:seen+count, :, :2] = coords
                predictions[seen:seen+count, :, 2:] = confidence
                boxes[seen:seen+count, :2] = centers
                boxes[seen:seen+count, 2:4] = scales
                boxes[seen:seen+count, 4] = np.prod(scales * 200, axis=1)
                boxes[seen:seen+count, 5] = meta['score'].numpy()
                paths.extend(meta['image'])
                seen += count
                print(f'{name}: batch{batch_index+1}/7 images={seen}', flush=True)
        assert seen == 100 and np.isfinite(predictions).all()
        metrics, ap = data.evaluate(config, predictions, str(target), boxes, paths, [], [])
        row = {
            'AP': float(ap * 100), 'metrics_fraction': {str(key): float(value) for key, value in metrics.items()},
            'crop_factors': factors, 'evaluation_wall_s': time.perf_counter() - started,
            'GPU_CNN_forward_s': gpu_seconds, 'CNN_forward_image_views': 100 * 2 * len(factors),
            'timing_scope': 'Forward timer excludes transfer/flip-back/fusion/decoding/I/O; wall timing not a randomized app benchmark',
            'predictions': str(target / 'results/keypoints_validate_infant_results_0.json'),
        }
        print(f'COMPLETED {name}: AP={row["AP"]:.8f}', flush=True)
        return row

    results['original_control'] = evaluate('original_control', original, [1.0])
    assert abs(results['original_control']['AP'] - 93.6071706887983) < .001, 'Original control drift; stop'
    results['trained_control'] = evaluate('trained_control', trained, [1.0])
    assert abs(results['trained_control']['AP'] - analysis.get('expected_trained_AP', 93.198637527681)) < .001, 'Trained control drift; stop'
    save_results()
    results['BN_buffers_restored'] = evaluate('BN_buffers_restored', restored, [1.0])
    results['BN_buffers_restored']['delta_AP_vs_trained'] = results['BN_buffers_restored']['AP'] - results['trained_control']['AP']
    save_results()
    results['three_crop_TTA'] = evaluate('three_crop_TTA', original, list(SCALES))
    for row in results.values():
        row['delta_AP_vs_original'] = row['AP'] - results['original_control']['AP']
    save_results()
    # Preserve the fixed diagnostic candidate regardless of whether its score improves.
    model.load_state_dict(restored, strict=True)
    restored_path = out / 'BN_buffers_restored.pth'
    torch.save(model.state_dict(), restored_path)
    results['BN_buffers_restored']['checkpoint'] = str(restored_path)
    results['BN_buffers_restored']['checkpoint_sha256'] = sha(restored_path)
    for path in (original_path, trained_path):
        assert sha(path) == protocol['weights'][str(path.relative_to(ROOT))], 'Source checkpoint changed'
    save_results()
    print(json.dumps(results, indent=2), flush=True)
    print('PASS: two diagnostics complete, matched controls verified, originals unchanged; STOP, no further training', flush=True)


if __name__ == '__main__':
    main()
