"""Geometry/normalization helpers for two prespecified, inference-only diagnostics."""
import copy
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib'))
from utils.transforms import get_affine_transform

SCALES = (0.9, 1.0, 1.1)


def bn_buffer_names(model):
    names = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            for suffix in ('running_mean', 'running_var', 'num_batches_tracked'):
                if getattr(module, suffix, None) is not None:
                    names.append(f'{name}.{suffix}' if name else suffix)
    return names


def restore_bn_buffers(model, original, trained):
    """Keep ALL trained parameters (including BN affine weights), restore only BN buffers."""
    model.load_state_dict(trained, strict=True)  # reject missing/extra/incorrect-shaped input
    names = bn_buffer_names(model)
    for key in names:
        if original[key].shape != trained[key].shape:
            raise ValueError(f'BN shape mismatch: {key}')
    return {key: (original[key].clone() if key in names else value)
            for key, value in trained.items()}, names


def alignment_matrix(center, scale, factor, size):
    """Scaled heatmap -> reference heatmap using the repo's actual align-corners conventions.

    Do not use naive factor*coordinates: this repo uses (src_width-1)/(dst_width-1).
    """
    if factor <= 0 or not np.isfinite(factor):
        raise ValueError('Positive finite crop factor required')
    reference = get_affine_transform(np.asarray(center), np.asarray(scale), 0, size)
    source = get_affine_transform(np.asarray(center), np.asarray(scale) * factor, 0, size)
    reference = np.vstack([reference, [0, 0, 1]])
    source = np.vstack([source, [0, 0, 1]])
    return (reference @ np.linalg.inv(source))[:2]


def merge_heatmaps(heatmaps, centers, scales, factors):
    """Uniform mean over available views, after warping to reference heatmap coordinates.

    Coverage normalization avoids treating out-of-view pixels as zero evidence at small-crop edges.
    """
    if len(heatmaps) != len(factors) or not heatmaps:
        raise ValueError('One heatmap batch per crop factor required')
    shape = heatmaps[0].shape
    if any(heatmap.shape != shape for heatmap in heatmaps) or len(shape) != 4:
        raise ValueError('Consistent N,C,H,W heatmap batches required')
    n, channels, height, width = shape
    if len(centers) != n or len(scales) != n:
        raise ValueError('Metadata batch mismatch')
    total = np.zeros(shape, np.float32)
    coverage = np.zeros((n, 1, height, width), np.float32)
    for factor, maps in zip(factors, heatmaps):
        if factor == 1.0:
            total += maps
            coverage += 1
            continue
        for index in range(n):
            matrix = alignment_matrix(centers[index], scales[index], factor, [width, height])
            valid = cv2.warpAffine(np.ones((height, width), np.float32), matrix, (width, height),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            coverage[index, 0] += valid
            for joint in range(channels):
                total[index, joint] += cv2.warpAffine(maps[index, joint], matrix, (width, height),
                                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    if np.any(coverage <= 0):
        raise ValueError('TTA leaves unsupported reference pixels')
    return total / coverage


def scaled_copy(dataset, factor):
    if dataset.is_train:
        raise ValueError('Inference-only dataset required')
    result = copy.copy(dataset)
    result.db = copy.deepcopy(dataset.db)
    for record in result.db:
        record['scale'] = record['scale'] * factor
    return result


def pose_error(gt, prediction, sigmas):
    """Per-person OKS and area-normalized joint distance, NOT confidence-ranked AP."""
    truth = np.asarray(gt['keypoints'], dtype=np.float64).reshape(-1, 3)
    estimate = np.asarray(prediction['keypoints'], dtype=np.float64).reshape(-1, 3)
    visible = truth[:, 2] > 0
    area = float(gt['area'])
    if area <= 0 or not visible.any():
        raise ValueError('Positive-area labeled GT required')
    squared = ((estimate[:, :2] - truth[:, :2]) ** 2).sum(axis=1)
    exponent = squared / ((np.asarray(sigmas) * 2) ** 2) / (area + np.spacing(1)) / 2
    oks = np.exp(-exponent[visible]).mean()
    distances = np.sqrt(squared / area)
    return float(oks), distances, visible
