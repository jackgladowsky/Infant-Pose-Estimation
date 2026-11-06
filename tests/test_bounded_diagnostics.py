from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from bounded_diagnostics import alignment_matrix, merge_heatmaps, pose_error, restore_bn_buffers, scaled_copy
from utils.transforms import get_affine_transform


class DiagnosticTests(unittest.TestCase):
    def test_restore_changes_only_running_buffers_not_affine_weights(self):
        model = torch.nn.Sequential(torch.nn.BatchNorm2d(2), torch.nn.Conv2d(2, 1, 1))
        original = {k: v.clone() for k, v in model.state_dict().items()}
        trained = {k: v.clone() + 1 for k, v in original.items()}
        restored, names = restore_bn_buffers(model, original, trained)
        self.assertEqual(set(names), {'0.running_mean', '0.running_var', '0.num_batches_tracked'})
        for key in trained:
            self.assertTrue(torch.equal(restored[key], original[key] if key in names else trained[key]))
        self.assertTrue(torch.equal(restored['0.weight'], trained['0.weight']))

    def test_bn_shape_mismatch_is_rejected(self):
        model = torch.nn.BatchNorm1d(2)
        trained = {k: v.clone() for k, v in model.state_dict().items()}
        original = {**trained, 'running_mean': torch.zeros(3)}
        with self.assertRaises(ValueError):
            restore_bn_buffers(model, original, trained)

    def test_alignment_uses_actual_repo_affine_convention(self):
        for factor in (.9, 1., 1.1):
            center, scale, size = np.array([317., 419.]), np.array([2.3, 3.1]), [72, 96]
            source = get_affine_transform(center, scale * factor, 0, size)
            reference = get_affine_transform(center, scale, 0, size)
            mapping = alignment_matrix(center, scale, factor, size)
            for point in ([250., 300., 1.], [317., 419., 1.], [400., 580., 1.]):
                observed = mapping @ np.r_[source @ point, 1.]
                np.testing.assert_allclose(observed, reference @ point, atol=1e-6)

    def test_single_reference_view_is_unchanged(self):
        maps = np.random.default_rng(9).normal(size=(2, 3, 96, 72)).astype(np.float32)
        result = merge_heatmaps([maps], np.array([[100, 150], [200, 300]]), np.ones((2, 2)), [1.])
        np.testing.assert_array_equal(result, maps)

    def test_edge_coverage_does_not_attenuate_constant_heatmaps(self):
        maps = np.ones((1, 2, 96, 72), np.float32)
        result = merge_heatmaps([maps, maps, maps], np.array([[100., 150.]]),
                               np.array([[2., 2.7]]), [.9, 1., 1.1])
        np.testing.assert_allclose(result, 1., atol=1e-6)

    def test_gaussian_peak_fuses_in_reference_coordinates(self):
        center, scale = np.array([400., 500.]), np.array([2.4, 3.2])
        reference_point = np.array([30.3, 40.6])
        inverse = get_affine_transform(center, scale, 0, [72, 96], inv=1)
        world = inverse @ np.r_[reference_point, 1.]
        y, x = np.mgrid[:96, :72]
        maps = []
        for factor in (.9, 1., 1.1):
            source = get_affine_transform(center, scale * factor, 0, [72, 96]) @ np.r_[world, 1.]
            maps.append(np.exp(-((x-source[0])**2 + (y-source[1])**2)/8).astype(np.float32)[None, None])
        result = merge_heatmaps(maps, center[None], scale[None], [.9, 1., 1.1])
        row, col = np.unravel_index(result[0, 0].argmax(), (96, 72))
        np.testing.assert_allclose([col, row], reference_point, atol=1.)

    def test_per_person_oks_ignores_unlabeled_joints(self):
        gt = {'area': 100, 'keypoints': [10, 20, 2, 30, 40, 0]}
        pred = {'keypoints': [10, 20, .7, 900, 900, .9]}
        oks, distance, visible = pose_error(gt, pred, [.1, .1])
        self.assertEqual(oks, 1.)
        self.assertEqual(distance[0], 0)
        self.assertFalse(visible[1])

    def test_scaled_dataset_does_not_modify_original(self):
        original = SimpleNamespace(is_train=False, db=[{'scale': np.array([2., 3.]), 'center': np.array([5., 6.])}])
        result = scaled_copy(original, .9)
        np.testing.assert_array_equal(original.db[0]['scale'], [2., 3.])
        np.testing.assert_allclose(result.db[0]['scale'], [1.8, 2.7])
        result.db[0]['center'][0] = 99
        self.assertEqual(original.db[0]['center'][0], 5)
        with self.assertRaises(ValueError):
            scaled_copy(SimpleNamespace(is_train=True, db=[]), .9)

    def test_invalid_or_unsupported_crop_fusion_is_rejected(self):
        with self.assertRaises(ValueError):
            alignment_matrix(np.ones(2), np.ones(2), 0, [72, 96])
        with self.assertRaises(ValueError):
            merge_heatmaps([np.ones((1, 1, 96, 72), np.float32)], np.array([[100., 200.]]),
                           np.ones((1, 2)), [.5])


if __name__ == '__main__':
    unittest.main()
