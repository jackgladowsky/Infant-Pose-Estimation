from pathlib import Path
import random
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from pilot_finetune import tree_digest
from run_small_screen import balanced_indices, configure_layers, set_augmentation_seed


class TinyPose(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4), torch.nn.Dropout(.5))
        self.stage4 = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4))
        self.final_layer = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.final_layer(self.stage4(self.trunk(x)))


class ScreenTests(unittest.TestCase):
    def test_layer_masks_and_frozen_state(self):
        for mode in ('head', 'stage4_head'):
            with self.subTest(mode=mode):
                model = TinyPose()
                frozen = configure_layers(model, mode)
                state = model.state_dict()
                before = tree_digest({name: state[name].clone() for name in frozen})
                self.assertFalse(model.trunk[1].training)
                self.assertFalse(model.trunk[2].training)
                self.assertEqual(model.stage4[1].training, mode == 'stage4_head')
                for name, parameter in model.named_parameters():
                    self.assertEqual(parameter.requires_grad,
                                     name.startswith('final_layer.') or (mode == 'stage4_head' and name.startswith('stage4.')))
                optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.01)
                initial_head = model.final_layer.weight.detach().clone()
                optimizer.zero_grad()
                model(torch.randn(8, 4)).square().mean().backward()
                optimizer.step()
                state = model.state_dict()
                self.assertEqual(before, tree_digest({name: state[name] for name in frozen}))
                self.assertFalse(torch.equal(initial_head, model.final_layer.weight))

    def test_batch_schedule_is_balanced_and_reproducible(self):
        domains = {False: [0, 1], True: [2, 3, 4]}
        for step in range(20):
            first = balanced_indices(domains, 7373, step, 8)
            random.random()
            self.assertEqual(first, balanced_indices(domains, 7373, step, 8))
            self.assertEqual(sum(index < 2 for index in first), 4)
        with self.assertRaises(ValueError):
            balanced_indices(domains, 7373, 1, 7)

    def test_augmentations_are_step_indexed_not_affected_by_previous_variant(self):
        set_augmentation_seed(7373, 17)
        expected = (random.random(), np.random.randn())
        random.seed(3)
        np.random.seed(91)
        set_augmentation_seed(7373, 17)
        self.assertEqual(expected, (random.random(), np.random.randn()))

    def test_invalid_trainable_mask_is_rejected(self):
        with self.assertRaises(ValueError):
            configure_layers(TinyPose(), 'all')


if __name__ == '__main__':
    unittest.main()
