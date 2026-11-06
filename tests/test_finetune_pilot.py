import copy
from pathlib import Path
import random
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from pilot_finetune import load_state, restore_rng, rng_state, save_state, tree_digest


class ResumeTests(unittest.TestCase):
    def test_rng_round_trip(self):
        state = rng_state()
        expected = (random.random(), np.random.rand(), torch.rand(4))
        restore_rng(state)
        actual = (random.random(), np.random.rand(), torch.rand(4))
        self.assertEqual(expected[:2], actual[:2])
        self.assertTrue(torch.equal(expected[2], actual[2]))

    def test_checkpoint_matches_uninterrupted_adam_and_rng(self):
        random.seed(91)
        np.random.seed(91)
        torch.manual_seed(91)
        model = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Dropout(.2), torch.nn.Linear(5, 1))
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        protocol = {'seed': 91, 'mode': 'unit'}
        def step():
            x = torch.rand(4, 3) * random.random() + np.random.rand()
            opt.zero_grad()
            model(x).square().mean().backward()
            opt.step()
        step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.pth'
            save_state(path, model, opt, 1, protocol)
            step()
            expected = tree_digest({'model': model.state_dict(), 'optimizer': opt.state_dict(), 'rng': rng_state()})
            self.assertEqual(load_state(path, model, opt, protocol), 1)
            step()
            self.assertEqual(expected, tree_digest({'model': model.state_dict(), 'optimizer': opt.state_dict(), 'rng': rng_state()}))
            with self.assertRaises(RuntimeError):
                save_state(path, model, opt, 2, protocol)
            before = tree_digest(model.state_dict())
            with self.assertRaises(RuntimeError):
                load_state(path, model, opt, {'wrong': True})
            self.assertEqual(before, tree_digest(model.state_dict()))

    def test_fingerprint_tracks_tensor_values_not_dict_insertion_order(self):
        a = {'weights': torch.tensor([1., 2.]), 'step': 2}
        b = {'step': 2, 'weights': a['weights'].clone()}
        self.assertEqual(tree_digest(a), tree_digest(b))
        b['weights'][0] += 1
        self.assertNotEqual(tree_digest(a), tree_digest(b))


if __name__ == '__main__':
    unittest.main()
