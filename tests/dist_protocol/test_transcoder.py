# -*- coding: utf-8 -*-
"""transcoder.v1：植入稀疏字典的找回（正例）、输入输出无关时不能「解释」（负对照）、指标公式与神经元基线。"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.dist_protocol import transcoder as T  # noqa: E402

try:
    import torch
except ImportError:                                  # pragma: no cover
    torch = None


def planted(n_feat=64, d_in=32, d_out=24, active=3, seed=0, unrelated=False):
    g = torch.Generator().manual_seed(seed)
    D_in = torch.nn.functional.normalize(torch.randn(n_feat, d_in, generator=g), dim=1)
    D_out = torch.nn.functional.normalize(torch.randn(n_feat, d_out, generator=g), dim=1)

    def codes(n):
        c = torch.zeros(n, n_feat)
        idx = torch.rand(n, n_feat, generator=g).topk(active, dim=1).indices
        c.scatter_(1, idx, torch.rand(n, active, generator=g) * 2 + 0.5)
        return c

    def sample(n):
        c = codes(n)
        y = (codes(n) if unrelated else c) @ D_out
        return c @ D_in, y
    return sample, D_out


@unittest.skipIf(torch is None, 'torch 未安装')
class TranscoderTests(unittest.TestCase):
    def test_planted_dictionary_recovered(self):
        sample, D_out = planted()
        t, curve = T.train(sample, d_in=32, d_out=24, m=128, k=6, steps=3000, batch=512, lr=3e-3, seed=1, log_every=0)
        x, y = sample(4000)
        with torch.no_grad():
            f = T.fvu(y.numpy(), t(x).numpy())
        self.assertLess(f, 0.1, f)
        cos = (D_out @ t.W_dec.T).max(dim=1).values      # 每个植入特征与最接近的学到的写出方向
        self.assertGreaterEqual(float((cos > 0.9).float().mean()), 0.8, cos)

    def test_unrelated_output_not_explained(self):
        sample, _ = planted(unrelated=True, seed=3)
        t, _ = T.train(sample, d_in=32, d_out=24, m=128, k=6, steps=1500, batch=512, lr=3e-3, seed=1, log_every=0)
        x, y = sample(4000)
        with torch.no_grad():
            self.assertGreater(T.fvu(y.numpy(), t(x).numpy()), 0.9)

    def test_topk_sparsity_and_state_roundtrip(self):
        sample, _ = planted()
        t, _ = T.train(sample, d_in=32, d_out=24, m=64, k=4, steps=50, batch=256, seed=0, log_every=0)
        x, _ = sample(100)
        a = t.encode(x)
        self.assertLessEqual(int((a != 0).sum(1).max()), 4)
        t2 = T.TopKTranscoder.from_state(t.state())
        self.assertTrue(torch.allclose(t(x), t2(x), atol=1e-6))

    def test_neuron_baseline(self):
        g = torch.Generator().manual_seed(0)
        A, W = torch.randn(20, 50, generator=g), torch.randn(50, 8, generator=g)
        self.assertTrue(torch.allclose(T.neuron_topk_recon(A, W, 50), A @ W, atol=1e-5))
        r1 = T.neuron_topk_recon(A, W, 1)
        i = (A.abs() * W.norm(dim=1)).argmax(1)
        self.assertTrue(torch.allclose(r1, A.gather(1, i[:, None]) * W[i], atol=1e-5))

    def test_metrics(self):
        self.assertAlmostEqual(T.loss_recovered(2.0, 2.0, 5.0), 1.0)
        self.assertAlmostEqual(T.loss_recovered(2.0, 5.0, 5.0), 0.0)
        y = np.random.default_rng(0).normal(size=(100, 4))
        self.assertAlmostEqual(T.fvu(y, y), 0.0)
        self.assertAlmostEqual(T.fvu(y, np.tile(y.mean(0), (100, 1))), 1.0)
        self.assertAlmostEqual(T.token_concentration([5, 5, 5, 1]), 0.75)


if __name__ == '__main__':
    unittest.main()
