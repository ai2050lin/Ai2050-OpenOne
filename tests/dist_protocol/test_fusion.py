# -*- coding: utf-8 -*-
"""fusion.v1：植入纯主效应 → 可加份额约 1；植入纯交互 → 约 0；随机矩阵 → 约 3/7（负对照）；份额与平方和自洽。"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.dist_protocol import fusion as F  # noqa: E402


class FusionTests(unittest.TestCase):
    def test_decompose(self):
        rng = np.random.default_rng(0)
        K = F.contrasts()
        self.assertTrue(np.allclose(K @ K.T, 8 * np.eye(7)))
        m = 200
        a, b, c = rng.normal(0, 1, (3, m))
        Y = np.outer(K[0], a) + np.outer(K[1], b) + np.outer(K[2], c) + rng.normal(0, 0.05, (8, m))
        d = F.decompose(Y)
        self.assertGreater(d['additive_share'], 0.98)
        self.assertAlmostEqual(sum(d[e] for e in F.EFFECTS), d['total'], places=6)
        Yi = np.outer(K[4], a) + rng.normal(0, 0.05, (8, m))          # 只有风格 × 语法交互
        di = F.decompose(Yi)
        self.assertLess(di['additive_share'], 0.02)
        self.assertGreater(di['sg'] / di['total'], 0.95)
        shares = [F.decompose(rng.normal(0, 1, (8, m)))['additive_share'] for _ in range(50)]
        self.assertLess(abs(np.mean(shares) - 3 / 7), 0.03)
        p = F.pooled([d, di])
        self.assertAlmostEqual(p['additive_share'] + p['interaction_share'], 1.0)
        lp = rng.normal(0, 1, (8, 30)) + rng.normal(0, 5, (8, 1))         # 各格归一化常数不同
        self.assertTrue(np.allclose(F.center_cells(lp).mean(1), 0))


if __name__ == '__main__':
    unittest.main()
