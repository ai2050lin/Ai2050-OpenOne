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

    def test_v2_slot_conditioned(self):
        rng = np.random.default_rng(1)
        k, m = 3, 150
        # 植入：槽位条件化的专家乘积 Y[s,l,g] = a_g + b_g·s + c_g·l
        a, b, c = rng.normal(0, 1, (3, k, m))
        Y = np.zeros((2, 2, k, m))
        for s_ in (0, 1):
            for l_ in (0, 1):
                Y[s_, l_] = a + (2 * s_ - 1) * b + (2 * l_ - 1) * c
        Y = Y.reshape(4 * k, m) + rng.normal(0, 0.05, (4 * k, m))
        ss = F.factorial_ss(Y, k)
        self.assertAlmostEqual(sum(ss[e] for e in F.EFFECTS), ss['total'], places=6)
        self.assertGreater(ss['sg'] + ss['lg'], 10 * (ss['sl'] + ss['slg']))
        for g, w in F.within_slot_ss(Y, k).items():
            self.assertGreater(F.within_additive(w), 0.98)
        # 负对照：只有风格 × 逻辑交互
        d = rng.normal(0, 1, m)
        Z = np.zeros((2, 2, k, m))
        for s_ in (0, 1):
            for l_ in (0, 1):
                Z[s_, l_] = (2 * s_ - 1) * (2 * l_ - 1) * d
        Z = Z.reshape(4 * k, m) + rng.normal(0, 0.05, (4 * k, m))
        self.assertTrue(all(F.within_additive(w) < 0.05 for w in F.within_slot_ss(Z, k).values()))
        zs = F.factorial_ss(Z, k)
        self.assertLess(zs['sg'] + zs['lg'], 0.1 * (zs['sl'] + zs['slg']))
        # 随机：槽位内可加期望 2 / 3
        r = np.mean([F.within_additive(F.within_slot_ss(rng.normal(0, 1, (4 * k, m)), k)[0]) for _ in range(40)])
        self.assertLess(abs(r - 2 / 3), 0.04)


if __name__ == '__main__':
    unittest.main()
