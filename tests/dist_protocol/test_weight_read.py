# -*- coding: utf-8 -*-
"""weight_read.v1：植入读方向的单元被找出、随机单元不被找出；植入写入读出子空间的头组合分数最高、随机头不高；
方向移除使读出与该方向正交且不动正交部分；组合分数低秩算法与显式算法一致；与真值无关的预测不过判据（负对照）。"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.dist_protocol import weight_read as W  # noqa: E402


class WeightReadTests(unittest.TestCase):
    def test_read_and_remove(self):
        rng = np.random.default_rng(0)
        d, n = 256, 400
        v = W.unit(rng.normal(0, 1, d))
        gamma = rng.uniform(0.5, 2.0, d)
        Wg, Wu = rng.normal(0, 1, (2, n, d))
        Wg[:20] += 8 * v / gamma                               # 植入：前 20 个单元的读出向量含 v
        sc = W.read_scores(Wg * gamma, Wu * gamma, v)
        r = W.ratio_vs_random(sc[:20].mean(), [sc[rng.choice(np.arange(20, n), 20, replace=False)].mean() for _ in range(100)])
        self.assertGreater(r['ratio'], 3)
        self.assertLess(r['p'], 0.02)
        rn = W.ratio_vs_random(sc[20:40].mean(), [sc[rng.choice(np.arange(40, n), 20, replace=False)].mean() for _ in range(100)])
        self.assertLess(rn['ratio'], 1.5)                      # 负对照：未植入的单元
        W2 = W.remove_direction(Wg[:20], gamma, v)
        self.assertLess(np.abs((W2 * gamma) @ v).max(), 1e-9)
        ortho = lambda M: (M * gamma) - np.outer((M * gamma) @ v, v)
        self.assertTrue(np.allclose(ortho(W2), ortho(Wg[:20])))

    def test_composition_and_verdict(self):
        rng = np.random.default_rng(1)
        d, r, k, H = 128, 16, 8, 40
        R = rng.normal(0, 1, (k, d))
        Q, _ = np.linalg.qr(R.T)                               # 读出子空间基 [d, k]
        heads = []
        alpha = np.r_[np.linspace(4, 1, 10), np.zeros(H - 10)]   # 植入：前 10 个头按强度递减写入读出子空间
        for h in range(H):
            Wo, Wv = rng.normal(0, 1, (d, r)), rng.normal(0, 1, (r, d))
            Wo = alpha[h] * Q @ rng.normal(0, 1, (k, r)) + Wo
            Wo /= np.linalg.norm(Wo @ Wv)                     # 各头 OV 范数相同：范数基线无信息
            heads.append((Wo, Wv))
        cs = np.array([W.composition_lowrank(R, Wo, Wv) for Wo, Wv in heads])
        self.assertAlmostEqual(cs[7], W.composition(R, heads[7][0] @ heads[7][1]), places=6)
        self.assertTrue(set(np.argsort(-cs)[:5]) <= set(range(10)))
        self.assertLess(cs[10:].max(), cs[:5].min())
        truth = alpha + rng.normal(0, 0.3, H)
        base = np.array([np.linalg.norm(Wo @ Wv) for Wo, Wv in heads])
        crit = dict(W.CRITERIA, H2_top_k=5, n_perm=500)
        self.assertTrue(W.head_verdict(cs, base, truth, crit)['pass'])
        self.assertFalse(W.head_verdict(rng.normal(0, 1, H), base, truth, crit)['pass'])   # 负对照：无关预测
        self.assertAlmostEqual(W.spearman(np.array([1, 2, 2, 3.]), np.array([1, 2, 2, 3.])), 1.0)


    def test_v2_remove_write_and_controls(self):
        rng = np.random.default_rng(3)
        d, r = 64, 8
        V = rng.normal(0, 1, (2, d))
        Wo = rng.normal(0, 1, (d, r))
        Wo2 = W.remove_write(Wo, V)
        self.assertLess(np.abs(V @ Wo2).max(), 1e-9)                 # 写出与 span(V) 正交
        Q, _ = np.linalg.qr(V.T)
        self.assertTrue(np.allclose(Wo2 - Wo, -Q @ (Q.T @ Wo)))       # 只改该子空间
        sel = [(3, 1), (3, 2), (5, 0)]
        for s_ in W.matched_heads(sel, 4, 5, 0):
            self.assertEqual([l for l, _ in s_], [3, 3, 5])
            self.assertFalse(set(s_) & set(sel))
            self.assertEqual(len(set(s_)), 3)
        self.assertTrue(W.over_random(0.3, [0.01, 0.05], 3, 0.2)['pass'])
        self.assertFalse(W.over_random(0.3, [0.2, 0.15], 3, 0.2)['pass'])   # 负对照：随机集合同样有效
        a = np.r_[10.0, np.zeros(20), np.ones(4)]
        self.assertAlmostEqual(W.locality(a, 24, 4), 1.0)
        self.assertAlmostEqual(W.locality(np.r_[5.0, np.ones(24)], 24, 4), 4 / 24)


if __name__ == '__main__':
    unittest.main()
