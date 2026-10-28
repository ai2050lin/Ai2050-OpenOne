# -*- coding: utf-8 -*-
"""goodfire_probe.v1 统计部分：植入轴找回、负对照（混合物、随机）、劈半、肩部留出、steering 单调与最小对。"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'shared' / 'dist_protocol'))
import goodfire_probe as G  # noqa: E402

D = 64


def world(seed=0, n=6000):
    """残差流 = 大的公共分量 μ + 噪声；植入一条轴 a：g = h·a 的两端就是 ±a。"""
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=D) * 8
    axis = G._unit(rng.normal(size=D))
    s = rng.normal(size=n)
    H = mu + rng.normal(size=(n, D)) + 3.0 * s[:, None] * axis
    g = (H - mu) @ axis
    texts = [(i // 4,) for i in range(n)]            # 每 4 个位置一段文本
    return H, g, mu, axis, texts


def poles(H, g, texts, k=100, m=100):
    o = np.argsort(-g)
    P, SP = o[:k], o[k:k + m]
    q = np.argsort(g)
    N, SN = q[:k], q[k:k + m]
    return dict(HP=H[P], HN=H[N], SP=H[SP], SN=H[SN], textP=[texts[i] for i in P], textN=[texts[i] for i in N],
                gP=g[P], gN=g[N])


class ProbeStats(unittest.TestCase):
    def test_coherence_random_baseline(self):
        rng = np.random.default_rng(1)
        c = G.coherence(rng.normal(size=(100, 256)))
        self.assertAlmostEqual(c, 0.1, delta=0.03)
        self.assertGreater(G.coherence(np.ones((100, 8)) + rng.normal(size=(100, 8)) * 0.1), 0.95)

    def test_planted_axis_recovered(self):
        H, g, mu, axis, texts = world()
        p = poles(H, g, texts)
        r = G.analyze(**p, mu=H.mean(0), read_vec=axis, write_vec=axis, g_mean=g.mean(), g_std=g.std())
        self.assertGreater(G.cos(r['delta'], axis), 0.95)
        for k in ('S2_coh', 'S3_k', 'S3_effective_texts', 'S4_split_half', 'S5_duality', 'S5_shoulder', 'S8_read_write'):
            self.assertTrue(r['checks'][k]['pass'], (k, r['checks'][k]))
        self.assertEqual(r['reading'], 'axis')

    def test_raw_h_without_centering_would_fake_coherence(self):
        H, g, mu, axis, texts = world()
        rng = np.random.default_rng(3)
        idx = rng.choice(len(H), 100, replace=False)
        self.assertGreater(G.coherence(H[idx]), 0.9)                # 公共分量：原始 h 上随机样本也"集中"
        self.assertLess(G.coherence(H[idx] - H.mean(0)), 0.2)       # 减去 μ 后回到随机基线附近

    def test_mixture_negative_pole_scattered(self):
        """负对照：正极集中但负极是随机位置 → 对偶检验不通过，判为混合物。"""
        H, g, mu, axis, texts = world()
        p = poles(H, g, texts)
        rng = np.random.default_rng(4)
        p['HN'] = H[rng.choice(len(H), 100, replace=False)]
        r = G.analyze(**p, mu=H.mean(0), read_vec=axis, write_vec=axis, g_mean=g.mean(), g_std=g.std())
        self.assertTrue(r['checks']['S2_coh']['pass'])
        self.assertFalse(r['checks']['S5_duality']['pass'])
        self.assertEqual(r['reading'], 'mixture')

    def test_shuffled_labels_fail(self):
        """负对照：两极都是随机位置 → 集中度、劈半都不通过。"""
        H, g, mu, axis, texts = world()
        rng = np.random.default_rng(5)
        gr = rng.permutation(g)
        r = G.analyze(**poles(H, gr, texts), mu=H.mean(0), read_vec=axis, write_vec=axis, g_mean=g.mean(), g_std=g.std())
        self.assertFalse(r['checks']['S2_coh']['pass'])
        self.assertFalse(r['checks']['S4_split_half']['pass'])
        self.assertEqual(G.verdict(r['checks'], {'pass': True}, None)['status'], 'not_passed')

    def test_effective_texts_and_read_write_gap(self):
        H, g, mu, axis, texts = world()
        p = poles(H, g, texts)
        p['textP'] = [(0,)] * 100                                   # 全部来自一段文本
        other = G._unit(np.random.default_rng(6).normal(size=D))
        r = G.analyze(**p, mu=H.mean(0), read_vec=axis, write_vec=other, g_mean=g.mean(), g_std=g.std())
        self.assertFalse(r['checks']['S3_effective_texts']['pass'])
        self.assertFalse(r['checks']['S8_read_write']['pass'])      # 读的轴 ≠ 写的轴

    def test_steering_monotone(self):
        alphas = [-2, -1, 0, 1, 2]
        good = np.array([[-2, -1, 0, 1, 2], [-1, -0.5, 0, 0.5, 1.5]], float)
        self.assertTrue(G.steering_monotone(good, alphas)['pass'])
        one_sided = np.array([[0, 0, 0, 1, 2], [0, 0, 0, 1, 2]], float)   # 只有 +α 有反应
        r = G.steering_monotone(one_sided, alphas)
        self.assertFalse(r['pass'])
        self.assertEqual(r['both_sides_frac'], 0.0)
        self.assertFalse(G.steering_monotone(good[:, ::-1], alphas)['pass'])

    def test_pairs_and_verdict(self):
        self.assertTrue(G.pairs_check([2, 3, 1], [0, -1, -2])['pass'])
        self.assertFalse(G.pairs_check([2, -3, 1], [0, -1, -2])['pass'])
        self.assertIsNone(G.pairs_check([], [])['pass'])
        H, g, mu, axis, texts = world()
        r = G.analyze(**poles(H, g, texts), mu=H.mean(0), read_vec=axis, write_vec=axis, g_mean=g.mean(), g_std=g.std())
        self.assertEqual(G.verdict(r['checks'], {'pass': True}, None)['status'], 'auto_passed_needs_S9')
        self.assertEqual(G.verdict(r['checks'], {'pass': True}, {'pass': True})['status'], 'candidate')
        self.assertEqual(G.verdict(r['checks'], {'pass': True}, {'pass': False})['status'], 'not_passed')
        self.assertIn('S7_steering', G.verdict(r['checks'], {'pass': False}, None)['failed'])


if __name__ == '__main__':
    unittest.main()
