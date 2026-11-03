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
        g0 = gamma.copy()
        g0[:5] = 0.0                                                   # γ 为 0 的维度：权重不动，读出仍与方向正交
        W3 = W.remove_direction(Wg[:20], g0, v)
        self.assertTrue(np.all(np.isfinite(W3)))
        self.assertTrue(np.allclose(W3[:, :5], Wg[:20, :5]))
        self.assertLess(np.abs((W3 * g0) @ v).max(), 1e-9)

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


    def test_v3_syntax_and_mass_ratio(self):
        for t in ['):\n', ':\n', ' return', ' if', ' +=', ' ==', ' =']:
            self.assertTrue(W.is_syntax(t), t)
        for t in [' acc', ' x', 'append', ' the', '\n', '1', ' returned']:
            self.assertFalse(W.is_syntax(t), t)
        cls = np.r_[False, np.array([True, False] * 10)]
        a = np.r_[50.0, np.where(cls[1:], 3.0, 1.0)]                     # 植入：类别上 3 倍注意力
        self.assertAlmostEqual(W.mass_ratio(a, 20, cls), 1.5)              # (30/40) / 0.5
        self.assertAlmostEqual(W.mass_ratio(np.r_[50.0, np.ones(20)], 20, cls), 1.0)   # 负对照：均匀注意力
        self.assertTrue(np.isnan(W.mass_ratio(a, 20, np.zeros(21, bool))))


    def test_v4_rope_and_alignment(self):
        rng = np.random.default_rng(4)
        hd = 16
        inv = 1.0 / (10000 ** (np.arange(0, hd, 2) / hd))
        q, k = rng.normal(0, 1, (2, hd))
        rope = lambda x, p_: x * np.cos(p_ * np.r_[inv, inv]) + W.rotate_half(x) * np.sin(p_ * np.r_[inv, inv])
        self.assertTrue(np.allclose(W.rope_matrix(inv, 3.0) @ q, rope(q, 3.0)))
        B = W.rope_avg(inv, range(1, 9))
        self.assertAlmostEqual(q @ B @ k, np.mean([rope(q, d) @ rope(k, 0.0) for d in range(1, 9)]), places=9)
        self.assertAlmostEqual(q @ B @ k, np.mean([rope(q, d + 5) @ rope(k, 5.0) for d in range(1, 9)]), places=9)  # 只依赖相对位置
        d = 32
        v = W.unit(rng.normal(0, 1, d))
        Wq = rng.normal(0, 1, (hd, d))
        Wk = rng.normal(0, 1, (hd, d))
        Wk_planted = Wk - np.outer(Wk @ v, v) + np.outer(np.linalg.solve(B, Wq @ v), v)   # 植入：v 写入的 key 与查询对齐
        ones = np.ones(hd)
        a_pl = W.qk_alignment(Wq, ones, Wk_planted, ones, np.ones(d), v, B)
        self.assertGreater(a_pl, 0.99)
        null = [W.qk_alignment(Wq, ones, Wk, ones, np.ones(d), W.unit(rng.normal(0, 1, d)), B) for _ in range(200)]
        self.assertGreater(W.z_vs(a_pl, null), 2)
        self.assertLess(abs(W.z_vs(W.qk_alignment(Wq, ones, Wk, ones, np.ones(d), v, B), null)), 3)   # 负对照：未植入
        self.assertTrue(W.monotone([1, 1, 2, 3]) and not W.monotone([1, 3, 2]) and W.monotone([3, 2, 2], False))


    def test_v5_neutral_and_persistence(self):
        a = W.neutral_segment('x', 6)
        self.assertEqual(a, W.neutral_segment('x', 6))
        self.assertEqual(len(a.split(', ')), 6)
        self.assertTrue(all(c.isdigit() for c in a.split(', ')))
        self.assertNotEqual(a, W.neutral_segment('y', 6))
        gaps = {4: 0.8, 16: 0.7, 64: 0.6}                                   # 植入：自持（几乎不衰减）
        self.assertGreaterEqual(W.persistence(gaps, 4, 64), 0.5)
        decay = {L: 0.8 * 0.5 ** ((L - 4) / 8) for L in (4, 16, 64)}        # 负对照：半衰期 8 个 token 的单稳态衰减
        self.assertLess(W.persistence(decay, 4, 64), 0.5)
        self.assertTrue(np.isnan(W.persistence({4: 0.0, 64: 0.1}, 4, 64)))


    def test_v6_local_integration(self):
        lo = W.unigram_logodds({1: 9, 2: 1}, {2: 9, 3: 1})
        self.assertGreater(lo[1], 0)
        self.assertLess(lo[2], 0)
        rng = np.random.default_rng(6)
        e = np.repeat(rng.choice([-2.0, 2.0], 12), 15) + rng.normal(0, 0.5, 180)
        self.assertTrue(np.allclose(W.window_feature(e, 1), e))
        self.assertAlmostEqual(W.window_feature(e, 4)[10], e[7:11].mean())
        f0 = W.matched_feature(e, 8.0, 0.0, np.zeros(len(e)))
        w = np.exp(-np.arange(11)[::-1] / 8.0)
        self.assertAlmostEqual(f0[10], (w * e[:11]).sum() / w.sum())
        y = 0.3 + 0.5 * f0 + rng.normal(0, 0.05, len(e))                 # 植入：τ = 8 的局部积分
        fits = {tau: W.r2(y, np.stack([np.ones_like(e), W.matched_feature(e, tau, 0.0, np.zeros(len(e)))], 1)
                          @ W.fit_linear(W.matched_feature(e, tau, 0.0, np.zeros(len(e))), y)) for tau in (1, 8, 64)}
        self.assertEqual(max(fits, key=fits.get), 8)
        self.assertGreater(fits[8] - fits[1], 0.1)
        yo = 0.3 + 0.5 * e + rng.normal(0, 0.05, len(e))                  # 负对照：只依赖本 token
        r_own = W.r2(yo, np.stack([np.ones_like(e), e], 1) @ W.fit_linear(e, yo))
        r_int = W.r2(yo, np.stack([np.ones_like(e), f0], 1) @ W.fit_linear(f0, yo))
        self.assertGreater(r_own, r_int)
        m_hat = np.sign(f0)
        fm = W.matched_feature(e, 8.0, 1.0, m_hat)
        self.assertGreater(np.abs(fm).mean(), np.abs(f0).mean())         # 模式匹配加权放大与当前模式同号的证据


    def test_gc2_remove_subspace(self):
        rng = np.random.default_rng(7)
        d = 40
        gamma = rng.uniform(0.5, 2.0, d)
        gamma[:3] = 0.0
        V = rng.normal(0, 1, (3, d))
        Wr = rng.normal(0, 1, (6, d))
        W2 = W.remove_subspace(Wr, gamma, V)
        Vok = np.where(gamma[None, :] != 0, V, 0.0)
        self.assertLess(np.abs((W2 * gamma) @ Vok.T).max(), 1e-9)       # 读出与支撑上的子空间正交
        self.assertTrue(np.allclose(W2[:, :3], Wr[:, :3]))               # γ = 0 维度不动
        W1 = W.remove_subspace(Wr, gamma, V[:1])
        self.assertTrue(np.allclose(W1, W.remove_direction(Wr, gamma, V[0])))   # 秩 1 与单方向移除一致


if __name__ == '__main__':
    unittest.main()
