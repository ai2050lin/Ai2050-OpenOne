# -*- coding: utf-8 -*-
"""assembly.v1：完全正确的预测 KL 为 0、恢复比例为 1；加噪声的预测比均匀好但不为 0；与真分布无关的预测不比均匀好（负对照）。"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.dist_protocol import assembly as S  # noqa: E402


class AssemblyTests(unittest.TestCase):
    def test_kl_and_recovery(self):
        rng = np.random.default_rng(0)
        T = rng.normal(0, 3, (300, 11))
        sc = S.family_scores('cat', T, {'exact': T, 'noisy': T + rng.normal(0, 1, T.shape), 'unrelated': rng.normal(0, 3, T.shape)})
        self.assertAlmostEqual(sc['exact'], 0.0, places=9)
        self.assertAlmostEqual(S.recovered(sc['exact'], sc['uniform']), 1.0)
        self.assertTrue(0 < sc['noisy'] < sc['uniform'])
        self.assertGreater(sc['unrelated'], sc['uniform'])
        m = rng.normal(0, 4, 500)
        sb = S.family_scores('bin', m, {'exact': m, 'half': m / 2})
        self.assertAlmostEqual(sb['exact'], 0.0, places=9)
        self.assertTrue(0 < sb['half'] < sb['uniform'])

    def test_v2_completion_and_transfer(self):
        rng = np.random.default_rng(1)
        V = 400
        b, c = rng.normal(0, 1, (2, V))
        def event():
            a = rng.normal(0, 2, V)
            Y = np.zeros((2, 2, V))
            for s_ in (0, 1):
                for l_ in (0, 1):
                    Y[s_, l_] = a + s_ * b + l_ * c + rng.normal(0, 0.05, V)
            return Y
        Y = event()
        cand = np.arange(60)
        k_add = S.kl_restricted(Y[1, 1], S.additive_completion(Y, 1, 1), cand)
        k_nb = S.kl_restricted(Y[1, 1], S.neighbor_mean(Y, 1, 1), cand)
        self.assertLess(k_add, 0.2 * k_nb)
        self.assertAlmostEqual(S.kl_restricted(Y[0, 0], Y[0, 0], cand), 0.0, places=12)
        Yd = np.stack([event() for _ in range(20)])
        ds, dl, dsl = S.slot_effects(Yd)
        self.assertGreater(np.corrcoef(ds, b)[0, 1], 0.99)
        self.assertLess(np.abs(dsl).mean(), 0.1)
        Yt = event()
        k_poe = S.kl_restricted(Yt[1, 1], Yt[0, 0] + ds + dl, cand)
        k_base = S.kl_restricted(Yt[1, 1], Yt[0, 0], cand)
        self.assertLess(k_poe, 0.1 * k_base)
        # 负对照：效应与事件无关（每个事件随机）时，迁移不比只用基础格好
        def event_rand():
            a = rng.normal(0, 2, V)
            bb, cc = rng.normal(0, 1, (2, V))
            Y = np.zeros((2, 2, V))
            for s_ in (0, 1):
                for l_ in (0, 1):
                    Y[s_, l_] = a + s_ * bb + l_ * cc
            return Y
        ds2, dl2, _ = S.slot_effects(np.stack([event_rand() for _ in range(20)]))
        Yt2 = event_rand()
        self.assertGreater(S.kl_restricted(Yt2[1, 1], Yt2[0, 0] + ds2 + dl2, cand), 0.8 * S.kl_restricted(Yt2[1, 1], Yt2[0, 0], cand))

    def test_v3_content_conditioned(self):
        rng = np.random.default_rng(2)
        a, b, c = rng.normal(0, 1, (3, 7))
        Z = np.stack([np.stack([a + s_ * b + l_ * c for l_ in (0, 1)]) for s_ in (0, 1)])
        ds, dl = S.own_effects(Z)
        self.assertTrue(np.allclose(ds, b) and np.allclose(dl, c))
        n, d, m = 16, 50, 80
        A = rng.normal(0, 1, (d, m))
        t0 = rng.normal(0, 1, m)
        X = rng.normal(0, 1, (n, d))
        Xt = rng.normal(0, 1, (16, d))
        def tgt(X_):
            Xc = X_ - X.mean(0)
            return t0 + (Xc / np.linalg.norm(Xc, axis=1, keepdims=True)) @ A
        T = tgt(X) + rng.normal(0, 0.05, (n, m))
        mdl = S.kr_fit(X, T)
        err_kr = ((S.kr_predict(mdl, Xt) - tgt(Xt)) ** 2).mean()
        err_mean = ((T.mean(0) - tgt(Xt)) ** 2).mean()
        self.assertLess(err_kr, 0.9 * err_mean)            # 植入：效应随内容线性变化 → 核岭回归优于平均效应
        # 负对照：效应与内容无关 → 留一选大 λ，预测接近平均效应，不比平均效应差多少
        Tn = t0 + rng.normal(0, 1, (n, m))
        mn = S.kr_fit(X, Tn)
        self.assertGreaterEqual(mn['lam'], 1.0)
        Tt = t0 + rng.normal(0, 1, (16, m))
        self.assertLess(((S.kr_predict(mn, Xt) - Tt) ** 2).mean(), 1.05 * ((Tn.mean(0) - Tt) ** 2).mean())


if __name__ == '__main__':
    unittest.main()
