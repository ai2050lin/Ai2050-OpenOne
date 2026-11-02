# -*- coding: utf-8 -*-
"""mode_field.v1：共用 token 位置植入的风格方向被读出（两两准确率高）；风格只体现在 token 身份时，共用位置读不出（负对照）；
token 配平方向找回植入的上下文方向且不被 token 身份带偏；AUC 与半衰期的正反例；专属词表。"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.dist_protocol import mode_field as F  # noqa: E402


def _make(rng, n=4000, d=40, ctx=1.0):
    """token 0–9 共用于全部风格；token 10+s 只出现于风格 s。状态 = token 嵌入 + ctx × 风格方向 + 噪声。"""
    E = rng.normal(0, 3, (14, d))
    D = rng.normal(0, 1, (4, d))
    y = rng.integers(0, 4, n)
    tok = np.where(rng.random(n) < 0.5, rng.integers(0, 10, n), 10 + y)
    X = E[tok] + ctx * D[y] + rng.normal(0, 0.5, (n, d))
    return X, y, tok, D, E


class ModeFieldTests(unittest.TestCase):
    def test_readout_planted_and_identity_only(self):
        rng = np.random.default_rng(0)
        X, y, tok, _, _ = _make(rng)
        m = F.ridge_multi(X[:3000], y[:3000], 4, 1e-2)
        acc = F.pairwise_acc(F.scores(m, X[3000:]), y[3000:], tok[3000:])
        self.assertTrue(all(v > 0.9 for v in acc.values()))
        X0, y0, tok0, _, _ = _make(rng, ctx=0.0)            # 风格只在专属 token 上
        m0 = F.ridge_multi(X0[:3000], y0[:3000], 4, 1e-2)
        acc0 = F.pairwise_acc(F.scores(m0, X0[3000:]), y0[3000:], tok0[3000:])
        self.assertTrue(all(abs(v - 0.5) < 0.08 for v in acc0.values()))

    def test_token_matched_direction(self):
        rng = np.random.default_rng(1)
        X, y, tok, D, E = _make(rng)
        v = F.token_matched_direction(X, y, tok, 3)
        target = D[3] - D[[0, 1, 2]].mean(0)
        self.assertGreater(v @ target / np.linalg.norm(v) / np.linalg.norm(target), 0.95)
        naive = X[y == 3].mean(0) - X[y != 3].mean(0)          # 朴素均值差被专属 token 带偏
        self.assertLess(naive @ target / np.linalg.norm(naive) / np.linalg.norm(target), 0.9)

    def test_auc_half_life_vocab(self):
        rng = np.random.default_rng(2)
        self.assertGreater(F.auc(rng.normal(2, 1, 200), rng.normal(0, 1, 300)), 0.85)
        self.assertLess(abs(F.auc(rng.normal(0, 1, 200), rng.normal(0, 1, 300)) - 0.5), 0.08)
        self.assertAlmostEqual(F.half_life([4, 2.5, 1.5, 1.0]), 1.5, places=6)
        self.assertEqual(F.half_life([4, 4, 4]), 3.0)
        ex = F.exclusive_vocab({'a': [1, 1, 1, 2, 5, 5, 5], 'b': [2, 2, 2, 3, 3, 3]}, 3)
        self.assertEqual(ex, {'a': [1, 5], 'b': [3]})

    def test_v2_window_band_switch_retain(self):
        import torch
        T = 8
        m = torch.triu(torch.full((1, 1, T, T), torch.finfo(torch.float32).min), 1)
        w = F.window_mask(m, 3)
        vis = (w[0, 0] == 0)
        self.assertTrue(vis[7, 0] and vis[7, 5] and vis[7, 7] and not vis[7, 4] and not vis[7, 1])
        self.assertTrue(bool(vis[2, :3].all()))
        self.assertFalse(bool(vis[3, 4]))                      # 仍保持因果
        self.assertTrue(F.band_localize(0.4, [0.3, 0.05, 0.02])['pass'])
        self.assertFalse(F.band_localize(0.4, [0.1, 0.1, 0.1, 0.1])['pass'])
        rng = np.random.default_rng(3)
        op = np.array([0, 1.5, 3.0])[None] + rng.normal(0, 0.3, (40, 3))
        rd = rng.normal(0, 0.3, (2, 40, 3))
        self.assertTrue(F.one_sided_verdict(op, rd)['pass'])
        self.assertFalse(F.one_sided_verdict(rd[0], rd)['pass'])     # 负对照：随机方向本身
        self.assertTrue(F.retain(2.0, 3.0, 1.8, 2.5)['pass'])
        self.assertFalse(F.retain(3.0, 3.0, 0.2, 2.5)['pass'])       # 胜者通吃

    def test_v3_restoration_select(self):
        rng = np.random.default_rng(4)
        n = 60
        joint = rng.normal(0, 0.2, n)
        single = joint + 3.0
        names = [f'{k}{l}' for l in range(10) for k in ('attn', 'mlp')]
        P_ = joint[:, None] + rng.normal(0, 0.2, (n, len(names)))
        P_[:, names.index('mlp7')] += 2.4                      # 植入：第 7 层 MLP 恢复约 0.8
        A = np.arange(n) % 2 == 0
        r = F.select_component(P_, joint, single, A, names)
        self.assertEqual(r['component'], 'mlp7')
        self.assertTrue(r['pass'] and r['mlp_pos_B'] > r['attn_pos_B'])
        r0 = F.select_component(joint[:, None] + rng.normal(0, 0.2, (n, len(names))), joint, single, A, names)
        self.assertFalse(r0['pass'])                           # 负对照：无组件恢复
        self.assertTrue(np.isnan(F.restoration(P_, joint, joint - 1)).all())

    def test_v4_units(self):
        rng = np.random.default_rng(5)
        n, U = 80, 300
        layer_of = np.repeat([0, 1, 2], 100)
        planted = [5, 17, 140, 260]
        d = rng.normal(0, 0.1, (n, U))
        d[:, planted] -= 1.0
        A = np.arange(n) % 2 == 0
        self.assertEqual(sorted(F.select_units(d, A, 4)), planted)
        self.assertEqual(F.select_units(np.zeros((n, U)) + rng.normal(0, 0.1, (n, U)), A, 4) == planted, False)   # 负对照
        rs = F.random_units_matched(planted, layer_of, 5, 0)
        for s in rs:
            self.assertEqual([layer_of[i] for i in s], [layer_of[i] for i in planted])
            self.assertFalse(set(s) & set(planted))
        self.assertTrue(F.unit_verdict(0.8, 0.6, [0.05, 0.1])['pass'])
        self.assertFalse(F.unit_verdict(0.8, 0.6, [0.3, 0.1])['pass'])
        self.assertFalse(F.unit_verdict(0.8, 0.2, [0.01])['pass'])

    def test_v5_segments(self):
        rng = np.random.default_rng(6)
        segs, Gs = [], []
        for _ in range(30):
            s = np.array(['code'] * 10 + ['comment'] * 8 + ['code'] * 10)
            G = np.where(s == 'code', 2.0, 0.0) + rng.normal(0, 0.3, len(s))
            G[10:12] = 2.0                                       # 植入：进入注释后 2 个 token 才关门
            segs.append(s); Gs.append(G)
        G, S = np.concatenate(Gs), np.concatenate(segs)
        self.assertGreater(F.segment_auc(G, S, 'code', 'comment'), 0.8)
        pr = F.boundary_profile(Gs, segs, 'code', 'comment', [-4, 6])
        self.assertEqual(pr['latency'], 2)
        self.assertEqual(F.boundary_profile(Gs, segs, 'comment', 'code', [-4, 6])['latency'], 0)
        Gn = [rng.normal(0, 1, len(s)) for s in segs]                # 负对照
        self.assertLess(abs(F.segment_auc(np.concatenate(Gn), S, 'code', 'comment') - 0.5), 0.08)


if __name__ == '__main__':
    unittest.main()
