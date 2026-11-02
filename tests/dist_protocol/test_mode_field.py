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


if __name__ == '__main__':
    unittest.main()
