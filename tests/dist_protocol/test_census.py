# -*- coding: utf-8 -*-
"""mlp_census.v1 累加器：植入单元的 top-k / bottom-k 与符号区找回，直方图与矩统计正确，分片合并等于单机累计。"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.dist_protocol import census as C  # noqa: E402


def data(seed=0, T=600, F=12):
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((T, F)).astype(np.float32)
    V = rng.standard_normal((T, F)).astype(np.float32)
    # 单元 0：在 7 个已知位置上 +×+ 强激活；单元 1：在 5 个已知位置上 +×− 强负激活
    hot = np.arange(7) * 37 + 5
    neg = np.arange(5) * 53 + 11
    G[hot, 0], V[hot, 0] = 8.0 + np.arange(7), 9.0
    G[neg, 1], V[neg, 1] = 7.0, -6.0 - np.arange(5)
    A = (G / (1 + np.exp(-G)) * V).astype(np.float32)
    return A, G, V, hot, neg


class CensusTests(unittest.TestCase):
    def acc(self, A, G, V, gidx, k=8, chunks=3):
        import torch
        acc = C.LayerAcc(A.shape[1], k=k)
        for part in np.array_split(np.arange(len(A)), chunks):
            acc.update(torch.tensor(A[part]), torch.tensor(G[part]), torch.tensor(V[part]), torch.tensor(gidx[part]))
        return acc.to_numpy()

    def test_planted_poles_and_regimes(self):
        A, G, V, hot, neg = data()
        gidx = C.global_index(2, np.arange(len(A), dtype=np.int64))
        d = self.acc(A, G, V, gidx)
        src, off = C.split_index(d['top_idx'][0, :7])
        self.assertTrue((src == 2).all())
        self.assertEqual(set(off.tolist()), set(hot.tolist()))
        self.assertTrue(np.all(np.diff(d['top_val'][0]) <= 0))           # 正极按激活值降序
        _, off_n = C.split_index(d['bot_idx'][1, :5])
        self.assertEqual(set(off_n.tolist()), set(neg.tolist()))
        self.assertTrue((d['top_g'][0, :7] > 0).all() and (d['top_v'][0, :7] > 0).all())
        self.assertTrue((d['bot_v'][1, :5] < 0).all())

    def test_hist_and_moments(self):
        A, G, V, _, _ = data(seed=1)
        d = self.acc(A, G, V, np.arange(len(A), dtype=np.int64))
        self.assertTrue((d['hist'].sum((1, 2)) == len(A)).all())
        self.assertTrue(np.allclose(d['s1'], A.astype(np.float64).sum(0), rtol=1e-6))
        self.assertTrue(np.allclose(d['s2'], (A.astype(np.float64) ** 2).sum(0), rtol=1e-6))
        self.assertTrue(np.allclose(d['amax'], A.max(0)) and np.allclose(d['amin'], A.min(0)))
        reg = d['hist'].sum(1)                                            # (F, 4) 符号区计数
        exp_pp = ((G > 0) & (V > 0)).sum(0)
        self.assertTrue((reg[:, 0] == exp_pp).all())
        pos = d['hist'][:, C.N_SIDE + 1:, :].sum((1, 2))
        self.assertTrue((pos == (A >= C.LO).sum(0)).all())

    def test_merge_equals_single_pass(self):
        A, G, V, _, _ = data(seed=2, T=900)
        gidx = np.arange(len(A), dtype=np.int64)
        full = self.acc(A, G, V, gidx, chunks=5)
        a = self.acc(A[:400], G[:400], V[:400], gidx[:400], chunks=2)
        b = self.acc(A[400:], G[400:], V[400:], gidx[400:], chunks=2)
        m = C.merge(a, b)
        for key in ('hist', 'n', 'top_idx', 'bot_idx'):
            self.assertTrue(np.array_equal(m[key], full[key]), key)
        for key in ('s1', 's2', 'amin', 'amax', 'top_val', 'bot_val'):
            self.assertTrue(np.allclose(m[key], full[key]), key)
        s = C.summarize_layer(m)
        self.assertEqual(s['n_tokens'], 900)
        self.assertAlmostEqual(sum(s['regime_share_mean'].values()), 1.0, places=6)


if __name__ == '__main__':
    unittest.main()
