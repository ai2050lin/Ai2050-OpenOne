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


if __name__ == '__main__':
    unittest.main()
