# -*- coding: utf-8 -*-
"""arbitration.v1：植入的仲裁规律（线索 + 知识强度）能被找回并在新实体上预测；知识强度与对数几率无关时 γ 的区间包含 0（负对照）。"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.dist_protocol import arbitration as A  # noqa: E402


def planted(n_ent, seed, gamma=-0.6):
    rng = np.random.default_rng(seed)
    rows = {k: [] for k in ('S', 'src', 'r', 's', 'LO', 'ent')}
    for e in range(n_ent):
        s = rng.gamma(2, 2)
        for S in (0, 1):
            for src in (0, 1):
                for r in (1, 2, 3):
                    lo = -1 + 4 * S + 0.5 * src + 1.5 * np.log(r) + gamma * s + 0.3 * S * s + rng.normal(0, 0.7)
                    for k, v in zip(('S', 'src', 'r', 's', 'LO', 'ent'), (S, src, r, s, lo, e)):
                        rows[k].append(v)
    return {k: np.array(v) for k, v in rows.items()}


class ArbitrationTests(unittest.TestCase):
    def test_planted_law(self):
        out = A.law_eval(planted(150, 1), planted(120, 2), crit={**A.CRITERIA, 'n_boot': 200})
        self.assertGreater(out['R2_conf'], 0.8)
        self.assertLess(out['ci95']['s'][1], 0)
        self.assertAlmostEqual(out['coef']['logr'], 1.5, delta=0.3)
        self.assertGreater(out['R2_conf'], out['R2_conf_cues_only'] + 0.05)

    def test_unrelated_strength(self):
        out = A.law_eval(planted(150, 1, gamma=0.0), planted(120, 2, gamma=0.0), crit={**A.CRITERIA, 'n_boot': 200})
        lo, hi = out['ci95']['s']
        self.assertTrue(lo < 0 < hi)


if __name__ == '__main__':
    unittest.main()
