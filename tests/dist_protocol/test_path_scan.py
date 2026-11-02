# -*- coding: utf-8 -*-
"""path_scan.v1：植入写入格在 A 选出、B 确认；全噪声时不确认（负对照）。植入的集中头被找回、占比高；分散作用时占比低（负对照）。
消融植入头使 LO 下降远超同层随机头；消融无关头时不成立（负对照）。"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.dist_protocol import path_scan as P  # noqa: E402


def _cells(rng, n, planted):
    k = np.array(P.CRITERIA_TS['k_grid'], float)
    out = {}
    for c in range(12):
        slope = 0.4 if c == planted else 0.0
        out[f'L{c}/end'] = {'op': slope * k[None] + rng.normal(0, 0.5, (n, len(k))),
                            'rand': rng.normal(0, 0.5, (2, n, len(k)))}
    return out


class PathScanTests(unittest.TestCase):
    def test_split_balanced(self):
        ids = [f'e{i}' for i in range(40)]
        cats = ['a'] * 20 + ['b'] * 20
        m = P.split_ab(ids, cats)
        self.assertEqual(m[:20].sum(), 10)
        self.assertEqual(m[20:].sum(), 10)
        self.assertTrue(np.array_equal(m, P.split_ab(ids, cats)))

    def test_truth_scan_planted_and_null(self):
        rng = np.random.default_rng(0)
        A = np.arange(200) % 2 == 0
        r = P.select_and_confirm(_cells(rng, 200, planted=7), A)
        self.assertEqual(r['cell'], 'L7/end')
        self.assertTrue(r['confirmed'])
        r0 = P.select_and_confirm(_cells(rng, 200, planted=-1), A)
        self.assertFalse(r0['confirmed'])

    def test_heads_planted_and_diffuse(self):
        rng = np.random.default_rng(1)
        n, L, H = 300, 8, 16
        planted = [5 * H + 3, 6 * H + 9, 7 * H + 1]
        D = rng.normal(0, 0.3, (n, L, H))
        for i in planted:
            D[:, i // H, i % H] += 2.0
        top = P.top_heads(D[:150], 3)
        self.assertEqual(sorted(top), sorted(planted))
        self.assertGreater(P.concentration(D[150:], top)['share'], 0.8)
        Dd = 0.05 + rng.normal(0, 0.3, (n, L, H))
        self.assertLess(P.concentration(Dd[150:], P.top_heads(Dd[:150], 3))['share'], 0.5)
        att = rng.uniform(0, 0.05, (n, L, H))
        for i in planted:
            att[:, i // H, i % H] = 0.6
        self.assertTrue(P.attn_focus(att, top)['pass'])
        self.assertFalse(P.attn_focus(att, P.random_matched(top, H, 1, 0)[0])['pass'])

    def test_ablation(self):
        rng = np.random.default_rng(2)
        base = rng.normal(3, 1, 200)
        top = base - 4 + rng.normal(0, 0.3, 200)
        rand = base[None] - 0.2 + rng.normal(0, 0.3, (5, 200))
        self.assertTrue(P.ablation_verdict(base, top, rand)['pass'])
        self.assertFalse(P.ablation_verdict(base, base - 0.2 + rng.normal(0, 0.3, 200), rand)['pass'])
        sets = P.random_matched([3, 17, 18], 16, 5, 0)
        for s in sets:
            self.assertEqual([i // 16 for i in s], [0, 1, 1])
            self.assertFalse(set(s) & {3, 17, 18})
            self.assertEqual(len(set(s)), 3)


if __name__ == '__main__':
    unittest.main()
