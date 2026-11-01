# -*- coding: utf-8 -*-
"""mode_atlas.v1：植入「主效应 + 否定 × 真假交互」的格结构能被找回（解耦 R²、可加份额、真值探针）；
没有交互时真值探针在机会水平（负对照）；效应分解能精确重建格均值。"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.dist_protocol import mode_atlas as M  # noqa: E402


def planted(n, seed, nt=1.0, d=48):
    rng = np.random.default_rng(seed)
    base = np.random.default_rng(7)
    m = {k: base.normal(size=d) for k in ('q', 'n', 't', 'nt')}
    H = np.zeros((n, 8, d))
    for e in range(n):
        content = rng.normal(0, 3, d)
        for c in range(8):
            s = M.cell_signs(c)
            H[e, c] = content + 2 * (s['q'] * m['q'] + s['n'] * m['n'] + s['t'] * m['t']) \
                + nt * 2 * (2 * s['n'] * s['t']) * m['nt'] + rng.normal(0, 0.5, d)
    return H


def truth_data(H):
    X = H.reshape(-1, H.shape[-1])
    cells = np.tile(np.arange(8), len(H))
    keep = (cells >> 2) == 0                                  # 陈述句
    truth = ((cells >> 1) & 1) ^ (cells & 1)                  # t XOR n
    groups = np.repeat(np.arange(len(H)), 8)
    return X[keep], truth[keep].astype(bool), groups[keep]


class ModeAtlasTests(unittest.TestCase):
    def test_effects_reconstruct_cell_means(self):
        G = M.center(planted(30, 0))
        eff = M.effects(G)
        np.testing.assert_allclose(M.predict(eff, list(eff)), G.mean(0), atol=1e-9)

    def test_planted_structure_recovered(self):
        Gd, Gc = M.center(planted(120, 1)), M.center(planted(100, 2))
        eff = M.effects(Gd)
        self.assertGreater(M.r2_cells(Gc, Gd.mean(0)), 0.8)
        sh = M.variance_shares(eff)
        self.assertLess(sh['qn'] + sh['qt'] + sh['qnt'], 0.05)
        self.assertGreater(sh['nt'], 0.1)
        self.assertLess(M.r2_cells(Gc, M.predict(eff, ['q', 'n', 't'])), M.r2_cells(Gc, Gd.mean(0)) - 0.05)
        Xd, yd, gd = truth_data(planted(120, 1)); Xc, yc, _ = truth_data(planted(100, 2))
        self.assertGreater(M.probe(Xd, yd, gd, Xc, yc)['acc'], 0.9)

    def test_no_interaction_truth_at_chance(self):
        Xd, yd, gd = truth_data(planted(120, 1, nt=0.0)); Xc, yc, _ = truth_data(planted(100, 2, nt=0.0))
        self.assertLess(M.probe(Xd, yd, gd, Xc, yc)['acc'], 0.65)

    def test_paired_gate(self):
        rng = np.random.default_rng(0)
        bias = -8 + rng.normal(0, 1, (200, 1))                      # 整体偏向「错」，绝对口径全错
        truth = np.array([0, 1, 1, 0])
        m = bias + 2.0 * truth[None] + rng.normal(0, 0.5, (200, 4))
        self.assertGreater(M.paired_gate(m)['acc'], 0.9)
        self.assertLess(M.paired_gate(bias + rng.normal(0, 0.5, (200, 4)))['acc'], 0.65)   # 负对照：不区分真假


if __name__ == '__main__':
    unittest.main()
