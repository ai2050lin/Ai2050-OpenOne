# -*- coding: utf-8 -*-
"""entity_atlas.v1：植入「大类 + 类别 + 个体差分 + 共享属性方向」的合成编码能被找回；属性与编码无关时算子不超过基线（负对照）；
跨语言迁移在共享算子时成立、在旋转后的独立空间里失效；RSA 对旋转不变、对打乱失效。"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.dist_protocol import entity_atlas as E  # noqa: E402


def planted(seed=0, d=64, attr_signal=3.0, unrelated=False):
    rng = np.random.default_rng(seed)
    sups, cats = ['animal', 'plant'], {'animal': ['bird', 'fish', 'mammal'], 'plant': ['fruit', 'veg']}
    u_sup = {s: rng.normal(0, 4, d) for s in sups}
    u_cat = {c: rng.normal(0, 2.5, d) for s in sups for c in cats[s]}
    colors = ['red', 'green', 'yellow', 'brown']
    a_col = {c: rng.normal(0, 1, d) for c in colors}
    ids, sup, cat, col, X = [], [], [], [], []
    for s in sups:
        for c in cats[s]:
            for i in range(40):
                k = colors[rng.integers(len(colors))]
                x = u_sup[s] + u_cat[c] + rng.normal(0, 1, d)
                x = x + (0 if unrelated else attr_signal) * a_col[k]
                ids.append(f'{c}.{i}'); sup.append(s); cat.append(c); col.append(k); X.append(x)
    X = np.array(X)
    split = [E.split_of(i) for i in ids]
    return X, np.array(sup), np.array(cat), np.array(col), np.array(split)


class EntityAtlasTests(unittest.TestCase):
    def test_hierarchy_recovered(self):
        X, sup, cat, col, split = planted()
        g = E.g1_layer(X, sup, cat, split)
        self.assertGreater(g['cat_acc_heldout'], 0.9)
        self.assertGreater(g['var_super'], g['var_cat_within_super'] * 0.5)

    def test_operator_recovers_cross_cutting_attribute(self):
        X, sup, cat, col, split = planted()
        layers = [np.random.default_rng(1).normal(size=X.shape), X]       # 第 0 层是噪声，层选择应选中第 1 层
        op = E.operator_eval(layers, col, split, cat)
        self.assertEqual(op['layer'], 1)
        self.assertGreater(op['heldout_bal_acc'], E.category_majority_acc(col, split, cat) + 0.3)
        dl = E.operator_eval(layers, col, split, cat, delta=True)
        self.assertGreater(dl['heldout_bal_acc'], 0.7)                    # 属性落在个体差分里

    def test_unrelated_attribute_not_above_baseline(self):
        X, sup, cat, col, split = planted(unrelated=True, seed=3)
        op = E.operator_eval([X], col, split, cat)
        self.assertLess(op['heldout_bal_acc'], 0.45)                      # 4 类机会 0.25

    def test_cross_language_transfer(self):
        X, sup, cat, col, split = planted(seed=5)
        rng = np.random.default_rng(9)
        shared = X + rng.normal(0, 0.5, X.shape)                         # 「英文」= 同一空间 + 噪声
        Q, _ = np.linalg.qr(rng.normal(size=(X.shape[1], X.shape[1])))
        rotated = X @ Q                                                   # 独立坐标：几何相同、坐标不同
        self.assertGreater(E.operator_eval([X], col, split, cat, X_transfer=[shared])['transfer_bal_acc'], 0.8)
        self.assertLess(E.operator_eval([X], col, split, cat, X_transfer=[rotated])['transfer_bal_acc'], 0.5)
        r = E.rsa(X, rotated, n_perm=50)
        self.assertGreater(r['spearman'], 0.99)                           # 相对结构不变
        self.assertGreater(r['spearman'], r['null_p95'])

    def test_orthogonality_and_pr(self):
        X, sup, cat, col, split = planted()
        o = E.hierarchy_orthogonality(X, sup, cat, n_shuffle=50)
        self.assertLess(o['mean_abs_cos'], 0.5)
        self.assertGreater(E.participation_ratio(np.random.default_rng(0).normal(size=(200, 20))), 10)


def planted_continuous(n, seed, d=48, related=True, A=None, U=None):
    """类别中心 + 个体差分；连续目标 = 共享算子 A 作用在个体差分上（related）或与编码无关的噪声。"""
    rng = np.random.default_rng(seed)
    cats = ['a', 'b', 'c', 'd']
    U = U if U is not None else {c: np.random.default_rng(100 + i).normal(0, 3, d) for i, c in enumerate(cats)}
    A = A if A is not None else np.random.default_rng(7).normal(0, 1, (d, 5))
    cat = np.array([cats[i % 4] for i in range(n)])
    delta = rng.normal(0, 1, (n, d))
    X = np.stack([U[c] for c in cat]) + delta
    Y = delta @ A + rng.normal(0, 0.5, (n, 5)) if related else rng.normal(0, 1, (n, 5))
    return X, Y, cat, A, U


class EntityAtlasV2Tests(unittest.TestCase):
    def test_shared_operator_confirmed_on_new_entities(self):
        Xd, Yd, cd, A, U = planted_continuous(200, 1)
        Xc, Yc, cc, _, _ = planted_continuous(150, 2, A=A, U=U)
        noise = np.random.default_rng(3).normal(size=Xd.shape)
        out = E.confirm_operator([noise, Xd], Yd, cd, [np.random.default_rng(4).normal(size=Xc.shape), Xc], Yc, cc,
                                 conf_other_layers=[None, Xc + np.random.default_rng(5).normal(0, 0.3, Xc.shape)])
        self.assertEqual(out['layer'], 1)
        self.assertGreater(out['R2_operator'], 0.7)
        self.assertTrue(out['supported_operator'] and out['supported_delta'] and out['supported_xlang'])

    def test_unrelated_target_not_confirmed(self):
        Xd, Yd, cd, A, U = planted_continuous(200, 1, related=False)
        Xc, Yc, cc, _, _ = planted_continuous(150, 2, related=False, A=A, U=U)
        out = E.confirm_operator([Xd], Yd, cd, [Xc], Yc, cc)
        self.assertFalse(out['supported_operator'])
        self.assertLess(out['R2_operator'], 0.1)


if __name__ == '__main__':
    unittest.main()
