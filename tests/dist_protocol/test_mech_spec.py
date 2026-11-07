# -*- coding: utf-8 -*-
"""mech_spec.v1：材料不重叠、排序构造、植入特异 / 通用找回、缺记录待检。"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared.dist_protocol import mech_bench as B
from shared.dist_protocol import mech_spec as S


def enc(text):
    return [hash(w) % 9973 for w in text.replace(',', ' ').split()]


class Materials(unittest.TestCase):
    def test_sealed_disjoint_from_mb(self):
        self.assertFalse(set(S.SEALED_NAMES) & set(B.NAMES))
        self.assertFalse(set(S.SEALED_CAPITALS) & set(B.CAPITALS))
        self.assertFalse({a for a, _ in S.SEALED_NOUNS} & {a for a, _ in B.NOUNS})
        for task, frames in S.SEALED_FRAMES.items():
            self.assertFalse(set(frames) & set(B.TASKS[task]['frames']))

    def test_build_is_deterministic_and_keeps_last_token(self):
        for task in B.TASKS:
            a = S.build_sealed(task, 40, enc)
            b = S.build_sealed(task, 40, enc)
            self.assertEqual([i['clean'] for i in a['items']], [i['clean'] for i in b['items']])
            self.assertEqual(len(a['items']), 40)
            for i in a['items']:
                self.assertEqual(i['clean_ids'][-1], i['corrupt_ids'][-1])


class Orders(unittest.TestCase):
    def test_contrast_pushes_shared_node_down(self):
        scores = {'a': np.array([10., 5, 1, 0]), 'b': np.array([10., 0, 1, 5]), 'c': np.array([10., 0, 1, 1])}
        order = S.order_of(S.contrast_score(scores, 'a'))
        self.assertEqual(order[0], 1)
        self.assertGreater(list(order).index(0), 0)

    def test_excluding_moves_banned_last_and_keeps_all(self):
        out = S.excluding(np.array([3, 1, 0, 2]), np.array([3]))
        self.assertEqual(out.tolist(), [1, 0, 2, 3])

    def test_iterative_keeps_group_nodes(self):
        """{0,1} 与 {2,3} 是两组必须同时在场的节点，其他节点无效。"""
        def value(alive):
            s = set(alive.tolist())
            return (1.0 if {0, 1} <= s else 0.0) + (1.0 if {2, 3} <= s else 0.0)

        def effect(alive):
            return [abs(value(alive) - value(np.array([x for x in alive if x != c]))) for c in alive]

        order = S.iterative_order(40, effect)
        self.assertEqual(sorted(order.tolist()), list(range(40)))
        self.assertTrue({0, 1, 2, 3} <= set(order[:8].tolist()))

    def test_ladder_and_primary(self):
        self.assertEqual(S.primary_k(1188), 12)
        self.assertEqual(S.ladder_counts(156), [1, 2, 4, 8])


def world(rng, target, off, n=120):
    """三项任务：集合 m 在本任务下降 target，在其他任务下降 off。"""
    raws, clusters = {}, {}
    for t in B.TASKS:
        full, empty = 3 + rng.normal(0, .1, n), rng.normal(0, .1, n)
        raw = {'clean': full, 'empty|cf': empty}
        raw[S.rkey('cf', 'm', 4)] = full - (full - empty) * target + rng.normal(0, .05, n)
        for src in B.TASKS:
            if src != t:
                raw[S.rkey('cf', 'm', 4, src=src)] = full - (full - empty) * off + rng.normal(0, .05, n)
        raws[t], clusters[t] = raw, [f'c{i // 6}' for i in range(n)]
    return raws, clusters


class Cells(unittest.TestCase):
    def cell(self, target, off, nat=1.0):
        raws, cl = world(np.random.default_rng(0), target, off)
        return S.spec_cell(raws, cl, 'rs_ioi', 'm', 4, natural=nat, natural_random=1.0, n_boot=200)

    def test_implanted_specific(self):
        self.assertEqual(self.cell(.8, .1)['status'], 'specific')

    def test_shared_is_not_specific(self):
        self.assertEqual(self.cell(.8, .8)['status'], 'not_specific')

    def test_weak_is_no_effect(self):
        self.assertEqual(self.cell(.05, 0)['status'], 'no_effect')

    def test_natural_damage_blocks(self):
        self.assertEqual(self.cell(.8, .1, nat=5.0)['status'], 'broad_natural')

    def test_missing_natural_and_task_pending(self):
        raws, cl = world(np.random.default_rng(0), .8, .1)
        self.assertEqual(S.spec_cell(raws, cl, 'rs_ioi', 'm', 4, n_boot=50)['status'], 'pending')
        raws.pop('gr_agree')
        self.assertEqual(S.spec_cell(raws, cl, 'rs_ioi', 'm', 4, n_boot=50)['status'], 'pending')

    def test_summary_needs_all_tasks(self):
        ok = {'status': 'specific'}
        self.assertEqual(S.summarize({'m': {'a': ok, 'b': ok}})['m']['verdict'], '待检')
        self.assertEqual(S.summarize({'m': {'a': ok, 'b': ok, 'c': {'status': 'no_effect'}}})['m']['verdict'], '支持')
        self.assertEqual(S.summarize({'m': {'a': ok, 'b': {'status': 'not_specific'}, 'c': {'status': 'no_effect'}}})['m']['verdict'], '未支持')


class Backbone(unittest.TestCase):
    def raw(self, good, base_f=.3, n=120):
        rng = np.random.default_rng(1)
        full, empty = 3 + rng.normal(0, .1, n), rng.normal(0, .1, n)
        base = empty + (full - empty) * base_f
        raw = {'clean': full, 'empty|cf': empty, 'bb_only': base}
        for k in [1, 4, 16]:
            f = min(1, k / 4) if good else 0.02 * k
            raw[S.bkey('m', k)] = base + (full - base) * f + rng.normal(0, .05, n)
            for d in range(3):
                raw[S.bkey('random', k, d)] = base + (full - base) * 0.02 * k + rng.normal(0, .05, n)
        return raw, [f'c{i // 6}' for i in range(n)]

    def test_small_set_found(self):
        raw, cl = self.raw(True)
        r = S.bb_score(raw, cl, [1, 4, 16], 'm', n_boot=200)
        self.assertTrue(r['beats_random'])
        self.assertEqual(r['k_f80'], 4)

    def test_random_like_negative(self):
        raw, cl = self.raw(False)
        self.assertFalse(S.bb_score(raw, cl, [1, 4, 16], 'm', n_boot=200)['beats_random'])

    def test_no_power_when_backbone_is_enough(self):
        raw, cl = self.raw(True, base_f=.95)
        self.assertEqual(S.bb_score(raw, cl, [1, 4, 16], 'm', n_boot=50)['status'], 'no_power')


class Group(unittest.TestCase):
    def raw(self, ig, ex, it, n=120):
        rng = np.random.default_rng(2)
        full, empty = 3 + rng.normal(0, .1, n), rng.normal(0, .1, n)
        raw = {'clean': full, 'empty|cf': empty}
        for b in B.BUDGETS:
            for m, v in (('atp_ig', ig), ('exact', ex), ('exact_iter', it), ('exact_denoise', ex)):
                f = 1.0 if b == 1.0 else v
                raw[S.gkey(m, b)] = empty + (full - empty) * f + rng.normal(0, .02, n)
        return raw, [f'c{i // 6}' for i in range(n)]

    def test_closed_gap_supported(self):
        raw, cl = self.raw(.5, .1, .45)
        r = S.group_score(raw, cl, n_boot=200)
        self.assertEqual(r['exact_iter']['verdict'], '支持')
        self.assertEqual(r['exact_denoise']['verdict'], '未支持')

    def test_no_gap(self):
        raw, cl = self.raw(.3, .3, .3)
        self.assertEqual(S.group_score(raw, cl, n_boot=50)['status'], 'no_gap')

    def test_missing_pending(self):
        raw, cl = self.raw(.5, .1, .45)
        raw.pop(S.gkey('exact_iter', 0.01))
        self.assertEqual(S.group_score(raw, cl, n_boot=50)['status'], 'pending')


if __name__ == '__main__':
    unittest.main()
