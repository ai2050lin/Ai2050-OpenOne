# -*- coding: utf-8 -*-
"""atlas.v1 方法验证：在三个模式族里植入一个共享的「疑问」方向（只在 mid/last 层），检查谱图挖掘能找回
共享编码轴、预注册概念假设得到支持、留一族迁移通过；负对照（各族的疑问方向互相独立）不得报告共享轴。
合成数据只用于验证方法，不代表任何真实模型。"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.dist_protocol import atlas  # noqa: E402
from shared.dist_protocol.suffstats import SuffStats  # noqa: E402

DIM = 64
LAYERS = ['first', 'mid', 'last']
FAMILIES = [  # (family, 主因子, 主因子水平, 句式因子, 句式水平, 疑问水平)
    ('KN-A', 'category', ['animal', 'fruit', 'tool', 'metal'], 'frame', ['decl', 'interrog', 'scene'], 'interrog'),
    ('KN-B', 'attr', ['color', 'taste', 'size'], 'frame', ['decl', 'question', 'habit'], 'question'),
    ('GR-C', 'polarity', ['aff', 'neg'], 'mood', ['decl', 'interrog'], 'interrog'),
]


def unit(rng):
    v = rng.standard_normal(DIM)
    return v / np.linalg.norm(v)


def make_family(spec, shared_q, seed, planted=True):
    fam, mf, mlv, ff, flv, qlv = spec
    rng = np.random.default_rng(seed)
    dirs = {(mf, lv): unit(rng) for lv in mlv}
    dirs.update({(ff, lv): unit(rng) for lv in flv})
    ss = SuffStats(LAYERS, DIM, [mf, ff])
    for k in range(24):                                  # 每个主水平 24 个填充词：16 个 discovery，8 个 lockbox
        for lv in mlv:
            filler = unit(rng) * 0.6
            split = 'discovery' if k < 16 else 'lockbox'
            for fl in flv:
                vecs = []
                for t in LAYERS:
                    fdir = dirs[(ff, fl)]
                    if fl == qlv and planted and t != 'first':
                        fdir = shared_q[t]
                    v = 1.0 * dirs[(mf, lv)] + 1.0 * fdir + filler + rng.standard_normal(DIM) * 0.15
                    vecs.append(v)
                ss.add(split, {mf: lv, ff: fl}, vecs)
    concepts = {f'{ff}={qlv}': '疑问'}
    return atlas.family_features(fam, 'TG-' + fam, ss, [mf, ff], concepts=concepts)


def build(planted, model_seed=0):
    rng = np.random.default_rng(100 + model_seed)
    shared = {t: unit(rng) for t in LAYERS}
    return [make_family(spec, shared, seed=10 * i + model_seed, planted=planted) for i, spec in enumerate(FAMILIES)]


class AtlasTests(unittest.TestCase):
    def test_recovers_planted_shared_axis(self):
        res = atlas.build_atlas({'m1': build(True)})
        g = res['graphs'][0]
        self.assertTrue(all(v['used'] for v in g['nulls'].values()))          # 经验零分布可用
        q_axes = [a for a in g['axes'] if '疑问' in a['concepts']]
        self.assertTrue(q_axes, g['axes'])
        self.assertEqual({a['layer'] for a in q_axes}, {'mid', 'last'})         # first 层没有植入，不应出现
        for a in q_axes:
            self.assertEqual(a['families'], ['GR-C', 'KN-A', 'KN-B'])
            self.assertTrue(a['supported'])
            self.assertGreater(a['transfer'], 0.55)   # 水平对比含本族其他水平的成分，理论上约 0.67
        hyp = {h['concept']: h for h in g['concept_hypotheses']}['疑问']
        self.assertEqual(hyp['status'], 'supported')
        kinds = {f['id']: f['kind'] for f in g['findings']}
        self.assertEqual(kinds['concept-疑问'], 'supported')
        self.assertEqual(len(g['coords']['mid']), len(g['nodes']) // 3)
        self.assertEqual(len(next(iter(g['coords']['mid'].values()))), 3)
        # 族内其他水平是各自独立的随机方向：不应形成跨族轴
        self.assertFalse([a for a in g['axes'] if not a['concepts']], [a['label'] for a in g['axes'] if not a['concepts']])

    def test_negative_control_independent_directions(self):
        g = atlas.build_atlas({'m1': build(False)})['graphs'][0]
        self.assertEqual(g['axes'], [])
        hyp = {h['concept']: h for h in g['concept_hypotheses']}['疑问']
        self.assertEqual(hyp['status'], 'negative')

    def test_cross_model_and_ai_digest(self):
        res = atlas.build_atlas({'m1': build(True, 0), 'm2': build(True, 1)})
        xm = res['cross_model']
        self.assertGreater(xm['edges_testable'], 0)
        self.assertGreaterEqual(xm['replication_rate'], 0.8)
        for g in res['graphs']:
            for a in g['axes']:
                self.assertEqual(len(a['replicated_in']), 1)
        text = atlas.digest(res, 'm1')
        self.assertIn('共享编码轴', text)
        self.assertIn('疑问', text)
        prompt = atlas.ai_prompt(res, '疑问方向是什么？', 'm1')
        self.assertIn('JSON', prompt)
        reply = '思考…\n[{"statement": "疑问语气在中层有统一编码", "evidence": ["KN-A:frame=interrog@mid"], ' \
                '"prediction": "英文疑问句也落在同一轴上", "new_family": {"family_id": "GR-Q-EN"}, "alternative": "句末标点"}]'
        parsed = atlas.parse_hypotheses(reply)
        self.assertEqual(parsed['status'], 'ok')
        self.assertEqual(parsed['hypotheses'][0]['status'], 'unverified')
        self.assertEqual(atlas.parse_hypotheses('没有 JSON')['status'], 'unparsed')

    def test_deflation_removes_baseline_leak_and_heldout_prediction(self):
        # 植入两个共享概念（疑问、陈述）。水平对比相对本族平均，本族基线里的「陈述」会让其他水平带上反向分量；
        # 逐轴收缩后，未声明的水平不应被并进共享轴。KN-B 的 question 只出现在留出句式，作为未见句式预测。
        rng = np.random.default_rng(7)
        shared = {c: {t: unit(rng) for t in LAYERS} for c in ('疑问', '陈述')}
        specs = [('KN-A', ['decl', 'interrog', 'scene', 'topic', 'concess'], 'interrog', None),
                 ('KN-B', ['decl', 'habit', 'compare', 'question'], 'question', 'question'),
                 ('GR-C', ['decl', 'interrog'], 'interrog', None)]
        fams = []
        for i, (fam, flv, qlv, held) in enumerate(specs):
            r = np.random.default_rng(50 + i)
            own = {lv: unit(r) for lv in flv}
            main = {lv: unit(r) for lv in ('x1', 'x2', 'x3')}
            ss = SuffStats(LAYERS, DIM, ['m', 'frame'])
            for k in range(24):
                for mlv in main:
                    filler = unit(r) * 0.6
                    for fl in flv:
                        split = 'lockbox_frame' if fl == held else ('discovery' if k < 16 else 'lockbox')
                        d = shared['疑问'] if fl == qlv else shared['陈述'] if fl == 'decl' else None
                        vecs = [main[mlv] + (d[t] if d else own[fl]) + filler + r.standard_normal(DIM) * 0.15 for t in LAYERS]
                        ss.add(split, {'m': mlv, 'frame': fl}, vecs)
            fams.append(atlas.family_features(fam, 'TG-' + fam, ss, ['m', 'frame'],
                                              concepts={f'frame={qlv}': '疑问', 'frame=decl': '陈述'}))
        g = atlas.build_atlas({'m1': fams})['graphs'][0]
        for a in g['axes']:
            levels = {x.split(':')[1].split('@')[0] for x in a['members']}
            self.assertTrue(levels <= {'frame=decl', 'frame=interrog'} or levels <= {'frame=decl'}
                            or levels <= {'frame=interrog'}, a['label'])
            self.assertEqual(len(a['concepts']), 1, a['label'])            # 每条轴只对应一个概念
        hyp = {h['concept']: h for h in g['concept_hypotheses']}
        self.assertEqual(hyp['陈述']['status'], 'supported')
        q = hyp['疑问']
        self.assertEqual(q['families'], ['GR-C', 'KN-A', 'KN-B'])
        best = next(p for p in q['layers'] if p['layer'] == q['best_layer'])
        self.assertEqual([h['id'].split('@')[0] for h in best['heldout']], ['KN-B:frame=question'])
        self.assertTrue(best['heldout'][0]['pass'], best)
        self.assertEqual(q['status'], 'supported')

    def test_mismatched_family_is_skipped(self):
        fams = build(True)
        fams[2] = dict(fams[2], dim=32)
        g = atlas.build_atlas({'m1': fams})['graphs'][0]
        self.assertEqual([s['family'] for s in g['skipped']], ['GR-C'])

    def test_families_with_layer_subsets_use_common_tags(self):
        # 真实运行中各族 suffstats 按体积预算保留的层不同：谱图应在共有层上建图，而不是排除整族
        fams = build(True)
        fams[0] = atlas._restrict_family(fams[0], ['mid', 'last'])
        g = atlas.build_atlas({'m1': fams})['graphs'][0]
        self.assertEqual(g['skipped'], [])
        self.assertEqual(g['layers'], ['mid', 'last'])
        self.assertEqual(g['layers_dropped'], {f: ['first'] for f in ('KN-A', 'KN-B', 'GR-C')})   # 每族记录被截掉的层
        self.assertTrue(all(n['layer'] in ('mid', 'last') for n in g['nodes']))
        q_axes = [a for a in g['axes'] if '疑问' in a['concepts']]
        self.assertEqual({a['layer'] for a in q_axes}, {'mid', 'last'})
        self.assertTrue(all(a['supported'] for a in q_axes))


if __name__ == '__main__':
    unittest.main()
