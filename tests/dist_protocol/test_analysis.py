# -*- coding: utf-8 -*-
"""analysis.v1 方法验证：用 runner_v3 合成模式（已知答案）检查分析能否找回植入的规律，
并用打乱类别的负对照检查不会报告假阳性。合成数据只用于验证方法，不代表任何真实模型。"""
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('AI2050_NATIVE_DIR', 'off')   # 测试不在用户目录写原生全场

from shared.dist_protocol import analysis, grammar  # noqa: E402


def synthetic_items(family='KN-ISA', shards=3):
    spec = json.loads((ROOT / 'shared/dist_protocol/grammars' / f'{family}.json').read_text(encoding='utf-8'))
    # 植入找回检验的是分析方法：固定为经典口径（只读 target，first/mid/last 三层），与文法当前的采集口径无关
    spec['layers'], spec['readouts'] = ['first', 'mid', 'last'], ['target']
    for key in ('components', 'mlp_units'):         # 可选采集口径依赖文法自身的读出位置，这里一并去掉
        spec.pop(key, None)
    corpus = grammar.generate(spec)
    corpus['synthetic_dim'] = 96
    tmp = Path(tempfile.mkdtemp())
    try:
        shutil.copy(ROOT / 'shared/dist_protocol/runner_v3.py', tmp / 'runner.py')
        (tmp / 'corpus.json').write_text(json.dumps(corpus, ensure_ascii=False), encoding='utf-8')
        blobs = []
        for i in range(shards):
            out = tmp / f'o{i}'
            subprocess.run([sys.executable, str(tmp / 'runner.py'), '--mode', 'synthetic', '--out', str(out)],
                           check=True, env=dict(os.environ, AI2050_SHARD=f'{i}/{shards}'), capture_output=True)
            blobs.append((out / 'items.npz').read_bytes())
        return corpus, analysis.load_items(blobs)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


class AnalysisRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus, cls.items = synthetic_items()
        cls.p = analysis.analyze(cls.items, cls.corpus['factors'], n_perm=100)
        cls.by_tag = {l['tag']: l for l in cls.p['layers']}

    def test_items_complete(self):
        n = sum(1 for it in self.corpus['items'] if not it.get('canary'))
        self.assertEqual(self.items.X.shape[1], n)
        self.assertEqual(self.items.space, 'full')

    def test_layer_division_recovered(self):
        te = {t: l['trace_eta2'] for t, l in self.by_tag.items()}
        self.assertGreater(te['first']['frame'], te['first']['category'])     # 早层句式主导
        self.assertGreater(te['mid']['category'], te['mid']['frame'])         # 中层类别主导
        self.assertGreater(self.by_tag['first']['lexical_excess'], 0.3)       # 早层词身份
        self.assertLess(self.by_tag['last']['lexical_excess'], 0.15)

    def test_category_readable_on_unseen(self):
        cat = self.by_tag['mid']['factors']['category']
        self.assertGreater(cat['splits']['lockbox']['probe_acc'], 0.8)
        self.assertLessEqual(cat['perm']['acc_p'], 0.02)
        self.assertGreater(cat['splits']['lockbox_frame']['probe_acc'], 0.8)
        first = self.by_tag['first']['factors']['category']['splits']['lockbox']['probe_acc']
        self.assertLess(first, 0.6)

    def test_invariance_and_additivity(self):
        inv = self.by_tag['mid']['invariance'][0]
        self.assertGreater(inv['mean_offdiag'], 0.6)
        for l in self.p['layers']:
            self.assertLess(abs(l['additivity']['reliable_interaction']), 0.25)

    def test_findings_and_3d(self):
        kinds = {f['id']: f['kind'] for f in self.p['findings']}
        self.assertEqual(kinds['readable-category'], 'supported')
        self.assertEqual(kinds['lexical'], 'warning')
        n = len(self.p['points'])
        for tag, c in self.p['coords'].items():
            self.assertEqual(set(c), {'lda:category', 'lda:frame', 'pca'})
            for pts in c.values():
                self.assertEqual(len(pts), n)
                self.assertEqual(len(pts[0]), 3)
        self.assertEqual(len(self.p['stack']['layers']), 3)

    def test_negative_control_shuffled_labels(self):
        items = copy.deepcopy(self.items)
        rng = np.random.default_rng(1)
        fills = sorted(set(items.filler))
        cats = [items.cond[list(items.filler).index(f)]['category'] for f in fills]
        new = dict(zip(fills, rng.permutation(cats)))
        for i, f in enumerate(items.filler):
            items.cond[i] = dict(items.cond[i], category=str(new[f]))
        p = analysis.analyze(items, self.corpus['factors'], n_perm=100)
        kinds = {f['id']: f['kind'] for f in p['findings']}
        self.assertEqual(kinds['readable-category'], 'negative')

    def test_degenerate_tag_skipped(self):
        # 真实运行中嵌入层在句末读出处所有样本相同（同一个 token）：该标签应标为退化并跳过，其余层照常分析
        items = copy.deepcopy(self.items)
        items.X[0][:] = items.X[0][0]
        p = analysis.analyze(items, self.corpus['factors'], n_perm=5)
        first = p['layers'][0]
        self.assertTrue(first.get('degenerate') and first.get('insufficient'))
        self.assertTrue(all('factors' in l for l in p['layers'][1:]))
        self.assertTrue(p['findings'])

    def test_low_rank_tag_keeps_shapes(self):
        # 真实运行中 cue@L0 只有两种不同向量（句式线索末 token 只有两种）：白化后只剩 1 维，3D 坐标与层堆叠仍应成形
        items = copy.deepcopy(self.items)
        a, b = items.X[0][0].copy(), items.X[0][1].copy()
        for i, fr in enumerate(items.frame):
            items.X[0][i] = a if hash(str(fr)) % 2 else b
        items.X[0][0], items.X[0][1] = a, b
        p = analysis.analyze(items, self.corpus['factors'], n_perm=5)
        tag = p['layers'][0]['tag']
        self.assertTrue(all(len(row) == 3 for row in p['coords'][tag]['lda:' + self.corpus['factors'][0]]))
        self.assertTrue(all(len(c) == 2 for c in p['stack']['layers'][0]['centroids']))


class FrameLevelFactorTests(unittest.TestCase):
    """二值句式级因子（GR-Q 的语气：每个句式只有一种语气）：探针准确率在 1.0 处饱和，按句式打乱时接近真实划分的错误标签
    也能拿满分，所以可读出判据改用留出投影间隔的置换 p。植入找回 + 正交伪标签负对照。"""

    @classmethod
    def setUpClass(cls):
        cls.corpus, cls.items = synthetic_items('GR-Q', shards=2)
        cls.p = analysis.analyze(cls.items, cls.corpus['factors'])

    def test_planted_frame_level_factor_recovered(self):
        kinds = {f['id']: f['kind'] for f in self.p['findings']}
        self.assertEqual(kinds['readable-mood'], 'supported')
        last = {l['tag']: l for l in self.p['layers']}['last']['factors']['mood']
        self.assertEqual(last['unit'], 'frame')
        self.assertEqual(last['splits']['lockbox']['probe_acc'], 1.0)
        self.assertLessEqual(last['perm']['sep_p'], analysis.CRITERIA['perm_alpha'])
        self.assertGreater(last['perm']['acc_p'], analysis.CRITERIA['perm_alpha'])   # 准确率置换已饱和：这正是改用间隔的原因

    def test_orthogonal_pseudo_labels_are_negative(self):
        # 负对照：在每个真实语气内部把句式交替分到两个伪水平，伪标签与植入的语气方向正交
        items = copy.deepcopy(self.items)
        frames = sorted(set(items.frame))
        mood_of = {fr: next(items.cond[i]['mood'] for i in range(len(items.frame)) if items.frame[i] == fr) for fr in frames}
        pseudo = {}
        for m in sorted(set(mood_of.values())):
            for k, fr in enumerate([fr for fr in frames if mood_of[fr] == m]):
                pseudo[fr] = 'A' if k % 2 == 0 else 'B'
        for i, fr in enumerate(items.frame):
            items.cond[i] = {'pseudo': pseudo[fr]}
        p = analysis.analyze(items, ['pseudo'])
        kinds = {f['id']: f['kind'] for f in p['findings']}
        self.assertEqual(kinds['readable-pseudo'], 'negative')

    def test_uncomputable_separation_gives_p_one(self):
        # 回归：某层两种语气的水平均值完全相同（向量只依赖填充词）时间隔无法计算，置换 p 必须为 1，不能取最小值
        items = copy.deepcopy(self.items)
        first = {}
        for i, fl in enumerate(items.filler):
            first.setdefault(fl, items.X[0][i].copy())
            items.X[0][i] = first[fl]
        p = analysis.analyze(items, self.corpus['factors'], n_perm=20)
        rec = p['layers'][0]['factors']['mood']
        self.assertIsNone(rec['splits']['lockbox']['separation'])
        self.assertEqual(rec['perm']['sep_p'], 1.0)


if __name__ == '__main__':
    unittest.main()
