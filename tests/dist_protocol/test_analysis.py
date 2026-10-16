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


if __name__ == '__main__':
    unittest.main()
