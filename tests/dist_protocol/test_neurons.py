# -*- coding: utf-8 -*-
"""neurons.v1 与 runner_v3 的 MLP 中间单元采集：合成模式写出 G、V（不进上传目录），植入的 +×+ 与 −×−（双负区）
单元能被选择性统计找回并标出符号区，与因子无关的伪标签负对照不入选，文法校验 mlp_units。合成数据只用于验证方法。"""
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
os.environ.setdefault('AI2050_NATIVE_DIR', 'off')

from shared.dist_protocol import grammar, neurons as N  # noqa: E402

FAST = {'n_perm': 100}


def gr_q_spec():
    spec = json.loads((ROOT / 'shared/dist_protocol/grammars/GR-Q.json').read_text(encoding='utf-8'))
    spec['mlp_units'] = {'readouts': ['final', 'probe_answer'], 'layers': 'all'}
    return spec


def run_synthetic(tmp):
    spec = gr_q_spec()
    corpus = grammar.generate(spec)
    corpus.update({'tm_id': 'TG-GR-Q', 'version': 1, 'synthetic_dim': 16, 'synthetic_layers': 4, 'synthetic_d_ff': 24})
    run = tmp / 'run'
    run.mkdir(parents=True)
    shutil.copy(ROOT / 'shared/dist_protocol/runner_v3.py', run / 'runner.py')
    (run / 'corpus.json').write_text(json.dumps(corpus, ensure_ascii=False), encoding='utf-8')
    env = dict(os.environ, AI2050_SHARD='0/1', AI2050_NATIVE_DIR=str(tmp / 'native'))
    subprocess.run([sys.executable, str(run / 'runner.py'), '--mode', 'synthetic', '--out', str(run / 'o')],
                   check=True, env=env, capture_output=True)
    summary = json.loads((run / 'o' / 'summary.json').read_text(encoding='utf-8'))
    return spec, corpus, summary, run


class MlpUnitsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.spec, cls.corpus, cls.summary, cls.run_dir = run_synthetic(cls.tmp)
        cls.items = {it['id']: it for it in cls.corpus['items'] if not it.get('canary')}
        cls.field = N.MlpField([cls.summary['native_field']['dir']])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_capture_written_locally_not_uploaded(self):
        m = self.summary['mlp_units']
        self.assertEqual(m['readouts'], ['final', 'probe_answer'])
        self.assertEqual(m['d_ff'], 24)
        self.assertFalse(m['native']['uploaded'])
        self.assertFalse((self.run_dir / 'o' / 'mlp').exists())
        self.assertEqual(len(m['double_negative_share_strong']), 2)
        self.assertEqual(self.field.d_ff, 24)
        self.assertEqual(len(self.field.ids), len(self.items))
        G, V = self.field.GV('final', self.field.blocks[0])
        self.assertEqual(G.shape, (len(self.items), 24))

    def test_planted_units_recovered_with_regimes(self):
        res = N.selectivity(self.field, self.items, 'mood', 'decl', 'interrog', criteria=FAST)
        self.assertEqual(res['unit'], 'frame')                       # 语气是句式级因子：按句式整体置换
        for t in res['tags']:
            sel = {u['unit']: u for u in t['units']}
            self.assertIn(0, sel, t)
            self.assertIn(1, sel, t)
            self.assertLessEqual(len(sel) - 2, 1)                    # 噪声单元至多偶然入选 1 个
            self.assertEqual(sel[0]['pos_level'], 'decl')
            self.assertGreater(sel[0]['pole_regime']['++'], 0.9)
            self.assertGreater(sel[0]['strong_regime']['++'], 0.9)
            self.assertGreater(sel[1]['pole_regime']['--'], 0.9)     # 双负区：G<0 且 V<0 给出正输出
            self.assertEqual(sel[1]['pos_level'], 'decl')

    def test_negative_control_pseudo_label(self):
        # 与语气无关的伪因子：按填充词编号奇偶分组（每个填充在两种语气中都出现）
        fake = {}
        for i, it in self.items.items():
            k = int(it['filler_id'].split('-')[-1])
            fake[i] = {**it, 'cond': {'pseudo': 'odd' if k % 2 else 'even'}}
        res = N.selectivity(self.field, fake, 'pseudo', 'odd', 'even', criteria=FAST)
        self.assertEqual(res['unit'], 'filler')
        self.assertEqual(sum(t['n_selected'] for t in res['tags']), 0)

    def test_residual_contribution(self):
        # 合成 down_proj：单元 0 的列范数为 10，其余为 1；植入单元 0 应在贡献排序中居首，分解的份额之和为 1
        rng = np.random.default_rng(0)
        Ws = {}
        for b in self.field.blocks:
            W = rng.standard_normal((16, 24))
            W /= np.linalg.norm(W, axis=0, keepdims=True)
            W[:, 0] *= 10.0
            Ws[b] = W
        res = N.selectivity(self.field, self.items, 'mood', 'decl', 'interrog', criteria=FAST, down=Ws.__getitem__)
        self.assertTrue(res['contribution'])
        for t in res['tags']:
            self.assertEqual(t['units_by_contrib'][0]['unit'], 0)
            self.assertAlmostEqual(sum(t['share_by_regime_all'].values()), 1.0, places=6)
            self.assertGreater(t['share_by_regime_all']['++'], 0.5)       # 单元 0（+×+）写入了大部分残差差异
            self.assertIn('--', t['top_contrib_dominant_regime'])
        plain = N.selectivity(self.field, self.items, 'mood', 'decl', 'interrog', criteria=FAST)
        self.assertNotIn('units_by_contrib', plain['tags'][0])

    def test_grammar_validation(self):
        self.assertEqual(grammar.validate(gr_q_spec()), [])
        spec = gr_q_spec(); spec['mlp_units']['readouts'] = ['cue']
        self.assertTrue(any('mlp_units.readouts' in e for e in grammar.validate(spec)))
        spec = gr_q_spec(); spec['mlp_units']['layers'] = [-1]
        self.assertTrue(any('mlp_units.layers' in e for e in grammar.validate(spec)))
        self.assertIn('mlp_units', grammar.generate(gr_q_spec(), include_canary=False))
        plain = gr_q_spec(); plain.pop('mlp_units')
        self.assertNotIn('mlp_units', grammar.generate(plain, include_canary=False))


if __name__ == '__main__':
    unittest.main()
