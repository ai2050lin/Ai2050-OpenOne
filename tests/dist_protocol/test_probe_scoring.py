# -*- coding: utf-8 -*-
"""行为门评分口径：probe 可接受答案集、对照集（auto:other / auto:same）、filler.answers_extra，
以及 runner 合成模式写出的 behavior.v1 字段。真实模式的完整答案对数概率需要模型，不在单元测试中运行。"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('AI2050_NATIVE_DIR', 'off')

from shared.dist_protocol import grammar  # noqa: E402

GRAMMARS = ROOT / 'shared' / 'dist_protocol' / 'grammars'


def load(name):
    return json.loads((GRAMMARS / f'{name}.json').read_text(encoding='utf-8'))


def study(corpus):
    return [it for it in corpus['items'] if not it.get('canary')]


class ProbeSetTests(unittest.TestCase):
    def test_auto_other_uses_other_categories_same_split(self):
        spec = load('KN-ISA')
        split_of = grammar.assign_splits(spec)
        by_id = {fl['id']: fl for fl in spec['fillers']}
        for it in study(grammar.generate(spec, include_canary=False)):
            p = it['probe']
            y = by_id[it['filler_id']]['slots']['Y']
            self.assertEqual(p['answers'], [y])
            self.assertNotIn(y, p['contrast'])
            pool = {fl['slots']['Y'] for fl in spec['fillers'] if split_of[fl['id']] == split_of[it['filler_id']]}
            self.assertTrue(set(p['contrast']) <= pool)          # 对照只来自同一划分
            self.assertEqual(len(p['contrast']), len(pool) - 1)

    def test_auto_same_and_answers_extra(self):
        spec = load('KN-ATTR')
        items = study(grammar.generate(spec, include_canary=False))
        ice = next(it for it in items if it['filler_id'] == next(
            fl['id'] for fl in spec['fillers'] if fl['slots']['X'] == '冰'))
        p = ice['probe']
        self.assertEqual(p['answers'][:3], ['冷', '冷色', '冷的'])
        self.assertIn('凉', p['answers'])                        # 近义答案可接受
        self.assertNotIn('凉', p['contrast'])                    # 且不进入对照
        self.assertFalse(set(p['answers']) & set(p['contrast']))
        attr = {fl['id']: fl['levels']['attr'] for fl in spec['fillers']}
        temp_vals = {fl['slots']['A'] for fl in spec['fillers'] if attr[fl['id']] == 'temp'}
        self.assertTrue({c.rstrip('色的') for c in p['contrast']} <= temp_vals)   # 只与同属性类型竞争

    def test_explicit_contrast_lists(self):
        items = study(grammar.generate(load('GR-NEG'), include_canary=False))
        neg = next(it for it in items if it['frame_id'] == 'neg-decl')
        self.assertEqual(neg['probe']['answers'], ['错', '错误'])
        self.assertEqual(neg['probe']['contrast'], ['对', '正确'])

    def test_legacy_probe_has_no_sets(self):
        spec = load('GR-NEG')
        for fr in spec['frames']:
            fr['probe'] = {'prompt': fr['probe']['prompt'], 'answer': fr['probe']['answer']}
        it = study(grammar.generate(spec, include_canary=False))[0]
        self.assertEqual(set(it['probe']), {'prompt', 'answer'})

    def test_validation(self):
        spec = load('GR-NEG')
        spec['frames'][0]['probe']['contrast'] = 'auto:nope'
        self.assertTrue(any('contrast' in e for e in grammar.validate(spec)))
        spec = load('GR-NEG')
        spec['frames'][0]['probe']['answers'] = []
        self.assertTrue(any('answers' in e for e in grammar.validate(spec)))
        spec = load('KN-ATTR')
        spec['fillers'][0]['answers_extra'] = 'x'
        self.assertTrue(any('answers_extra' in e for e in grammar.validate(spec)))
        spec = load('GR-NEG')
        spec['frames'][0]['probe']['answers'] = ['{Q}']
        self.assertTrue(any('缺少' in e for e in grammar.validate(spec)))


class SyntheticBehaviorTests(unittest.TestCase):
    def test_behavior_records_mode(self):
        corpus = grammar.generate(load('GR-NEG'))
        corpus.update({'tm_id': 'TM-PROBE', 'synthetic_dim': 16, 'synthetic_layers': 3})
        tmp = Path(tempfile.mkdtemp())
        try:
            shutil.copy(ROOT / 'shared/dist_protocol/runner_v3.py', tmp / 'runner.py')
            (tmp / 'corpus.json').write_text(json.dumps(corpus, ensure_ascii=False), encoding='utf-8')
            subprocess.run([sys.executable, str(tmp / 'runner.py'), '--mode', 'synthetic', '--out', str(tmp / 'o')],
                           check=True, capture_output=True, env=dict(os.environ, AI2050_SHARD='0/1'))
            b = json.loads((tmp / 'o' / 'behavior.json').read_text(encoding='utf-8'))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        self.assertTrue(b['items'])
        for rec in b['items']:
            self.assertEqual(rec['mode'], 'forced_choice')
            self.assertIn('margin', rec)
            self.assertIn('first_top1', rec)


if __name__ == '__main__':
    unittest.main()
