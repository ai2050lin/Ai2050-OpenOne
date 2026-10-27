# -*- coding: utf-8 -*-
"""语言谱图拼图（pieces.v1）：语言树校验与旧族映射、可读出判据与 analysis.findings 一致、
植入结构找回（等级 3）、打乱标签负对照、嵌入层基线（token 身份不升级、各样本相同时可升级、缺基线不升级）、
跨模型 / 跨族聚合为等级 4。合成数据只用于验证方法，不代表任何真实模型。"""
import copy
import json
import os
import re
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault('AI2050_NATIVE_DIR', 'off')

from shared.dist_protocol import analysis, pieces as PC  # noqa: E402
from test_analysis import synthetic_items  # noqa: E402


def _relabel(items):
    """合成运行的层标签是 target@first/mid/last：按层序改成 target@L0/L1/L2（L0 充当嵌入层基线）。"""
    items = copy.deepcopy(items)
    items.layer_tags = [f'target@L{i}' for i in range(len(items.layer_tags))]
    return items


def _piece(ext, factor='category', model='m1', family='KN-ISA'):
    return PC.make_piece(ext, family_id=family, tm_id=f'TG-{family}', factor=factor, item='KN.ISA', model=model)


class TreeTests(unittest.TestCase):
    def test_tree_valid_and_library_factors_mapped(self):
        tree = PC.load_tree()
        for f in sorted((ROOT / 'shared/dist_protocol/grammars').glob('*.json')):
            g = json.loads(f.read_text(encoding='utf-8'))
            for factor in g['factors']:
                item = PC.factor_item(tree, g['family_id'], factor, g.get('lang_items'))
                self.assertIsNotNone(item, f'{g["family_id"]}.{factor} 没有映射到语言项')

    def test_invalid_tree_reported(self):
        tree = PC.load_tree()
        bad = copy.deepcopy(tree)
        bad['items'][0]['category'] = 'NO.such'
        bad['family_map']['GR-Q'] = {'mood': 'NO.item'}
        errs = PC.validate_tree(bad)
        self.assertTrue(any('NO.such' in e for e in errs))
        self.assertTrue(any('NO.item' in e for e in errs))

    def test_grammar_lang_items_override(self):
        tree = PC.load_tree()
        self.assertEqual(PC.factor_item(tree, 'GR-Q', 'mood'), 'GR.Q')
        self.assertEqual(PC.factor_item(tree, 'GR-Q', 'mood', {'mood': 'RS.NEG'}), 'RS.NEG')
        self.assertIsNone(PC.factor_item(tree, 'NEW-FAM', 'x'))


class RouteTRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus, raw = synthetic_items()
        cls.items = _relabel(raw)
        cls.p = analysis.analyze(cls.items, cls.corpus['factors'], n_perm=100)
        cls.ext = PC.extract_analysis(cls.p)

    def test_readable_matches_findings(self):
        # pieces.layer_readable 复制了 analysis.findings 的判据：两者给出的可读出层必须相同
        for f in self.corpus['factors']:
            fd = next(x for x in self.p['findings'] if x['id'] == f'readable-{f}')
            if fd['kind'] == 'supported':
                listed = set(re.search(r' 在 (.+?) 层可读出', fd['text']).group(1).split('、'))
            else:
                listed = set()
            mine = {f'target@L{c["layer"]}' for c in self.ext['factors'][f]['target']['curve'] if c['readable']}
            self.assertEqual(mine, listed, f)

    def test_planted_category_recovered(self):
        piece = _piece(self.ext)
        ro = piece['readouts']['target']
        self.assertEqual(piece['level'], 3)
        self.assertEqual(ro['baseline']['state'], 'not_separable')   # 合成早层类别不可读出（L0 基线）
        self.assertIn(1, ro['readable_layers'])
        self.assertEqual(ro['peak']['layer'], 1)                     # 植入在中层
        self.assertIn('L1', PC.rule_text(piece))

    def test_negative_control_shuffled_labels(self):
        items = copy.deepcopy(self.items)
        rng = np.random.default_rng(1)
        fills = sorted(set(items.filler))
        cats = [items.cond[list(items.filler).index(f)]['category'] for f in fills]
        new = dict(zip(fills, rng.permutation(cats)))
        for i, f in enumerate(items.filler):
            items.cond[i] = dict(items.cond[i], category=str(new[f]))
        ext = PC.extract_analysis(analysis.analyze(items, self.corpus['factors'], n_perm=100))
        self.assertLess(_piece(ext)['level'], 3)

    def test_token_identity_baseline_caps_level(self):
        # 嵌入层已可分（把中层向量复制到 L0，模拟读出位置 token 不同）：不计为留出复现
        items = copy.deepcopy(self.items)
        items.X[0] = items.X[1].copy()
        ext = PC.extract_analysis(analysis.analyze(items, self.corpus['factors'], n_perm=100))
        piece = _piece(ext)
        self.assertTrue(piece['readouts']['target']['token_identity'])
        self.assertEqual(piece['readouts']['target']['baseline']['state'], 'separable')
        self.assertEqual(piece['level'], 2)
        self.assertIn('token 身份', PC.rule_text(piece))

    def test_identical_baseline_allows_level3(self):
        # 嵌入层上所有样本向量相同（读出 token 相同）：基线为 identical，中层可读出就是上下文计算
        items = copy.deepcopy(self.items)
        items.X[0][:] = items.X[0][0]
        ext = PC.extract_analysis(analysis.analyze(items, self.corpus['factors'], n_perm=100))
        piece = _piece(ext)
        self.assertEqual(piece['readouts']['target']['baseline']['state'], 'identical')
        self.assertEqual(piece['level'], 3)

    def test_missing_baseline_blocks_level3(self):
        p = copy.deepcopy(self.p)
        p['layers'] = [l for l in p['layers'] if l['tag'] != 'target@L0']
        piece = _piece(PC.extract_analysis(p))
        self.assertEqual(piece['readouts']['target']['baseline']['state'], 'missing')
        self.assertEqual(piece['level'], 2)

    def test_scatter_at_peak_layer(self):
        sc = self.ext['scatter']['category']
        self.assertEqual((sc['readout'], sc['layer']), ('target', 1))           # 峰值层（植入在中层）
        self.assertEqual(sc['tag'], 'target@L1')
        self.assertLessEqual(len(sc['pts']), PC.MAX_SCATTER)
        self.assertEqual(set(sc['levels']), {str(p[2]) for p in sc['pts']})
        self.assertIsNotNone(_piece(self.ext)['scatter'])

    def test_unavailable_analysis_gives_no_piece(self):
        self.assertIsNone(_piece(PC.extract_analysis({'available': False})))


class AssembleTests(unittest.TestCase):
    @staticmethod
    def _fake_ext(level):
        curve = [{'layer': 0, 'sep': 0.1, 'acc': 0.5, 'eta2': 0.0, 'readable': False},
                 {'layer': 1, 'sep': 2.0, 'acc': 1.0, 'eta2': 0.3, 'readable': level >= 3}]
        ro = {'level': level, 'readable_layers': [1] if level >= 3 else [], 'segments': [[1, 1]] if level >= 3 else [],
              'peak': {'layer': 1, 'sep': 2.0, 'acc': 1.0, 'rep': 0.9} if level >= 3 else None,
              'baseline': {'layer': 0, 'state': 'not_separable', 'acc': 0.5, 'sep': 0.1}, 'token_identity': False,
              'frame_general': None, 'lexical_layers': [], 'length_layers': [], 'curve': curve}
        return {'factors': {'category': {'target': ro}}}

    def test_cross_model_level4_and_single_model_level3(self):
        tree = PC.load_tree()
        one = [_piece(self._fake_ext(3), model='m1')]
        out = PC.assemble(tree, one, {})
        self.assertEqual(out['items']['KN.ISA']['status']['read'], 3)
        two = one + [_piece(self._fake_ext(3), model='m2')]
        out = PC.assemble(tree, two, {})
        st = out['items']['KN.ISA']['status']
        self.assertEqual(st['read'], 4)
        self.assertTrue(st['cross_model'])
        weak = one + [_piece(self._fake_ext(2), model='m2')]             # 另一个模型只到 2：不算跨模型
        self.assertEqual(PC.assemble(tree, weak, {})['items']['KN.ISA']['status']['read'], 3)

    def test_cross_family_within_model(self):
        tree = PC.load_tree()
        ps = [_piece(self._fake_ext(3), family='KN-ISA'), _piece(self._fake_ext(3), family='KN-OTHER')]
        st = PC.assemble(tree, ps, {})['items']['KN.ISA']['status']
        self.assertTrue(st['cross_family'])
        self.assertEqual(st['by_model']['m1']['read'], 4)

    def test_tree_status_overlay_and_unmapped(self):
        tree = PC.load_tree()
        ps = [_piece(self._fake_ext(3)), PC.make_piece(self._fake_ext(3), family_id='X', tm_id='TG-X', factor='category',
                                                      item=None, model='m1')]
        sk = {'m1': {'n_layers': 2}}
        out = PC.assemble(tree, ps, sk)
        brief = {it['id']: it['status'] for it in out['tree']['items']}
        self.assertEqual(brief['KN.ISA']['read'], 3)
        self.assertEqual(brief['RS.COND']['read'], 0)
        self.assertTrue(brief['RS.COND']['planned'])
        self.assertEqual(len(out['tree']['unmapped']), 1)               # 没有语言项的 piece 单独列出，不计入覆盖
        ov = out['overlays']['m1']
        self.assertEqual(ov['n_layers'], 2)
        self.assertEqual([r['item'] for r in ov['route_T']], ['KN.ISA'])
        self.assertEqual(ov['route_T'][0]['readable'], [False, True])
        self.assertEqual(ov['route_N'], [])

    def test_skeleton_from_summary(self):
        s = {'n_hidden_states': 37, 'dim': 2560, 'precision': 'bf16', 'model_snapshot': {'sha256': 'ab' * 32},
             'mlp_units': {'d_ff': 9728}, 'final_layer_norm': 'post_final_norm'}
        sk = PC.skeleton_from_summary('qwen3-4b|cfg|bf16', s, [{'model_key': 'abababababababab-bf16', 'run': 'r', 'n_tokens': 5}])
        self.assertEqual((sk['n_layers'], sk['d_model'], sk['d_ff']), (36, 2560, 9728))
        self.assertEqual((sk['census_key'], sk['census_run']), ('abababababababab-bf16', 'r'))
        self.assertIsNone(PC.skeleton_from_summary('x', {})['n_layers'])   # 缺字段留空，不猜
        cfg = {'num_attention_heads': 32, 'num_key_value_heads': 8, 'head_dim': 128, 'private': 'x'}
        sk2 = PC.skeleton_from_summary('qwen3-4b|cfg|bf16', s, [{'model_key': 'abababababababab-bf16', 'run': 'r', 'n_tokens': 5}],
                                       census_config=lambda mk, run: cfg if (mk, run) == ('abababababababab-bf16', 'r') else None)
        self.assertEqual(sk2['config'], {'num_attention_heads': 32, 'num_key_value_heads': 8, 'head_dim': 128})
        self.assertIsNone(sk['config'])                                     # 没有普查 config：注意力头数等留空

    def test_examples_minimal_pairs_from_discovery_only(self):
        def it(fill, mood, frame, split='discovery'):
            return {'filler_id': fill, 'cond': {'mood': mood}, 'frame_id': frame, 'split': split, 'text': f'{fill}-{mood}-{frame}',
                    'probe': {'prompt': 'p', 'answers': ['a'], 'contrast': ['b']}}
        items = [it('f1', 'decl', 'A'), it('f1', 'interrog', 'B'), it('f2', 'decl', 'A'),          # f2 只有一个水平：不是最小对
                 it('f3', 'decl', 'A', 'lockbox'), it('f3', 'interrog', 'B', 'lockbox')]            # 留出材料不展示
        ex = PC.examples_of(items, 'mood')
        self.assertEqual(ex['levels'], ['decl', 'interrog'])
        self.assertEqual({r['filler'] for r in ex['rows']}, {'f1'})
        self.assertEqual(set(ex['probes']), {'decl', 'interrog'})
        self.assertTrue(ex['minimal_pairs'])

    def test_examples_for_filler_level_factor(self):
        # 名词类别这类填充级因子：每个词只属于一个水平，没有最小对；每个水平各取几句（只取 discovery）
        items = [{'filler_id': f'{c}{i}', 'cond': {'cat': c}, 'frame_id': 'A', 'split': 'discovery', 'text': f'{c}{i}'}
                 for c in ('animal', 'tool') for i in range(6)] + [{'filler_id': 'z', 'cond': {'cat': 'tool'}, 'frame_id': 'A', 'split': 'lockbox', 'text': 'z'}]
        ex = PC.examples_of(items, 'cat')
        self.assertFalse(ex['minimal_pairs'])
        self.assertEqual({r['level'] for r in ex['rows']}, {'animal', 'tool'})
        self.assertNotIn('z', {r['filler'] for r in ex['rows']})
        self.assertEqual(sum(1 for r in ex['rows'] if r['level'] == 'animal'), 4)

    def test_examples_and_selectivity_attached_to_items(self):
        tree = PC.load_tree()
        ex = {('TG-GR-Q', 'mood'): ('GR-Q', {'levels': ['decl', 'interrog'], 'rows': [], 'probes': {}})}
        sel = [{'file': 'GR-Q-v10.json', 'item': 'GR.Q', 'by_readout': {}}, {'file': 'Z.json', 'item': None, 'by_readout': {}}]
        out = PC.assemble(tree, [], {}, ex, sel)
        self.assertEqual([e['tm_id'] for e in out['items']['GR.Q']['examples']], ['TG-GR-Q'])
        self.assertEqual([s['file'] for s in out['items']['GR.Q']['selectivity']], ['GR-Q-v10.json'])
        self.assertEqual(out['items']['KN.ISA']['selectivity'], [])


if __name__ == '__main__':
    unittest.main()
