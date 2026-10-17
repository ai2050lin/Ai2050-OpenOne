# -*- coding: utf-8 -*-
"""entangle.v1 方法验证：植入「低秩纠缠 + 由线索门控的解纠缠」的合成场能被找回，纯加性负对照不报纠缠；
逐名词行为门与平衡检查；ENT-XF 材料；runner_v3 部件采集（合成模式，及可用时的极小随机 Qwen3 真实前向闭合）。
合成数据只用于验证方法，不代表任何真实模型。"""
import importlib.util
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

from shared.dist_protocol import entangle as E, grammar  # noqa: E402

PAIRS = [('color', 'lang'), ('color', 'task'), ('cat', 'lang'), ('lang', 'task'), ('polarity', 'task'),
         ('cat', 'polarity'), ('noun', 'lang')]
ALIGN = {'color': 'task=color'}
FAST = dict(n_perm=100, n_split=4, n_boot=100)


def _by(res):
    return {(t['readout'], t['hs_index']): t['pairs'] for t in res['tags']}


class PlantedRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.field, cls.design, cls.truth = E.synthetic_field(seed=0, planted=True)
        cls.res = E.analyze(cls.field, cls.design, PAIRS, task_alignment=ALIGN, **FAST)
        cls.tab = _by(cls.res)
        cls.att_cl = E.attribute(cls.field, cls.design, ('color', 'lang'), 'target', task_alignment=ALIGN, n_split=4)
        cls.att_ct = E.attribute(cls.field, cls.design, ('color', 'task'), 'target', task_alignment=ALIGN, n_split=4)

    def test_criteria_returned(self):
        self.assertEqual(self.res['criteria'], E.CRITERIA)
        self.assertEqual(self.res['schema'], 'entangle.v1')

    def test_entanglement_appears_after_entangler(self):
        for li in range(0, self.truth['entangle_from_hs']):
            self.assertNotEqual(self.tab[('target', li)]['color×lang']['verdict'], 'entangled', li)
        for li in range(self.truth['entangle_from_hs'], self.truth['n_blocks'] + 1):
            p = self.tab[('target', li)]['color×lang']
            self.assertEqual(p['verdict'], 'entangled', li)
            self.assertLessEqual(p['p']['I'], E.CRITERIA['alpha'])
        p4 = self.tab[('target', 4)]['color×lang']
        self.assertEqual(p4['E2']['signif_rank'], self.truth['planted_rank'])     # 植入秩 2
        self.assertGreater(p4['E1'], 0.25)

    def test_disentangler_reduces_interaction(self):
        e4 = self.tab[('target', 4)]['color×lang']['E1']
        e6 = self.tab[('target', 6)]['color×lang']['E1']
        self.assertLess(e6, e4)                                                     # task=color 时被抵消

    def test_new_directions_vs_gain_modulation(self):
        cl = self.tab[('target', 4)]['color×lang']
        ct = self.tab[('target', 6)]['color×task']
        self.assertLess(cl['E3']['frac_I_in_main'], 0.3)                          # 纠缠写在新方向
        self.assertGreater(ct['E3']['frac_I_in_main'], 0.5)                       # 门控 = 主效应方向上的增益
        self.assertTrue(cl['rotated'])
        self.assertGreaterEqual(ct['E4']['A_given_B']['gain_ratio'], E.CRITERIA['gain_ratio_min'])

    def test_interference_and_gating(self):
        self.assertTrue(self.tab[('target', 4)]['color×lang']['interfered'])
        self.assertTrue(self.tab[('target', 6)]['color×task']['gated'])
        self.assertFalse(self.tab[('target', 3)]['color×lang']['interfered'])

    def test_token_identity_baseline(self):
        nl0 = self.tab[('target', 0)]['noun×lang']
        self.assertEqual(nl0['verdict'], 'token_identity')
        for li in range(0, 7):
            self.assertNotEqual(self.tab[('target', li)]['noun×lang']['verdict'], 'entangled')
        self.assertLess(self.tab[('target', 6)]['noun×lang']['E1'], nl0['E1'])   # 语言无关概念码稀释 token 身份

    def test_attribution_closure_and_shares(self):
        for a in (self.att_cl, self.att_ct):
            for v in a['closure_rel_err'].values():
                self.assertLess(v, 1e-8)
            tot = sum(e['share_I'] for e in a['components'])
            self.assertAlmostEqual(tot, 1.0, places=6)

    def test_attribution_finds_planted_components(self):
        self.assertIn(self.truth['entangler'], [e['component'] for e in self.att_cl['entanglers']])
        self.assertIn(self.truth['disentangler'], [e['component'] for e in self.att_cl['disentanglers']])
        self.assertIn(self.truth['disentangler'], [e['component'] for e in self.att_ct['gaters']])
        self.assertEqual(self.att_cl['top_by_share_I'][0]['component'], self.truth['entangler'])
        heads = {e['component'] for e in self.att_cl['entanglers'] + self.att_cl['disentanglers']}
        self.assertLessEqual(heads, {self.truth['entangler'], self.truth['disentangler']})

    def test_findings(self):
        kinds = {f['id']: f['kind'] for f in self.res['findings']}
        self.assertEqual(kinds['entangled:color×lang'], 'supported')
        self.assertEqual(kinds['gated:color×task'], 'supported')
        self.assertEqual(kinds['additive:cat×lang'], 'unsupported')
        self.assertEqual(kinds['token:noun×lang'], 'warning')


class AdditiveNegativeControlTests(unittest.TestCase):
    """纯加性（含共享方向与 token 身份）的合成场：任何读出 × 层、任何因子对都不应报纠缠 / 干扰 / 门控，也不应标出部件。"""

    def test_no_entanglement_reported(self):
        for seed in (0, 1):
            field, design, _ = E.synthetic_field(seed=seed, planted=False)
            res = E.analyze(field, design, PAIRS, task_alignment=ALIGN, **FAST)
            for t in res['tags']:
                for k, p in t['pairs'].items():
                    self.assertNotEqual(p.get('verdict'), 'entangled', (seed, k, t['hs_index']))
                    self.assertFalse(p.get('interfered'), (seed, k, t['hs_index']))
                    self.assertFalse(p.get('gated'), (seed, k, t['hs_index']))
            for pair in (('color', 'lang'), ('color', 'task')):
                a = E.attribute(field, design, pair, 'target', task_alignment=ALIGN, n_split=4)
                self.assertEqual(a['entanglers'] + a['disentanglers'] + a['gaters'], [], (seed, pair))


class GateAndBalanceTests(unittest.TestCase):
    def setUp(self):
        self.field, self.design, _ = E.synthetic_field(seed=3, planted=False, per_cat=9)

    def test_noun_gate_drops_whole_nouns(self):
        d = self.design
        corr = np.where(np.isin(d.labels['task'], ['color', 'size']), 1, -1)
        bad = sorted(set(d.noun[d.split == 'discovery']))[:2]
        corr[(d.noun == bad[0]) & (d.labels['task'] == 'color') & (d.labels['lang'] == 'en')] = 0
        corr[(d.noun == bad[1]) & (d.labels['task'] == 'size')] = 0
        d2, rep = E.gate_nouns(E.Design(**{**d.__dict__, 'correct': corr}))
        self.assertEqual(rep['dropped_nouns'], bad)
        self.assertFalse(d2.keep[np.isin(d2.noun, bad)].any())
        self.assertTrue(d2.keep[~np.isin(d2.noun, bad)].all())
        idx = np.where(d2.keep & (d2.split == 'discovery'))[0]
        L = E.layout(d2, 'lang', 'task', idx)
        self.assertEqual(len(L.units), len(set(d.noun[d.split == 'discovery'])) - 2)

    def test_per_attribute_gate_groups(self):
        d = self.design
        corr = np.where(np.isin(d.labels['task'], ['color', 'size']), 1, -1)
        nouns = sorted(set(d.noun[d.split == 'discovery']))
        col_bad, size_bad = nouns[0], nouns[1]
        corr[(d.noun == col_bad) & (d.labels['task'] == 'color') & (d.labels['lang'] == 'zh')] = 0
        corr[(d.noun == size_bad) & (d.labels['task'] == 'size')] = 0
        d2 = E.Design(**{**d.__dict__, 'correct': corr})
        pairs = [('color', 'task'), ('color', 'lang'), ('lang', 'task'), ('cat', 'lang')]
        groups = E.gate_groups(d2, pairs, 'per_attribute', {'color': 'task=color', 'size': 'task=size'})
        by = {g['name']: g for g in groups}
        self.assertEqual(sorted(by), ['all_probes', 'color'])
        self.assertEqual(by['color']['pairs'], [('color', 'task'), ('color', 'lang')])
        self.assertEqual(by['color']['report']['dropped_nouns'], [col_bad])      # 大小失败的名词留在颜色因子对里
        self.assertTrue(by['color']['design'].keep[d2.noun == size_bad].all())
        self.assertEqual(by['all_probes']['report']['dropped_nouns'], sorted([col_bad, size_bad]))
        # 旧策略：全部因子对共用 all_probes 门
        (g,) = E.gate_groups(d2, pairs, 'all_probes')
        self.assertEqual(g['report']['dropped_nouns'], sorted([col_bad, size_bad]))
        with self.assertRaises(ValueError):
            E.gate_groups(d2, pairs, 'per_attribute')
        res = E.analyze_gated(self.field, groups, n_perm=20, n_split=2, n_boot=10,
                              tags=[self.field.tags()[0]])
        gates = {k: p.get('gate') for k, p in res['tags'][0]['pairs'].items()}
        self.assertEqual(gates, {'color×task': 'color', 'color×lang': 'color', 'lang×task': 'all_probes',
                                 'cat×lang': 'all_probes'})

    def test_degenerate_readout_does_not_crash(self):
        d = self.design
        X = np.where((d.labels['lang'] == 'zh')[:, None], 1.0, -1.0) * np.ones((len(d.ids), 8))  # 句末 token 式的退化读出
        f = E.ArrayField({('final', 0): X})
        res = E.analyze(f, d, [('cat', 'lang'), ('lang', 'task')], n_perm=20, n_split=2, n_boot=10)
        p = res['tags'][0]['pairs']['lang×task']
        self.assertNotEqual(p.get('verdict'), 'entangled')

    def test_single_class_test_set_not_testable(self):
        rng = np.random.default_rng(0)
        Ztr, Zte = rng.standard_normal((20, 4)), rng.standard_normal((6, 4))
        ytr = np.array(['a'] * 10 + ['b'] * 10)
        self.assertIsNone(E._margins(Ztr, ytr, Zte, np.array(['a'] * 6), ['a', 'b'], True, Zte.mean(0)))
        self.assertIsNotNone(E._margins(Ztr, ytr, Zte, np.array(['a', 'b'] * 3), ['a', 'b'], True, Zte.mean(0)))

    def test_rare_attr_level_dropped_not_pair(self):
        d = self.design
        disc = d.split == 'discovery'
        c0 = sorted(set(d.noun[disc & (d.labels['color'] == 'c0')]))
        keep = d.keep & ~np.isin(d.noun, c0[1:])                   # c0 只剩 1 个名词
        d2 = E.Design(**{**d.__dict__, 'keep': keep})
        L = E.layout(d2, 'color', 'lang', np.where(keep & disc)[0])
        self.assertEqual(L.a_levels, ['c1', 'c2'])
        self.assertIn('c0', L.note)
        self.assertFalse(np.isin(d.noun[L.item_idx], c0).any())
        Y = E.cube(L, self.field.X('target', 3))
        self.assertEqual(Y.shape[:3], (len(L.units), 2, 2))

    def test_partial_cell_exclusion_removes_unbalanced_units(self):
        d = self.design
        cat0 = sorted(set(d.labels['cat']))[0]
        cell = f'cat={cat0}|lang=zh|polarity=aff|task=color'
        d2, rep = E.gate_nouns(d, policy='none', exclude_cells=[cell])
        self.assertGreater(rep['n_items_excluded_by_cells'], 0)
        idx = np.where(d2.keep & (d2.split == 'discovery'))[0]
        L = E.layout(d2, 'lang', 'task', idx)          # cat0 名词的 (zh, color) 格只剩否定句 → 这些名词整体退出
        kept_cats = {d.labels['cat'][np.where(d.noun == u)[0][0]] for u in L.units}
        self.assertNotIn(cat0, kept_cats)
        self.assertIn('单位完整', L.note)
        self.assertIsNone(E.layout(d2, 'cat', 'lang', idx[np.isin(d2.labels['cat'][idx], [cat0])]))


def _load_builder():
    spec = importlib.util.spec_from_file_location('build_ent_grammar', ROOT / 'scripts' / 'build_ent_grammar.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class EntMaterialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads((ROOT / 'shared/dist_protocol/grammars/ENT-XF.json').read_text(encoding='utf-8'))
        cls.builder = _load_builder()
        cls.rows = cls.builder.read_lexicon()
        cls.corpus = grammar.generate(cls.spec)

    def test_file_matches_builder(self):
        self.assertEqual(self.builder.build_spec(self.rows), self.spec)
        errors, _ = self.builder.check_material(self.spec, self.rows)
        self.assertEqual(errors, [])

    def test_fully_crossed_balanced(self):
        self.assertEqual(len(self.rows), 210)
        cats = {}
        for r in self.rows:
            cats[r['cat']] = cats.get(r['cat'], 0) + 1
        self.assertEqual(set(cats.values()), {30})
        items = [it for it in self.corpus['items'] if not it.get('canary')]
        self.assertEqual(len(items), 210 * 24)
        per = {}
        for it in items:
            per.setdefault(it['filler_id'], set()).add(it['frame_id'])
        self.assertTrue(all(len(v) == 24 for v in per.values()))
        self.assertTrue(all(v for v in self.corpus['visibility']['target'].values()))
        self.assertTrue(all(v for v in self.corpus['visibility']['final'].values()))
        self.assertEqual(self.corpus['components'], {'readouts': ['target', 'final'], 'layers': 'all'})

    def test_probes_and_splits(self):
        split_of = {}
        for it in self.corpus['items']:
            if it.get('canary'):
                continue
            self.assertEqual(split_of.setdefault(it['filler_id'], it['split']), it['split'])
            p = it.get('probe')
            if it['cond']['task'] == 'none':
                self.assertIsNone(p)
            else:
                self.assertTrue(p['contrast'])
                self.assertFalse(set(p['answers']) & set(p['contrast']))
        d = E.design_from_corpus(self.corpus, self.spec)
        self.assertEqual(d.kind['cat'], 'noun_attr')
        self.assertEqual(d.kind['color'], 'noun_attr')
        self.assertEqual(d.kind['lang'], 'frame')
        idx = np.where(d.split == 'discovery')[0]
        for A, B in self.spec['entangle']['pairs']:
            L = E.layout(d, A, B, idx)
            self.assertIsNotNone(L, (A, B))
            self.assertEqual(L.note, '', (A, B))

    def test_v2_answer_sets_and_gate(self):
        """v2：备选可接受颜色、表面形式、zh 大小 probe 的规范分词形式与按属性分开的行为门。"""
        self.assertEqual(self.spec['grammar_version'], 2)
        ent = self.spec['entangle']
        self.assertEqual(ent['noun_gate'], 'per_attribute')
        self.assertEqual(ent['gate_probes'], {'color': 'task=color', 'size': 'task=size'})
        probes = {}
        for it in self.corpus['items']:
            if it.get('probe') and not it.get('canary'):
                probes[(it['filler_id'], it['cond']['lang'], it['cond']['task'])] = it['probe']
        coco = probes[('fruit.coconut', 'zh', 'color')]
        self.assertEqual(coco['answers'], ['棕色', '褐色', '绿色'])
        self.assertNotIn('绿色', coco['contrast'])
        self.assertIn('灰色', coco['contrast'])
        jelly = probes[('aquatic.jellyfish', 'en', 'color')]
        self.assertEqual(jelly['answers'], [' white', ' transparent'])
        self.assertNotIn(' transparent', probes[('fruit.banana', 'en', 'color')]['contrast'])
        self.assertIn(' golden', probes[('fruit.banana', 'en', 'color')]['answers'])
        ele = probes[('mammal.elephant', 'zh', 'size')]
        self.assertTrue(ele['prompt'].endswith('通常'))
        self.assertEqual((ele['answers'], ele['contrast']), (['更大', '更高'], ['更小', '更矮']))
        mouse = probes[('mammal.mouse', 'en', 'size')]
        self.assertEqual((mouse['answers'], mouse['contrast']), ([' smaller', ' shorter'], [' bigger', ' larger', ' taller']))
        self.assertTrue(all(r['color'] not in r['color_alt'] and len(r['color_alt']) <= 2 for r in self.rows))
        grape = probes[('fruit.grape', 'zh', 'color')]
        self.assertEqual(grape['answers'], ['紫色', '绿色', '红色'])
        self.assertFalse({'紫色', '绿色', '红色'} & set(grape['contrast']))
        for (_, lang, task), p in probes.items():           # 可接受集与对照集不相交
            self.assertFalse(set(p['answers']) & set(p['contrast']))


def _run_runner(corpus, out_dir, native, shard='0/1', mode='synthetic', env_extra=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / 'shared/dist_protocol/runner_v3.py', out_dir / 'runner.py')
    (out_dir / 'corpus.json').write_text(json.dumps(corpus, ensure_ascii=False), encoding='utf-8')
    env = dict(os.environ, AI2050_SHARD=shard, AI2050_NATIVE_DIR=str(native), **(env_extra or {}))
    subprocess.run([sys.executable, str(out_dir / 'runner.py'), '--mode', mode, '--out', str(out_dir / 'o')],
                   check=True, env=env, capture_output=True)
    return json.loads((out_dir / 'o' / 'summary.json').read_text(encoding='utf-8'))


def _small_corpus(n_nouns=14, with_components=True):
    spec = json.loads((ROOT / 'shared/dist_protocol/grammars/ENT-XF.json').read_text(encoding='utf-8'))
    keep = []
    for cat in sorted({fl['levels']['cat'] for fl in spec['fillers']})[:2]:
        keep += [fl for fl in spec['fillers'] if fl['levels']['cat'] == cat][:n_nouns // 2]
    spec['fillers'] = keep
    if not with_components:
        spec.pop('components')
    corpus = grammar.generate(spec)
    corpus['synthetic_dim'] = 24
    corpus['synthetic_layers'] = 5
    return spec, corpus


class RunnerComponentTests(unittest.TestCase):
    def test_synthetic_components_close_and_load(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            spec, corpus = _small_corpus()
            s = _run_runner(corpus, tmp / 'run', tmp / 'native')
            c = s['components']
            self.assertEqual(c['readouts'], ['target', 'final'])
            self.assertEqual(c['blocks'], [0, 1, 2, 3])
            self.assertFalse(c['native']['uploaded'])
            self.assertFalse((tmp / 'run' / 'o' / 'components').exists())          # 部件不进上传目录
            shard = Path(s['native_field']['dir'])
            f = E.NativeField([str(shard)])
            self.assertEqual(f.blocks('target'), [0, 1, 2, 3])
            for r in ('target', 'final'):
                for b in f.blocks(r):
                    Z, M = f.comp(r, b)
                    delta = f.X(r, b + 1) - f.X(r, b)
                    recon = Z @ f.W_O(b).T + M
                    self.assertLess(float(np.abs(delta - recon).max()), 0.05)    # float16 存储误差
            f.post_norm_last = False
            d = E.design_from_corpus(corpus, spec, f.ids)
            a = E.attribute(f, d, ('lang', 'task'), 'final', n_split=2)
            self.assertLess(a['closure_rel_err']['I'], 0.05)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_no_components_backward_compatible(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            _, corpus = _small_corpus(with_components=False)
            self.assertNotIn('components', corpus)
            s = _run_runner(corpus, tmp / 'run', tmp / 'native')
            self.assertNotIn('components', s)
            self.assertFalse(list((tmp / 'native').rglob('components.json')))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_items_budget_respects_upload_cap(self):
        from shared.dist_protocol import runner_v3 as rv
        self.assertEqual(rv.items_budget({'shards': 4}), rv.ITEM_CORPUS_BUDGET_BYTES)
        self.assertEqual(rv.items_budget({'shards': 1}), rv.ITEM_SHARD_CAP_BYTES)    # 单分片试跑
        space, k = rv.item_projection(336, 74, 2560, rv.items_budget({'shards': 1}))
        self.assertLessEqual(336 * 74 * k * 2, rv.ITEM_SHARD_CAP_BYTES)

    def test_suffstats_budget_override(self):
        from shared.dist_protocol import runner_v3 as rv
        self.assertEqual(rv.suffstats_layer_subset(37, 2, 252, 2560), [0, 8, 16, 24, 32, 36])
        self.assertEqual(rv.suffstats_layer_subset(37, 2, 252, 2560, 56 << 20), [0, 36])


TOKENIZER_DIR = ROOT / 'models' / 'hf' / 'qwen3-4b'


def _has_torch():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except Exception:
        return False


@unittest.skipUnless(_has_torch() and (TOKENIZER_DIR / 'tokenizer.json').exists(),
                     '需要 torch / transformers 与本地 Qwen3 tokenizer')
class TinyQwenRealModeTests(unittest.TestCase):
    """极小随机初始化 Qwen3（2 层）+ 真实 tokenizer：真实模式的钩子采集与 hs 闭合。随机权重，无研究意义。"""

    def test_real_mode_component_closure(self):
        import torch
        from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
        tmp = Path(tempfile.mkdtemp())
        try:
            tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
            cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                              num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=256,
                              tie_word_embeddings=True)
            torch.manual_seed(0)
            model = Qwen3ForCausalLM(cfg)
            model.save_pretrained(tmp / 'model')
            tok.save_pretrained(tmp / 'model')
            spec, corpus = _small_corpus(n_nouns=2)
            corpus['items'] = [it for it in corpus['items'] if not it.get('canary')][:12] + \
                [it for it in corpus['items'] if it.get('canary')][:1]
            s = _run_runner(corpus, tmp / 'run', tmp / 'native', mode='real',
                            env_extra={'AI2050_RUNNER_MODEL': str(tmp / 'model'), 'AI2050_PRECISION': 'fp32'})
            self.assertEqual(s['status'], 'real', s.get('real_error'))
            c = s['components']
            self.assertEqual((c['n_heads'], c['head_dim'], c['blocks']), (4, 16, [0, 1]))
            self.assertLess(c['closure']['rel_err_max'], 1e-4)
            f = E.NativeField([s['native_field']['dir']])
            Z, M = f.comp('target', 0)
            self.assertEqual(Z.shape, (12, 64))
            recon = Z @ f.W_O(0).T + M
            self.assertLess(float(np.abs(f.X('target', 1) - f.X('target', 0) - recon).max()), 2e-2)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
