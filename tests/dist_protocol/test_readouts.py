# -*- coding: utf-8 -*-
"""因果可见性检查、多读出位置 / 全层采集的数据格式（items.v2 / suffstats.v2）及向后兼容。
合成模式只用于验证协议与格式，不代表任何真实模型。"""
import copy
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
os.environ.setdefault('AI2050_NATIVE_DIR', 'off')   # 测试不在用户目录写原生全场

from shared.dist_protocol import analysis, grammar  # noqa: E402
from shared.dist_protocol.suffstats import SuffStats  # noqa: E402

GRAMMARS = sorted((ROOT / 'shared' / 'dist_protocol' / 'grammars').glob('*.json'))
RUNNER = ROOT / 'shared' / 'dist_protocol' / 'runner_v3.py'


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def base_spec(frames, readouts=None):
    fillers = [{'id': f'f{i}', 'levels': {'category': 'a' if i % 2 else 'b'},
                'slots': {'X': f'词{i}', 'Y': '甲类' if i % 2 else '乙类'}} for i in range(6)]
    spec = {'schema_version': grammar.SCHEMA_VERSION, 'family_id': 'T-VIS', 'factors': ['category', 'frame'],
            'frames': frames, 'fillers': fillers}
    if readouts:
        spec['readouts'] = readouts
    return spec


def runner_module():
    spec = importlib.util.spec_from_file_location('runner_v3_under_test', RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_synthetic(corpus, shards, env=None):
    tmp = Path(tempfile.mkdtemp())
    try:
        shutil.copy(RUNNER, tmp / 'runner.py')
        (tmp / 'corpus.json').write_text(json.dumps(corpus, ensure_ascii=False), encoding='utf-8')
        outs = []
        for i in range(shards):
            out = tmp / f'o{i}'
            subprocess.run([sys.executable, str(tmp / 'runner.py'), '--mode', 'synthetic', '--out', str(out)],
                           check=True, env=dict(os.environ, AI2050_SHARD=f'{i}/{shards}', **(env or {})),
                           capture_output=True)
            outs.append({p.name: p.read_bytes() for p in out.iterdir()})
        return outs
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


class CausalVisibilityTests(unittest.TestCase):
    def test_sentence_initial_target_is_invisible(self):
        # 反例：KN-ISA v1 的 decl / interrog —— 目标 X 在句首，两句式在 X 处前缀相同
        spec = base_spec([
            {'id': 'decl', 'levels': {'frame': 'decl'}, 'text': '{X}是一种{Y}。', 'target': 'X'},
            {'id': 'interrog', 'levels': {'frame': 'interrog'}, 'text': '{X}是不是一种{Y}？', 'target': 'X'}])
        errors = grammar.validate(spec)
        self.assertTrue(any('因子 frame 在读出位置不可见' in e for e in errors), errors)
        self.assertFalse(grammar.preview(spec)['ok'])

    def test_prefix_makes_factor_visible(self):
        # 正例：句式差异在目标之前
        spec = base_spec([
            {'id': 'decl', 'levels': {'frame': 'decl'}, 'text': '众所周知，{X}是一种{Y}。', 'target': 'X'},
            {'id': 'interrog', 'levels': {'frame': 'interrog'}, 'text': '请问{X}是不是一种{Y}？', 'target': 'X'}])
        self.assertEqual(grammar.validate(spec), [])
        vis = grammar.causal_visibility(spec)
        self.assertTrue(vis['matrix']['target']['frame'])
        self.assertTrue(vis['matrix']['target']['category'])
        self.assertEqual(vis['warnings'], [])

    def test_partial_visibility_warns_not_errors(self):
        # 目标处不可见、句末可见：不报错，但对 target 读出告警并在矩阵中标为不可见
        spec = base_spec([
            {'id': 'decl', 'levels': {'frame': 'decl'}, 'text': '{X}是一种{Y}。', 'target': 'X'},
            {'id': 'interrog', 'levels': {'frame': 'interrog'}, 'text': '{X}是不是一种{Y}？', 'target': 'X'}],
            readouts=['target', 'final'])
        self.assertEqual(grammar.validate(spec), [])
        pv = grammar.preview(spec)
        self.assertTrue(pv['ok'])
        self.assertFalse(pv['visibility']['matrix']['target']['frame'])
        self.assertTrue(pv['visibility']['matrix']['final']['frame'])
        self.assertTrue(any('读出位置 target 不可见' in w for w in pv['warnings']))
        corpus = grammar.generate(spec, include_canary=False)
        self.assertFalse(corpus['visibility']['target']['frame'])

    def test_label_and_answer_slot_before_target_warns(self):
        spec = base_spec([
            {'id': 'decl', 'levels': {'frame': 'decl'}, 'text': '众所周知，{X}是一种{Y}。', 'target': 'X'},
            {'id': 'topic', 'levels': {'frame': 'topic'}, 'text': '说到{Y}，{X}就是其中之一。', 'target': 'X',
             'probe': {'prompt': '{X}通常被归为', 'answer': '{Y}'}}])
        self.assertEqual(grammar.label_slots(spec), ['Y'])
        warns = grammar.causal_visibility(spec)['warnings']
        self.assertTrue(any('topic' in w and '{Y}' in w and '类别标签槽' in w for w in warns), warns)
        self.assertFalse(any('decl' in w for w in warns))

    def test_prefix_confound_warning(self):
        # 一方有句首前缀、另一方没有 → 告警（不报错）；等长前缀 → 不告警
        spec = base_spec([
            {'id': 'decl', 'levels': {'frame': 'decl'}, 'text': '{X}是{Y}。', 'target': 'X', 'cue': '是'},
            {'id': 'q', 'levels': {'frame': 'q'}, 'text': '我想问，{X}是{Y}吗？', 'target': 'X', 'cue': '是'}])
        self.assertEqual(grammar.validate(spec), [])
        vis = grammar.causal_visibility(spec)
        self.assertTrue(any('前缀 / 位置混杂' in w for w in vis['prefix_warnings']), vis)
        self.assertTrue(any('前缀 / 位置混杂' in w for w in grammar.preview(spec)['warnings']))
        spec['frames'][0]['text'] = '我想说，{X}是{Y}。'
        self.assertEqual(grammar.causal_visibility(spec)['prefix_warnings'], [])

    def test_library_grammars_fully_visible(self):
        for p in GRAMMARS:
            with self.subTest(p=p.name):
                spec = load(p)
                self.assertEqual(grammar.validate(spec), [])
                vis = grammar.causal_visibility(spec)
                self.assertEqual(vis['errors'], [])
                self.assertEqual(vis['warnings'], [])
                for r, row in vis['matrix'].items():
                    self.assertTrue(all(row.values()), f'{p.name} {r}: {row}')

    def test_readout_declarations_validated(self):
        spec = base_spec([
            {'id': 'decl', 'levels': {'frame': 'decl'}, 'text': '众所周知，{X}是一种{Y}。', 'target': 'X'},
            {'id': 'q', 'levels': {'frame': 'q'}, 'text': '请问{X}是不是一种{Y}？', 'target': 'X'}],
            readouts=['target', 'cue'])
        self.assertTrue(any('没有声明 cue' in e for e in grammar.validate(spec)))
        spec['frames'][0]['cue'] = '不存在的片段'
        spec['frames'][1]['cue'] = '是不是一种'
        self.assertTrue(any('cue 必须是' in e for e in grammar.validate(spec)))
        spec['readouts'] = ['target', 'probe_answer']
        self.assertTrue(any('没有 probe' in e for e in grammar.validate(spec)))
        spec['readouts'] = ['target', 'nowhere']
        self.assertTrue(any('readouts 必须是' in e for e in grammar.validate(spec)))


class ReadoutPositionTests(unittest.TestCase):
    def test_cue_and_last_occurrence_offsets(self):
        for p in GRAMMARS:
            corpus = grammar.generate(load(p), include_canary=False)
            for it in corpus['items']:
                self.assertEqual(it['text'][it['target_char']:it['target_char'] + len(it['target'])], it['target'])
                if 'cue' in it:
                    self.assertEqual(it['text'][it['cue_char']:it['cue_char'] + len(it['cue'])], it['cue'])
        role = grammar.generate(load(ROOT / 'shared/dist_protocol/grammars/GR-ROLE.json'), include_canary=False)
        for it in role['items']:
            self.assertEqual(it['target_char'], it['text'].rfind(it['target']))       # 句后复指的 N
        isa = grammar.generate(load(ROOT / 'shared/dist_protocol/grammars/KN-ISA.json'), include_canary=False)
        self.assertEqual(isa['layers'], ['all'])
        self.assertEqual(isa['readouts'], ['target', 'cue', 'final', 'probe_answer'])
        for it in isa['items']:
            self.assertLess(it['target_char'], it['cue_char'])                         # cue 在类别词之前、目标之后
            self.assertTrue(it['text'][it['cue_char'] + len(it['cue']):].startswith(it['probe']['answer']))


class MultiReadoutSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = grammar.generate(load(ROOT / 'shared/dist_protocol/grammars/KN-ISA.json'))
        cls.corpus.update({'tm_id': 'TM-TEST', 'synthetic_dim': 32, 'synthetic_layers': 5})
        cls.outs = run_synthetic(cls.corpus, 2)

    def test_items_v2_tags_and_lens(self):
        readouts = ['target', 'cue', 'final', 'probe_answer']
        z = np.load(__import__('io').BytesIO(self.outs[0]['items.npz']), allow_pickle=False)
        self.assertEqual(str(z['schema']), 'items.v2')
        tags = [str(t) for t in z['layer_tags']]
        self.assertEqual(tags, [f'{r}@L{i}' for r in readouts for i in range(5)])
        self.assertEqual([str(x) for x in z['tag_readout']], [r for r in readouts for _ in range(5)])
        self.assertEqual([int(x) for x in z['tag_layer']], list(range(5)) * 4)
        n = len(z['ids'])
        self.assertEqual(z['X'].shape[:2], (20, n))
        self.assertEqual(list(z['lens_layers']), list(range(5)))
        self.assertEqual(z['lens_next_logprob'].shape, (4, 5, n))
        self.assertEqual(z['lens_answer_logprob'].shape, (4, 5, n))
        self.assertEqual(z['lens_topk_ids'].shape, (4, 5, n, 5))
        self.assertEqual(z['lens_next_id'].shape, (4, n))
        self.assertIn('lens_vocab_text', z.files)

    def test_suffstats_v2_merge_and_summary(self):
        merged = None
        for o in self.outs:
            ss = SuffStats.from_npz_bytes(o['suffstats.npz'])
            self.assertEqual(ss.tag_readout[0], 'target')
            self.assertEqual(ss.tag_layer[:5], list(range(5)))
            merged = ss if merged is None else merged.merge(ss)
        n_study = sum(1 for it in self.corpus['items'] if not it.get('canary'))
        self.assertEqual(int(merged.n.sum()), n_study)
        again = SuffStats.from_npz_bytes(merged.to_npz_bytes())
        self.assertEqual(again.tag_readout, merged.tag_readout)
        s = json.loads(self.outs[0]['summary.json'])
        self.assertEqual(s['status'], 'synthetic')
        self.assertEqual(s['readouts'], ['target', 'cue', 'final', 'probe_answer'])
        self.assertEqual([x['hs_index'] for x in s['layer_plan']], list(range(5)))
        self.assertEqual(s['suffstats_layers'], [f'L{i}' for i in range(5)])
        self.assertIn('answer_logprob_mean', s['lens'])

    def test_analysis_loads_v2_items(self):
        items = analysis.load_items([o['items.npz'] for o in self.outs])
        n_study = sum(1 for it in self.corpus['items'] if not it.get('canary'))
        self.assertEqual(items.X.shape[1], n_study)
        self.assertEqual(len(items.layer_tags), 20)

    def test_embedding_layer_has_no_frame_signal_at_target(self):
        # 合成数据的 L0（嵌入层）在 target 处只有词身份：句式因子的 η² 应接近机会水平
        merged = None
        for o in self.outs:
            ss = SuffStats.from_npz_bytes(o['suffstats.npz'])
            merged = ss if merged is None else merged.merge(ss)
        from shared.dist_protocol.suffstats import summarize
        layers = {l['tag']: l for l in summarize(merged)['layers']}
        l0 = layers['target@L0']['by_split']['discovery']['factors']['frame']['excess']
        l2 = layers['target@L2']['by_split']['discovery']['factors']['frame']['excess']
        self.assertLess(l0, 0.05)
        self.assertGreater(l2, l0 + 0.05)


class NativeFieldAndProjectionTests(unittest.TestCase):
    """原生全场按层落盘（不上传）+ items 投影维数只由语料决定（分片大小不等也能合并）。"""

    @classmethod
    def setUpClass(cls):
        cls.native = Path(tempfile.mkdtemp())
        cls.corpus = grammar.generate(load(ROOT / 'shared/dist_protocol/grammars/KN-ATTR.json'))
        cls.corpus.update({'tm_id': 'TM-NATIVE', 'version': 3, 'synthetic_dim': 256, 'synthetic_layers': 3,
                           'item_budget_bytes': 300_000})        # 强制随机投影：12 标签 × 180 条 × 256 维 > 预算
        cls.shards = 7                                             # 180 条研究样本 → 分片 26/25 条，大小不等
        cls.outs = run_synthetic(cls.corpus, cls.shards, env={'AI2050_NATIVE_DIR': str(cls.native)})
        cls.rn = runner_module()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.native, ignore_errors=True)

    def items(self, o):
        return np.load(__import__('io').BytesIO(o['items.npz']), allow_pickle=False)

    def test_projection_fixed_by_corpus(self):
        sizes = {len(self.items(o)['ids']) for o in self.outs}
        self.assertGreater(len(sizes), 1)
        spaces = {str(self.items(o)['space']) for o in self.outs}
        self.assertEqual(spaces, {'jl128'})
        n_study = sum(1 for it in self.corpus['items'] if not it.get('canary'))
        self.assertEqual(self.rn.item_projection(n_study, 12, 256, 300_000), ('jl128', 128))
        merged = analysis.load_items([o['items.npz'] for o in self.outs])
        self.assertEqual(merged.X.shape, (12, n_study, 128))
        self.assertEqual(len(set(merged.ids)), n_study)

    def test_native_files_match_manifest(self):
        n_seen = 0
        for o in self.outs:
            s = json.loads(o['summary.json'])
            nf = s['native_field']
            d = Path(nf['dir'])
            self.assertTrue(str(d).startswith(str(self.native)))
            mbytes = (d / 'manifest.json').read_bytes()
            self.assertEqual(__import__('hashlib').sha256(mbytes).hexdigest(), nf['manifest_sha256'])
            m = json.loads(mbytes)
            z = self.items(o)
            ids = [str(i) for i in z['ids']]
            self.assertEqual(m['ids'], ids)
            self.assertEqual(m['readouts'], ['target', 'cue', 'final', 'probe_answer'])
            self.assertEqual(m['tag_layer'], [0, 1, 2] * 4)
            self.assertEqual([f['hs_index'] for f in m['files']], [0, 1, 2])
            for f in m['files']:
                data = (d / f['path']).read_bytes()
                self.assertEqual(__import__('hashlib').sha256(data).hexdigest(), f['sha256'])
                X = np.load(__import__('io').BytesIO(data))['X']
                self.assertEqual(list(X.shape), [4, len(ids), 256])
                self.assertEqual(f['shape'], [4, len(ids), 256])
                self.assertEqual(X.dtype, np.float16)
            # 上传的投影向量 = 原生向量 × 语料决定的投影矩阵（target@L1）
            R = self.rn.jl_matrix(self.rn.corpus_seed(self.corpus), 256, 128)
            X1 = np.load(d / 'L1.npz')['X'][0].astype(np.float32)
            tags = [str(t) for t in z['layer_tags']]
            np.testing.assert_allclose(z['X'][tags.index('target@L1')].astype(np.float32), X1 @ R, rtol=2e-2, atol=2e-2)
            self.assertFalse(any(p.endswith('.npz') and p.startswith('L') for p in o))      # 原生文件不在上传目录
            n_seen += len(ids)
        self.assertEqual(n_seen, sum(1 for it in self.corpus['items'] if not it.get('canary')))

    def test_native_can_be_disabled(self):
        out = run_synthetic(self.corpus, 3, env={'AI2050_NATIVE_DIR': 'off'})[0]
        self.assertEqual(json.loads(out['summary.json'])['native_field'], {'disabled': True})


class BackwardCompatTests(unittest.TestCase):
    def test_legacy_corpus_keeps_tags(self):
        spec = load(ROOT / 'shared/dist_protocol/grammars/GR-NEG.json')
        spec['layers'], spec['readouts'] = ['first', 'mid', 'last'], ['target']
        corpus = grammar.generate(spec)
        corpus.pop('readouts')                                    # 旧语料没有 readouts / visibility 字段
        corpus.pop('visibility')
        corpus['synthetic_dim'] = 16
        out = run_synthetic(corpus, 1)[0]
        ss = SuffStats.from_npz_bytes(out['suffstats.npz'])
        self.assertEqual(ss.layer_tags, ['first', 'mid', 'last'])
        z = np.load(__import__('io').BytesIO(out['items.npz']), allow_pickle=False)
        self.assertEqual([str(t) for t in z['layer_tags']], ['first', 'mid', 'last'])

    def test_suffstats_v1_roundtrip(self):
        ss = SuffStats(['first', 'last'], 4, ['a'])
        ss.add('discovery', {'a': 'x'}, [np.ones(4), np.zeros(4)])
        data = ss.to_npz_bytes()
        z = np.load(__import__('io').BytesIO(data), allow_pickle=False)
        self.assertEqual(str(z['schema']), 'suffstats.v1')
        back = SuffStats.from_npz_bytes(data)
        self.assertIsNone(back.tag_readout)
        self.assertEqual(int(back.n.sum()), 1)

    def test_layer_subset_and_snapshot(self):
        rn = runner_module()
        keep = rn.suffstats_layer_subset(37, 4, 80, 2560)
        self.assertEqual(keep[0], 0)
        self.assertEqual(keep[-1], 36)
        self.assertLessEqual(len(keep) * 4 * 80 * 2560 * 16, rn.SUFFSTATS_BUDGET_BYTES)
        self.assertEqual(keep, rn.suffstats_layer_subset(37, 4, 80, 2560))     # 确定性：各分片相同
        self.assertEqual(rn.suffstats_layer_subset(9, 4, 80, 32), list(range(9)))
        tmp = Path(tempfile.mkdtemp())
        try:
            (tmp / 'model.safetensors').write_bytes(b'w' * 10)
            (tmp / 'tokenizer.json').write_text('{}', encoding='utf-8')
            (tmp / 'notes.txt').write_text('ignored', encoding='utf-8')
            a = rn.model_snapshot(tmp)
            self.assertEqual(set(a['files']), {'model.safetensors', 'tokenizer.json'})
            self.assertEqual(a, rn.model_snapshot(tmp))
            (tmp / 'tokenizer.json').write_text('{"x":1}', encoding='utf-8')
            self.assertNotEqual(a['sha256'], rn.model_snapshot(tmp)['sha256'])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
