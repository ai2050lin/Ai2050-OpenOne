# -*- coding: utf-8 -*-
"""dist_protocol 单元测试（标准库 unittest + numpy，无需 GPU / fastapi）。

运行：python -m unittest discover -s tests/dist_protocol -v   （仓库根目录）
"""
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

from shared.dist_protocol import canary, grammar  # noqa: E402
from shared.dist_protocol.suffstats import SuffStats, decompose, replicate_consistency, summarize  # noqa: E402

GRAMMARS = sorted((ROOT / 'shared' / 'dist_protocol' / 'grammars').glob('*.json'))


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


class GrammarTests(unittest.TestCase):
    def test_library_valid(self):
        self.assertGreaterEqual(len(GRAMMARS), 4)
        for p in GRAMMARS:
            with self.subTest(p=p.name):
                self.assertEqual(grammar.validate(load(p)), [])
                pv = grammar.preview(load(p))
                self.assertTrue(pv['ok'], pv.get('errors'))
                self.assertGreater(pv['by_split']['lockbox'], 0)

    def test_deterministic_and_no_lexical_leak(self):
        spec = load(GRAMMARS[0])
        a, b = grammar.generate(spec), grammar.generate(spec)
        self.assertEqual(a['items'], b['items'])
        by_filler = {}
        for it in a['items']:
            if it.get('canary'):
                continue
            if it['split'] == grammar.FRAME_SPLIT:
                continue
            by_filler.setdefault(it['filler_id'], set()).add(it['split'])
        for fid, splits in by_filler.items():
            self.assertEqual(len(splits), 1, f'{fid} 跨划分泄漏: {splits}')

    def test_every_level_in_every_split(self):
        for p in GRAMMARS:
            spec = load(p)
            c = grammar.generate(spec, include_canary=False)
            for f in spec['factors']:
                for sp in grammar.SPLITS:
                    levels = {it['cond'][f] for it in c['items'] if it['split'] == sp}
                    all_levels = {it['cond'][f] for it in c['items'] if it['split'] != grammar.FRAME_SPLIT}
                    self.assertEqual(levels, all_levels, f'{p.name}:{f}:{sp}')

    def test_target_offset_points_at_slot(self):
        for p in GRAMMARS:
            for it in grammar.generate(load(p), include_canary=False)['items']:
                s = it['target_char']
                self.assertEqual(it['text'][s:s + len(it['target'])], it['target'])

    def test_validation_errors(self):
        spec = load(GRAMMARS[0])
        spec['frames'][0]['target'] = 'Q'
        self.assertTrue(any('target' in e for e in grammar.validate(spec)))
        spec = load(GRAMMARS[0])
        spec['splits']['heldout_frames'] = [f['id'] for f in spec['frames']]
        self.assertTrue(grammar.validate(spec))

    def test_pilot_limit(self):
        spec = load(GRAMMARS[0])
        c = grammar.generate(grammar.pilot_spec(spec), per_cell_limit=2, include_canary=False)
        counts = {}
        for it in c['items']:
            k = (it['split'], grammar.cond_key(it['cond']))
            counts[k] = counts.get(k, 0) + 1
        self.assertLessEqual(max(counts.values()), 2)


def make_ss(rng, conds, n_per, dim=24, layers=('first', 'last'), split='discovery', effect=None):
    ss = SuffStats(list(layers), dim, sorted(conds[0]))
    data = {}
    for cond in conds:
        for _ in range(n_per):
            base = effect(cond) if effect else 0
            vecs = [base + rng.standard_normal(dim) for _ in layers]
            ss.add(split, cond, vecs)
            data.setdefault(json.dumps(cond, sort_keys=True), []).append(vecs)
    return ss, data


class SuffStatsTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.conds = [{'a': a, 'b': b} for a in ('x', 'y', 'z') for b in ('p', 'q')]

    def test_merge_equals_single_pass(self):
        ss1, d1 = make_ss(self.rng, self.conds, 4)
        ss2, d2 = make_ss(self.rng, self.conds, 5)
        merged = ss1.merge(ss2)
        # 直接在全部样本上计算 η²(a)
        allv, labels = [], []
        for d in (d1, d2):
            for k, vs in d.items():
                for v in vs:
                    allv.append(v[1]); labels.append(json.loads(k)['a'])
        allv = np.stack(allv)
        g = allv.mean(0)
        sst = ((allv - g) ** 2).sum(0)
        ssa = sum((np.array(labels) == lv).sum() * (allv[np.array(labels) == lv].mean(0) - g) ** 2 for lv in set(labels))
        conds, n, s, q = merged.collapse(['discovery'])
        dec = decompose(conds, n, s[1], q[1], ['a', 'b'])
        np.testing.assert_allclose(dec['factors']['a']['eta2'], ssa / sst, rtol=1e-9, atol=1e-12)
        self.assertEqual(int(n.sum()), 6 * 9)

    def test_npz_roundtrip(self):
        ss, _ = make_ss(self.rng, self.conds, 3)
        back = SuffStats.from_npz_bytes(ss.to_npz_bytes())
        np.testing.assert_array_equal(back.s, ss.s)
        self.assertEqual(back.cells, ss.cells)
        self.assertEqual(replicate_consistency(ss, back), 0.0)

    def test_replication_detects_planted_effect(self):
        dirs = {lv: self.rng.standard_normal(24) * 3 for lv in ('x', 'y', 'z')}
        eff = lambda c: dirs[c['a']]  # noqa: E731
        disc, _ = make_ss(self.rng, self.conds, 8, effect=eff)
        lock, _ = make_ss(self.rng, self.conds, 8, effect=eff, split='lockbox')
        payload = summarize(disc.merge(lock))
        last = payload['layers'][-1]
        self.assertGreater(last['replication']['lockbox']['a']['mean_cos'], 0.9)
        self.assertLess(abs(last['replication']['lockbox']['b']['mean_cos'] or 0), 0.9)
        self.assertGreater(last['by_split']['discovery']['factors']['a']['excess'], 0.3)
        self.assertIsNotNone(last['rdm_agreement']['lockbox']['pearson'])

    def test_merge_rejects_mismatched_dims(self):
        a, _ = make_ss(self.rng, self.conds, 2, dim=8)
        b, _ = make_ss(self.rng, self.conds, 2, dim=9)
        with self.assertRaises(ValueError):
            a.merge(b)


class CanaryTests(unittest.TestCase):
    def test_compare(self):
        v = np.linspace(-1, 1, 64)
        ref = {'entries': {'c1': {'last': canary.summarize_vector(v)}}}
        same = {'entries': {'c1': {'last': canary.summarize_vector(v * (1 + 1e-3))}}}
        far = {'entries': {'c1': {'last': canary.summarize_vector(v * 1.2)}}}
        self.assertEqual(canary.compare(ref, same, 'bf16')['status'], 'ok')
        self.assertEqual(canary.compare(ref, far, 'bf16')['status'], 'drift')
        self.assertEqual(canary.compare(ref, same, 'fp32')['status'], 'drift')
        self.assertEqual(canary.evaluate(None, same, 'bf16')[0], 'reference')
        self.assertEqual(canary.evaluate(ref, {}, 'bf16')[0], 'missing')


class RunnerSyntheticTests(unittest.TestCase):
    def test_shards_merge_to_full_corpus(self):
        spec = load(ROOT / 'shared' / 'dist_protocol' / 'grammars' / 'KN-ISA.json')
        corpus = grammar.generate(spec)
        corpus['tm_id'] = 'TM-TEST'
        corpus['synthetic_dim'] = 32
        tmp = Path(tempfile.mkdtemp())
        try:
            shutil.copy(ROOT / 'shared' / 'dist_protocol' / 'runner_v3.py', tmp / 'runner.py')
            (tmp / 'corpus.json').write_text(json.dumps(corpus, ensure_ascii=False), encoding='utf-8')
            merged = None
            for i in range(3):
                env = dict(os.environ, AI2050_SHARD=f'{i}/3')
                out = tmp / f'out{i}'
                subprocess.run([sys.executable, str(tmp / 'runner.py'), '--mode', 'synthetic', '--out', str(out)],
                               check=True, env=env, capture_output=True)
                summary = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
                self.assertEqual(summary['status'], 'synthetic')
                self.assertEqual(summary['shard'], f'{i}/3')
                for f in ('suffstats.npz', 'behavior.json', 'canary.json'):
                    self.assertTrue((out / f).is_file())
                ss = SuffStats.from_npz_bytes((out / 'suffstats.npz').read_bytes())
                merged = ss if merged is None else merged.merge(ss)
            n_study = sum(1 for it in corpus['items'] if not it.get('canary'))
            self.assertEqual(int(merged.n.sum()), n_study)
            payload = summarize(merged)
            self.assertIn('lockbox', payload['layers'][-1]['replication'])
            self.assertGreater(payload['layers'][-1]['replication']['lockbox']['category']['mean_cos'], 0.5)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_smoke_mode_writes_no_stats(self):
        corpus = grammar.generate(load(GRAMMARS[0]))
        tmp = Path(tempfile.mkdtemp())
        try:
            shutil.copy(ROOT / 'shared' / 'dist_protocol' / 'runner_v3.py', tmp / 'runner.py')
            (tmp / 'corpus.json').write_text(json.dumps(corpus, ensure_ascii=False), encoding='utf-8')
            env = {k: v for k, v in os.environ.items() if k != 'AI2050_RUNNER_MODEL'}
            subprocess.run([sys.executable, str(tmp / 'runner.py'), '--out', str(tmp / 'o')], check=True,
                           env=env, capture_output=True)
            self.assertEqual(json.loads((tmp / 'o' / 'summary.json').read_text(encoding='utf-8'))['status'], 'smoke')
            self.assertFalse((tmp / 'o' / 'suffstats.npz').exists())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
