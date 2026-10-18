# -*- coding: utf-8 -*-
"""成对行为门（prereg.gate_mode=paired）：文法校验，植入"区分 + 整体偏移"后成对口径找回区分而绝对口径失败，
无区分时成对口径不通过（负对照），答案集不互换的句子对计为无效。"""
import hashlib
import json
import os
import random
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.dist_protocol import grammar  # noqa: E402
from shared.dist_protocol import pipeline  # noqa: E402
from shared.dist_protocol.pipeline import paired_behavior  # noqa: E402

GRAMMARS = ROOT / 'shared' / 'dist_protocol' / 'grammars'


def gr_q(mode='paired'):
    spec = json.loads((GRAMMARS / 'GR-Q.json').read_text(encoding='utf-8'))
    spec['prereg'] = {**spec.get('prereg', {}), 'gate_mode': mode}
    if mode == 'paired':
        spec['prereg']['gate_factor'] = 'mood'
    else:
        spec['prereg'].pop('gate_factor', None)
    return spec


def behavior(corpus, offset, delta, noise, seed=0):
    """u = log P(不确定) − log P(确定) = offset ± delta + 噪声（疑问 +delta，陈述 −delta）；
    疑问句的可接受答案是「不确定」，故 m = u；陈述句 m = −u。"""
    rng = random.Random(seed)
    out = []
    for it in corpus['items']:
        if it.get('canary') or not it.get('probe'):
            continue
        q = it['cond']['mood'] == 'interrog'
        u = offset + (delta if q else -delta) + rng.gauss(0, noise)
        m = u if q else -u
        out.append({'id': it['id'], 'cell': grammar.cond_key(it['cond']), 'mode': 'forced_choice',
                    'correct': m > 0, 'margin': m})
    return out


def absolute(items):
    acc = {}
    for b in items:
        r = acc.setdefault(b['cell'], [0, 0])
        r[0] += b['correct']; r[1] += 1
    return {c: a / n for c, (a, n) in acc.items()}


class PairedGateValidationTests(unittest.TestCase):
    def test_valid_and_invalid_specs(self):
        self.assertEqual(grammar.validate(gr_q('paired')), [])
        self.assertEqual(grammar.validate(gr_q('absolute')), [])
        spec = gr_q('paired'); spec['prereg']['gate_mode'] = 'pairwise'
        self.assertTrue(any('gate_mode' in e for e in grammar.validate(spec)))
        spec = gr_q('paired'); spec['prereg']['gate_factor'] = 'nope'
        self.assertTrue(any('gate_factor' in e for e in grammar.validate(spec)))
        spec = gr_q('absolute'); spec['prereg']['gate_factor'] = 'mood'
        self.assertTrue(any('gate_factor' in e for e in grammar.validate(spec)))
        spec = gr_q('paired'); spec['frames'][0]['probe']['contrast'] = ['确定', '是']
        self.assertTrue(any('互换' in e for e in grammar.validate(spec)))
        spec = gr_q('paired'); spec['frames'][0]['probe'].pop('contrast')
        self.assertTrue(any('contrast' in e for e in grammar.validate(spec)))

    def test_gate_mode_changes_design_sha(self):
        self.assertNotEqual(grammar.spec_sha(gr_q('paired')), grammar.spec_sha(gr_q('absolute')))


class PairedGateScoringTests(unittest.TestCase):
    def setUp(self):
        self.corpus = grammar.generate(gr_q('paired'), include_canary=True)

    def test_planted_discrimination_with_offset(self):
        # 模型清楚区分语气（delta=2），但整体偏向「确定」（offset=−5）：绝对口径疑问格失败，成对口径通过
        items = behavior(self.corpus, offset=-5.0, delta=2.0, noise=0.5)
        ab = absolute(items)
        self.assertLess(ab['mood=interrog'], 0.1)
        self.assertGreater(ab['mood=decl'], 0.9)
        pr = paired_behavior(self.corpus['items'], items, 'mood')
        for cell in ('mood=interrog', 'mood=decl'):
            self.assertGreater(pr[cell]['n'], 0)
            self.assertEqual(pr[cell]['invalid'], 0)
            self.assertGreater(pr[cell]['correct'] / pr[cell]['n'], 0.95)

    def test_negative_control_no_discrimination(self):
        # 不区分语气（delta=0），只有噪声：成对口径接近机会水平，低于预注册阈值 0.7
        items = behavior(self.corpus, offset=0.0, delta=0.0, noise=1.0, seed=3)
        pr = paired_behavior(self.corpus['items'], items, 'mood')
        for cell in ('mood=interrog', 'mood=decl'):
            acc = pr[cell]['correct'] / pr[cell]['n']
            self.assertLess(acc, 0.65)
            self.assertGreater(acc, 0.35)

    def test_pairs_stay_within_filler_and_split(self):
        items = behavior(self.corpus, offset=0.0, delta=1.0, noise=0.1)
        pr = paired_behavior(self.corpus['items'], items, 'mood')
        meta = {it['id']: it for it in self.corpus['items'] if not it.get('canary')}
        expect = 0
        groups = {}
        for b in items:
            it = meta[b['id']]
            groups.setdefault((it['split'], it['filler_id']), {}).setdefault(it['cond']['mood'], 0)
            groups[(it['split'], it['filler_id'])][it['cond']['mood']] += 1
        for g in groups.values():
            expect += g.get('interrog', 0) * g.get('decl', 0)
        self.assertEqual(pr['mood=interrog']['n'], expect)
        self.assertEqual(pr['mood=decl']['n'], expect)

    def test_unswapped_answer_sets_are_invalid(self):
        items = behavior(self.corpus, offset=0.0, delta=1.0, noise=0.1)
        corpus = json.loads(json.dumps(self.corpus))
        for it in corpus['items']:
            if it.get('probe') and it['cond']['mood'] == 'decl':
                it['probe']['contrast'] = ['不确定', '没有']
        pr = paired_behavior(corpus['items'], items, 'mood')
        self.assertEqual(pr['mood=interrog']['n'], 0)
        self.assertGreater(pr['mood=interrog']['invalid'], 0)

    def test_missing_margin_is_invalid(self):
        items = behavior(self.corpus, offset=0.0, delta=1.0, noise=0.1)
        for b in items:
            b.pop('margin')
        pr = paired_behavior(self.corpus['items'], items, 'mood')
        self.assertEqual(pr['mood=decl']['n'], 0)


class PairedGateAggregateTests(unittest.TestCase):
    """端到端：合成 runner 产物换上植入的 behavior.json，经 pipeline.aggregate 按成对口径判定行为门。"""

    def setUp(self):
        from tests.dist_protocol.test_pipeline_db import SCHEMA
        os.environ.setdefault('AI2050_NATIVE_DIR', 'off')
        self.tmp = Path(tempfile.mkdtemp())
        self.db = sqlite3.connect(':memory:', check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def tearDown(self):
        self.db.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _load(self, mode, offset, delta):
        spec = gr_q(mode)
        spec['shards'] = 1
        corpus = grammar.generate(spec)
        corpus.update({'tm_id': 'TG-GR-Q', 'version': 1, 'mode': 'final', 'synthetic_dim': 16})
        bundle = self.tmp / ('bundle-' + mode)
        bundle.mkdir()
        (bundle / 'corpus.json').write_text(json.dumps(corpus, ensure_ascii=False), encoding='utf-8')
        shutil.copy(ROOT / 'shared/dist_protocol/runner_v3.py', bundle / 'runner.py')
        self.db.execute("DELETE FROM templates"); self.db.execute("DELETE FROM results")
        self.db.execute("INSERT INTO templates VALUES('TG-GR-Q',1,'q','','','','','DESIGN-'||?,'open',?,0,'{}')",
                        (mode, str(bundle)))
        out = self.tmp / ('out-' + mode)
        env = dict(os.environ, AI2050_SHARD='0/1', AI2050_NATIVE_DIR='off')
        subprocess.run([sys.executable, str(bundle / 'runner.py'), '--mode', 'synthetic', '--out', str(out)],
                       check=True, env=env, capture_output=True)
        files = {f.name: f.read_bytes() for f in sorted(out.iterdir())}
        planted = behavior(corpus, offset=offset, delta=delta, noise=0.5)
        files['behavior.json'] = json.dumps({'schema': 'behavior.v1', 'items': planted}).encode()
        summary = json.loads(files['summary.json'])
        cells = {}
        for b in planted:
            r = cells.setdefault(b['cell'], {'n': 0, 'correct': 0})
            r['n'] += 1; r['correct'] += int(b['correct'])
        summary['behavior_by_cell'] = cells
        files['summary.json'] = json.dumps(summary).encode()
        table = json.dumps(list(files)).encode()
        blob = len(table).to_bytes(4, 'big') + table + b''.join(files.values())
        sha = hashlib.sha256(blob).hexdigest()
        (self.tmp / 'objects' / sha[:2]).mkdir(parents=True, exist_ok=True)
        (self.tmp / 'objects' / sha[:2] / sha).write_bytes(blob)
        summary['_manifest'] = [{'path': k, 'sha256': hashlib.sha256(v).hexdigest(), 'size': len(v)} for k, v in files.items()]
        summary['_upload_meta'] = {'precision': 'bf16'}
        summary['_canary'] = {'status': 'reference'}
        self.db.execute('INSERT INTO results VALUES(?,?,?,?,?,?,?,?,?,?,?,?,0)',
                        (sha, 'TG-GR-Q', 1, 'N0', 'm-' + mode, 'r1', 0, 'real', 'DESIGN-' + mode, len(blob),
                         json.dumps(summary, ensure_ascii=False), 0.0))
        self.db.commit()
        return pipeline.aggregate(self.db, self.tmp, 'TG-GR-Q')['buckets'][0]

    def test_paired_mode_passes_offset_model(self):
        b = self._load('paired', offset=-5.0, delta=2.0)
        self.assertEqual(b['gate_mode'], 'paired')
        self.assertEqual(b['gate_excluded_cells'], [])
        q = b['behavior']['mood=interrog']
        self.assertEqual(q['mode'], 'paired')
        self.assertLess(q['absolute_acc'], 0.1)
        self.assertGreater(q['acc'], 0.95)

    def test_absolute_mode_excludes_same_data(self):
        b = self._load('absolute', offset=-5.0, delta=2.0)
        self.assertEqual(b['gate_mode'], 'absolute')
        self.assertEqual(b['gate_excluded_cells'], ['mood=interrog'])

    def test_paired_mode_excludes_non_discriminating_model(self):
        b = self._load('paired', offset=-5.0, delta=0.0)
        self.assertEqual(sorted(b['gate_excluded_cells']), ['mood=decl', 'mood=interrog'])


if __name__ == '__main__':
    unittest.main()
