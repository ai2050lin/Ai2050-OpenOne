# -*- coding: utf-8 -*-
"""pipeline 中心节点派生视图测试：用临时 SQLite + 合成 runner 产物（status=synthetic）模拟分布式结果库。

只验证合并/门链/审计/覆盖的计算口径；写入时把合成结果标记为 kind=real 仅限本测试的临时库。
"""
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('AI2050_NATIVE_DIR', 'off')   # 测试不在用户目录写原生全场

from shared.dist_protocol import grammar, pipeline  # noqa: E402

SCHEMA = '''
CREATE TABLE nodes(node_id TEXT PRIMARY KEY, token_hash TEXT, name TEXT, gpu TEXT, model TEXT, dtype TEXT,
                   trust TEXT, created_at REAL, last_seen REAL);
CREATE TABLE templates(tm_id TEXT PRIMARY KEY, version INTEGER, name TEXT, dim TEXT, description TEXT,
                       metric_pin TEXT, model_pin TEXT, design_sha TEXT, status TEXT, bundle_dir TEXT,
                       created_at REAL, research_adapter_json TEXT DEFAULT '{}');
CREATE TABLE tasks(task_id TEXT PRIMARY KEY, tm_id TEXT, node_id TEXT, status TEXT, claimed_at REAL,
                   lease_until REAL, done_at REAL, result_sha TEXT, note TEXT, work_unit_id TEXT,
                   research_ref_json TEXT DEFAULT '{}', shard TEXT);
CREATE TABLE results(sha TEXT PRIMARY KEY, tm_id TEXT, version INTEGER, node_id TEXT, model_id TEXT,
                     model_rev TEXT, seed INTEGER, kind TEXT, design_sha TEXT, size INTEGER,
                     summary_json TEXT, uploaded_at REAL, downloads INTEGER DEFAULT 0);
CREATE TABLE audits(result_sha TEXT PRIMARY KEY, status TEXT, reviewer TEXT, note TEXT, checks_json TEXT, created_at REAL);
CREATE TABLE result_replicas(sha TEXT, node_id TEXT, task_id TEXT, model_id TEXT, model_rev TEXT,
                             precision TEXT, uploaded_at REAL, note TEXT);
'''


class PipelineDbTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.db = sqlite3.connect(':memory:', check_same_thread=False)
        cls.db.row_factory = sqlite3.Row
        cls.db.executescript(SCHEMA)
        spec = json.loads((ROOT / 'shared/dist_protocol/grammars/GR-ROLE.json').read_text(encoding='utf-8'))
        corpus = grammar.generate(spec)
        corpus.pop('_warnings', None)
        corpus.update({'tm_id': 'TG-GR-ROLE', 'version': 1, 'mode': 'final', 'synthetic_dim': 32})
        bundle = cls.tmp / 'bundle'
        bundle.mkdir()
        (bundle / 'corpus.json').write_text(json.dumps(corpus, ensure_ascii=False), encoding='utf-8')
        shutil.copy(ROOT / 'shared/dist_protocol/runner_v3.py', bundle / 'runner.py')
        cls.db.execute("INSERT INTO templates VALUES('TG-GR-ROLE',1,'role','','','','','DESIGN','open',?,0,'{}')",
                       (str(bundle),))
        cls.shas = []
        for i, model in enumerate(['m-a', 'm-a', 'm-b', 'm-b']):
            shard = f'{i % 2}/2'
            out = cls.tmp / f'out{i}'
            env = dict(os.environ, AI2050_SHARD=shard, AI2050_SYNTHETIC_SALT=model)
            subprocess.run([sys.executable, str(bundle / 'runner.py'), '--mode', 'synthetic', '--out', str(out)],
                           check=True, env=env, capture_output=True)
            files = {p.name: p.read_bytes() for p in sorted(out.iterdir())}
            table = json.dumps(list(files)).encode()
            blob = len(table).to_bytes(4, 'big') + table + b''.join(files.values())
            sha = hashlib.sha256(blob).hexdigest()
            (cls.tmp / 'objects' / sha[:2]).mkdir(parents=True, exist_ok=True)
            (cls.tmp / 'objects' / sha[:2] / sha).write_bytes(blob)
            summary = json.loads(files['summary.json'])
            summary['_manifest'] = [{'path': k, 'sha256': hashlib.sha256(v).hexdigest(), 'size': len(v)} for k, v in files.items()]
            summary['_upload_meta'] = {'precision': 'bf16'}
            summary['_canary'] = {'status': 'reference' if i in (0, 2) else 'ok'}
            cls.db.execute('INSERT INTO results VALUES(?,?,?,?,?,?,?,?,?,?,?,?,0)',
                           (sha, 'TG-GR-ROLE', 1, f'N{i}', model, 'r1', 0, 'real', 'DESIGN', len(blob),
                            json.dumps(summary, ensure_ascii=False), float(i)))
            cls.shas.append(sha)
        cls.db.commit()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_aggregate_two_buckets_complete(self):
        agg = pipeline.aggregate(self.db, self.tmp, 'TG-GR-ROLE', gate=False)
        self.assertEqual(len(agg['buckets']), 2)
        for b in agg['buckets']:
            self.assertTrue(b['complete'])
            n = sum(v['N'] for v in b['agg']['cells_by_split'].values())
            self.assertEqual(n, 180)

    def test_gate_chain_cross_model(self):
        agg = pipeline.aggregate(self.db, self.tmp, 'TG-GR-ROLE', gate=False)
        gates = {g['id']: g for g in pipeline.gate_chain(agg)}
        self.assertEqual(gates['instrument']['status'], 'pass')
        self.assertEqual(gates['repeat']['status'], 'pending')
        self.assertEqual(gates['selectivity']['status'], 'not_run')
        self.assertIn(gates['lockbox']['status'], ('pass', 'fail'))

    def test_audit_flow_excludes_rejected(self):
        q = pipeline.audit_queue(self.db)
        self.assertEqual(q['counts']['pending'], 4)
        self.assertTrue(all(c['id'] != 'artifacts' or c['status'] == 'pass' for i in q['items'] for c in i['checks']))
        pipeline.record_audit(self.db, threading.Lock(), self.shas[1], 'rejected', 't', '', [])
        agg = pipeline.aggregate(self.db, self.tmp, 'TG-GR-ROLE', bucket='m-a|r1|bf16', gate=False)
        self.assertFalse(agg['buckets'][0]['complete'])
        self.db.execute('DELETE FROM audits')
        self.db.commit()
        pipeline._CACHE.clear()

    def test_coverage_and_overview(self):
        cov = pipeline.coverage(self.db, self.tmp)
        row = cov['rows'][0]
        self.assertEqual(row['protocol'], 'v3')
        self.assertEqual(row['cells']['repr']['status'], 'done')
        ov = pipeline.overview(self.db, self.tmp)
        self.assertEqual(ov['stages'][0]['value'], 1)
        sb = pipeline.scoreboard(self.db, self.tmp)
        self.assertEqual(len(sb['templates']), 1)
        self.assertTrue(sb['scores'])

    def test_catalog_export_and_atlas(self):
        import io
        import zipfile
        lib = [json.loads((ROOT / f'shared/dist_protocol/grammars/{f}.json').read_text(encoding='utf-8'))
               for f in ('GR-ROLE', 'KN-ISA')]
        cat = pipeline.catalog(self.db, self.tmp, lib)
        items = {i['id']: i for i in cat['items']}
        self.assertEqual(items['GR-ROLE']['status'], 'done')
        self.assertEqual(items['GR-ROLE']['category'], '语法系统')
        self.assertEqual(len(items['GR-ROLE']['models']), 2)
        self.assertIn('--tm-id TG-GR-ROLE', items['GR-ROLE']['claim_command'])
        self.assertEqual(items['KN-ISA']['status'], 'draft')                    # 文法库里有、尚未冻结
        self.assertEqual(sum(c['total'] for c in cat['categories']), 2)
        data, name = pipeline.export_pack(self.db, self.tmp, 'TG-GR-ROLE', bucket='m-a|r1|bf16', gate=False)
        z = zipfile.ZipFile(io.BytesIO(data))
        names = set(z.namelist())
        self.assertTrue({'manifest.json', 'README.md', 'corpus.json', 'agg.json', 'analysis.json', 'suffstats.npz'} <= names)
        self.assertEqual(len([n for n in names if n.startswith('items/')]), 2)
        man = json.loads(z.read('manifest.json'))
        self.assertEqual(man['schema'], 'respack.v1')
        self.assertEqual(len(man['files']), len(names) - 1)
        self.assertTrue(name.startswith('respack_TG-GR-ROLE'))
        with self.assertRaises(LookupError):
            pipeline.export_pack(self.db, self.tmp, 'TG-GR-ROLE', bucket='nope')
        at = pipeline.atlas_payload(self.db, self.tmp, gate=False)
        self.assertEqual(sorted(at['models']), ['m-a|r1|bf16', 'm-b|r1|bf16'])
        for g in at['graphs']:
            self.assertEqual(g['families'], ['GR-ROLE'])
            self.assertEqual(g['axes'], [])                                     # 只有一个模式族：没有跨族轴
            self.assertTrue(g['nodes'])
        pack, _ = pipeline.atlas_pack(at, 'm-a|r1|bf16')
        self.assertEqual(set(zipfile.ZipFile(io.BytesIO(pack)).namelist()), {'manifest.json', 'atlas.json', 'digest.md', 'prompt.md'})


if __name__ == '__main__':
    unittest.main()
