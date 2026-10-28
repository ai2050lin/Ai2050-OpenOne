# -*- coding: utf-8 -*-
"""数据管理（deploy/data_admin_service.py，data_admin.v1）：只有管理员能看；上传按节点所有者归到账号，
无主节点单列；副本与普查分片分别计数；筛选与详情；存储占用按目录统计。"""
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'deploy')]


class DataAdminTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.env = {k: os.environ.get(k) for k in ('AI2050_SNAPSHOT_DISABLE', 'AI2050_NATIVE_DIR', 'AI2050_REGISTRATION')}
        os.environ['AI2050_SNAPSHOT_DISABLE'] = '1'
        os.environ['AI2050_NATIVE_DIR'] = str(cls.tmp / 'native')
        os.environ.pop('AI2050_REGISTRATION', None)
        import distributed_service as D
        import census_service as CS
        import data_admin_service as DA
        import platform_service as PS
        D._DATA_DIR = cls.tmp / 'dist'
        D._DATA_DIR.mkdir()
        D._conn = None
        D._tls = threading.local()
        CS._schema_done = False
        DA._storage_cache.update(at=0.0, key='', data=None)
        cls.D, cls.CS, cls.DA = D, CS, DA
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        app = FastAPI()
        for r in (D.router, CS.router, DA.router, PS.auth_router):
            app.include_router(r)
        cls.c = TestClient(app)
        cls.admin = cls._register('boss')          # 第一个账号是管理员
        cls.member = cls._register('alice')
        cls.alice_id = cls.c.get('/api/auth/me', headers=cls.h(cls.member)).json()['user']['user_id']
        code = cls.c.post('/api/auth/node-code', headers=cls.h(cls.member)).json()['code']
        cls.node_a = cls.c.post('/api/nodes/register', json={'name': 'alice-gpu', 'bind_code': code}).json()['node_id']
        cls.node_x = cls.c.post('/api/nodes/register', json={'name': 'stray'}).json()['node_id']
        cls._seed()

    @classmethod
    def tearDownClass(cls):
        cls.D._conn = None
        cls.D._tls = threading.local()
        for k, v in cls.env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(cls.tmp, ignore_errors=True)

    @staticmethod
    def h(tok):
        return {'Authorization': f'Bearer {tok}'}

    @classmethod
    def _register(cls, name):
        r = cls.c.post('/api/auth/register', json={'username': name, 'password': 'pw-' + name * 3})
        assert r.status_code == 200, r.text
        return r.json()['token']

    @classmethod
    def _seed(cls):
        db = cls.D._db()
        now = time.time()
        summary = {'status': 'real', 'shard': 's0', '_manifest': [{'path': 'summary.json', 'size': 10, 'sha256': 'f' * 64}],
                   '_canary': {'status': 'reference'}}
        rows = [('a' * 64, 'TM-A', cls.node_a, 'qwen3-4b', 'real', 1000, now - 30),
                ('b' * 64, 'TM-B', cls.node_x, 'qwen3-4b', 'synthetic', 300, now - 20)]
        for sha, tm, node, model, kind, size, t in rows:
            db.execute('INSERT INTO results(sha,tm_id,version,node_id,model_id,model_rev,seed,kind,design_sha,size,summary_json,uploaded_at,downloads) '
                       'VALUES(?,?,?,?,?,?,?,?,?,?,?,?,0)', (sha, tm, None, node, model, 'r1', 0, kind, 'd' * 64, size,
                                                             json.dumps(summary), t))
        db.execute('INSERT INTO result_replicas(sha,node_id,task_id,model_id,model_rev,precision,uploaded_at,note) VALUES(?,?,?,?,?,?,?,?)',
                   ('a' * 64, cls.node_x, '', 'qwen3-4b', 'r1', 'bf16', now - 10, 'byte_identical_replicate'))
        db.execute("INSERT INTO audits(result_sha,status,reviewer,note,checks_json,created_at) VALUES(?,?,?,?,?,?)",
                   ('a' * 64, 'passed', 'boss', '', '{}', now))
        cdb = cls.CS.db()
        cdb.execute("INSERT INTO census_jobs(job_id,status,model_key,snapshot_sha256,job_sha256,created_at) VALUES('J1','open','qwen3-4b','s','j',?)", (now,))
        for lease, st, tok, nbytes, t in (('L1', 'accepted', 500, 4096, now - 5), ('L2', 'rejected', None, 100, now - 1)):
            cdb.execute("INSERT INTO census_leases(lease_id,job_id,shard,node_id,status,claimed_at,lease_until,done_at,bytes,result_sha) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?)", (lease, 'J1', 't0-g0', cls.node_a, 'done', t - 60, t + 3600, t, nbytes, 'c' * 64))
            cdb.execute("INSERT INTO census_results(lease_id,sha,job_id,shard,node_id,kind,status,n_tokens,summary_json,validation_json,created_at) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?,?)", (lease, 'c' * 64, 'J1', 't0-g0', cls.node_a, 'real', st, tok, '{}',
                                                         json.dumps({'ok': st == 'accepted', 'errors': [] if st == 'accepted' else ['bad']}), t))
        db.commit()
        obj = cls.D._DATA_DIR / 'objects' / 'aa'
        obj.mkdir(parents=True, exist_ok=True)
        (obj / ('a' * 64)).write_bytes(b'x' * 1000)

    def test_admin_only(self):
        for path in ('/api/admin/data/overview', '/api/admin/data/uploads', f"/api/admin/data/uploads/result/{'a' * 64}"):
            self.assertEqual(self.c.get(path).status_code, 401, path)
            self.assertEqual(self.c.get(path, headers=self.h(self.member)).status_code, 401, path)
            self.assertEqual(self.c.get(path, headers=self.h(self.admin)).status_code, 200, path)
        self.assertEqual(self.c.get('/api/admin/data/overview', headers={'X-Admin-Token': self.D.ADMIN_TOKEN}).status_code, 200)

    def test_overview_attributes_uploads_to_accounts(self):
        d = self.c.get('/api/admin/data/overview?refresh=true', headers=self.h(self.admin)).json()
        self.assertEqual(d['schema'], 'data_admin.v1')
        c = d['counts']
        self.assertEqual(c['users']['total'], 2)
        self.assertEqual(c['nodes'], {'total': 2, 'online': 2, 'unowned': 1})
        self.assertEqual(c['results']['total'], 2)
        self.assertEqual(c['results']['bytes'], 1300)
        self.assertEqual(c['results']['replicas'], 1)
        self.assertEqual(c['results']['unaudited'], 1)
        self.assertEqual(c['census']['results'], {'accepted': 1, 'rejected': 1})
        self.assertEqual(c['census']['tokens'], 500)          # 只计已接受的分片
        users = {(u['user_id'] or '-'): u for u in d['by_user']}
        a = users[self.alice_id]
        self.assertEqual((a['nodes'], a['results'], a['results_bytes'], a['census'], a['census_accepted'], a['census_rejected']),
                         (1, 1, 1000, 2, 1, 1))
        self.assertEqual(a['census_bytes'], 4196)
        stray = users['-']
        self.assertEqual(stray['display_name'], '无主节点')
        self.assertEqual((stray['nodes'], stray['results'], stray['replicas'], stray['synthetic']), (1, 1, 1, 1))
        boss = [u for u in d['by_user'] if u['username'] == 'boss'][0]
        self.assertEqual((boss['nodes'], boss['results'], boss['census']), (0, 0, 0))
        parts = {p['id']: p for p in d['storage']['parts']}
        self.assertEqual(parts['objects'], {'id': 'objects', 'label': parts['objects']['label'], 'files': 1, 'bytes': 1000})
        self.assertGreater(parts['db']['bytes'], 0)
        self.assertEqual(d['storage']['total_bytes'], sum(p['bytes'] for p in d['storage']['parts']))

    def test_uploads_filters(self):
        get = lambda q='': self.c.get('/api/admin/data/uploads' + q, headers=self.h(self.admin)).json()  # noqa: E731
        allx = get()
        self.assertEqual(allx['total'], 5)                    # 2 结果 + 1 副本 + 2 普查
        times = [x['at'] for x in allx['items']]
        self.assertEqual(times, sorted(times, reverse=True))
        self.assertEqual({x['source'] for x in get('?source=census')['items']}, {'census'})
        mine = get(f'?user={self.alice_id}')
        self.assertEqual(mine['total'], 3)
        self.assertTrue(all(x['owner']['username'] == 'alice' for x in mine['items']))
        stray = get('?user=-')
        self.assertEqual({x['source'] for x in stray['items']}, {'result', 'replica'})
        self.assertTrue(all(x['owner'] is None for x in stray['items']))
        self.assertEqual(get('?q=tm-b')['total'], 1)
        self.assertEqual(get('?q=ALICE-gpu')['total'], 3)
        self.assertEqual(get('?status=rejected')['items'][0]['id'], 'L2')
        self.assertEqual(get('?kind=synthetic')['total'], 1)
        page = get('?limit=2&offset=2')
        self.assertEqual(len(page['items']), 2)
        self.assertEqual(page['items'][0]['id'], allx['items'][2]['id'])
        self.assertEqual(self.c.get('/api/admin/data/uploads?source=bogus', headers=self.h(self.admin)).status_code, 422)

    def test_details(self):
        r = self.c.get(f"/api/admin/data/uploads/result/{'a' * 64}", headers=self.h(self.admin)).json()
        self.assertEqual((r['tm_id'], r['kind'], r['bytes']), ('TM-A', 'real', 1000))
        self.assertTrue(r['object_present'])
        self.assertEqual(r['files'][0]['path'], 'summary.json')
        self.assertEqual(r['audit']['status'], 'passed')
        self.assertEqual(r['replicas'][0]['node_id'], self.node_x)
        self.assertEqual(r['node']['username'], 'alice')
        self.assertNotIn('_manifest', r['summary'])
        rep = [x for x in self.c.get('/api/admin/data/uploads?source=replica', headers=self.h(self.admin)).json()['items']][0]
        self.assertEqual(self.c.get(f"/api/admin/data/uploads/replica/{rep['id']}", headers=self.h(self.admin)).json()['sha'], 'a' * 64)
        cz = self.c.get('/api/admin/data/uploads/census/L2', headers=self.h(self.admin)).json()
        self.assertEqual((cz['status'], cz['validation']['errors'], cz['bytes']), ('rejected', ['bad'], 100))
        self.assertFalse(cz['object_present'])
        self.assertEqual(self.c.get('/api/admin/data/uploads/census/NOPE', headers=self.h(self.admin)).status_code, 404)
        self.assertEqual(self.c.get('/api/admin/data/uploads/other/x', headers=self.h(self.admin)).status_code, 404)

    def test_read_only(self):
        before = self.c.get('/api/admin/data/uploads', headers=self.h(self.admin)).json()['total']
        for m in ('post', 'put', 'delete'):
            self.assertEqual(getattr(self.c, m)('/api/admin/data/uploads', headers=self.h(self.admin)).status_code, 405)
        self.assertEqual(self.c.get('/api/admin/data/uploads', headers=self.h(self.admin)).json()['total'], before)


if __name__ == '__main__':
    unittest.main()
