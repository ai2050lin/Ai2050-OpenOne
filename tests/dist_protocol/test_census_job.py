# -*- coding: utf-8 -*-
"""census_job.v1：普查分片的规划、节点 runner（synthetic）、中心校验（篡改负对照）、复算比较、合并等价性，
以及中心端到端流程（建作业 → 两个账号的节点领取 → 分块上传 → 校验 → 复算 → 合并 → 单元卡端点可读）与开放平台端点。"""
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'deploy'))

from shared.dist_protocol import census as C          # noqa: E402
from shared.dist_protocol import census_job as CJ     # noqa: E402
from shared.dist_protocol import census_runner as CR  # noqa: E402

SEP = 7


def synth_streams(seed=0, n_src=2, length=5000):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_src):
        t = rng.integers(10, 500, size=length).astype(np.int32)
        t[rng.random(length) < 0.02] = SEP
        out.append(t)
    return out


def make_job(tmp: Path, job_id='J1', replica_rate=0.0, **kw):
    streams = synth_streams()
    cman = {'separator_id': SEP, 'sources': [{'source': 'a', 'file': 'a.npz'}, {'source': 'b', 'file': 'b.npz'}]}
    model = {'model_key': 'mk-test', 'name': 'toy', 'snapshot_sha256': 'f' * 64, 'files': {}}
    job, files = CJ.build_job(job_id, model=model, corpus_fp='fp0', corpus_manifest=cman, streams=streams, n_layers=5,
                              d_ff=12, tokens_per_slice=2048, seq_len=64, skip_head=4, k=6, group_size=2,
                              replica_rate=replica_rate, batch=4, **kw)
    job['code'] = {}
    d = tmp / job_id
    d.mkdir(parents=True, exist_ok=True)
    for n, b in files.items():
        (d / n).write_bytes(b)
    (d / 'job.json').write_text(json.dumps(job), encoding='utf-8')
    return job, d, streams


def run_shard(job, d: Path, sid: str, out: Path):
    sh = CJ.shard_spec(job, sid)
    s = CR.run(job, sid, d / f'tokens_S{sh["slice"]}.npz', '', out, device='cpu', mode='synthetic')
    return (out / 'shard.npz').read_bytes(), s


class CensusJobUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.job, cls.dir, cls.streams = make_job(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_plan_and_expected_counts(self):
        job = self.job
        self.assertEqual(job['layer_groups'], [[0, 1], [2, 3], [4]])
        self.assertEqual(len(job['shards']), job['n_slices'] * 3)
        for s in job['slices']:
            with np.load(self.dir / s['file']) as z:
                sl = {k: z[k] for k in z.files}
            g = CJ.counted_gidx(sl, 4, SEP)
            self.assertEqual(len(g), s['expected_n'])
            src, off = C.split_index(g)
            for a, b in zip(src, off):                    # 位置还原回原始 token 流
                self.assertNotEqual(self.streams[a][b], SEP)

    def test_runner_valid_and_tamper_rejected(self):
        data, summ = run_shard(self.job, self.dir, 'S1-G0', self.tmp / 'o1')
        self.assertEqual(summ['n_tokens_counted'], summ['expected_n'])
        layers = CJ.load_shard_npz(data)
        with np.load(self.dir / 'tokens_S1.npz') as z:
            counted = CJ.counted_gidx({k: z[k] for k in z.files}, 4, SEP)
        self.assertTrue(CJ.validate_shard(self.job, 'S1-G0', layers, counted)['ok'])
        bad = {L: dict(d) for L, d in layers.items()}
        bad[0]['n'] = np.array(int(bad[0]['n']) - 1)
        self.assertFalse(CJ.validate_shard(self.job, 'S1-G0', bad, counted)['ok'])
        bad = {L: dict(d) for L, d in layers.items()}
        bad[1]['top_idx'] = bad[1]['top_idx'] + 1_000_000          # 极值位置不在该切片
        self.assertFalse(CJ.validate_shard(self.job, 'S1-G0', bad, counted)['ok'])
        bad = {L: dict(d) for L, d in layers.items()}
        h = bad[0]['hist'].copy(); h[0, 0, 0] += 5; bad[0]['hist'] = h
        self.assertFalse(CJ.validate_shard(self.job, 'S1-G0', bad, counted)['ok'])
        self.assertFalse(CJ.validate_shard(self.job, 'S1-G0', {0: layers[0]}, counted)['ok'])   # 少一层
        r = CJ.validate_shard(self.job, 'S0-G0', layers, None)                                    # 切片 0 要求读出
        self.assertFalse(r['ok'])

    def test_replica_compare(self):
        a, _ = run_shard(self.job, self.dir, 'S0-G1', self.tmp / 'r1')
        b, _ = run_shard(self.job, self.dir, 'S0-G1', self.tmp / 'r2')
        la, lb = CJ.load_shard_npz(a), CJ.load_shard_npz(b)
        self.assertTrue(CJ.compare_replicas(la, lb)['ok'])
        pert = {L: dict(d) for L, d in lb.items()}
        pert[2]['s1'] = pert[2]['s1'] + 0.5 * np.sqrt(np.maximum(pert[2]['s2'], 1)) * np.sqrt(float(pert[2]['n']))
        self.assertFalse(CJ.compare_replicas(la, pert)['ok'])

    def test_merge_equals_single_pass(self):
        import torch
        job = self.job
        res = []
        for sh in job['shards']:
            data, _ = run_shard(job, self.dir, sh['shard'], self.tmp / f"m{sh['shard']}")
            res.append((sh['shard'], CJ.load_shard_npz(data)))
        merged = CJ.merge_job(job, res)
        # 单机一次累计：全部切片串起来
        sl = [np.load(self.dir / s['file']) for s in job['slices']]
        W = np.concatenate([z['windows'] for z in sl]); src = np.concatenate([z['src'] for z in sl])
        off = np.concatenate([z['win_off'] for z in sl])
        acc = C.LayerAcc(12, k=6)
        pos = np.arange(64)
        mask = (pos[None, :] >= 4) & (W != SEP)
        g = ((src.astype(np.int64)[:, None] << C.SRC_SHIFT) | (off[:, None] + pos[None, :]))[mask]
        t = torch.tensor(W[mask], dtype=torch.float32)[:, None]
        u = torch.arange(12, dtype=torch.float32)[None, :]
        G = torch.sin(t * 0.37 + u * 0.11 + 3); V = torch.cos(t * 0.53 - u * 0.07 + 1.5) * 2
        acc.update(torch.nn.functional.silu(G) * V, G, V, torch.tensor(g))
        ref = acc.to_numpy()
        m = merged[3]
        self.assertEqual(int(m['n']), int(ref['n']))
        np.testing.assert_array_equal(m['hist'], ref['hist'])
        np.testing.assert_allclose(m['s1'], ref['s1'], rtol=1e-6)
        np.testing.assert_allclose(m['top_val'], ref['top_val'], rtol=1e-6)   # 伪激活只依赖 token，并列值的位置顺序可不同
        np.testing.assert_allclose(m['amin'], ref['amin'], rtol=1e-6)
        self.assertIn('readout_top', merged[0])

    def test_snapshot_functions_agree(self):
        d = self.tmp / 'model'
        d.mkdir()
        (d / 'config.json').write_text('{"a": 1}')
        (d / 'model.safetensors').write_bytes(b'x' * 1000)
        (d / 'tokenizer.json').write_text('{}')
        (d / 'README.md').write_text('ignored')
        from shared.dist_protocol.runner_v3 import model_snapshot as rv3
        import node_agent
        os.environ['AI2050_HOME'] = str(self.tmp / 'home')
        try:
            self.assertEqual(rv3(d)['sha256'], CR.model_snapshot(d)['sha256'])
            self.assertEqual(rv3(d)['sha256'], node_agent.model_snapshot(d)['sha256'])
        finally:
            os.environ.pop('AI2050_HOME', None)


class CenterE2ETests(unittest.TestCase):
    """中心端到端：沙盒（允许 synthetic）+ 临时数据目录。"""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.env = {k: os.environ.get(k) for k in ('AI2050_DIST_ALLOW_SYNTHETIC', 'AI2050_NATIVE_DIR', 'AI2050_SNAPSHOT_DISABLE',
                                                   'AI2050_PUBLIC_URL')}
        os.environ['AI2050_DIST_ALLOW_SYNTHETIC'] = '1'
        os.environ['AI2050_NATIVE_DIR'] = str(cls.tmp / 'native')
        os.environ['AI2050_SNAPSHOT_DISABLE'] = '1'
        os.environ['AI2050_PUBLIC_URL'] = 'http://center.test'
        cdir = cls.tmp / 'native' / '_census' / 'corpus' / 'fp0'
        cdir.mkdir(parents=True)
        for name, t in zip(('a', 'b'), synth_streams()):
            buf = io.BytesIO(); np.savez(buf, tokens=t, doc_starts=np.array([0])); (cdir / f'tokens_{name}.npz').write_bytes(buf.getvalue())
        (cdir / 'manifest.json').write_text(json.dumps({'schema': 'census_corpus.v1', 'separator_id': SEP,
                                                        'sources': [{'source': 'a', 'file': 'tokens_a.npz'}, {'source': 'b', 'file': 'tokens_b.npz'}]}))
        mdir = cls.tmp / 'model'
        mdir.mkdir()
        (mdir / 'config.json').write_text(json.dumps({'num_hidden_layers': 5, 'intermediate_size': 12}))
        tok = ROOT / 'models' / 'hf' / 'qwen3-4b' / 'tokenizer.json'
        (mdir / 'tokenizer.json').write_bytes(tok.read_bytes() if tok.is_file() else b'{}')
        import distributed_service as D
        import census_service as CS
        import platform_service as PS
        D._DATA_DIR = cls.tmp / 'dist'
        D._DATA_DIR.mkdir()
        D._conn = None
        D._tls = threading.local()
        CS._schema_done = False
        CS._job_cached.cache_clear(); CS._counted.cache_clear()
        cls.D, cls.CS, cls.PS = D, CS, PS
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        app = FastAPI()
        for r in (D.router, CS.router, PS.router, PS.auth_router):
            app.include_router(r)
        cls.c = TestClient(app)
        cls.mdir = mdir

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

    def _user_node(self, name):
        r = self.c.post('/api/auth/register', json={'username': name, 'password': 'pw-' + name * 3, 'display_name': name.upper()})
        self.assertEqual(r.status_code, 200, r.text)
        tok = r.json()['token']
        code = self.c.post('/api/auth/node-code', headers={'Authorization': f'Bearer {tok}'}).json()['code']
        n = self.c.post('/api/nodes/register', json={'name': name, 'bind_code': code}).json()
        self.assertTrue(n['owner_bound'])
        return tok, n['node_token']

    def _do_lease(self, node_token, out):
        r = self.c.post('/api/census/claim', json={'mode': 'synthetic'}, headers={'X-Node-Token': node_token})
        if r.status_code == 204:
            return None
        lease = r.json()
        d = out / lease['lease_id']
        d.mkdir(parents=True)
        for f in lease['files']:
            b = self.c.get(f['url']).content
            self.assertEqual(CJ.sha256_bytes(b), f['sha256'])
            (d / f['name']).write_bytes(b)
        job = json.loads((d / 'job.json').read_text(encoding='utf-8'))
        sys.path.insert(0, str(d))
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(f'cr_{lease["lease_id"]}', d / 'census_runner.py')
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        finally:
            sys.path.remove(str(d))
        s = mod.run(job, lease['shard'], d / f"tokens_S{lease['spec']['slice']}.npz", '', d / 'out', device='cpu', mode='synthetic')
        data = (d / 'out' / 'shard.npz').read_bytes()
        for seq, i in enumerate(range(0, len(data), 50_000)):
            rr = self.c.post(f"/api/census/leases/{lease['lease_id']}/chunk?seq={seq}", content=data[i:i + 50_000],
                             headers={'X-Node-Token': node_token, 'Content-Type': 'application/octet-stream'})
            self.assertEqual(rr.status_code, 200, rr.text)
        fin = self.c.post(f"/api/census/leases/{lease['lease_id']}/finish", json={'sha256': CJ.sha256_bytes(data), 'summary': s},
                          headers={'X-Node-Token': node_token}).json()
        # 节点没收到响应而重试 finish：按已登记的结论回答（节点据 status 判断是否被接受）
        again = self.c.post(f"/api/census/leases/{lease['lease_id']}/finish", json={'sha256': CJ.sha256_bytes(data), 'summary': s},
                            headers={'X-Node-Token': node_token}).json()
        self.assertTrue(again.get('idempotent_replay'), again)
        self.assertEqual(again['status'], fin['status'])
        return lease, fin

    def test_end_to_end(self):
        admin_tok, n1 = self._user_node('alice')       # 第一个账号是管理员
        _, n2 = self._user_node('bob')
        body = {'job_id': 'CJ-test', 'model_dir': str(self.mdir), 'snapshot': {'sha256': 'e' * 64, 'files': {}},
                'corpus_fp': 'fp0', 'tokens_per_slice': 2048, 'max_slices': 2, 'seq_len': 64, 'skip_head': 4, 'k': 6,
                'group_size': 2, 'replica_rate': 1.0, 'batch': 4, 'model_name': 'toy'}
        self.assertEqual(self.c.post('/api/census/jobs', json=body).status_code, 401)
        r = self.c.post('/api/census/jobs', json=body, headers={'X-Admin-Token': admin_tok})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()['n_shards'], 6)
        cfg = self.c.get('/api/platform/config').json()
        self.assertIn('e' * 64, [m['snapshot_sha256'] for m in cfg['models']])      # 开放作业的模型（另有缺省参与模型）
        self.assertIn('census_shard', cfg['task_kinds'])
        out = self.tmp / 'work'
        # alice 做完全部 6 个分片；同一节点不能复算自己的分片
        fins = []
        while True:
            x = self._do_lease(n1, out)
            if x is None:
                break
            fins.append(x)
        self.assertEqual(len(fins), 6)
        self.assertTrue(all(f['status'] == 'accepted' for _, f in fins), fins)
        prog = self.c.get('/api/census/jobs/CJ-test').json()
        self.assertEqual(prog['counts'], {'replica_pending': 6})
        # bob 复算：另一个账号
        while self._do_lease(n2, out) is not None:
            pass
        prog = self.c.get('/api/census/jobs/CJ-test').json()
        self.assertEqual(prog['counts'], {'done': 6}, prog['counts'])
        self.assertTrue(all(s['replica_ok'] for s in prog['shards']))
        # 篡改的上传被拒收（重复领取已无分片 → 204）
        r = self.c.post('/api/census/claim', json={'mode': 'synthetic'}, headers={'X-Node-Token': n1})
        self.assertEqual(r.status_code, 204)
        # 合并 → 原生普查目录 → 单元卡端点可读
        m = self.c.post('/api/census/jobs/CJ-test/merge', headers={'X-Admin-Token': admin_tok}).json()
        self.assertEqual(m['slices'], [0, 1])
        man = json.loads((Path(m['dir']) / 'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(man['schema'], 'mlp_census.v1')
        self.assertEqual(man['n_layers'], 5)
        self.assertEqual(sorted(f['block'] for f in man['files']), [0, 1, 2, 3, 4])
        from server import neuron_census_service as NS
        self.assertIn('CJ-test', [x['run'] for x in NS.census_runs()])
        # 贡献榜与个人页
        contrib = self.c.get('/api/platform/contributors').json()['items']
        self.assertEqual({x['name']: x['census_shards'] for x in contrib}, {})   # synthetic 不计入贡献
        ov = self.c.get('/api/platform/overview').json()
        self.assertEqual(ov['census_jobs'][0]['counts'], {'done': 6})
        my = self.c.get('/api/auth/my-contributions', headers={'Authorization': f'Bearer {admin_tok}'}).json()
        self.assertEqual(len(my['nodes'][0]['census']), 6)

    def test_tampered_upload_rejected(self):
        tok = self.c.post('/api/auth/register', json={'username': 'carol', 'password': 'pw-carolcarol'}).json()['token']
        code = self.c.post('/api/auth/node-code', headers={'Authorization': f'Bearer {tok}'}).json()['code']
        node = self.c.post('/api/nodes/register', json={'name': 'c', 'bind_code': code}).json()['node_token']
        admin = self.c.post('/api/auth/login', json={'username': 'alice', 'password': 'pw-' + 'alice' * 3})
        if admin.status_code != 200:
            self.skipTest('依赖 test_end_to_end 先建管理员')
        body = {'job_id': 'CJ-tamper', 'model_dir': str(self.mdir), 'snapshot': {'sha256': 'd' * 64, 'files': {}},
                'corpus_fp': 'fp0', 'tokens_per_slice': 2048, 'max_slices': 1, 'seq_len': 64, 'skip_head': 4, 'k': 6,
                'group_size': 5, 'replica_rate': 0.0, 'batch': 4}
        self.assertEqual(self.c.post('/api/census/jobs', json=body, headers={'X-Admin-Token': admin.json()['token']}).status_code, 200)
        lease = self.c.post('/api/census/claim', json={'mode': 'synthetic', 'job_id': 'CJ-tamper'}, headers={'X-Node-Token': node}).json()
        job = json.loads(self.c.get(f"/api/census/jobs/CJ-tamper/files/job.json").content)
        d = self.tmp / 'tamper'
        d.mkdir()
        (d / 'tokens.npz').write_bytes(self.c.get(f"/api/census/jobs/CJ-tamper/files/tokens_S0.npz").content)
        s = CR.run(job, lease['shard'], d / 'tokens.npz', '', d / 'o', device='cpu', mode='synthetic')
        layers = CJ.load_shard_npz((d / 'o' / 'shard.npz').read_bytes())
        layers[2]['s1'] = layers[2]['s1'] * 1.5          # 被改过的统计量：自洽性仍可能通过，但这里改 n
        layers[2]['n'] = np.array(int(layers[2]['n']) + 3)
        data = CJ.npz_bytes({f'L{L}__{k}': v for L, d_ in layers.items() for k, v in d_.items()}, compressed=True)
        s['runner_sha256'] = job['code']['census_runner.py']; s['census_sha256'] = job['code']['census.py']
        self.c.post(f"/api/census/leases/{lease['lease_id']}/chunk?seq=0", content=data,
                    headers={'X-Node-Token': node, 'Content-Type': 'application/octet-stream'})
        fin = self.c.post(f"/api/census/leases/{lease['lease_id']}/finish", json={'sha256': CJ.sha256_bytes(data), 'summary': s},
                          headers={'X-Node-Token': node}).json()
        self.assertEqual(fin['status'], 'rejected')
        self.assertTrue(any('计入 token 数' in e for e in fin['errors']), fin['errors'])

    def test_kit_zip(self):
        r = self.c.get('/api/platform/kit')
        self.assertEqual(r.status_code, 200)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        names = set(z.namelist())
        for n in ('node_agent.py', 'node_service.py', 'goodfire_runner.py', 'goodfire_probe.py', 'config.json', 'requirements.txt',
                  'README.txt', 'start_windows.bat', 'start_linux.sh'):
            self.assertIn(f'ai2050-node/{n}', names)
        cfg = json.loads(z.read('ai2050-node/config.json'))
        self.assertEqual(cfg['server'], 'http://center.test')
        self.assertEqual(cfg['bind_code'], '')                  # 未登录：不带绑定码
        tok = self.c.post('/api/auth/register', json={'username': 'dave', 'password': 'pw-davedave'}).json()['token']
        z = zipfile.ZipFile(io.BytesIO(self.c.get('/api/auth/kit', headers={'Authorization': f'Bearer {tok}'}).content))
        code = json.loads(z.read('ai2050-node/config.json'))['bind_code']
        self.assertTrue(code.startswith('B-'))
        n = self.c.post('/api/nodes/register', json={'name': 'dave-box', 'bind_code': code}).json()
        self.assertTrue(n['owner_bound'])
        self.assertEqual(self.c.post('/api/nodes/register', json={'name': 'x', 'bind_code': code}).status_code, 400)   # 一次性


if __name__ == '__main__':
    unittest.main()
