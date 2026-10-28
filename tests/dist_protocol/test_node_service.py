# -*- coding: utf-8 -*-
"""node_service.v1（本机节点服务，design/contributor_client_v1.md P1）：
nvidia-smi 解析、时间窗、配置白名单、资源门、显存估计、退避；本机接口的访问关卡（令牌、Host、Origin）；
以及对沙盒中心的端到端流程（绑定码注册 → 开始 → 领取普查分片 → synthetic 运行 → 上传 → 中心校验通过 → 停止）。"""
import io
import json
import os
import random
import shutil
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'deploy'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import node_service as S  # noqa: E402

SMI = '''0, NVIDIA GeForce RTX 5080, 16303, 1024, 6, 610.62
1, NVIDIA RTX A6000, 49140, 40000, 97, 610.62
garbage line
'''
FAKE = {'gpus': lambda: S.parse_nvidia_smi(SMI), 'battery': lambda: None, 'disk': lambda p: 100.0,
        'runtime': lambda py: {'ok': True, 'python': py}}


def lt(h, m):
    return time.struct_time((2026, 10, 27, h, m, 0, 1, 300, 0))


class PureTests(unittest.TestCase):
    def test_parse_nvidia_smi(self):
        g = S.parse_nvidia_smi(SMI)
        self.assertEqual([x['index'] for x in g], [0, 1])
        self.assertEqual(g[0]['name'], 'NVIDIA GeForce RTX 5080')
        self.assertAlmostEqual(g[0]['vram_gib'], 15.9)
        self.assertEqual(g[1]['util'], 97.0)
        self.assertEqual(S.parse_nvidia_smi(''), [])

    def test_window(self):
        self.assertTrue(S.in_window('', lt(3, 0)))
        self.assertTrue(S.in_window('09:00-18:00', lt(9, 0)))
        self.assertFalse(S.in_window('09:00-18:00', lt(18, 0)))
        self.assertTrue(S.in_window('22:00-08:00', lt(23, 30)))      # 跨午夜
        self.assertTrue(S.in_window('22:00-08:00', lt(7, 59)))
        self.assertFalse(S.in_window('22:00-08:00', lt(12, 0)))
        for bad in ('9-18', '25:00-01:00', '08:00-08:00', 'x'):
            with self.assertRaises(ValueError):
                S.parse_window(bad)

    def test_validate_config(self):
        base = json.loads(json.dumps(S.DEFAULT_CONFIG))
        c = S.validate_config(base, {'server': 'https://server.ai2050.com/', 'limits': {'hours': '22:00-08:00'}})
        self.assertEqual(c['server'], 'https://server.ai2050.com')
        self.assertEqual(c['limits']['hours'], '22:00-08:00')
        self.assertTrue(c['limits']['pause_on_battery'])               # 其余限制保留缺省
        self.assertEqual(base['server'], '')                            # 不改原配置
        for bad in ({'python': 'evil.exe'}, {'server': 'ftp://x'}, {'device': 'cuda:x'}, {'task_kinds': ['rm']},
                    {'limits': {'yield_gpu_util': 150}}, {'limits': {'hours': '9-18'}}, {'limits': {'shell': 1}},
                    {'limits': {'pause_on_battery': 'yes'}}, {'model_dirs': ['a']}):
            with self.assertRaises(ValueError, msg=bad):
                S.validate_config(base, bad)

    def test_gates(self):
        cfg = S.validate_config(json.loads(json.dumps(S.DEFAULT_CONFIG)), {})
        gpus = S.parse_nvidia_smi(SMI)
        ok = dict(now_struct=lt(12, 0), gpus=gpus, battery=False, disk_free_gib=100)
        self.assertEqual(S.gate_reasons(cfg, **ok), [])
        self.assertEqual(len(S.gate_reasons(cfg, **{**ok, 'battery': True})), 1)
        self.assertEqual(len(S.gate_reasons(cfg, **{**ok, 'disk_free_gib': 1.0})), 1)
        self.assertEqual(len(S.gate_reasons(cfg, **{**ok, 'gpus': []})), 1)              # 没有 NVIDIA GPU
        busy = S.validate_config(cfg, {'device': 'cuda:1'})                               # 卡 1 利用率 97%
        self.assertIn('GPU 正被其他程序使用', S.gate_reasons(busy, **ok)[0])
        self.assertEqual(S.gate_reasons(busy, **{**ok, 'gpus': []}, synthetic=True), [])  # 联调不看 GPU
        night = S.validate_config(cfg, {'limits': {'hours': '22:00-08:00'}})
        self.assertEqual(len(S.gate_reasons(night, **ok)), 1)
        nobat = S.validate_config(cfg, {'limits': {'pause_on_battery': False}})
        self.assertEqual(S.gate_reasons(nobat, **{**ok, 'battery': True}), [])

    def test_model_fits(self):
        cfg = S.validate_config(json.loads(json.dumps(S.DEFAULT_CONFIG)), {})
        gpus = S.parse_nvidia_smi(SMI)
        self.assertTrue(S.model_fits(8.0, cfg, gpus))                 # 8 × 1.2 ≤ 15.9
        self.assertFalse(S.model_fits(14.0, cfg, gpus))
        self.assertFalse(S.model_fits(8.0, S.validate_config(cfg, {'limits': {'max_vram_gib': 8}}), gpus))
        self.assertTrue(S.model_fits(0, cfg, gpus))

    def test_log_survives_gbk_console(self):
        """中文控制台（GBK，严格编码）打印 ✓ 不能让任务失败（曾导致已上传的分片被报为失败）。"""
        import node_agent as A
        old, buf = sys.stdout, io.BytesIO()
        out = sys.stdout = io.TextIOWrapper(buf, encoding='gbk', errors='strict')
        try:
            A.log('上传完成，中心校验通过 ✓')
            sys.stdout.flush()
        finally:
            sys.stdout = old
        self.assertIn('上传完成'.encode('gbk'), buf.getvalue())
        out.detach()

    def test_backoff(self):
        b = S.Backoff(random.Random(0))
        idle = [b.next('idle') for _ in range(8)]
        self.assertTrue(48 <= idle[0] <= 72)
        self.assertTrue(all(a <= c * 1.5 + 1 for a, c in zip(idle, idle[1:])))
        self.assertTrue(all(x <= 900 * 1.2 for x in idle))
        self.assertGreaterEqual(idle[-1], 900 * 0.8)                   # 到上限
        e = b.next('error')                                            # 换类型：另一类计数清零
        self.assertTrue(24 <= e <= 36)
        self.assertTrue(48 <= b.next('idle') <= 72)
        b.reset()
        self.assertTrue(24 <= b.next('error') <= 36)


def call(port, path, method='GET', body=None, token='', host=None, origin=None):
    req = urllib.request.Request(f'http://127.0.0.1:{port}{path}', method=method,
                                 data=json.dumps(body).encode('utf-8') if body is not None else None)
    req.add_header('Content-Type', 'application/json')
    if token:
        req.add_header('X-Local-Token', token)
    if host:
        req.add_header('Host', host)
    if origin:
        req.add_header('Origin', origin)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode('utf-8') or '{}')


def start_service(home, **kw):
    box = {}
    ev = threading.Event()

    def ready(httpd, svc, token):
        box.update(httpd=httpd, svc=svc, token=token, port=httpd.server_address[1])
        ev.set()
    t = threading.Thread(target=S.serve, args=(home, 0), kwargs={'ready': ready, 'probes': FAKE, **kw}, daemon=True)
    t.start()
    assert ev.wait(10)
    box['thread'] = t
    return box


class LocalApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.b = start_service(self.tmp / 'svc', allow_origins=('http://localhost:5173',))

    def tearDown(self):
        call(self.b['port'], '/shutdown', 'POST', {}, token=self.b['token'])
        self.b['thread'].join(10)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_service_file(self):
        info = json.loads((self.tmp / 'svc' / 'service.json').read_text(encoding='utf-8'))
        self.assertEqual(info['port'], self.b['port'])
        self.assertEqual(info['token'], self.b['token'])

    def test_guards(self):
        p, tok = self.b['port'], self.b['token']
        self.assertEqual(call(p, '/status')[0], 401)                                         # 没有令牌
        self.assertEqual(call(p, '/status', token='wrong')[0], 401)
        self.assertEqual(call(p, '/status', token=tok, host=f'evil.example:{p}')[0], 403)   # DNS 重绑定
        self.assertEqual(call(p, '/status', token=tok, origin='https://evil.example')[0], 403)
        self.assertEqual(call(p, '/status', token=tok, origin='http://localhost:5173')[0], 200)   # 显式放行
        self.assertEqual(call(p, '/status', token=tok, host=f'localhost:{p}')[0], 200)
        st, s = call(p, '/status', token=tok)
        self.assertEqual((st, s['state'], s['bound']), (200, 'stopped', False))
        self.assertEqual(call(p, '/nope', token=tok)[0], 404)

    def test_config_and_controls(self):
        p, tok = self.b['port'], self.b['token']
        st, c = call(p, '/config', 'PUT', {'limits': {'hours': '22:00-08:00'}}, token=tok)
        self.assertEqual((st, c['limits']['hours']), (200, '22:00-08:00'))
        self.assertEqual(call(p, '/config', 'PUT', {'python': 'x'}, token=tok)[0], 400)
        saved = json.loads((self.tmp / 'svc' / 'config.json').read_text(encoding='utf-8'))
        self.assertEqual(saved['limits']['hours'], '22:00-08:00')
        self.assertEqual(call(p, '/start', 'POST', {}, token=tok)[0], 409)                 # 未注册不能开始
        self.assertEqual(call(p, '/bind', 'POST', {'bind_code': 'x'}, token=tok)[0], 400)  # 没有中心地址
        st, d = call(p, '/device', token=tok)
        self.assertEqual(d['gpus'][0]['name'], 'NVIDIA GeForce RTX 5080')
        self.assertTrue(d['runtime']['ok'])
        self.assertEqual(call(p, '/logs?after=0', token=tok)[0], 200)
        self.assertEqual(call(p, '/history', token=tok)[1], {'items': []})


class SandboxCenterE2ETests(unittest.TestCase):
    """沙盒中心（允许 synthetic）以真实 HTTP 运行，本机服务经 urllib 访问它。"""

    @classmethod
    def setUpClass(cls):
        import uvicorn
        from test_census_job import SEP, synth_streams
        cls.tmp = Path(tempfile.mkdtemp())
        cls.env = {k: os.environ.get(k) for k in ('AI2050_DIST_ALLOW_SYNTHETIC', 'AI2050_NATIVE_DIR', 'AI2050_SNAPSHOT_DISABLE',
                                                   'AI2050_PUBLIC_URL')}
        os.environ['AI2050_DIST_ALLOW_SYNTHETIC'] = '1'
        os.environ['AI2050_NATIVE_DIR'] = str(cls.tmp / 'native')
        os.environ['AI2050_SNAPSHOT_DISABLE'] = '1'
        cdir = cls.tmp / 'native' / '_census' / 'corpus' / 'fp0'
        cdir.mkdir(parents=True)
        for name, t in zip(('a', 'b'), synth_streams()):
            buf = io.BytesIO(); np.savez(buf, tokens=t, doc_starts=np.array([0])); (cdir / f'tokens_{name}.npz').write_bytes(buf.getvalue())
        (cdir / 'manifest.json').write_text(json.dumps({'schema': 'census_corpus.v1', 'separator_id': SEP,
                                                        'sources': [{'source': 'a', 'file': 'tokens_a.npz'}, {'source': 'b', 'file': 'tokens_b.npz'}]}))
        mdir = cls.tmp / 'model'
        mdir.mkdir()
        (mdir / 'config.json').write_text(json.dumps({'num_hidden_layers': 4, 'intermediate_size': 12}))
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
        cls.D = D
        from fastapi import FastAPI
        app = FastAPI()
        for r in (D.router, CS.router, PS.router, PS.auth_router):
            app.include_router(r)
        cls.srv = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=0, log_level='warning'))
        cls.srv_thread = threading.Thread(target=cls.srv.run, daemon=True)
        cls.srv_thread.start()
        for _ in range(200):
            if cls.srv.started:
                break
            time.sleep(0.05)
        port = cls.srv.servers[0].sockets[0].getsockname()[1]
        cls.center = f'http://127.0.0.1:{port}'
        os.environ['AI2050_PUBLIC_URL'] = cls.center
        cls.mdir = mdir

    @classmethod
    def tearDownClass(cls):
        cls.srv.should_exit = True
        cls.srv_thread.join(10)
        cls.D._conn = None
        cls.D._tls = threading.local()
        for k, v in cls.env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def http(self, path, method='GET', body=None, headers=None):
        req = urllib.request.Request(self.center + path, method=method,
                                     data=json.dumps(body).encode('utf-8') if body is not None else None)
        req.add_header('Content-Type', 'application/json')
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode('utf-8'))

    def test_bind_start_run_upload(self):
        admin = self.http('/api/auth/register', 'POST', {'username': 'erin', 'password': 'pw-erinerin', 'display_name': 'Erin'})['token']
        code = self.http('/api/auth/node-code', 'POST', {}, {'Authorization': f'Bearer {admin}'})['code']
        body = {'job_id': 'CJ-svc', 'model_dir': str(self.mdir), 'snapshot': {'sha256': 'f' * 64, 'files': {}},
                'corpus_fp': 'fp0', 'tokens_per_slice': 2048, 'max_slices': 1, 'seq_len': 64, 'skip_head': 4, 'k': 6,
                'group_size': 2, 'replica_rate': 0.0, 'batch': 4, 'model_name': 'toy'}
        self.assertEqual(self.http('/api/census/jobs', 'POST', body, {'X-Admin-Token': admin})['n_shards'], 2)
        b = start_service(self.tmp / 'svc', synthetic=True, server_override=self.center)
        p, tok = b['port'], b['token']
        try:
            st, r = call(p, '/bind', 'POST', {'bind_code': code}, token=tok)
            self.assertEqual(st, 200, r)
            self.assertTrue(r['owner_bound'])
            self.assertTrue(call(p, '/status', token=tok)[1]['bound'])
            self.assertEqual(call(p, '/start', 'POST', {}, token=tok)[0], 200)
            deadline = time.time() + 120
            while time.time() < deadline:
                s = call(p, '/status', token=tok)[1]
                if s['counts']['done'] + s['counts']['failed'] + s['counts']['rejected'] >= 2:
                    break
                time.sleep(0.5)
            self.assertEqual(s['counts'], {'done': 2, 'failed': 0, 'rejected': 0}, s)
            hist = call(p, '/history', token=tok)[1]['items']
            self.assertEqual(sorted(h['status'] for h in hist), ['accepted', 'accepted'])
            self.assertEqual({h['job'] for h in hist}, {'CJ-svc'})
            prog = self.http('/api/census/jobs/CJ-svc')
            self.assertEqual(prog['counts'], {'done': 2}, prog['counts'])
            logs = call(p, '/logs?after=0', token=tok)[1]['items']
            self.assertFalse(any(code in x['msg'] for x in logs))                  # 绑定码不进日志
            self.assertEqual(call(p, '/stop', 'POST', {}, token=tok)[1]['state'], 'stopped')
        finally:
            call(p, '/shutdown', 'POST', {}, token=tok)
            b['thread'].join(10)


class FakeHF:
    """本机假的 Hugging Face：文件列表 API + 支持 Range 的 resolve；可设置损坏或不支持续传。"""
    REV = 'a' * 40

    def __init__(self):
        import hashlib
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        self.files = {'config.json': b'{"num_hidden_layers": 2}', 'model.safetensors': bytes(range(256)) * 4096}
        self.corrupt, self.no_range, self.ranges = set(), False, []
        lfs = {'model.safetensors': hashlib.sha256(self.files['model.safetensors']).hexdigest()}
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):  # noqa: N802
                u = urlparse(self.path)
                if u.path == f'/api/models/org/toy/revision/{fake.REV}':
                    sib = [{'rfilename': n, 'size': len(b), **({'lfs': {'sha256': lfs[n], 'size': len(b)}} if n in lfs else {})}
                           for n, b in fake.files.items()]
                    sib += [{'rfilename': 'sub/x.bin', 'size': 3}, {'rfilename': '..', 'size': 1}, {'rfilename': '.gitattributes', 'size': 1}]
                    body = json.dumps({'siblings': sib}).encode()
                    self.send_response(200); self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
                    return
                name = u.path.rsplit('/', 1)[-1]
                if not u.path.startswith(f'/org/toy/resolve/{fake.REV}/') or name not in fake.files:
                    self.send_response(404); self.end_headers(); return
                data = fake.files[name]
                if name in fake.corrupt:
                    data = b'\x00' * len(data)
                rng = self.headers.get('Range')
                fake.ranges.append((name, rng))
                start = int(rng[6:].split('-')[0]) if rng and not fake.no_range else 0
                self.send_response(206 if start else 200)
                self.send_header('Content-Length', str(len(data) - start)); self.end_headers()
                self.wfile.write(data[start:])

        self.httpd = ThreadingHTTPServer(('127.0.0.1', 0), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f'http://127.0.0.1:{self.httpd.server_address[1]}'

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


from urllib.parse import urlparse  # noqa: E402


class ModelDownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.env = os.environ.get('AI2050_HOME')
        os.environ['AI2050_HOME'] = str(self.tmp / 'home')      # model_snapshot 的哈希缓存
        self.hf = FakeHF()

    def tearDown(self):
        self.hf.close()
        if self.env is None:
            os.environ.pop('AI2050_HOME', None)
        else:
            os.environ['AI2050_HOME'] = self.env
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_safe_filename(self):
        for ok in ('model.safetensors', 'tokenizer.json', 'model-00001-of-00002.safetensors'):
            self.assertTrue(S.safe_filename(ok), ok)
        for bad in ('..', '../x', 'a/b', 'a\\b', 'C:x', '.gitattributes', '', 'a\x00b'):
            self.assertFalse(S.safe_filename(bad), bad)

    def test_file_list_skips_nested_and_unsafe(self):
        files = S.hf_file_list(self.hf.url, 'org/toy', FakeHF.REV)
        self.assertEqual(sorted(f['name'] for f in files), ['config.json', 'model.safetensors'])
        self.assertTrue(next(f for f in files if f['name'] == 'model.safetensors')['sha256'])
        with self.assertRaises(ValueError):
            S.hf_file_list(self.hf.url, '../etc', FakeHF.REV)

    def test_resume_from_part(self):
        data = self.hf.files['model.safetensors']
        dest = self.tmp / 'model.safetensors'
        (self.tmp / 'model.safetensors.part').write_bytes(data[:300_000])
        url = f'{self.hf.url}/org/toy/resolve/{FakeHF.REV}/model.safetensors'
        import hashlib
        got = S.download_file(url, dest, size=len(data), sha256=hashlib.sha256(data).hexdigest())
        self.assertEqual(dest.read_bytes(), data)
        self.assertEqual(got, hashlib.sha256(data).hexdigest())
        self.assertEqual(self.hf.ranges[-1], ('model.safetensors', 'bytes=300000-'))
        self.assertFalse((self.tmp / 'model.safetensors.part').exists())

    def test_server_without_range_restarts(self):
        self.hf.no_range = True
        data = self.hf.files['model.safetensors']
        (self.tmp / 'model.safetensors.part').write_bytes(b'junk' * 10)
        seen = []
        S.download_file(f'{self.hf.url}/org/toy/resolve/{FakeHF.REV}/model.safetensors', self.tmp / 'model.safetensors',
                        size=len(data), on_bytes=seen.append)
        self.assertEqual((self.tmp / 'model.safetensors').read_bytes(), data)
        self.assertEqual(sum(seen), len(data))                   # 进度计数在重新开始时扣回

    def test_sha_mismatch_rejected(self):
        self.hf.corrupt.add('model.safetensors')
        data = self.hf.files['model.safetensors']
        import hashlib
        with self.assertRaises(RuntimeError):
            S.download_file(f'{self.hf.url}/org/toy/resolve/{FakeHF.REV}/model.safetensors', self.tmp / 'm.safetensors',
                            size=len(data), sha256=hashlib.sha256(data).hexdigest())
        self.assertFalse((self.tmp / 'm.safetensors').exists())
        self.assertFalse((self.tmp / 'm.safetensors.part').exists())

    def _service(self, disk=100.0):
        ref = self.tmp / 'ref'
        ref.mkdir()
        for n, b in self.hf.files.items():
            (ref / n).write_bytes(b)
        want = S.A.model_snapshot(ref)['sha256']
        svc = S.NodeService(self.tmp / 'svc', probes={**FAKE, 'disk': lambda p: disk})
        svc.update_config({'models_root': str(self.tmp / 'models')})
        svc.cfg['hf_endpoint'] = self.hf.url                     # 测试用本机来源（真实配置只接受白名单）
        svc.pcfg = {'models': [{'snapshot_sha256': want, 'name': 'toy', 'hf_repo': 'org/toy', 'hf_revision': FakeHF.REV}]}
        return svc, want

    def _wait(self, svc, snap, until=('ok', 'mismatch', 'error', 'cancelled')):
        deadline = time.time() + 30
        while time.time() < deadline:
            st = (svc.checks.get(snap) or {}).get('status')
            if st in until:
                return svc.checks[snap]
            time.sleep(0.05)
        self.fail(f'download did not finish: {svc.checks.get(snap)}')

    def test_service_download_and_verify(self):
        svc, want = self._service()
        try:
            self.assertEqual(svc.download_model(want)['status'], 'downloading')
            r = self._wait(svc, want)
            self.assertEqual(r['status'], 'ok', r)
            d = Path(svc.cfg['model_dirs'][want])
            self.assertEqual(d, self.tmp / 'models' / 'toy')
            self.assertEqual(sorted(p.name for p in d.iterdir()), ['config.json', 'model.safetensors'])
            cache = json.loads((self.tmp / 'home' / 'snapshot_cache.json').read_text())
            self.assertTrue(any('model.safetensors' in k for k in cache))
            self.assertEqual(svc.models()[0]['status'], 'ok')
            n = len(self.hf.ranges)
            svc.download_model(want)                              # 已齐全：不再请求文件
            self._wait(svc, want, until=('ok',))
            self.assertEqual(len(self.hf.ranges), n)
        finally:
            svc.shutdown()

    def test_corrupt_download_is_error_not_ok(self):
        self.hf.corrupt.add('model.safetensors')
        svc, want = self._service()
        try:
            svc.download_model(want)
            r = self._wait(svc, want)
            self.assertEqual(r['status'], 'error')
            self.assertNotIn(want, svc.cfg['model_dirs'])
        finally:
            svc.shutdown()

    def test_refuses_when_disk_too_small(self):
        svc, want = self._service(disk=1.0)
        try:
            svc.download_model(want)
            r = self._wait(svc, want)
            self.assertEqual(r['status'], 'error')
            self.assertIn('磁盘空间不够', r['error'])
            self.assertEqual(self.hf.ranges, [])
        finally:
            svc.shutdown()

    def test_unknown_snapshot_and_config_whitelist(self):
        svc, _ = self._service()
        try:
            with self.assertRaises(ValueError):
                svc.download_model('0' * 64)
            with self.assertRaises(ValueError):
                svc.update_config({'hf_endpoint': 'http://evil.example'})
            with self.assertRaises(ValueError):
                svc.update_config({'models_root': 'relative/dir'})
            self.assertEqual(svc.update_config({'hf_endpoint': 'https://hf-mirror.com'})['hf_endpoint'], 'https://hf-mirror.com')
        finally:
            svc.shutdown()


FAKE_RUNNER = r'''
import argparse, json, sys, time
ap = argparse.ArgumentParser()
for k in ('--model', '--layer', '--unit', '--job', '--spec', '--out', '--k', '--shoulder', '--device'):
    ap.add_argument(k)
ap.add_argument('--slices', nargs='+')
a = ap.parse_args()
spec = json.load(open(a.spec, encoding='utf-8'))
if spec['prompts'] == ['hang']:
    time.sleep(60)
print('@@progress ' + json.dumps({'stage': 'S1', 'done': 1, 'total': 1}), flush=True)
json.dump({'schema': 'goodfire_probe.v1', 'verdict': {'status': 'auto_passed_needs_S9', 'failed': []},
           'params': {'layer': int(a.layer), 'unit': int(a.unit)}, 'slices': a.slices,
           'job': json.load(open(a.job, encoding='utf-8'))}, open(a.out, 'w', encoding='utf-8'))
'''


class GoodfireProbeServiceTests(unittest.TestCase):
    """神经元测试：语料来自（假的）中心普查作业并按 sha256 核对，runner 用假脚本；校验、互斥、取消与结果读取。"""
    SNAP = 'ab' * 32

    def setUp(self):
        import hashlib
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        self.tmp = Path(tempfile.mkdtemp())
        self.files = {'job.json': json.dumps({'job_id': 'CJ-1', 'seq_len': 8, 'skip_head': 1, 'separator_id': 0}).encode(),
                      'tokens_S0.npz': b'slice-0', 'tokens_S1.npz': b'slice-1'}
        self.tamper = set()
        slices = [{'slice': i, 'file': f'tokens_S{i}.npz', 'sha256': hashlib.sha256(self.files[f'tokens_S{i}.npz']).hexdigest()} for i in (0, 1)]
        t = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):  # noqa: N802
                if self.path == '/api/census/jobs':
                    body = {'jobs': [{'job_id': 'CJ-other', 'model': {'snapshot_sha256': 'cd' * 32}},
                                     {'job_id': 'CJ-1', 'model': {'snapshot_sha256': t.SNAP}}]}
                elif self.path == '/api/census/jobs/CJ-1':
                    body = {'job_id': 'CJ-1', 'slices': slices}
                elif self.path.startswith('/api/census/jobs/CJ-1/files/'):
                    name = self.path.rsplit('/', 1)[-1]
                    data = b'evil' if name in t.tamper else t.files.get(name)
                    if data is None:
                        self.send_response(404); self.end_headers(); return
                    self.send_response(200); self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)
                    return
                else:
                    self.send_response(404); self.end_headers(); return
                b = json.dumps(body).encode()
                self.send_response(200); self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)

        self.httpd = ThreadingHTTPServer(('127.0.0.1', 0), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        runner = self.tmp / 'fake_runner.py'
        runner.write_text(FAKE_RUNNER, encoding='utf-8')
        (self.tmp / 'model').mkdir()
        self.svc = S.NodeService(self.tmp / 'svc', python=sys.executable, probes={**FAKE, 'gf_runner': str(runner)})
        self.svc.update_config({'server': f'http://127.0.0.1:{self.httpd.server_address[1]}', 'model_dirs': {self.SNAP: str(self.tmp / 'model')}})

    def tearDown(self):
        self.svc.shutdown()
        self.httpd.shutdown()
        self.httpd.server_close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def wait(self, until=('done', 'failed', 'cancelled')):
        deadline = time.time() + 30
        while time.time() < deadline:
            st = self.svc.probe_status()
            if st['status'] in until:
                return st
            time.sleep(0.05)
        self.fail(f'probe did not finish: {self.svc.probe_status()}')

    def test_run_downloads_verified_corpus_and_lists_result(self):
        st = self.svc.start_probe({'snapshot': self.SNAP, 'layer': 30, 'unit': 9517, 'n_slices': 2, 'prompts': ['你好'],
                                   'pairs': [{'a': '香港', 'b': '巴黎'}]})
        self.assertEqual(st['status'], 'preparing')
        st = self.wait()
        self.assertEqual(st['status'], 'done', st)
        self.assertEqual(st['verdict']['status'], 'auto_passed_needs_S9')
        res = self.svc.get_probe(st['id'])
        self.assertEqual(res['params'], {'layer': 30, 'unit': 9517})
        self.assertEqual([Path(x).name for x in res['slices']], ['tokens_S0.npz', 'tokens_S1.npz'])
        self.assertEqual(res['job']['job_id'], 'CJ-1')                     # 选中与模型快照一致的作业
        self.assertEqual([p['id'] for p in self.svc.list_probes()], [st['id']])
        with self.assertRaises(ValueError):
            self.svc.get_probe('../../etc')

    def test_tampered_slice_rejected(self):
        self.tamper.add('tokens_S1.npz')
        self.svc.start_probe({'snapshot': self.SNAP, 'layer': 1, 'unit': 2, 'n_slices': 2})
        st = self.wait()
        self.assertEqual(st['status'], 'failed')
        self.assertIn('sha256', st['error'])
        self.assertFalse((self.tmp / 'svc' / 'corpus' / 'CJ-1' / 'tokens_S1.npz').exists())

    def test_validation_and_exclusion(self):
        bad = [{'snapshot': 'x', 'layer': 1, 'unit': 1}, {'snapshot': self.SNAP, 'layer': -1, 'unit': 1},
               {'snapshot': self.SNAP, 'layer': 1, 'unit': 1, 'alphas': [0, 1, 2]},           # 只有正向
               {'snapshot': self.SNAP, 'layer': 1, 'unit': 1, 'k': 5},
               {'snapshot': self.SNAP, 'layer': 1, 'unit': 1, 'pairs': [{'a': 'x'}]},
               {'snapshot': 'cd' * 32, 'layer': 1, 'unit': 1}]                                # 没有核对过的模型目录
        for b in bad:
            with self.assertRaises(ValueError, msg=b):
                self.svc.start_probe(b)
        self.svc.current = {'kind': 'census_shard'}
        with self.assertRaises(PermissionError):
            self.svc.start_probe({'snapshot': self.SNAP, 'layer': 1, 'unit': 1})
        self.svc.current = None

    def test_cancel_and_contribution_waits(self):
        self.svc.start_probe({'snapshot': self.SNAP, 'layer': 1, 'unit': 1, 'n_slices': 1, 'prompts': ['hang']})
        self.wait(until=('running',))
        self.assertTrue(any('神经元测试' in r for r in self.svc.gates()))       # 测试期间贡献循环不领取
        with self.assertRaises(PermissionError):
            self.svc.start_probe({'snapshot': self.SNAP, 'layer': 1, 'unit': 1})
        self.assertEqual(self.svc.cancel_probe()['status'], 'cancelled')
        self.assertFalse(self.svc.probe_busy())


if __name__ == '__main__':
    unittest.main()
