# -*- coding: utf-8 -*-
"""结果上传的抗中断：中心把分块落盘（重启后可续传）、同一 seq 重传覆盖、finish 重放只入库一次、
内容不合格丢弃而分块不全保留、过期上传清理、上传编号不能越出上传目录；
节点只对网络错误、5xx、429 重试，4xx 不重试，「停止」立即结束重试等待。"""
import base64
import hashlib
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


def pack(files: dict) -> bytes:
    table = json.dumps(list(files), ensure_ascii=False).encode('utf-8')
    return len(table).to_bytes(4, 'big') + table + b''.join(files.values())


class CenterUploadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.env = {k: os.environ.get(k) for k in ('AI2050_SNAPSHOT_DISABLE', 'AI2050_NATIVE_DIR')}
        os.environ['AI2050_SNAPSHOT_DISABLE'] = '1'
        os.environ['AI2050_NATIVE_DIR'] = str(cls.tmp / 'native')
        import distributed_service as D
        D._DATA_DIR = cls.tmp / 'dist'
        D._DATA_DIR.mkdir()
        D._conn = None
        D._tls = threading.local()
        cls.D = D
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        app = FastAPI()
        app.include_router(D.router)
        cls.c = TestClient(app)
        reg = cls.c.post('/api/nodes/register', json={'name': 'n1'}).json()
        cls.h = {'X-Node-Token': reg['node_token']}
        cls.h2 = {'X-Node-Token': cls.c.post('/api/nodes/register', json={'name': 'n2'}).json()['node_token']}
        row = D._db().execute('SELECT tm_id, design_sha FROM templates LIMIT 1').fetchone()
        cls.tm_id, cls.design = row['tm_id'], row['design_sha']

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

    def init(self, files: dict, headers=None) -> str:
        manifest = [{'path': k, 'sha256': hashlib.sha256(v).hexdigest(), 'size': len(v)} for k, v in files.items()]
        r = self.c.post('/api/results/upload/init', headers=headers or self.h, json={
            'tm_id': self.tm_id, 'design_sha': self.design, 'model_id': 'm', 'kind': 'smoke',
            'summary': {'note': 'upload test'}, 'manifest': manifest})
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()['upload_id']

    def chunk(self, uid: str, seq: int, data: bytes, headers=None):
        return self.c.post(f'/api/results/upload/{uid}/chunk', headers=headers or self.h,
                           json={'seq': seq, 'data_b64': base64.b64encode(data).decode('ascii')})

    def finish(self, uid: str, headers=None):
        return self.c.post(f'/api/results/upload/{uid}/finish', headers=headers or self.h)

    def n_results(self) -> int:
        return self.D._db().execute('SELECT COUNT(*) AS n FROM results').fetchone()['n']

    def test_chunks_on_disk_resend_and_replay(self):
        files = {'summary.json': b'{"a": 1}', 'extra.bin': os.urandom(3000)}
        blob = pack(files)
        uid = self.init(files)
        half = len(blob) // 2
        self.assertEqual(self.chunk(uid, 0, b'garbage').status_code, 200)
        self.assertEqual(self.chunk(uid, 0, blob[:half]).status_code, 200)     # 同一 seq 重传覆盖
        # 分块在磁盘上，不在进程内存里：中心重启后同一 upload_id 照常可用
        self.assertTrue((self.D._DATA_DIR / 'uploads' / uid / '000000.part').is_file())
        self.assertEqual(self.chunk(uid, 1, blob[half:]).status_code, 200)
        self.assertEqual(self.chunk(uid, 0, b'x', headers=self.h2).status_code, 403)
        before = self.n_results()
        first = self.finish(uid)
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(first.json()['sha'], hashlib.sha256(blob).hexdigest())
        again = self.finish(uid)                                                  # 节点没收到响应而重试
        self.assertEqual(again.status_code, 200, again.text)
        self.assertTrue(again.json()['idempotent_replay'])
        self.assertEqual(again.json()['sha'], first.json()['sha'])
        self.assertEqual(self.n_results(), before + 1)
        self.assertEqual(list((self.D._DATA_DIR / 'uploads' / uid).glob('*.part')), [])
        self.assertEqual(self.chunk(uid, 0, b'late').status_code, 409)

    def test_incomplete_kept_bad_content_dropped(self):
        files = {'summary.json': b'{"b": 2}'}
        uid = self.init(files)
        self.assertEqual(self.finish(uid).status_code, 409)                    # 一块都没传：保留，可以继续传
        self.assertEqual(self.chunk(uid, 0, pack(files)).status_code, 200)
        self.assertEqual(self.finish(uid).status_code, 200)

        uid = self.init({'summary.json': b'{"c": 3}'})
        self.assertEqual(self.chunk(uid, 0, pack({'summary.json': b'{"c": 4}'})).status_code, 200)
        self.assertEqual(self.finish(uid).status_code, 422)                    # sha256 不符：丢弃
        self.assertEqual(self.finish(uid).status_code, 404)
        self.assertEqual(self.chunk(uid, 99, b'x').status_code, 404)

    def test_bad_ids_and_range(self):
        for bad in ('..', 'U-..%2F..%2Fx', 'U-zzzzzzzzzz', 'U-0123456789ab'):
            self.assertIn(self.chunk(bad, 0, b'x').status_code, (404, 405), bad)
        uid = self.init({'summary.json': b'{}'})
        self.assertEqual(self.chunk(uid, 50, b'x').status_code, 422)

    def test_expired_uploads_removed(self):
        uid = self.init({'summary.json': b'{"old": 1}'})
        d = self.D._DATA_DIR / 'uploads' / uid
        old = time.time() - self.D.UPLOAD_TTL - 60
        os.utime(d, (old, old))
        self.init({'summary.json': b'{"new": 1}'})                              # 每次登记新上传时清理过期的
        self.assertFalse(d.exists())
        self.assertEqual(self.chunk(uid, 0, b'x').status_code, 404)


class NodeRetryTests(unittest.TestCase):
    def setUp(self):
        import node_agent as A
        self.A = A
        self.delays = A.RETRY_DELAYS
        A.RETRY_DELAYS = (0, 0, 0)
        A.ABORT.clear()

    def tearDown(self):
        self.A.RETRY_DELAYS = self.delays
        self.A.ABORT.clear()

    def test_transient_retried_then_succeeds(self):
        calls = []

        def fn():
            calls.append(1)
            if len(calls) < 3:
                raise self.A.TransientError('502')
            return 'ok'
        self.assertEqual(self.A.with_retry(fn, 'x'), 'ok')
        self.assertEqual(len(calls), 3)

    def test_gives_up_after_all_attempts(self):
        calls = []

        def fn():
            calls.append(1)
            raise self.A.TransientError('down')
        with self.assertRaises(self.A.TransientError):
            self.A.with_retry(fn, 'x')
        self.assertEqual(len(calls), len(self.A.RETRY_DELAYS) + 1)

    def test_client_errors_not_retried(self):
        calls = []

        def fn():
            calls.append(1)
            raise RuntimeError('422 design_sha mismatch')
        with self.assertRaises(RuntimeError):
            self.A.with_retry(fn, 'x')
        self.assertEqual(len(calls), 1)

    def test_abort_stops_waiting(self):
        self.A.RETRY_DELAYS = (30,)
        self.A.ABORT.set()
        t0 = time.time()
        with self.assertRaises(self.A.TransientError):
            self.A.with_retry(lambda: (_ for _ in ()).throw(self.A.TransientError('down')), 'x')
        self.assertLess(time.time() - t0, 5)

    def test_error_classification(self):
        import io
        import urllib.error
        for code, transient in ((502, True), (503, True), (429, True), (404, False), (409, False), (422, False)):
            e = urllib.error.HTTPError('http://x', code, 'm', {}, io.BytesIO(b''))
            err = self.A._http_error('POST', 'http://x', e, b'{"detail": "d"}')
            self.assertEqual(isinstance(err, self.A.TransientError), transient, code)
            self.assertIsInstance(err, RuntimeError)
        with self.assertRaises(self.A.TransientError):                        # 端口没人监听：网络错误
            self.A.http_json('http://127.0.0.1:9/api/x', timeout=2)


if __name__ == '__main__':
    unittest.main()
