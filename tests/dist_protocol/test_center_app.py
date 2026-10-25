# -*- coding: utf-8 -*-
"""开放平台中心入口（deploy/center_app.py）：公开中心上本机工具端点（/api/live/*、/api/models/*）需要管理令牌，
本机中心（AI2050_LOCAL_TOOLS=1）放开；公开配置如实报告 local_tools。"""
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'deploy')]


class CenterAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.env = {k: os.environ.get(k) for k in ('AI2050_LOCAL_TOOLS', 'AI2050_SNAPSHOT_DISABLE', 'AI2050_FRONTEND_DIST')}
        os.environ['AI2050_SNAPSHOT_DISABLE'] = '1'
        os.environ['AI2050_FRONTEND_DIST'] = str(Path(cls.tmp) / 'nofrontend')
        import distributed_service as D
        D._DATA_DIR = Path(cls.tmp) / 'dist'
        D._DATA_DIR.mkdir()
        D._conn = None
        D._tls = threading.local()
        cls.D = D

    @classmethod
    def tearDownClass(cls):
        cls.D._conn = None
        cls.D._tls = threading.local()
        for k, v in cls.env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def client(self, local: bool):
        from fastapi.testclient import TestClient
        import center_app
        if local:
            os.environ['AI2050_LOCAL_TOOLS'] = '1'
        else:
            os.environ.pop('AI2050_LOCAL_TOOLS', None)
        return TestClient(center_app.create_app())

    def test_public_center_guards_local_tools(self):
        c = self.client(local=False)
        self.assertEqual(c.get('/api/health').status_code, 200)
        self.assertFalse(c.get('/api/platform/config').json()['local_tools'])
        self.assertEqual(c.get('/api/models/local').status_code, 403)
        self.assertEqual(c.post('/api/models/download', json={'repo_id': 'Qwen/Qwen3-4B'}).status_code, 403)
        self.assertEqual(c.post('/api/live/connect', json={'model_path': '/etc'}).status_code, 403)
        ok = c.get('/api/models/local', headers={'X-Admin-Token': self.D.ADMIN_TOKEN})
        self.assertEqual(ok.status_code, 200)

    def test_local_center_allows_tools(self):
        c = self.client(local=True)
        self.assertTrue(c.get('/api/platform/config').json()['local_tools'])
        self.assertEqual(c.get('/api/models/local').status_code, 200)


if __name__ == '__main__':
    unittest.main()
