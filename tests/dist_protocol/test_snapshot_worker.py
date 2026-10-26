# -*- coding: utf-8 -*-
"""后台计算子进程（deploy/snapshot_worker.py）：快照写文件、数据指纹不变时跳过、数据变化后重算；
网站进程（platform_service.snapshot）只读文件，没有文件时返回 pending。"""
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'deploy')]


class SnapshotWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = {k: os.environ.get(k) for k in ('AI2050_SNAPSHOT_DISABLE',)}
        os.environ['AI2050_SNAPSHOT_DISABLE'] = '1'          # 测试里不拉起子进程，直接调用 run_once
        import distributed_service as D
        import platform_service as PS
        import snapshot_worker as W
        cls.D, cls.PS, cls.W = D, PS, W

    def setUp(self):                                           # 每个测试独立的数据目录
        self.tmp = Path(tempfile.mkdtemp())
        self.D._DATA_DIR = self.tmp
        self.D._conn = None
        self.D._tls = threading.local()
        self.PS._file_cache.clear()
        import census_service as CS
        CS._schema_done = False                                # 换了数据库：普查表要重新建

    @classmethod
    def tearDownClass(cls):
        cls.D._conn = None
        cls.D._tls = threading.local()
        for k, v in cls.env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_write_skip_recompute_and_read(self):
        self.assertTrue(self.PS.snapshot('catalog')['pending'])          # 还没有文件
        first = self.W.run_once(log=lambda *_: None)
        self.assertEqual(sorted(first['computed']), sorted(self.W.NAMES))
        for n in self.W.NAMES:
            self.assertTrue((self.tmp / 'snapshots' / f'{n}.json').is_file())
        got = self.PS.snapshot('catalog')
        self.assertFalse(got['pending'])
        self.assertEqual(got['data']['schema'], 'catalog.v1')
        second = self.W.run_once(log=lambda *_: None)                  # 指纹没变：全部跳过
        self.assertEqual(second['computed'], [])
        self.assertEqual(sorted(second['skipped']), sorted(self.W.NAMES))
        self.assertEqual(second['analyses'], 'skipped')
        db = self.D._db()
        with self.D._db_lock:
            db.execute("INSERT INTO nodes(node_id,token_hash,name,created_at,last_seen) VALUES('NODE-T','x','t',?,?)",
                       (time.time(), time.time()))
            db.commit()
        third = self.W.run_once(log=lambda *_: None)                   # 数据变化：重算
        self.assertEqual(sorted(third['computed']), sorted(self.W.NAMES))

    def test_datapack_build_skip_and_serve(self):
        import gzip
        import json
        import datapack as DP
        self.assertEqual(DP.canonical('/api/neurons/space?model_key=m&layer=30'), '/api/neurons/space?layer=30&model_key=m')
        self.assertEqual(DP.canonical('/api/x?b=2&a=%E4%B8%AD'), '/api/x?a=中&b=2')     # 与客户端 dataStore.canonical 同一规则
        self.W.run_once(force=True, log=lambda *_: None)
        meta = DP.read_meta(self.tmp)
        self.assertEqual(meta['schema'], 'datapack.v1')
        self.assertEqual(meta['errors'], [])
        self.assertEqual(self.PS._datapack_meta()['version'], meta['version'])           # 站点配置带版本，客户端据此提示更新
        resp = self.PS.datapack_download()
        self.assertEqual(Path(resp.path), self.tmp / 'datapack' / 'datapack.json.gz')
        doc = json.loads(gzip.decompress(Path(resp.path).read_bytes()))
        self.assertEqual(doc['version'], meta['version'])
        for key in ('/api/catalog', '/api/templates', '/api/platform/overview', '/api/platform/snapshot/catalog'):
            self.assertIn(key, doc['entries'])
        self.assertFalse(doc['entries']['/api/platform/snapshot/catalog']['pending'])
        again = self.W.run_once(log=lambda *_: None)                  # 指纹没变：数据包不重建
        self.assertEqual(again['datapack'], 'skipped')


if __name__ == '__main__':
    unittest.main()
