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
        cls.tmp = Path(tempfile.mkdtemp())
        D._DATA_DIR = cls.tmp
        D._conn = None
        D._tls = threading.local()
        PS._file_cache.clear()

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


if __name__ == '__main__':
    unittest.main()
