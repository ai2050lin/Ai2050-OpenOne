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
        cls.env = {k: os.environ.get(k) for k in ('AI2050_SNAPSHOT_DISABLE', 'AI2050_NATIVE_DIR')}
        os.environ['AI2050_SNAPSHOT_DISABLE'] = '1'          # 测试里不拉起子进程，直接调用 run_once
        os.environ['AI2050_NATIVE_DIR'] = tempfile.mkdtemp()  # 不读本机真实普查（否则会生成单元卡附加包）
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

    def test_step_killed_last_time_is_skipped(self):
        import json
        self.W.run_once(log=lambda *_: None)
        fp = self.W.fingerprint()
        (self.tmp / 'snapshots' / 'catalog.json').unlink()
        (self.tmp / 'snapshots' / '_running.json').write_text(       # 模拟上次在 catalog 中被系统杀掉
            json.dumps({'stage': 'catalog', 'fingerprint': fp}), encoding='utf-8')
        out = self.W.run_once(log=lambda *_: None)
        self.assertNotIn('catalog', out['computed'])                   # 同一数据指纹下不再重试
        self.assertFalse((self.tmp / 'snapshots' / 'catalog.json').exists())
        self.assertFalse((self.tmp / 'snapshots' / '_running.json').exists())
        skipped = json.loads((self.tmp / 'snapshots' / '_skipped.json').read_text(encoding='utf-8'))
        self.assertIn('catalog', skipped['stages'])
        db = self.D._db()                                              # 数据变化：再试一次
        with self.D._db_lock:
            db.execute("INSERT INTO nodes(node_id,token_hash,name,created_at,last_seen) VALUES('NODE-K','x','k',?,?)",
                       (time.time(), time.time()))
            db.commit()
        self.assertIn('catalog', self.W.run_once(log=lambda *_: None)['computed'])

    def test_extra_packs_build_skip_and_serve(self):
        import gzip
        import json
        import datapack as DP
        self.assertEqual(DP.pack_hint('/api/agg/TG-GR-Q/v2?gate=true'), 'tm:TG-GR-Q')
        self.assertEqual(DP.pack_hint('/api/analysis/TG-GR-Q?bucket=b&gate=false'), 'tm:TG-GR-Q')
        self.assertEqual(DP.pack_hint('/api/neurons/unit?model_key=mk&block=3&unit=7&run=r'), 'cards:mk')
        self.assertIsNone(DP.pack_hint('/api/catalog'))
        plan = [{'id': 'tm:TM-X', 'kind': 'template', 'tm_id': 'TM-X', 'label': '实验响应 · TM-X'}]
        calls = []

        raw = self.tmp / 'analysis.json'
        raw.write_text('{"available": true, "n": 3}', encoding='utf-8')

        def fake_entries(tm):
            calls.append(tm)
            return {DP.canonical(f'/api/agg/{tm}/v2?gate=true'): {'tm_id': tm},
                    DP.canonical(f'/api/analysis/{tm}?gate=true&bucket=b'): DP.RawFile(raw)}, []
        version = {'v': 'v1'}
        orig = DP.extras_plan, DP.template_entries, DP.template_version
        DP.extras_plan, DP.template_entries, DP.template_version = (lambda: plan), fake_entries, (lambda tm: version['v'])
        try:
            DP.build_extras(self.tmp, 'c1', log=lambda *_: None)
            DP.build_extras(self.tmp, 'c1', log=lambda *_: None)                              # 同版本：跳过
            self.assertEqual(calls, ['TM-X'])
            version['v'] = 'v2'
            DP.build_extras(self.tmp, 'c1', log=lambda *_: None)                              # 数据变了：重建
            self.assertEqual(calls, ['TM-X', 'TM-X'])
            meta = {e['id']: e for e in DP.extras_meta(self.tmp)}
            self.assertEqual(meta['tm:TM-X']['version'], 'v2')
            resp = self.PS.datapack_extra_download('tm:TM-X')
            doc = json.loads(gzip.decompress(Path(resp.path).read_bytes()))
            self.assertEqual(doc['schema'], 'datapack_extra.v1')
            self.assertIn('/api/agg/TM-X/v2?gate=true', doc['entries'])
            self.assertEqual(doc['entries']['/api/analysis/TM-X?bucket=b&gate=true'], {'available': True, 'n': 3})   # 原样拷贝
            plan.clear()                                                # 模式族下线：附加包随之删除
            DP.build_extras(self.tmp, 'c1', log=lambda *_: None)
            self.assertEqual(DP.extras_meta(self.tmp), [])
            self.assertFalse(Path(resp.path).exists())
        finally:
            DP.extras_plan, DP.template_entries, DP.template_version = orig

    def test_lang_atlas_endpoints_and_datapack(self):
        import gzip
        import json
        from fastapi import HTTPException
        self.assertTrue(self.PS.lang_tree_ep()['pending'])              # 还没有谱图快照
        out = self.W.run_once(log=lambda *_: None)
        self.assertEqual(out['lang_atlas'], 'computed')
        tree = self.PS.lang_tree_ep()
        self.assertFalse(tree['pending'])
        self.assertEqual(tree['schema'], 'lang_tree_status.v1')
        self.assertEqual({t['id'] for t in tree['types']} >= {'KN', 'RS', 'GR'}, True)
        self.assertTrue(all(it['status']['read'] == 0 for it in tree['items']))   # 空数据库：全部未研究
        item = self.PS.lang_item_ep('GR.Q')
        self.assertEqual(item['item']['type'], 'GR')
        self.assertEqual(item['pieces'], [])
        with self.assertRaises(HTTPException):
            self.PS.lang_item_ep('NO.such')
        self.assertEqual(self.PS.model_skeleton_ep()['models'], [])
        with self.assertRaises(HTTPException):
            self.PS.space_overlay_ep('no-model')
        doc = json.loads(gzip.decompress((self.tmp / 'datapack' / 'datapack.json.gz').read_bytes()))
        self.assertIn('/api/lang/tree', doc['entries'])                 # 谱图生成后数据包随之重建
        self.assertIn('/api/lang/item/GR.Q', doc['entries'])
        self.assertIn('/api/model/skeleton', doc['entries'])
        self.assertTrue(doc['version'].endswith(self.W.stored_fingerprint('lang_atlas')))
        again = self.W.run_once(log=lambda *_: None)                   # 数据、语言树与代码都没变：不重建
        self.assertEqual((again['lang_atlas'], again['datapack']), ('skipped', 'skipped'))

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
