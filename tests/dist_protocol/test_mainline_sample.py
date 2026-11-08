# -*- coding: utf-8 -*-
"""关联机制工作台的冻结样本端点（mainline_sample_service，Phase 346）：
合成的小型原生目录上核对形状与 float16 字节往返一致；缺原生数据时 404 并说明原因；越界 404。
合成数据只用于验证端点，不代表任何真实模型。"""
import base64
import importlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _decode(o):
    return np.frombuffer(base64.b64decode(o['b64']), dtype='<f2').reshape(o['shape'])


class MainlineSampleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        os.environ['AI2050_NATIVE_DIR'] = str(self.tmp)
        import server.mainline_sample_service as M
        self.M = importlib.reload(M)
        self.d = self.tmp / '_mainline' / 'MA-H2' / M.SNAP

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        os.environ.pop('AI2050_NATIVE_DIR', None)

    def _make(self, n=3, T=12):
        self.d.mkdir(parents=True)
        rng = np.random.default_rng(0)
        pos = ['ans', 'qname', 'scope', 'tname', 'tcol', 'lastcol']
        (self.d / 'manifest.json').write_text(json.dumps({'model_snapshot': 'x' * 64, 'positions': pos, 'mlp_boundary': 'A'}), encoding='utf-8')
        (self.d / 'meta.json').write_text(json.dumps([{'cell': ['doc', 'first', 2, 0, 'single'], 'entity': f'e{i}', 'K': 'red', 'P': 'blue'} for i in range(n)]), encoding='utf-8')
        self.hs = rng.normal(size=(n, 6, 4, 8)).astype(np.float16)
        np.save(self.d / 'hs.npy', self.hs)
        np.save(self.d / 'mlp.npy', rng.normal(size=(n, 3, 10)).astype(np.float16))
        np.save(self.d / 'attn.npy', rng.random(size=(n, 3, 2, T)).astype(np.float16))
        np.save(self.d / 'lens.npy', rng.normal(size=(n, 4, 3)).astype(np.float32))
        np.save(self.d / 'ntok.npy', np.array([T, T - 2, T - 1], dtype=np.int32)[:n])
        np.save(self.d / 'tokpos.npy', np.tile(np.arange(6, dtype=np.int32), (n, 1)))

    def test_missing_native_is_404_with_reason(self):
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as e:
            self.M.samples()
        self.assertEqual(e.exception.status_code, 404)
        self.assertIn('采集节点', e.exception.detail)

    def test_shapes_and_round_trip(self):
        self._make()
        s = self.M.samples()
        self.assertEqual(s['n'], 3)
        self.assertEqual(s['items'][1]['ntok'], 10)
        one = self.M.sample(1)
        np.testing.assert_array_equal(_decode(one['hs']), self.hs[1])
        self.assertEqual(_decode(one['attn']).shape, (3, 2, 10))      # 只截到该条的 token 数
        self.assertEqual(len(one['lens']), 4)
        from fastapi import HTTPException
        with self.assertRaises(HTTPException):
            self.M.sample(9)


if __name__ == '__main__':
    unittest.main()
