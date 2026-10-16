# -*- coding: utf-8 -*-
"""runner_v3 的模型放置与权重读取辅助函数：AI2050_MAX_MEMORY 解析、模块设备、稠密权重（含 bitsandbytes 4-bit 反量化）。
真实的分层摆放需要模型，在 Qwen3-14B / GLM4 的小样本核对中检查；这里只测纯函数。"""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / 'shared' / 'dist_protocol' / 'runner_v3.py'


def runner_module():
    spec = importlib.util.spec_from_file_location('runner_v3_placement', RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class PlacementHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rn = runner_module()

    def test_parse_max_memory(self):
        self.assertEqual(self.rn.parse_max_memory('0=12GiB,cpu=20GiB'), {0: '12GiB', 'cpu': '20GiB'})
        self.assertEqual(self.rn.parse_max_memory(' 0 = 12GiB , cpu=20GiB '), {0: '12GiB', 'cpu': '20GiB'})
        self.assertIsNone(self.rn.parse_max_memory(''))
        with self.assertRaises(ValueError):
            self.rn.parse_max_memory('0:12GiB')

    def test_dense_weight_plain_linear(self):
        import torch
        lin = torch.nn.Linear(8, 4, bias=False)
        w = self.rn.dense_weight(lin)
        self.assertEqual(w.dtype, torch.float32)
        self.assertTrue(torch.equal(w, lin.weight.detach().float()))
        self.assertEqual(self.rn.module_device(lin).type, 'cpu')

    def test_dense_weight_rejects_meta(self):
        import torch
        with torch.device('meta'):
            lin = torch.nn.Linear(8, 4, bias=False)
        with self.assertRaises(RuntimeError):
            self.rn.dense_weight(lin)
        self.assertIsNone(self.rn.module_device(lin))

    def test_dense_weight_dequantizes_nf4(self):
        import torch
        if not torch.cuda.is_available() or importlib.util.find_spec('bitsandbytes') is None:
            self.skipTest('需要 CUDA 与 bitsandbytes')
        import bitsandbytes as bnb
        torch.manual_seed(0)
        W = torch.randn(32, 64) * 0.05
        lin = bnb.nn.Linear4bit(64, 32, bias=False, quant_type='nf4', compute_dtype=torch.bfloat16)
        lin.weight = bnb.nn.Params4bit(W.clone(), requires_grad=False, quant_type='nf4')
        lin = lin.cuda()
        self.assertNotEqual(tuple(lin.weight.shape), (32, 64))          # 打包后的形状：直接读会出错
        dense = self.rn.dense_weight(lin)
        self.assertEqual(tuple(dense.shape), (32, 64))
        rel = float((dense - W).norm() / W.norm())
        self.assertLess(rel, 0.15)                                        # NF4 量化误差量级
        x = torch.randn(3, 64)
        y_mod = lin(x.cuda().to(torch.bfloat16)).float().cpu()
        self.assertLess(float((y_mod - x @ dense.T).norm() / y_mod.norm()), 0.05)   # 与模块前向一致


if __name__ == '__main__':
    unittest.main()
