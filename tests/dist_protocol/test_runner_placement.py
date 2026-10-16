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

    def test_dense_weight_meta_reads_checkpoint(self):
        # 分层摆放时卸载层只剩 meta 占位：没有检查点参数名时报错；给出时从 safetensors 读取原始权重
        import json
        import shutil
        import tempfile
        import torch
        from safetensors.torch import save_file
        with torch.device('meta'):
            lin = torch.nn.Linear(8, 4, bias=False)
        with self.assertRaises(RuntimeError):
            self.rn.dense_weight(lin)
        self.assertIsNone(self.rn.module_device(lin))
        tmp = Path(tempfile.mkdtemp())
        try:
            W = torch.randn(4, 8).to(torch.bfloat16)
            save_file({'model.layers.3.self_attn.o_proj.weight': W, 'other': torch.zeros(2)}, str(tmp / 'part-1.safetensors'))
            (tmp / 'model.safetensors.index.json').write_text(json.dumps(
                {'weight_map': {'model.layers.3.self_attn.o_proj.weight': 'part-1.safetensors', 'other': 'part-1.safetensors'}}),
                encoding='utf-8')
            got = self.rn.dense_weight(lin, (str(tmp), 'model.layers.3.self_attn.o_proj.weight'))
            self.assertTrue(torch.equal(got, W.float()))
            with self.assertRaises(RuntimeError):                              # 形状不一致
                self.rn.dense_weight(lin, (str(tmp), 'other'))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

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
