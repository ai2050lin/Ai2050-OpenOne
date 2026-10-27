# -*- coding: utf-8 -*-
"""逐层神经元摘要（neuron_layers.v1）：植入六类单元按 L30-v1 规则找回分型、命中 token 众数、强激活率口径，
以及层文档 / 层塔的合并。合成数组只用于验证方法，不代表任何真实模型。"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared.dist_protocol import census as C  # noqa: E402
from shared.dist_protocol import neuron_layers as NL  # noqa: E402


def planted():
    """六个单元：0 单 token 检测器、1 上下文 token 检测器、2 稀疏强激活、3 稠密、4 负极主导、5 其他。"""
    F, n, K = 6, 10000, 32
    sb = NL.strong_bin_index()
    hist = np.zeros((F, 2 * C.N_SIDE + 1, 4), dtype=np.int64)
    zero = C.N_SIDE                                   # 中间箱（|A| 很小）
    pos_strong, neg_strong = sb + 1, 5
    plan = {0: {pos_strong: 500}, 1: {pos_strong: 500}, 2: {pos_strong + 10: 5}, 3: {pos_strong: 2000},
            4: {neg_strong: 500, pos_strong: 100}, 5: {pos_strong: 500}}
    for j, bins in plan.items():
        for b, c in bins.items():
            hist[j, b, 0] = c
        hist[j, zero, 0] = n - sum(bins.values())
    stream = (np.arange(5000) % 997 + 1000).astype(np.int32)  # 默认每个位置 token 都不同
    top_idx = np.full((F, K), -1, dtype=np.int64)
    for j in range(F):
        offs = 1000 + 100 * j + 2 * np.arange(K)               # 各单元的极值位置互不重叠
        top_idx[j] = offs                                       # source 0：gidx 就是偏移
    stream[top_idx[0]] = 7                                      # 单元 0：命中 token 全是 7
    stream[top_idx[1] - 1] = 9                                  # 单元 1：前一个 token 全是 9，命中 token 各不相同
    amax = np.array([5, 5, 50, 5, 5, 5], dtype=np.float32)
    z = {'hist': hist, 'n': np.array(n), 's1': np.zeros(F), 's2': np.full(F, float(n)),
         'amin': -amax, 'amax': amax, 'top_idx': top_idx, 'bot_idx': np.full((F, K), -1, dtype=np.int64),
         'readout_top': np.arange(F * 10, dtype=np.int32).reshape(F, 10)}
    return z, [stream]


class NeuronLayerTests(unittest.TestCase):
    def test_planted_types_recovered(self):
        z, streams = planted()
        base = NL.summarize(z, streams, lambda i: f't{i}')
        col = base['columns']
        self.assertEqual(col['type'], [0, 1, 2, 3, 4, 5])
        self.assertEqual(col['hit'][0], 't7')                        # T1：命中 token 众数
        self.assertEqual(col['hit'][1], 't9')                        # T2：前一个 token 众数
        self.assertAlmostEqual(col['dens'][3], 0.2, places=6)        # 强激活率按普查直方图口径
        self.assertGreater(col['neg'][4], col['dens'][4])
        self.assertEqual(col['ro'][2], 't20')                        # 直接读出首位 token
        self.assertEqual(base['n_units'], 6)

    def test_concentration_below_threshold_is_not_detector(self):
        z, streams = planted()
        stream = streams[0].copy()
        stream[z['top_idx'][0][:20]] = np.arange(20) + 3000            # 单元 0 的命中 token 只剩 12/32 相同
        base = NL.summarize(z, [stream], lambda i: f't{i}')
        self.assertNotEqual(base['columns']['type'][0], 0)
        self.assertLess(base['columns']['pconc'][0], 0.75)

    def test_layer_and_tower_docs(self):
        z, streams = planted()
        base = NL.summarize(z, streams, lambda i: f't{i}')
        sel = [{'file': 'X.json', 'tm_id': 'TG-GR-Q', 'factor': 'mood', 'hi': 'interrog', 'lo': 'decl', 'item': 'GR.Q',
                'by_readout': {'final': [3, 0]}, 'top': {'final': {'0': [[1, 2.5, 'interrog']]}}}]
        doc = NL.layer_doc(base, model_key='mk', run='r', block=0, explained={2: '文本可复现'},
                           selectivity=[{'file': 'X.json', 'n_selected': {'final': 3}, 'top': {'final': [[1, 2.5, 'interrog']]}}])
        self.assertEqual(doc['explained'], {'2': '文本可复现'})
        self.assertEqual(len(doc['type_names']), 6)
        tw = NL.tower_doc('mk', 'r', 2, 6, {0: base}, {0: {2: '文本可复现', 3: '未解释'}}, sel)
        self.assertEqual(tw['blocks'][0]['types'], [1, 1, 1, 1, 1, 1])
        self.assertEqual(tw['blocks'][0]['explained'], {'文本可复现': 1, '未解释': 1})
        self.assertFalse(tw['blocks'][1]['summary'])                   # 没有摘要的块如实标出
        self.assertEqual(tw['selectivity'][0]['by_readout'], {'final': [3, 0]})


if __name__ == '__main__':
    unittest.main()
