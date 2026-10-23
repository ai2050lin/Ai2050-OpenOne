# -*- coding: utf-8 -*-
"""MLP 神经元自然语料普查（mlp_census.v1）的累加器：逐层、逐单元的可合并统计。

对每层每个 MLP 中间单元 j（单元输出 A = silu(G) ⊙ V），在 GPU 上流式累计：
  - hist：按激活值分箱（symlog：|A| < 1e-3 为零箱，两侧各 32 个对数箱覆盖 1e-3 … 1e2）× 符号区（++ / −− / +− / −+）的计数；
  - n、ΣA、ΣA²（float64）、min、max；
  - 正极 top-k 与负极 bottom-k：激活值、G、V 与全局 token 位置（source_id << 40 | 来源内偏移）。
合并口径：计数与和逐项相加，min/max 取极值，top-k 按数值合并——分片合并结果与单机一次累计相同（测试覆盖）。
证据边界：这是自然语料上的描述统计，不说明单元被模型使用；极值片段天然有选择偏差。
"""
from __future__ import annotations

from typing import Dict

import numpy as np

N_SIDE = 32
LO, HI = 1e-3, 1e2
N_BINS = 2 * N_SIDE + 1                     # 负侧 0..31（|A| 由大到小），零箱 32，正侧 33..64（|A| 由小到大）
REGIMES = ('++', '--', '+-', '-+')
SRC_SHIFT = 40


def bin_edges() -> np.ndarray:
    """正侧箱的下边界（N_SIDE+1 个，最后一个为上界）；负侧对称。"""
    return np.geomspace(LO, HI, N_SIDE + 1)


def global_index(source_id: int, offset):
    return (int(source_id) << SRC_SHIFT) | offset


def split_index(gidx):
    gidx = np.asarray(gidx, dtype=np.int64)
    return gidx >> SRC_SHIFT, gidx & ((1 << SRC_SHIFT) - 1)


class LayerAcc:
    """单层累加器（torch 张量，可在 GPU 上）。"""

    def __init__(self, n_units: int, k: int = 32, device='cpu'):
        import torch
        self.F, self.k, self.device = int(n_units), int(k), device
        f = self.F
        self.hist = torch.zeros(f * N_BINS * 4, dtype=torch.int64, device=device)
        self.n = 0
        self.s1 = torch.zeros(f, dtype=torch.float64, device=device)
        self.s2 = torch.zeros(f, dtype=torch.float64, device=device)
        self.amin = torch.full((f,), float('inf'), device=device)
        self.amax = torch.full((f,), float('-inf'), device=device)
        self.top = {'val': torch.full((f, k), float('-inf'), device=device), 'idx': torch.full((f, k), -1, dtype=torch.int64, device=device),
                    'g': torch.zeros((f, k), device=device), 'v': torch.zeros((f, k), device=device)}
        self.bot = {'val': torch.full((f, k), float('inf'), device=device), 'idx': torch.full((f, k), -1, dtype=torch.int64, device=device),
                    'g': torch.zeros((f, k), device=device), 'v': torch.zeros((f, k), device=device)}
        self._step = (np.log10(HI) - np.log10(LO)) / N_SIDE

    def _bins(self, A):
        import torch
        a = A.abs()
        k = torch.clamp(torch.floor((torch.log10(a.clamp_min(LO)) - np.log10(LO)) / self._step), 0, N_SIDE - 1).long()
        b = torch.where(A > 0, N_SIDE + 1 + k, N_SIDE - 1 - k)
        return torch.where(a < LO, torch.full_like(b, N_SIDE), b)

    @staticmethod
    def _regime(G, V):
        import torch
        return torch.where(G > 0, torch.where(V > 0, 0, 2), torch.where(V < 0, 1, 3))

    def update(self, A, G, V, gidx):
        """A、G、V：(T, F) float32；gidx：(T,) int64 全局 token 位置。"""
        import torch
        T = A.shape[0]
        if T == 0:
            return
        comb = self._bins(A) * 4 + self._regime(G, V)                                   # (T, F)
        flat = comb + (torch.arange(self.F, device=A.device) * (N_BINS * 4))[None, :]
        self.hist += torch.bincount(flat.reshape(-1), minlength=self.F * N_BINS * 4)
        self.n += T
        self.s1 += A.sum(0, dtype=torch.float64)
        self.s2 += (A.double() ** 2).sum(0)
        self.amin = torch.minimum(self.amin, A.min(0).values)
        self.amax = torch.maximum(self.amax, A.max(0).values)
        ix = gidx.to(A.device)[None, :].expand(self.F, T)
        for st, sign in ((self.top, 1.0), (self.bot, -1.0)):
            val = torch.cat([st['val'], A.T], 1)
            sel = torch.topk(sign * val, self.k, dim=1).indices
            st['val'] = torch.gather(val, 1, sel)
            st['idx'] = torch.gather(torch.cat([st['idx'], ix], 1), 1, sel)
            st['g'] = torch.gather(torch.cat([st['g'], G.T], 1), 1, sel)
            st['v'] = torch.gather(torch.cat([st['v'], V.T], 1), 1, sel)

    def to_numpy(self) -> Dict[str, np.ndarray]:
        out = {'hist': self.hist.view(self.F, N_BINS, 4).cpu().numpy().astype(np.int64),
               'n': np.array(self.n, dtype=np.int64), 's1': self.s1.cpu().numpy(), 's2': self.s2.cpu().numpy(),
               'amin': self.amin.cpu().numpy(), 'amax': self.amax.cpu().numpy()}
        for name, st in (('top', self.top), ('bot', self.bot)):
            out[f'{name}_val'] = st['val'].cpu().numpy().astype(np.float32)
            out[f'{name}_idx'] = st['idx'].cpu().numpy()
            out[f'{name}_g'] = st['g'].cpu().numpy().astype(np.float16)
            out[f'{name}_v'] = st['v'].cpu().numpy().astype(np.float16)
        return out


def merge(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """两个 to_numpy() 结果的合并（同一层、同一单元数与 k）。"""
    out = {'hist': a['hist'] + b['hist'], 'n': a['n'] + b['n'], 's1': a['s1'] + b['s1'], 's2': a['s2'] + b['s2'],
           'amin': np.minimum(a['amin'], b['amin']), 'amax': np.maximum(a['amax'], b['amax'])}
    for name, sign in (('top', 1.0), ('bot', -1.0)):
        val = np.concatenate([a[f'{name}_val'], b[f'{name}_val']], 1)
        k = a[f'{name}_val'].shape[1]
        sel = np.argsort(-sign * val, axis=1, kind='stable')[:, :k]
        for f in ('val', 'idx', 'g', 'v'):
            out[f'{name}_{f}'] = np.take_along_axis(np.concatenate([a[f'{name}_{f}'], b[f'{name}_{f}']], 1), sel, 1)
    return out


def summarize_layer(d: Dict[str, np.ndarray]) -> Dict[str, object]:
    """一层的描述摘要：激活率、符号区占比、激活形状（强激活是否集中在少数 token）。"""
    hist = d['hist'].astype(np.float64)                         # (F, 65, 4)
    n = float(d['n'])
    pos = hist[:, N_SIDE + 1:, :].sum((1, 2)) / n
    reg = hist.sum(1) / n                                       # (F, 4)
    strong = hist[:, N_SIDE + 1 + N_SIDE // 2:, :].sum((1, 2)) / n   # A > 10^(−3 + 2.5) ≈ 0.32
    return {'n_tokens': int(n), 'pos_rate_median': float(np.median(pos)),
            'regime_share_mean': dict(zip(REGIMES, map(float, reg.mean(0)))),
            'strong_rate_median': float(np.median(strong)), 'amax_median': float(np.median(d['amax'])),
            'amin_median': float(np.median(d['amin']))}
