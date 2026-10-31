# -*- coding: utf-8 -*-
"""TopK transcoder（transcoder.v1）：引擎 A 的字典原子，以及「原子比较」的保真度指标。

transcoder 学习一层 MLP 的输入 → 输出映射：
    a = TopK_k( ReLU( (x − b_pre) W_enc + b_enc ) )        a ∈ R^m，每个 token 恰好 ≤ k 个非零
    ŷ = a W_dec + b_dec                                     ŷ ≈ y = MLP(x)
x 是 MLP 的实际输入（post_attention_layernorm 的输出），y 是 MLP 的实际输出。与 SAE 不同，它把 MLP 的计算换成
稀疏特征，特征的 W_dec 行就是该特征写入残差流的方向，因此能给出「特征 → 下游」的计算边。

训练（Gao et al. 2024 的 TopK 做法）：损失 = ‖y − ŷ‖² / Var(y) + aux_coef · AuxK 损失；AuxK 用长期未激活
（dead）特征中前 aux_k 个去拟合残差，减少死特征。W_dec 行在每步之后归一化为单位长度（激活量吸收尺度）。

保真度指标（与「神经元作原子」同口径比较）：
    FVU = Σ‖y − ŷ‖² / Σ‖y − ȳ‖²                     （未解释方差比例，越小越好）
    损失恢复 = (CE_ablate − CE_splice) / (CE_ablate − CE_clean)   （1 = 换上后模型损失不变；0 = 等于消融整层 MLP）
    神经元基线：同一 token 只保留贡献 |A_i|·‖W_down,i‖ 最大的 k 个 MLP 中间单元（A = silu(G)⊙V），其余置零后经 down_proj。
判据固定在 CRITERIA，随结果返回。
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, Optional

import numpy as np

SCHEMA = 'transcoder.v1'
CRITERIA = {
    'schema': SCHEMA,
    # 原子比较（同一留出 token、同一 k）：特征原子胜出 = FVU 更低 且 损失恢复更高
    'atom_compare': 'feature wins iff fvu_feature < fvu_neuron and loss_recovered_feature > loss_recovered_neuron at the same k',
    'dead_after_tokens': 1_000_000,     # 训练中连续这么多 token 未激活记为 dead
    'concentration_top': 100,           # 集中度：最强 100 次激活中出现最多的 token 占比
}


def fvu(y: np.ndarray, y_hat: np.ndarray) -> float:
    y = np.asarray(y, np.float64)
    y_hat = np.asarray(y_hat, np.float64)
    return float(((y - y_hat) ** 2).sum() / max(((y - y.mean(0)) ** 2).sum(), 1e-30))


def loss_recovered(ce_clean: float, ce_splice: float, ce_ablate: float) -> float:
    return float((ce_ablate - ce_splice) / (ce_ablate - ce_clean)) if ce_ablate != ce_clean else float('nan')


def token_concentration(top_tokens: Iterable[int]) -> float:
    """最强激活位置上 token 的集中度：出现最多的 token 占比（1 = 全是同一个 token）。"""
    t = list(top_tokens)
    if not t:
        return float('nan')
    _, c = np.unique(np.asarray(t), return_counts=True)
    return float(c.max() / len(t))


def _torch():
    import torch
    return torch


class TopKTranscoder:
    """纯参数容器 + 前向；训练见 train()。参数为 torch 张量（float32）。"""

    def __init__(self, d_in: int, d_out: int, m: int, k: int, *, device='cpu', seed: int = 0):
        torch = _torch()
        g = torch.Generator().manual_seed(seed)
        self.d_in, self.d_out, self.m, self.k = d_in, d_out, m, k
        w = torch.randn(m, d_out, generator=g)
        w = w / w.norm(dim=1, keepdim=True)
        self.W_dec = w.to(device)
        self.W_enc = (w.T.clone() * math.sqrt(d_out / d_in) if d_in == d_out else torch.randn(d_in, m, generator=g) / math.sqrt(d_in)).to(device)
        self.b_enc = torch.zeros(m, device=device)
        self.b_pre = torch.zeros(d_in, device=device)
        self.b_dec = torch.zeros(d_out, device=device)
        self.x_scale = 1.0          # 输入、输出各一个标量尺度：训练前按均值范数标准化，保存时记录
        self.y_scale = 1.0

    def params(self):
        return [self.W_enc, self.b_enc, self.W_dec, self.b_dec, self.b_pre]

    def pre_acts(self, x):
        return (x / self.x_scale - self.b_pre) @ self.W_enc + self.b_enc

    def encode(self, x, k: Optional[int] = None):
        torch = _torch()
        z = torch.relu(self.pre_acts(x))
        v, i = z.topk(k or self.k, dim=-1)
        return torch.zeros_like(z).scatter_(-1, i, v)

    def decode(self, a):
        return (a @ self.W_dec + self.b_dec) * self.y_scale

    def __call__(self, x, k: Optional[int] = None):
        return self.decode(self.encode(x, k))

    def state(self) -> Dict[str, np.ndarray]:
        return {'W_enc': self.W_enc.cpu().numpy(), 'b_enc': self.b_enc.cpu().numpy(), 'W_dec': self.W_dec.cpu().numpy(),
                'b_dec': self.b_dec.cpu().numpy(), 'b_pre': self.b_pre.cpu().numpy(),
                'meta': np.array([self.d_in, self.d_out, self.m, self.k]), 'scales': np.array([self.x_scale, self.y_scale])}

    @classmethod
    def from_state(cls, st, device='cpu') -> 'TopKTranscoder':
        torch = _torch()
        d_in, d_out, m, k = (int(v) for v in st['meta'])
        t = cls(d_in, d_out, m, k, device=device)
        for name in ('W_enc', 'b_enc', 'W_dec', 'b_dec', 'b_pre'):
            setattr(t, name, torch.as_tensor(np.asarray(st[name]), dtype=torch.float32, device=device))
        t.x_scale, t.y_scale = (float(v) for v in st['scales'])
        return t


def train(sample: Callable[[int], tuple], *, d_in: int, d_out: int, m: int, k: int, steps: int, batch: int = 4096,
          lr: float = 2e-4, aux_k: int = 256, aux_coef: float = 1 / 32, dead_after: Optional[int] = None,
          device: str = 'cpu', seed: int = 0, log: Optional[Callable[[dict], None]] = None, log_every: int = 200):
    """sample(n) → (x[n, d_in], y[n, d_out])，torch 张量（任意浮点类型，在 device 上）。返回 (transcoder, 训练曲线)。"""
    torch = _torch()
    torch.manual_seed(seed)
    t = TopKTranscoder(d_in, d_out, m, k, device=device, seed=seed)
    with torch.no_grad():                               # 标量尺度与偏置初始化：用前几批的均值
        xs, ys = zip(*(sample(batch) for _ in range(4)))
        x0, y0 = torch.cat(xs).float(), torch.cat(ys).float()
        t.x_scale = float(x0.norm(dim=1).mean() / math.sqrt(d_in))
        t.y_scale = float(y0.norm(dim=1).mean() / math.sqrt(d_out))
        t.b_pre.copy_((x0 / t.x_scale).mean(0))
        t.b_dec.copy_((y0 / t.y_scale).mean(0))
    for p in t.params():
        p.requires_grad_(True)
    opt = torch.optim.Adam(t.params(), lr=lr, betas=(0.9, 0.999))
    since = torch.zeros(m, device=device)
    dead_after = dead_after or CRITERIA['dead_after_tokens']
    curve = []
    for step in range(steps):
        x, y = sample(batch)
        x, y = x.float(), y.float() / t.y_scale
        z = torch.relu(t.pre_acts(x))
        v, i = z.topk(k, dim=-1)
        a = torch.zeros_like(z).scatter(-1, i, v)
        y_hat = a @ t.W_dec + t.b_dec
        err = y - y_hat
        var = (y - y.mean(0)).pow(2).sum(-1).mean()
        loss = err.pow(2).sum(-1).mean() / var
        dead = since >= dead_after
        aux = torch.zeros((), device=device)
        if dead.any():                                  # AuxK：只用死特征去拟合当前残差
            zd = z.masked_fill(~dead, 0.0)
            kk = min(aux_k, int(dead.sum()))
            vd, idd = zd.topk(kk, dim=-1)
            e_hat = torch.zeros_like(zd).scatter(-1, idd, vd) @ t.W_dec
            aux = (err.detach() - e_hat).pow(2).sum(-1).mean() / var
            loss = loss + aux_coef * aux
        opt.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():                           # 去掉梯度中与 W_dec 行平行的分量，再归一化
            t.W_dec.grad -= (t.W_dec.grad * t.W_dec).sum(-1, keepdim=True) * t.W_dec
        opt.step()
        with torch.no_grad():
            t.W_dec /= t.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
            fired = torch.zeros(m, dtype=torch.bool, device=device)
            fired[i.reshape(-1)] = True
            since = torch.where(fired, torch.zeros_like(since), since + batch)
        if log_every and (step % log_every == 0 or step == steps - 1):
            rec = {'step': step, 'fvu': float((err.pow(2).sum(-1).mean() / var).detach()), 'aux': float(aux.detach()),
                   'dead_frac': float(dead.float().mean())}
            curve.append(rec)
            if log:
                log(rec)
    for p in t.params():
        p.requires_grad_(False)
    return t, curve


def neuron_topk_recon(A, W_down_T, k: int):
    """神经元基线：A[n, d_ff]（down_proj 输入），W_down_T[d_ff, d_out]；每个 token 只保留 |A_i|·‖W_down,i‖ 最大的 k 个。"""
    torch = _torch()
    score = A.abs() * W_down_T.norm(dim=1)[None, :]
    i = score.topk(k, dim=-1).indices
    keep = torch.zeros_like(A).scatter_(-1, i, A.gather(-1, i))
    return keep @ W_down_T
