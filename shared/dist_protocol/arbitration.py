# -*- coding: utf-8 -*-
"""仲裁规律（arbitration.v1）：上下文与知识冲突时，对数几率怎样随线索与知识强度变化（G4）。纯 numpy；判据固定在 CRITERIA。

每个样本：实体 e（模型先验颜色 P、上下文给出的冲突颜色 K）× 条件（重复次数 r ≥ 1、来源 src、提问范围 scope）。
  LO = log P(K 首 token) − log P(P 首 token)（答案位置）；s = 知识强度 = 无上下文颜色问答中 P 与 K 的完整答案对数概率差（EA-v2 目标）。
规律（只在发现集实体上最小二乘拟合，确认集评估一次）：
  LO ≈ β0 + β_S·S + β_src·src + β_r·log r + β_rS·S·log r + γ·s + γ_S·S·s，  S = 1 表示「在这本书里」（上下文范围），0 表示「在现实世界里」。
  γ 是现实范围里知识强度的作用，γ + γ_S 是上下文范围里的作用；按实体重抽样（bootstrap）给 95% 区间。
"""
from __future__ import annotations

from typing import Dict

import numpy as np

SCHEMA = 'arbitration.v1'
CRITERIA = {
    'schema': SCHEMA,
    'H1_win_min': 0.7,          # r = 1、童话来源：上下文范围 LO > 0 的比例、现实范围 LO < 0 的比例，均 ≥ 0.7
    'H2_r2_min': 0.5,           # 确认集上完整规律的 R²
    'H4_r2_drop_max': 0.10,     # 用实体编码预测的知识强度代替实际强度：R² 下降 ≤ 0.10
    'n_boot': 1000,
    'k_grid': [-6, -3, 0, 3, 6], 'n_random': 2, 'min_swing': 1.0,
    'seed': 0,
}
TERMS = ['1', 'S', 'src', 'logr', 'S_logr', 's', 'S_s']


def design(S, src, r, s) -> np.ndarray:
    S, src, r, s = (np.asarray(x, float) for x in (S, src, r, s))
    lr = np.log(r)
    return np.column_stack([np.ones_like(S), S, src, lr, S * lr, s, S * s])


def fit(X, y) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


def r2(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(1 - ((y - p) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-30))


def bootstrap(X, y, groups, n_boot: int, seed: int) -> Dict[str, list]:
    """按实体重抽样的系数 95% 区间；另给上下文范围的知识作用 γ + γ_S。"""
    ug = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in ug}
    rng = np.random.default_rng(seed)
    B = []
    for _ in range(n_boot):
        rows = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        B.append(fit(X[rows], y[rows]))
    B = np.array(B)
    out = {t: [float(np.quantile(B[:, i], 0.025)), float(np.quantile(B[:, i], 0.975))] for i, t in enumerate(TERMS)}
    g_ctx = B[:, TERMS.index('s')] + B[:, TERMS.index('S_s')]
    out['gamma_context'] = [float(np.quantile(g_ctx, 0.025)), float(np.quantile(g_ctx, 0.975))]
    return out


def law_eval(disc: Dict[str, np.ndarray], conf: Dict[str, np.ndarray], s_key='s', crit=CRITERIA) -> Dict[str, object]:
    """disc / conf：键 S、src、r、s（或 s_key 指定的强度列）、LO、ent。"""
    Xd = design(disc['S'], disc['src'], disc['r'], disc[s_key])
    Xc = design(conf['S'], conf['src'], conf['r'], conf[s_key])
    b = fit(Xd, disc['LO'])
    cues = [TERMS.index(t) for t in ('1', 'S', 'src', 'logr', 'S_logr')]
    bc = fit(Xd[:, cues], disc['LO'])
    ci = bootstrap(Xd, disc['LO'], disc['ent'], crit['n_boot'], crit['seed'])
    return {'coef': dict(zip(TERMS, map(float, b))), 'ci95': ci,
            'R2_conf': r2(conf['LO'], Xc @ b), 'R2_conf_cues_only': r2(conf['LO'], Xc[:, cues] @ bc),
            'R2_disc': r2(disc['LO'], Xd @ b)}
