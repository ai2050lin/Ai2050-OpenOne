# -*- coding: utf-8 -*-
"""模式向量图谱（mode_atlas.v1）：G3 句型模式是否是与内容解耦、可叠加的方向，以及组合（否定 × 真假）是否被编码。纯 numpy；判据固定在 CRITERIA。

输入：H = [n_entity, 8, d]，同一实体在 2×2×2 因子格（q 疑问、n 否定、t 类别真假；格序 c = 4q + 2n + t）下、句末同一 token 处的残差。
  去掉实体均值：G[e, c] = H[e, c] − mean_c H[e, c]（内容的主体被扣除，剩下的是格之间的差）。
  效应（对比编码 s = ±1/2）：对因子子集 S，m_S = mean_e Σ_c (Π_{F∈S} 2 s_F(c)) G[e, c] / 8 · 2^{|S|}，即标准 2^3 析因效应；
  格均值模型：Ĝ[c] = 发现集 mean_e G[e, c]（与内容无关的全部格结构，主效应 + 全部交互）；
  主效应模型：Ĝ[c] = Σ_{F} s_F(c) m_F。
  R²(模型) = 1 − Σ‖G − Ĝ‖² / Σ‖G‖²，在确认集实体上计算。
  · 解耦：格均值模型的 R²——格之间的差有多少与具体实体无关；
  · 可加：主效应模型 R² / 格均值模型 R²；
  · 组合：真值 = t XOR n 在纯可加编码里线性不可分；句末状态上的线性探针若能读出真值，说明编码含 n×t 交互（组合）。
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, Sequence

import numpy as np

SCHEMA = 'mode_atlas.v1'
FACTORS = ('q', 'n', 't')
CRITERIA = {
    'schema': SCHEMA,
    'lambdas_rel': [1e-3, 1e-2, 1e-1, 1.0, 10.0],
    'H1_decouple_r2_min': 0.5,     # 确认集格均值模型 R² ≥ 0.5（两种语言）
    'H2_q_interaction_max': 0.10,  # 含 q 的交互（qn、qt、qnt）占格均值结构方差的比例 ≤ 0.10（两种语言）
    'H3_truth_acc_min': 0.75,      # 陈述句（q = 0）句末真值探针在确认集的平衡准确率（两种语言）
    'H3_gate_min': 0.7,            #   只在行为门（裸陈述句真假判断）准确率 ≥ 0.7 的语言上解释
    'H4_cos_min': 0.3,             # 跨语言：同一因子中英效应向量的 cos ≥ 0.3 且大于跨因子 |cos|
    'H4_transfer_min': 0.75,       #         中文发现集训练的探针用于英文确认集的平衡准确率（q 与 n）
    'seed': 0,
}


def cell_signs(c: int) -> Dict[str, float]:
    return {'q': ((c >> 2) & 1) - 0.5, 'n': ((c >> 1) & 1) - 0.5, 't': (c & 1) - 0.5}


def center(H: np.ndarray) -> np.ndarray:
    return H - H.mean(1, keepdims=True)


def effects(G: np.ndarray) -> Dict[str, np.ndarray]:
    """全部 7 个析因效应向量（主效应与交互），键如 'q'、'nt'、'qnt'。"""
    M = G.mean(0)                                   # [8, d]
    out = {}
    for r in (1, 2, 3):
        for S in combinations(FACTORS, r):
            w = np.array([np.prod([2 * cell_signs(c)[f] for f in S]) for c in range(8)])
            out[''.join(S)] = (w[:, None] * M).sum(0) / 8 * 2
    return out


def predict(eff: Dict[str, np.ndarray], keys: Sequence[str]) -> np.ndarray:
    """由效应向量重建 8 格的格结构（对比编码：格 c 的值 = Σ_S Π_{F∈S}(2 s_F) · m_S / 2）。"""
    return np.stack([sum(np.prod([2 * cell_signs(c)[f] for f in k]) * eff[k] / 2 for k in keys) for c in range(8)])


def r2_cells(G: np.ndarray, P: np.ndarray) -> float:
    tot = float((G ** 2).sum())
    if tot <= 1e-12 * G.size:                     # 各格完全相同（如 hs[0] 在同一 token 上）：没有格结构可解释
        return float('nan')
    return float(1 - ((G - P[None]) ** 2).sum() / tot)


def split_half_r2(G: np.ndarray, seed: int = 0) -> float:
    p = np.random.default_rng(seed).permutation(len(G))
    a, b = p[: len(p) // 2], p[len(p) // 2:]
    return float(np.mean([r2_cells(G[b], G[a].mean(0)), r2_cells(G[a], G[b].mean(0))]))


def variance_shares(eff: Dict[str, np.ndarray]) -> Dict[str, float]:
    tot = sum(float((v ** 2).sum()) for v in eff.values())
    return {k: float((v ** 2).sum()) / max(tot, 1e-30) for k, v in eff.items()}


def _ridge(X, y, lam_rel):
    mu = X.mean(0)
    Xc = X - mu
    K = Xc @ Xc.T
    lam = lam_rel * np.trace(K) / Xc.shape[1] + 1e-8      # 下限：全部样本相同时 w = 0，探针退化为常数
    a = np.linalg.solve(K + lam * np.eye(len(K)), y - y.mean())
    return Xc.T @ a, mu, y.mean()


def balanced_acc(y, p) -> float:
    y, p = np.asarray(y, bool), np.asarray(p, bool)
    return float(0.5 * (np.mean(p[y]) + np.mean(~p[~y])))


def probe(X_tr, y_tr, groups_tr, X_te, y_te, crit=CRITERIA, X_te2=None, y_te2=None):
    """±1 岭回归探针；λ 由按实体分组的 5 折交叉验证选定；返回确认集（与可选的第二测试集）的平衡准确率。"""
    y = np.where(np.asarray(y_tr, bool), 1.0, -1.0)
    ug = np.unique(groups_tr)
    parts = np.array_split(np.random.default_rng(crit['seed']).permutation(ug), 5)
    best = None
    for lam in crit['lambdas_rel']:
        acc = []
        for p in parts:
            te = np.isin(groups_tr, p)
            w, mu, b = _ridge(X_tr[~te], y[~te], lam)
            acc.append(balanced_acc(y[te] > 0, (X_tr[te] - mu) @ w + b > 0))
        if best is None or np.mean(acc) > best[0]:
            best = (float(np.mean(acc)), lam)
    w, mu, b = _ridge(X_tr, y, best[1])
    out = {'cv_acc': best[0], 'lambda_rel': best[1], 'acc': balanced_acc(y_te, (X_te - mu) @ w + b > 0)}
    if X_te2 is not None:
        out['acc_transfer'] = balanced_acc(y_te2, (X_te2 - mu) @ w + b > 0)
    return out


def cos(a, b) -> float:
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


# ---------------- MV-v2：成对行为门、跨句式框架、中层跨语言、真值方向写入 ----------------
CRITERIA_V2 = {
    'schema': 'mode_atlas.v2',
    'lambdas_rel': CRITERIA['lambdas_rel'],
    'mid_band': list(range(12, 22)),   # 中层段：MV-v1 事后诊断中跨语言复用最强的层段（第 12–21 层），预注册后不再改
    'H1_paired_gate_min': 0.7,         # 成对行为门：同一实体、同一 n，真句的「对−错」间隔 > 假句，比例 ≥ 0.7
    'H2_truth_acc_min': 0.75,          # 新框架确认集真值探针（层由新框架发现集交叉验证在中层段内选定）
    'H3_xframe_truth_min': 0.75,       # 旧框架（MV-v1）发现集训练的真值探针用于新框架确认集
    'H3_xframe_cos_min': 0.5,          # 新旧框架同一效应向量（n、t、nt）的 cos
    'H4_cos_min': 0.3, 'H4_transfer_min': 0.75,
    'steer_layers': [13, 17, 21], 'k_grid': [-6, -3, 0, 3, 6], 'n_random': 2, 'min_swing': 1.0,
    'n_steer_entities': 56,
    'seed': 0,
}


def paired_gate(margin: np.ndarray) -> Dict[str, float]:
    """margin：[n_entity, 4]，格序 2n + t，值 = 「对」− 「错」的完整答案对数概率差。
    成对：同一实体、同一 n，真句（t XOR n = 1）的间隔大于假句即判对。"""
    m = np.asarray(margin, float)
    aff = m[:, 1] > m[:, 0]                     # 肯定：t=1 为真
    neg = m[:, 2] > m[:, 3]                     # 否定：t=0 为真
    return {'affirm': float(aff.mean()), 'negated': float(neg.mean()), 'acc': float(np.concatenate([aff, neg]).mean())}
