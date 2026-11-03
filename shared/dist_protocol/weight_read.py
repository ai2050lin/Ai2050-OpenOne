# -*- coding: utf-8 -*-
"""权重阅读与封闭式编辑（weight_read.v1，WS-v1）：只读权重提出假设，少量前向判定。纯 numpy；判据固定在 CRITERIA。

参考「权重空间编程」（Takehi 等，2026-11）的做法：先从权重结构推出部件读什么、写什么、由谁供给，再用前向与一次封闭式权重编辑检验。
  读出向量：MLP 单元 i 的 gate / up 行乘以该块 post_attention_layernorm 的权重 γ，即残差空间里它读的方向 a_i = γ ⊙ W[i]。
  组合分数（Elhage 等 2021 的 Frobenius 形式）：注意力头 h 的 OV 矩阵 W_OV = W_O^h W_V^h diag(γ_in) 写入的内容被读出矩阵 R 读到的比例
      CS_h = ‖R W_OV‖_F / (‖R‖_F ‖W_OV‖_F)；只依赖权重，不看头注意哪里。
  方向移除：a' = a − (a·v) v，W' = a' / γ（单位向量 v；只改被选单元的行）。
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

SCHEMA = 'weight_read.v1'
CRITERIA = {
    'schema': SCHEMA,
    'n_random_sets': 200,
    'H1_read_ratio_min': 3.0, 'H1_p_max': 0.01,          # 32 个门控单元读代码方向的平均分数 ≥ 同层随机集合的 3 倍，经验 p ≤ 0.01
    'H2_spearman_min': 0.3, 'H2_top_k': 16, 'H2_p_max': 0.05,  # 组合分数与前向实测 |T_h| 的 Spearman ≥ 0.3，且前 16 重合置换 p ≤ 0.05，且优于范数基线
    'H3_restore_frac_min': 0.5,                           # 封闭式编辑的恢复 ≥ MF-v4 激活修补恢复的 0.5 倍
    'H3_control_max_frac': 1 / 3,                         # 随机方向编辑、随机单元编辑各自的恢复 ≤ 真实编辑的 1/3
    'H4_nll_rise_max': 0.01,                              # 自然文本平均 NLL 相对上升 ≤ 1%（全部与代码文本分别）
    'n_perm': 2000,
    'seed': 0,
}


def unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def cosines(A: np.ndarray, v: np.ndarray) -> np.ndarray:
    """A：[n, d] 行向量；返回每行与 v 的余弦。"""
    return (A @ unit(v)) / (np.linalg.norm(A, axis=1) + 1e-12)


def read_scores(Ag: np.ndarray, Au: np.ndarray, v: np.ndarray) -> np.ndarray:
    """gate 与 up 读出向量对方向 v 的合并读分数 sqrt(cos_g² + cos_u²)。"""
    return np.sqrt(cosines(Ag, v) ** 2 + cosines(Au, v) ** 2)


def ratio_vs_random(score_top: float, score_rand: Sequence[float]) -> Dict[str, float]:
    r = np.asarray(score_rand, float)
    return {'top': float(score_top), 'random_mean': float(r.mean()), 'ratio': float(score_top / max(r.mean(), 1e-12)),
            'p': float((1 + (r >= score_top).sum()) / (1 + len(r)))}


def composition(R: np.ndarray, Wov: np.ndarray) -> float:
    """R：[k, d] 读出行；Wov：[d, d]（或低秩因子乘积）。"""
    return float(np.linalg.norm(R @ Wov) / (np.linalg.norm(R) * np.linalg.norm(Wov) + 1e-12))


def composition_lowrank(R: np.ndarray, Wo: np.ndarray, Wv: np.ndarray) -> float:
    """W_OV = Wo @ Wv（Wo：[d, r]，Wv：[r, d]），不显式构造 d×d。"""
    RW = (R @ Wo) @ Wv
    _, s_o = np.linalg.qr(Wo)
    nov = np.linalg.norm(s_o @ Wv)                       # ‖Wo Wv‖_F = ‖R_o Wv‖_F（Wo = Q R_o）
    return float(np.linalg.norm(RW) / (np.linalg.norm(R) * nov + 1e-12))


def remove_direction(W_rows: np.ndarray, gamma: np.ndarray, v: np.ndarray) -> np.ndarray:
    """在残差读出空间里移除方向 v：a = γ ⊙ w，a' = a − (a·v̂) v̂，返回 w' = a' / γ。"""
    vh = unit(v)
    A = W_rows * gamma[None, :]
    A2 = A - np.outer(A @ vh, vh)
    return A2 / gamma[None, :]


def rankdata(x: np.ndarray) -> np.ndarray:
    o = np.argsort(x, kind='mergesort')
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    xs = np.asarray(x)[o]
    i = 0
    while i < len(x):                                   # 并列取平均秩
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(rankdata(np.asarray(a, float)), rankdata(np.asarray(b, float)))[0, 1])


def topk_overlap(pred: np.ndarray, truth: np.ndarray, k: int, n_perm: int, seed: int) -> Dict[str, float]:
    """预测前 k 与真值前 k 的重合个数及置换 p（打乱预测分数）。"""
    tk = set(np.argsort(-truth)[:k].tolist())
    ov = len(tk & set(np.argsort(-pred)[:k].tolist()))
    rng = np.random.default_rng(seed)
    null = [len(tk & set(rng.permutation(len(pred))[:k].tolist())) for _ in range(n_perm)]
    return {'overlap': ov, 'k': k, 'null_mean': float(np.mean(null)), 'p': float((1 + sum(n >= ov for n in null)) / (1 + n_perm))}


def head_verdict(cs: np.ndarray, base: np.ndarray, truth: np.ndarray, crit=CRITERIA) -> Dict[str, object]:
    rho, rho_b = spearman(cs, truth), spearman(base, truth)
    ov = topk_overlap(cs, truth, crit['H2_top_k'], crit['n_perm'], crit['seed'])
    return {'spearman': rho, 'spearman_norm_baseline': rho_b, 'overlap': ov,
            'pass': bool(rho >= crit['H2_spearman_min'] and rho > rho_b and ov['p'] <= crit['H2_p_max'])}
