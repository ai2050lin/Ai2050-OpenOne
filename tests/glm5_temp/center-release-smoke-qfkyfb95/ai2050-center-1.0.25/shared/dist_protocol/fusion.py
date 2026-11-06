# -*- coding: utf-8 -*-
"""多维融合（fusion.v1）：风格 × 逻辑 × 语法的 2×2×2 因子设计中，下一词分布与内部状态能否分解为各因子主效应之和。纯 numpy；判据固定在 CRITERIA。

方差分解：对每个条目，Y 为 8 个格（c = 4s + 2l + g）× m 列的矩阵（输出：候选 token 的对数概率，按格中心化；状态：隐向量）。
用 2^3 正交对比得到 7 个效应（s、l、g、sl、sg、lg、slg）各自的平方和；可加份额 = 三个主效应平方和 / 格间总平方和。
无结构的随机矩阵期望可加份额 = 3 / 7 ≈ 0.43。
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

SCHEMA = 'fusion.v1'
EFFECTS = ['s', 'l', 'g', 'sl', 'sg', 'lg', 'slg']
CRITERIA = {
    'schema': SCHEMA,
    'top_k_per_cell': 20,         # 候选 token = 8 个格各自前 20 的并集
    'H0_main_share_min': 0.05,    # 行为门：三个因子各自的主效应份额（确认集合并）≥ 0.05，否则该因子未起作用
    'H1_out_additive_min': 0.8,   # 输出：确认集合并的可加份额 ≥ 0.8（专家乘积式融合）
    'H2_state_additive_min': 0.8, # 状态：中层段 [L/3, 2L/3] 各层「去掉语法主效应后」的可加份额 (s + l) / (总 − g) 平均 ≥ 0.8
                                  #（末位 token 只随语法因子不同，语法主效应主要是 token 身份；随机期望 2 / 6）
    'H3_interaction_growth': 1.5, # 同一口径的交互份额：最终层 ≥ 中层段平均的 1.5 倍（融合在后段发生）
    'seed': 0,
}


def contrasts() -> np.ndarray:
    """[7, 8]：每行一个效应的 ±1 对比（格 c 的位：s = bit2，l = bit1，g = bit0）。"""
    bits = np.array([[c >> 2 & 1, c >> 1 & 1, c & 1] for c in range(8)]) * 2 - 1
    s, l, g = bits.T
    return np.stack([s, l, g, s * l, s * g, l * g, s * l * g]).astype(float)


def decompose(Y: np.ndarray) -> Dict[str, float]:
    """Y：[8, m]。返回各效应平方和、格间总平方和与可加份额。"""
    Y = np.asarray(Y, float)
    Yc = Y - Y.mean(0, keepdims=True)
    K = contrasts()
    est = K @ Yc / 8.0                        # [7, m]
    ss = 8.0 * (est ** 2).sum(1)              # 每个效应的平方和
    tot = float((Yc ** 2).sum())
    out = {e: float(v) for e, v in zip(EFFECTS, ss)}
    out['total'] = tot
    out['additive_share'] = float(ss[:3].sum() / tot) if tot > 0 else float('nan')
    return out


def pooled(decs: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """多个条目合并：各效应平方和相加后求份额。"""
    tot = sum(d['total'] for d in decs)
    out = {e: float(sum(d[e] for d in decs) / tot) for e in EFFECTS}
    out['additive_share'] = out['s'] + out['l'] + out['g']
    out['interaction_share'] = 1 - out['additive_share']
    rest = 1 - out['g']
    out['additive_ex_g'] = (out['s'] + out['l']) / rest if rest > 0 else float('nan')
    out['interaction_ex_g'] = 1 - out['additive_ex_g']
    return out


def center_cells(logp: np.ndarray) -> np.ndarray:
    """[8, m] 对数概率按格（行）中心化，去掉各格的归一化常数。"""
    return logp - logp.mean(1, keepdims=True)


# ---------------- FU-v2：槽位条件化的专家乘积（2 × 2 × k 设计）----------------
# 主假设（FU-v1 事后发现，本版预注册确认）：固定语法槽位后，风格与逻辑对下一词对数概率的作用近似相加；
# 因子之间的交互主要是「因子 × 槽位」，而不是「风格 × 逻辑」。
CRITERIA_V2 = {
    'schema': 'fusion.v2',
    'top_k_per_cell': 20,
    'H0_within_share_min': 0.1,    # 行为门（每种语言，各槽位合并）：槽位内风格与逻辑各自占槽位内平方和 ≥ 0.1
    'H1_within_additive_min': 0.8, # 输出：槽位内可加份额 (s + l) / (s + l + sl) ≥ 0.8
    'H1_min_cells': 5,             #   （槽位 × 语言）6 格中至少 5 格
    'H2_state_additive_min': 0.8,  # 状态（最终层隐状态）：同一口径，6 格中至少 5 格
    'H3_slot_over_sl': 3.0,        # 输出（每种语言）：SS(风格×槽位) + SS(逻辑×槽位) ≥ 3 × [SS(风格×逻辑) + SS(三阶)]
    'seed': 0,
}


def factorial_ss(Y: np.ndarray, k: int) -> Dict[str, float]:
    """Y：[2·2·k, m]，格序 c = 2k·s + k·l + g。均衡全因子设计（每格一个观测）的效应平方和（按列相加）。"""
    Y = np.asarray(Y, float).reshape(2, 2, k, -1)
    mu = Y.mean((0, 1, 2), keepdims=True)
    ms, ml, mg = Y.mean((1, 2), keepdims=True), Y.mean((0, 2), keepdims=True), Y.mean((0, 1), keepdims=True)
    msl, msg, mlg = Y.mean(2, keepdims=True), Y.mean(1, keepdims=True), Y.mean(0, keepdims=True)
    eff = {'s': ms - mu, 'l': ml - mu, 'g': mg - mu,
           'sl': msl - ms - ml + mu, 'sg': msg - ms - mg + mu, 'lg': mlg - ml - mg + mu,
           'slg': Y - msl - msg - mlg + ms + ml + mg - mu}
    out = {e: float((np.broadcast_to(v, Y.shape) ** 2).sum()) for e, v in eff.items()}
    out['total'] = float(((Y - mu) ** 2).sum())
    return out


def within_slot_ss(Y: np.ndarray, k: int) -> Dict[int, Dict[str, float]]:
    """每个槽位 g 内的 2×2（风格 × 逻辑）平方和。"""
    Y = np.asarray(Y, float).reshape(2, 2, k, -1)
    out = {}
    for g in range(k):
        Z = Y[:, :, g]
        mu = Z.mean((0, 1), keepdims=True)
        ms, ml = Z.mean(1, keepdims=True), Z.mean(0, keepdims=True)
        out[g] = {'s': float((np.broadcast_to(ms - mu, Z.shape) ** 2).sum()), 'l': float((np.broadcast_to(ml - mu, Z.shape) ** 2).sum()),
                  'sl': float(((Z - ms - ml + mu) ** 2).sum())}
    return out


def within_additive(ss: Dict[str, float]) -> float:
    t = ss['s'] + ss['l'] + ss['sl']
    return float((ss['s'] + ss['l']) / t) if t > 0 else float('nan')
