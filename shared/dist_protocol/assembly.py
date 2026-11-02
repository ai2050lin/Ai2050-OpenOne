# -*- coding: utf-8 -*-
"""图谱拼装与预测保真度（assembly.v1）：只用图谱部件预测留出实体上模型的答案分布，与模型真实分布比较（G5）。纯 numpy；判据固定在 CRITERIA。

答案分布限定在候选答案上：多选（颜色 11 种）为完整答案对数概率的 softmax；二选（大小、重量、含水、可食、仲裁 K / P、真假判断）为间隔的 sigmoid。
保真度 = 平均 KL(p_模型 ‖ p_预测)（nat，确认集逐题平均）；恢复比例 = 1 − KL_图谱 / KL_均匀。
预测时可用的输入：提示的设计因子（句式、范围、重复次数、来源、否定、所述类别）、实体在中性句式里的编码（另一次前向，不是被预测的状态）、
只在发现集上拟合的图谱参数。不可用：被预测提示本身的状态、logits 或模型答案。
"""
from __future__ import annotations

from typing import Dict

import numpy as np

SCHEMA = 'assembly.v1'
CRITERIA = {
    'schema': SCHEMA,
    'H1_min_cells': 11,          # 14 个「任务族 × 语言」中，图谱 KL 同时低于 hs[0]（词汇身份）与类别（或无知识）基线的至少 11 个
    'H2_recovered_min': 0.5,     # 每种语言：全部任务族合并的恢复比例（相对均匀分布）≥ 0.5
    'seed': 0,
}


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def kl_categorical(logit_true: np.ndarray, logit_pred: np.ndarray) -> np.ndarray:
    lt, lp = log_softmax(logit_true), log_softmax(logit_pred)
    return (np.exp(lt) * (lt - lp)).sum(-1)


def kl_bernoulli(margin_true: np.ndarray, margin_pred: np.ndarray) -> np.ndarray:
    mt, mp = np.asarray(margin_true, float), np.asarray(margin_pred, float)
    return kl_categorical(np.stack([mt, np.zeros_like(mt)], -1), np.stack([mp, np.zeros_like(mp)], -1))


def family_scores(kind: str, true: np.ndarray, preds: Dict[str, np.ndarray]) -> Dict[str, float]:
    """kind ∈ {'cat', 'bin'}；preds：方法名 → 预测（多选为 logits，二选为间隔）。另加均匀分布。"""
    f = kl_categorical if kind == 'cat' else kl_bernoulli
    out = {k: float(f(true, p).mean()) for k, p in preds.items()}
    out['uniform'] = float(f(true, np.zeros_like(np.asarray(true, float))).mean())
    return out


def recovered(kl_method: float, kl_uniform: float) -> float:
    return float(1 - kl_method / max(kl_uniform, 1e-30))


# ---------------- AM-v2：用「槽位条件化的专家乘积」预测未见组合 ----------------
# 每个事件 × 槽位是一个 2×2（风格 s × 逻辑 l）；Y[s, l] 为全词表对数概率。
# P1 事件内补全：Ŷ[s,l] = Y[1−s,l] + Y[s,1−l] − Y[1−s,1−l]（加法补全），对照「两个相邻格的平均」。
# P2 跨事件迁移：新事件只看到 (日常, 因果) 格 Y00；Ŷ[s,l] = Y00 + s·Δs_g + l·Δl_g，Δ 只在发现集事件上按槽位估计。
#    对照：只用 Y00；Δ 不分槽位（三槽位平均）；加上发现集估计的交互项 s·l·Δsl_g。
# KL 在被预测格的候选集（该事件 12 个真实格各自前 20 的并集）上，真分布与预测各自重新归一化。
CRITERIA_V2 = {
    'schema': 'assembly.v2',
    'top_k_cand': 20,
    'H1_ratio_max': 0.5, 'H1_min_cells': 5,   # P1：加法补全 KL ≤ 0.5 × 相邻平均 KL，（语言 × 槽位）6 格中至少 5 格
    'H2_recovered_min': 0.5,                  # P2：1 − KL_专家乘积 / KL_只用基础格 ≥ 0.5（每种语言）
    'H3_slot_gain_min': 0.1,                  # P2：分槽位的 Δ 比不分槽位的 KL 低 ≥ 10%（每种语言）
    'H4_interaction_gain_max': 0.1,           # P2：加交互项使 KL 降低不超过 10%（可加已足够；每种语言）
    'seed': 0,
}


def kl_restricted(true_lp: np.ndarray, pred_lp: np.ndarray, cand: np.ndarray) -> float:
    """真分布与预测都限制到候选集并重新归一化后的 KL(p_true ‖ p_pred)。"""
    t = log_softmax(np.asarray(true_lp, float)[cand])
    p = log_softmax(np.asarray(pred_lp, float)[cand])
    return float((np.exp(t) * (t - p)).sum())


def additive_completion(Y: np.ndarray, s: int, l: int) -> np.ndarray:
    """Y：[2, 2, V]。用另外三格加法补全 (s, l)。"""
    return Y[1 - s, l] + Y[s, 1 - l] - Y[1 - s, 1 - l]


def neighbor_mean(Y: np.ndarray, s: int, l: int) -> np.ndarray:
    return 0.5 * (Y[1 - s, l] + Y[s, 1 - l])


def slot_effects(Yd: np.ndarray):
    """Yd：[n_disc, 2, 2, V]。返回 (Δs, Δl, Δsl)，均为 [V]。"""
    ds = (Yd[:, 1] - Yd[:, 0]).mean((0, 1))
    dl = (Yd[:, :, 1] - Yd[:, :, 0]).mean((0, 1))
    dsl = (Yd[:, 1, 1] - Yd[:, 1, 0] - Yd[:, 0, 1] + Yd[:, 0, 0]).mean(0)
    return ds, dl, dsl
