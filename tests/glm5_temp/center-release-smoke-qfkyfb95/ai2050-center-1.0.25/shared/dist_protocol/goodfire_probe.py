# -*- coding: utf-8 -*-
"""Goodfire 式原生神经元逆向分析（goodfire_probe.v1）的统计部分：纯 numpy，不依赖 torch。

方法来源：Goodfire《Neural Networks Are Already Clean》（Cammarata）的找特征流程，本项目拆解见
research/gpt5/docs/GOODFIRE_FIND_FEATURE_STEPS_v2.md（S1–S9）。GPU 部分（S1 两极采集、S6 词表投影、S7 steering、
S9 最小对前向）在 goodfire_runner.py；这里只做可单元测试的向量运算与判据。

记号（单元 j，第 l 层）：
  g   = MLP 门控的 pre-activation（gate_proj 输出的第 j 维，SiLU 之前）；按 g 取两极，不按 A = silu(G)⊙V
  h   = 残差流在 post_attention_layernorm 之前的向量（MLP 读的是 norm(h)）
  μ   = 本次语料全部计入位置上 h 的均值；集中度与归族都在 h − μ 上算（残差流的公共分量会把一切都抬成"集中"）
  P/N = g 最大 / 最小的 k 个位置；肩部 = 排名 k+1 … k+m 的位置（留出检验，不参与 Δ）
  Δ   = mean(h[P]) − mean(h[N])

判据固定在 CRITERIA 中并随结果返回；不得在看到结果后调整。判据通过只说明这一格的读出与检验结论，
不自动晋级证据等级（AGENTS.md §6.1 第 7 条）。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

SCHEMA = 'goodfire_probe.v1'

CRITERIA = {
    'coh_P_min': 0.30,          # S2 正极集中度（h − μ 上）；随机基线 ≈ 1/√k
    'coh_ratio_to_random': 3.0, # 同时须 ≥ 3 × 1/√k
    'eff_text_ratio': 0.2,      # S3 有效文本段数 ≥ k/5（同一窗口段内的位置高度相关）
    'k_min': 20,
    'split_half_min': 0.5,      # S4 劈半 cos（校正前）
    'R_min': 0.6,               # S5 对偶对称比 coh_N / coh_P
    'shoulder_acc_min': 0.9,    # S5 留出肩部落在正确一侧的比例
    'read_write_gap_max': 0.2,  # S8 cos(读, Δ) − cos(写, Δ) 的上限；超过记为"看得见但可能控不住"
    'steer_monotone_min': 0.8,  # S7 各提示上「Δ 正端词 − 负端词」logit 差随 α 单调上升的比例
    'pairs_sign_min': 0.9,      # S9 最小对 z 差符号正确率
    'split_seed': 20261028,
}


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v * 0.0


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_unit(np.asarray(a, np.float64)), _unit(np.asarray(b, np.float64))))


def coherence(H: np.ndarray) -> float:
    """‖mean(h)‖ / mean(‖h‖)：k 个方向一致时 → 1，随机时 ≈ 1/√k。H 应已减去语料均值 μ。"""
    H = np.asarray(H, np.float64)
    norms = np.linalg.norm(H, axis=1)
    den = float(norms.mean()) if len(norms) else 0.0
    return float(np.linalg.norm(H.mean(0)) / den) if den > 1e-12 else 0.0


def split_half(HP: np.ndarray, HN: np.ndarray, seed: int) -> Dict[str, float]:
    """把 P、N 各随机劈成两半分别算 Δ，比较方向；Spearman-Brown 校正得到全量信度。"""
    rng = np.random.default_rng(seed)
    ip, iN = rng.permutation(len(HP)), rng.permutation(len(HN))
    a = HP[ip[: len(ip) // 2]].mean(0) - HN[iN[: len(iN) // 2]].mean(0)
    b = HP[ip[len(ip) // 2:]].mean(0) - HN[iN[len(iN) // 2:]].mean(0)
    c = cos(a, b)
    return {'cos': c, 'reliability': float(2 * c / (1 + c)) if c > -1 else -1.0}


def shoulder_check(delta: np.ndarray, mP: np.ndarray, mN: np.ndarray, SP: np.ndarray, SN: np.ndarray) -> Dict[str, Any]:
    """留出肩部：投影到 Δ̂ 上（以两极均值的中点为零点），正肩应在正侧、负肩应在负侧。"""
    d = _unit(delta)
    mid = (mP + mN) / 2
    pp = (np.asarray(SP, np.float64) - mid) @ d if len(SP) else np.zeros(0)
    pn = (np.asarray(SN, np.float64) - mid) @ d if len(SN) else np.zeros(0)
    n = len(pp) + len(pn)
    acc = float(((pp > 0).sum() + (pn < 0).sum()) / n) if n else None
    return {'n': int(n), 'acc': acc, 'pos_acc': float((pp > 0).mean()) if len(pp) else None,
            'neg_acc': float((pn < 0).mean()) if len(pn) else None}


def analyze(*, HP: np.ndarray, HN: np.ndarray, SP: np.ndarray, SN: np.ndarray, mu: np.ndarray,
            textP: List, textN: List, read_vec: np.ndarray, write_vec: np.ndarray,
            gP: np.ndarray, gN: np.ndarray, g_mean: float, g_std: float) -> Dict[str, Any]:
    """S2–S5、S8 的统计与判据。HP/HN：两极的 h（k × d）；SP/SN：肩部的 h；textP/N：每个位置的独立文本段标识；
    read_vec = γ_post ⊙ W_gate[j]（门控实际读的残差方向），write_vec = W_down[:, j]。"""
    C = CRITERIA
    k = int(min(len(HP), len(HN)))
    cP, cN = np.asarray(HP, np.float64) - mu, np.asarray(HN, np.float64) - mu
    sP, sN = np.asarray(SP, np.float64) - mu, np.asarray(SN, np.float64) - mu
    base = 1.0 / np.sqrt(max(k, 1))
    coh_P, coh_N = coherence(cP), coherence(cN)
    mP, mN = cP.mean(0), cN.mean(0)
    delta = mP - mN
    sh = split_half(cP, cN, C['split_seed']) if k >= 4 else {'cos': None, 'reliability': None}
    shoulder = shoulder_check(delta, mP, mN, sP, sN)
    effP, effN = len({tuple(t) for t in textP}), len({tuple(t) for t in textN})
    cr, cw = cos(read_vec, delta), cos(write_vec, delta)
    R = coh_N / coh_P if coh_P > 1e-12 else 0.0
    checks = {
        'S2_coh': {'value': coh_P, 'random': base, 'pass': coh_P >= C['coh_P_min'] and coh_P >= C['coh_ratio_to_random'] * base},
        'S3_k': {'value': k, 'pass': k >= C['k_min']},
        'S3_effective_texts': {'value': min(effP, effN), 'pos': effP, 'neg': effN, 'need': int(np.ceil(C['eff_text_ratio'] * k)),
                               'pass': min(effP, effN) >= C['eff_text_ratio'] * k},
        'S4_split_half': {'value': sh['cos'], 'reliability': sh['reliability'], 'pass': sh['cos'] is not None and sh['cos'] >= C['split_half_min']},
        'S5_duality': {'value': R, 'coh_P': coh_P, 'coh_N': coh_N, 'pass': R >= C['R_min'] and coh_N >= C['coh_ratio_to_random'] * base},
        'S5_shoulder': {'value': shoulder['acc'], **shoulder, 'pass': shoulder['acc'] is not None and shoulder['acc'] >= C['shoulder_acc_min']},
        'S8_read_write': {'value': cr - cw, 'cos_read': cr, 'cos_write': cw, 'pass': (cr - cw) <= C['read_write_gap_max']},
    }
    for c in checks.values():
        c['pass'] = bool(c['pass'])                 # numpy bool 不能直接写 JSON
    # S5 判读：真轴 / 混合物（负极散）/ 数据不足（两极都散）
    if checks['S2_coh']['pass'] and checks['S5_duality']['pass']:
        reading = 'axis'
    elif checks['S2_coh']['pass']:
        reading = 'mixture'
    elif R >= C['R_min']:
        reading = 'insufficient_data'
    else:
        reading = 'unreadable'
    return {'delta': delta, 'delta_norm': float(np.linalg.norm(delta)), 'checks': checks, 'reading': reading,
            'g': {'pos_min': float(np.min(gP)) if len(gP) else None, 'neg_max': float(np.max(gN)) if len(gN) else None,
                  'mean': float(g_mean), 'std': float(g_std),
                  'z_pos_min': float((np.min(gP) - g_mean) / g_std) if len(gP) and g_std > 0 else None,
                  'z_neg_max': float((np.max(gN) - g_mean) / g_std) if len(gN) and g_std > 0 else None}}


def steering_monotone(shift: np.ndarray, alphas: List[float]) -> Dict[str, Any]:
    """S7：shift[p, a] = 第 p 个提示在 α_a 下「Δ 正端词平均 logit − 负端词平均 logit」。
    单调 = 按 α 从小到大逐项不下降且首尾严格上升；正负两向都要有变化（只有一侧有反应说明在破坏分布而不是移动轴）。"""
    order = np.argsort(alphas)
    S = np.asarray(shift, np.float64)[:, order]
    a = np.asarray(alphas, np.float64)[order]
    zero = int(np.argmin(np.abs(a)))
    mono = [bool(np.all(np.diff(r) >= -1e-6) and r[-1] > r[0]) for r in S]
    both = [bool(r[-1] > r[zero] and r[0] < r[zero]) for r in S] if 0 < zero < len(a) - 1 else [False] * len(S)
    frac = float(np.mean(mono)) if mono else 0.0
    return {'monotone_frac': frac, 'both_sides_frac': float(np.mean(both)) if both else 0.0,
            'pass': frac >= CRITERIA['steer_monotone_min'] and (float(np.mean(both)) if both else 0.0) >= CRITERIA['steer_monotone_min']}


def pairs_check(z_a: List[float], z_b: List[float]) -> Dict[str, Any]:
    """S9：最小对，A 句预期在正端、B 句在负端；z = (g − 语料均值) / 语料标准差（各句取 g 的最大值）。"""
    d = np.asarray(z_a, np.float64) - np.asarray(z_b, np.float64)
    if not len(d):
        return {'n': 0, 'sign_acc': None, 'mean_dz': None, 'pass': None}
    acc = float((d > 0).mean())
    return {'n': int(len(d)), 'sign_acc': acc, 'mean_dz': float(d.mean()), 'pass': acc >= CRITERIA['pairs_sign_min']}


def verdict(checks: Dict[str, Dict[str, Any]], steer: Optional[Dict[str, Any]], pairs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """汇总：自动步全部通过才「可入册候选」；S9 未做时标为待人工。只描述本次检验，不是证据等级。"""
    failed = [k for k, c in checks.items() if not c['pass']] + ([] if steer and steer.get('pass') else ['S7_steering'])
    if pairs and pairs.get('pass') is False:
        failed.append('S9_pairs')
    status = 'not_passed' if failed else 'candidate' if pairs and pairs.get('pass') else 'auto_passed_needs_S9'
    return {'status': status, 'failed': failed}
