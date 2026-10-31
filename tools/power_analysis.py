# -*- coding: utf-8 -*-
"""P0.5 功效分析模板（G5-DP01 P0.5 / 审计 B4）：McNemar 配对口径。

每个 Phase 预注册必须含「预期效应量 → 所需 n → 检验力 ≥0.8」三栏。
口径：消融 vs base 是逐题配对二值结果；判别力来自翻转题（discordant）。
规划参数 p_d = 预期翻转率（两方向翻转题占总题数比例，保守缺省 0.10）。

  n_required = (z_{1-α/2} + z_{power})² · p_d / Δ²        （Δ 为绝对正确率变化，比例）

只依赖标准库 statistics.NormalDist。CLI：python tools/power_analysis.py
"""
from __future__ import annotations

import math
from statistics import NormalDist


def n_required(delta: float, p_d: float = 0.10, alpha: float = 0.05, power: float = 0.8) -> int:
    """预期效应 delta（如 0.03=3pt）、翻转率 p_d → 所需每类题数。"""
    if not 0 < delta < 1:
        raise ValueError(f'delta 须在 (0,1)：{delta}')
    za = NormalDist().inv_cdf(1 - alpha / 2)
    zb = NormalDist().inv_cdf(power)
    return math.ceil(((za + zb) ** 2) * p_d / (delta ** 2))


def achieved_power(n: int, delta: float, p_d: float = 0.10, alpha: float = 0.05) -> float:
    """给定 n 与效应，反算检验力。"""
    za = NormalDist().inv_cdf(1 - alpha / 2)
    lam = abs(delta) * math.sqrt(n / p_d)
    return NormalDist().cdf(lam - za)


def plan_table(deltas=(0.02, 0.03, 0.05), p_d=0.10, alpha=0.05, power=0.8) -> list:
    return [{'delta_pt': round(d * 100, 1), 'p_d': p_d,
             'n_required': n_required(d, p_d, alpha, power),
             'power_at_n1000': round(achieved_power(1000, d, p_d, alpha), 3)}
            for d in deltas]


def check_phase(expected_delta: float, actual_n: int, p_d: float = 0.10,
                alpha: float = 0.05) -> dict:
    """预注册检查：预期效应下该 n 是否达标（power ≥0.8）。"""
    need = n_required(expected_delta, p_d, alpha)
    pw = achieved_power(actual_n, expected_delta, p_d, alpha)
    return {'expected_delta_pt': round(expected_delta * 100, 1), 'n': actual_n,
            'p_d': p_d, 'n_required': need, 'achieved_power': round(pw, 3),
            'pass': bool(pw >= 0.8)}


if __name__ == '__main__':
    import json
    print(json.dumps({'plan_table': plan_table(),
                      'phase_check_example': check_phase(0.03, 1000)},
                     ensure_ascii=False, indent=1))
