# -*- coding: utf-8 -*-
"""路径扫描（path_scan.v1）：PS-v1 的两项检验，纯 numpy；判据固定在 CRITERIA_TS / CRITERIA_AH 并随结果返回。

TS（真值写入扫描）：XM-v1 中 GLM4-9B 的真值写入（C6）在 3 个相对深度的句末位置都不成立。本检验在「层 × 写入位置」全网格上扫描，
  干预实体按类别确定性分成 A / B 两半：只用 A 选格（A 上 steer_verdict 成立的格中摆幅最大者），在 B 上确认。
AH（仲裁上下文搬运头）：CF-v2 / XM-v1 中「书里」范围的上下文答案主要由后段注意力直接写出。本检验把注意力直接作用拆到单个头：
  前 k 个头（只按发现集选定）在确认集上占多少正向直接作用、是否注视上下文颜色词、消融后上下文胜出是否下降（与同层随机头对照）、
  「现实」范围下这些头是否被关小。
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Sequence

import numpy as np

SCHEMA = 'path_scan.v1'
CRITERIA_TS = {
    'schema': SCHEMA + '/truth_scan',
    'layer_step': 2,                    # 层网格 range(0, L, 2)
    'positions': ['end', 'span', 'final'],   # 句末标点 / 陈述句全部 token / 答案位置（提问后缀的最后一个 token）
    'k_grid': [-6, -3, 0, 3, 6], 'n_random': 2, 'min_swing': 1.0,
    'gate_min': 0.7,                    # 成对行为门（XM-v1 C4）≥ 0.7 的语言才做 H1 / H2 判定，其余只描述
    'seed': 0,
}
CRITERIA_AH = {
    'schema': SCHEMA + '/arbitration_heads',
    'top_k': 10,                        # 发现集「书里」范围平均直接作用最大的 10 个头
    'H1_share_min': 0.5,                # 确认集：前 k 头占全部正向头直接作用之和的比例
    'H2_attn_abs_min': 0.2,             # 前 k 头从答案位置注视上下文颜色词 token 的平均注意力
    'H2_attn_ratio_min': 3.0,           #   且 ≥ 全部头平均的 3 倍
    'n_random_sets': 5,                 # 同层随机头集合（每个前 k 头换成同层另一个非前 k 头）
    'H3_drop_ratio_min': 3.0,           # 消融前 k 头（答案位置输出置零）：书里 LO 平均下降 ≥ 每个随机集合下降的 3 倍
    'H3_win_drop_min': 0.2,             #   且上下文胜出率下降 ≥ 0.2
    'H4_ratio_max': 0.5,                # 现实范围下前 k 头直接作用之和 ≤ 书里的 0.5 倍
    'seed': 0,
}


# ---------------- TS ----------------

def split_ab(ids: Sequence[str], cats: Sequence[str], tag: str = 'PS-v1') -> np.ndarray:
    """每个类别内按 sha256(tag|id) 排序，前一半为 A（返回布尔掩码）。"""
    ids, cats = list(ids), list(cats)
    mask = np.zeros(len(ids), bool)
    for c in sorted(set(cats)):
        idx = [i for i in range(len(ids)) if cats[i] == c]
        idx.sort(key=lambda i: hashlib.sha256(f'{tag}|{ids[i]}'.encode()).hexdigest())
        mask[idx[:len(idx) // 2]] = True
    return mask


def steer_verdict(delta_op: np.ndarray, delta_rand: np.ndarray, k_grid, min_swing: float) -> Dict[str, object]:
    """同 entity_atlas.steer_verdict：单调、两端异号、摆幅 ≥ min_swing 且大于每个随机方向。"""
    k = np.asarray(k_grid)
    hi, lo = int(np.argmax(k)), int(np.argmin(k))
    m = delta_op.mean(0)
    swing = float(m[hi] - m[lo])
    rand_sw = [float(r.mean(0)[hi] - r.mean(0)[lo]) for r in delta_rand]
    mono = bool(np.all(np.diff(m[np.argsort(k)]) > 0))
    ok = bool(mono and m[hi] > 0 > m[lo] and swing >= min_swing and swing > max(rand_sw, default=-np.inf))
    return {'mean_delta_by_k': [float(x) for x in m], 'swing': swing, 'random_swings': rand_sw, 'monotonic': mono,
            'supported_write': ok}


def select_and_confirm(cells: Dict[str, Dict[str, np.ndarray]], item_A: np.ndarray, crit=CRITERIA_TS) -> Dict[str, object]:
    """cells：格名 → {'op': [n_item, n_k], 'rand': [n_rand, n_item, n_k]}；item_A：逐条目是否属于 A。
    在 A 上逐格判定，取成立格中摆幅最大者，在 B 上用同一判定确认。"""
    k, ms = crit['k_grid'], crit['min_swing']
    vA = {c: steer_verdict(d['op'][item_A], d['rand'][:, item_A], k, ms) for c, d in cells.items()}
    ok = [c for c, v in vA.items() if v['supported_write']]
    out = {'n_cells': len(cells), 'n_supported_A': len(ok), 'swing_A': {c: v['swing'] for c, v in vA.items()}}
    if not ok:
        out.update({'cell': None, 'confirmed': False})
        return out
    best = max(ok, key=lambda c: vA[c]['swing'])
    vB = steer_verdict(cells[best]['op'][~item_A], cells[best]['rand'][:, ~item_A], k, ms)
    out.update({'cell': best, 'A': vA[best], 'B': vB, 'confirmed': bool(vB['supported_write'])})
    return out


# ---------------- AH ----------------

def top_heads(dla_disc: np.ndarray, k: int) -> List[int]:
    """dla_disc：[n, L, H]；返回平均直接作用最大的 k 个头（展平索引 l·H + h，降序）。"""
    m = dla_disc.reshape(len(dla_disc), -1).mean(0)
    return [int(i) for i in np.argsort(-m)[:k]]


def concentration(dla: np.ndarray, idx: Sequence[int]) -> Dict[str, object]:
    """前 k 头占全部正向头平均直接作用之和的比例；另给达到 50% / 80% 所需的头数（按本集合自身排序，仅描述）。"""
    m = dla.reshape(len(dla), -1).mean(0)
    pos = float(m[m > 0].sum())
    srt = np.sort(m[m > 0])[::-1]
    cum = np.cumsum(srt) / max(pos, 1e-30)
    need = lambda q: int(np.searchsorted(cum, q) + 1) if len(cum) else 0
    return {'share': float(m[list(idx)].sum() / max(pos, 1e-30)), 'top_sum': float(m[list(idx)].sum()), 'positive_sum': pos,
            'all_sum': float(m.sum()), 'heads_for_50': need(0.5), 'heads_for_80': need(0.8)}


def attn_focus(att: np.ndarray, idx: Sequence[int], crit=CRITERIA_AH) -> Dict[str, object]:
    """att：[n, L, H] 答案位置对目标 token 的注意力之和。"""
    m = att.reshape(len(att), -1).mean(0)
    top, allm = float(m[list(idx)].mean()), float(m.mean())
    return {'top_mean': top, 'all_mean': allm, 'ratio': top / max(allm, 1e-30),
            'pass': bool(top >= crit['H2_attn_abs_min'] and top >= crit['H2_attn_ratio_min'] * allm)}


def random_matched(idx: Sequence[int], H: int, n_sets: int, seed: int) -> List[List[int]]:
    """每个前 k 头换成同层另一个非前 k 头。"""
    rng = np.random.default_rng(seed)
    top = set(idx)
    out = []
    for _ in range(n_sets):
        s = []
        for i in idx:
            l = i // H
            pool = [l * H + h for h in range(H) if l * H + h not in top and l * H + h not in s]
            s.append(int(rng.choice(pool)))
        out.append(s)
    return out


def ablation_verdict(base: np.ndarray, top: np.ndarray, rand: np.ndarray, crit=CRITERIA_AH) -> Dict[str, object]:
    """base / top：[n] 书里范围 LO（K − P）；rand：[n_sets, n]。"""
    drop = float((base - top).mean())
    rdrop = [float((base - r).mean()) for r in rand]
    win0, win1 = float((base > 0).mean()), float((top > 0).mean())
    ok = bool(drop > 0 and drop >= crit['H3_drop_ratio_min'] * max(max(rdrop), 1e-9) and win0 - win1 >= crit['H3_win_drop_min'])
    return {'lo_drop': drop, 'random_drops': rdrop, 'win_base': win0, 'win_ablated': win1,
            'win_random': [float((r > 0).mean()) for r in rand], 'pass': ok}
