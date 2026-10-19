# -*- coding: utf-8 -*-
"""MLP 中间单元的选择性统计（neurons.v1）：模板因子 → 原生神经元。

输入是 runner_v3 在节点本地写出的 MLP 中间单元（<原生全场目录>/shard-*/mlp/B<块>.npz，G、V: (P, N, d_ff)），
单元输出 A = silu(G) ⊙ V。对一个二值对照（因子的两个水平，或一个水平对其余水平）：

  1. 只在 discovery 上计算每个单元的标准化均值差 d = (mean_hi − mean_lo) / pooled_sd；
  2. 零分布：把水平标签按取值单位（句式级因子按句式，其余按填充词）整体置换，记录每次置换中全部单元 |d| 的最大值，
     取 1 − perm_alpha 分位作为同一块内的族错误阈值；
  3. 入选：|d_disc| ≥ max(阈值, d_min)，且在留出划分（confirm / lockbox / lockbox_frame）上同号、|d_held| ≥ held_min；
  4. 对入选单元报告两极（均值较高 / 较低的水平）、各水平均值，以及两种符号区占比（G、V 的 +×+、−×−、+×−、−×+）：
     pole_regime 为均值较高一极全部条目中的占比（该单元的区分靠哪个符号区实现）；strong_regime 为强激活
     （|A| ≥ 该单元 strong_q 分位，即 top-k 列表）中的占比。双负区输出幅度小，常在 pole_regime 中占主导而不进入强激活。

判据固定在 CRITERIA 中并随结果返回，不得在看到结果后调整。选择性是相关性观测（"该单元在这个对照上可区分"），
不说明单元被模型因果使用；因果需要另做消融或干预。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

CRITERIA = {'d_min': 0.8, 'held_min': 0.5, 'perm_alpha': 0.01, 'n_perm': 200, 'strong_q': 0.99, 'top_k': 20}
HELD = ('confirm', 'lockbox', 'lockbox_frame')


def silu(x):
    return x / (1.0 + np.exp(-x))


class MlpField:
    """多个分片的 MLP 中间单元，按条目拼接；按 (读出, 块) 惰性读取。"""

    def __init__(self, shard_dirs: Sequence[str]):
        self.dirs = [Path(d) / 'mlp' if (Path(d) / 'mlp' / 'mlp.json').exists() else Path(d) for d in shard_dirs]
        self.man = [json.loads((d / 'mlp.json').read_text(encoding='utf-8')) for d in self.dirs]
        self.ids = [i for m in self.man for i in m['ids']]
        self.readouts = list(self.man[0]['readouts'])
        self.blocks = list(self.man[0]['blocks'])
        self.d_ff = int(self.man[0]['d_ff'])
        for m in self.man[1:]:
            if m['readouts'] != self.readouts or m['blocks'] != self.blocks or int(m['d_ff']) != self.d_ff:
                raise ValueError('各分片的 MLP 采集口径不一致')

    def GV(self, readout: str, block: int):
        p = self.readouts.index(readout)
        Gs, Vs = [], []
        for d in self.dirs:
            with np.load(d / f'B{block}.npz') as z:
                Gs.append(np.asarray(z['G'][p], dtype=np.float32))
                Vs.append(np.asarray(z['V'][p], dtype=np.float32))
        return np.concatenate(Gs), np.concatenate(Vs)


def _std_diff(A, hi, lo):
    a, b = A[hi], A[lo]
    if len(a) < 2 or len(b) < 2:
        return np.full(A.shape[1], np.nan)
    sd = np.sqrt(((len(a) - 1) * a.var(0, ddof=1) + (len(b) - 1) * b.var(0, ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean(0) - b.mean(0)) / np.maximum(sd, 1e-6)


def design(ids: Sequence[str], items_by_id: Dict[str, dict], factor: str, hi: str, lo: Optional[str] = None):
    """标签向量：+1 = hi 水平，−1 = lo 水平（lo 为 None 时为其余水平），0 = 不参与；以及划分与取值单位。"""
    lab, split, unit = np.zeros(len(ids), int), [], []
    for k, i in enumerate(ids):
        it = items_by_id.get(i)
        if it is None or factor not in (it.get('cond') or {}):
            split.append(None); unit.append(None)
            continue
        v = str(it['cond'][factor])
        lab[k] = 1 if v == hi else (-1 if (lo is None or v == lo) else 0)
        split.append(it.get('split'))
        unit.append((it.get('frame_id'), it.get('filler_id')))
    frames = {}
    for k, u in enumerate(unit):
        if u is not None and lab[k] != 0:
            frames.setdefault(u[0], set()).add(int(lab[k]))
    frame_level = bool(frames) and all(len(v) == 1 for v in frames.values())
    groups = [None if u is None else (u[0] if frame_level else u[1]) for u in unit]
    return lab, np.array(split, dtype=object), groups, ('frame' if frame_level else 'filler')


def _perm_labels(lab, groups, mask, rng):
    """在 mask 内按取值单位整体置换标签（保持每个单位内条目共享同一标签）。"""
    out = lab.copy()
    idx = np.where(mask & (lab != 0))[0]
    gs = sorted({groups[i] for i in idx}, key=str)
    g_lab = {}
    for i in idx:
        g_lab.setdefault(groups[i], []).append(lab[i])
    vals = [int(np.sign(np.mean(g_lab[g]))) for g in gs]
    rng.shuffle(vals)
    new = dict(zip(gs, vals))
    for i in idx:
        out[i] = new[groups[i]]
    return out


def selectivity(field: MlpField, items_by_id: Dict[str, dict], factor: str, hi: str, lo: Optional[str] = None,
                readouts: Optional[Sequence[str]] = None, blocks: Optional[Sequence[int]] = None,
                seed: int = 0, criteria: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    C = dict(CRITERIA, **(criteria or {}))
    lab, split, groups, unit_kind = design(field.ids, items_by_id, factor, hi, lo)
    disc = np.array([s == 'discovery' for s in split])
    held = np.array([s in HELD for s in split])
    out = {'schema': 'neurons.v1', 'factor': factor, 'hi': hi, 'lo': lo if lo is not None else '其余水平',
           'unit': unit_kind, 'criteria': C,
           'n': {'discovery_hi': int(((lab == 1) & disc).sum()), 'discovery_lo': int(((lab == -1) & disc).sum()),
                 'held_hi': int(((lab == 1) & held).sum()), 'held_lo': int(((lab == -1) & held).sum())},
           'tags': []}
    rng = np.random.default_rng(seed)
    perms = [_perm_labels(lab, groups, disc, rng) for _ in range(int(C['n_perm']))]
    for r in (readouts or field.readouts):
        for b in (blocks or field.blocks):
            G, V = field.GV(r, b)
            A = silu(G) * V
            dd = _std_diff(A, (lab == 1) & disc, (lab == -1) & disc)
            dh = _std_diff(A, (lab == 1) & held, (lab == -1) & held)
            null = np.array([np.nanmax(np.abs(_std_diff(A, (pl == 1) & disc, (pl == -1) & disc))) for pl in perms])
            thr = float(np.quantile(null, 1 - float(C['perm_alpha'])))
            cut = max(thr, float(C['d_min']))
            sel = np.where((np.abs(dd) >= cut) & (np.sign(dh) == np.sign(dd)) & (np.abs(dh) >= float(C['held_min'])))[0]
            order = sel[np.argsort(-np.abs(dd[sel]))]
            q = np.quantile(np.abs(A), float(C['strong_q']), axis=0)
            units = []
            for j in order[:int(C['top_k'])]:
                strong = np.abs(A[:, j]) >= q[j]
                g, v = G[strong, j], V[strong, j]
                n_s = max(int(strong.sum()), 1)
                m_hi, m_lo = float(A[lab == 1, j].mean()), float(A[lab == -1, j].mean())
                pole = (lab == 1) if m_hi > m_lo else (lab == -1)        # 均值较高一极的全部条目
                gp, vp = G[pole, j], V[pole, j]
                n_p = max(int(pole.sum()), 1)
                units.append({'unit': int(j), 'd_disc': float(dd[j]), 'd_held': float(dh[j]),
                              'pos_level': hi if m_hi > m_lo else out['lo'], 'neg_level': out['lo'] if m_hi > m_lo else hi,
                              'mean_hi': m_hi, 'mean_lo': m_lo,
                              'strong_regime': {'++': float(((g > 0) & (v > 0)).sum() / n_s), '--': float(((g < 0) & (v < 0)).sum() / n_s),
                                                '+-': float(((g > 0) & (v < 0)).sum() / n_s), '-+': float(((g < 0) & (v > 0)).sum() / n_s)},
                              'pole_regime': {'++': float(((gp > 0) & (vp > 0)).sum() / n_p), '--': float(((gp < 0) & (vp < 0)).sum() / n_p),
                                              '+-': float(((gp > 0) & (vp < 0)).sum() / n_p), '-+': float(((gp < 0) & (vp > 0)).sum() / n_p)},
                              'strong_hi_share': float((lab[strong] == 1).sum() / n_s)})
            out['tags'].append({'readout': r, 'block': int(b), 'null_max_d': thr, 'cut': cut,
                                'n_selected': int(len(sel)), 'n_units': int(A.shape[1]),
                                'n_pos_hi': int((dd[sel] > 0).sum()), 'n_pos_lo': int((dd[sel] < 0).sum()),
                                'units': units})
    out['summary'] = {r: {'n_selected_by_block': {t['block']: t['n_selected'] for t in out['tags'] if t['readout'] == r}}
                      for r in (readouts or field.readouts)}
    return out
