# -*- coding: utf-8 -*-
"""逐层神经元摘要（neuron_layers.v1）：从普查合并结果（mlp_census.v1 的 L{block}.npz）为一个块的全部 MLP 中间单元
生成按真实索引排列的列数据，供 v8 客户端「神经元」主线的层详情阵列着色、排序与悬停。

每个单元的字段（列式，长度 = d_ff）：
  dens   强激活率 P(A > 10^-0.5)（普查直方图口径）       neg   负向强激活率 P(A < −10^-0.5)
  amax   最大激活                                         std   标准差
  regime 主导符号区（0..3 = ++ / −− / +− / −+，按 token 数）
  type   分型 0..5 = T1..T6（规则同 L30-v1 预注册：正极命中 token 集中度 ≥ 0.75 → T1；前一个 token 集中度 ≥ 0.75 → T2；
         强激活率 < 0.001 且尾部比 amax/std ≥ 20 → T3；强激活率 ≥ 0.10 → T4；负向强激活率 > 正向 → T5；其余 T6）
  pconc  正极命中 token 集中度        hit  正极命中 token 的众数（T2 时为前一个 token 的众数）
  ro     直接读出推高最多的 token（W_U(γ⊙W_down[:, j]) 首位）
第 30 块以外的分型是同一规则的描述性应用（只有 L30-v1 是预注册结果）；分型不是证据等级。
"""
from __future__ import annotations

import hashlib
import inspect
from collections import Counter
from typing import Any, Callable, Dict, List, Sequence

import numpy as np

from . import census as C

SCHEMA = 'neuron_layers.v1'
TYPES = [('T1', '单 token 检测器'), ('T2', '上下文 token 检测器'), ('T3', '稀疏强激活'), ('T4', '稠密'), ('T5', '负极主导'), ('T6', '其他')]
RULES = {'conc_min': 0.75, 'sparse_rate': 0.001, 'tail_min': 20.0, 'dense_rate': 0.10, 'strong': 10 ** -0.5,
         'source': 'L30-v1 预注册分型规则（research/neurons/L30-v1.prereg.json）；其他块为描述性应用'}


def strong_bin_index() -> int:
    """正侧第一个下边界 ≥ 10^-0.5 的箱（与 census.summarize_layer 的 strong 口径一致）。"""
    return C.N_SIDE + 1 + C.N_SIDE // 2


def _concentration(M: np.ndarray):
    """每行（一个单元的 k 个极值位置的 token id）众数所占比例与众数；-1 为空位。"""
    out = np.zeros(len(M)); mode = np.full(len(M), -1, dtype=np.int64)
    for i, row in enumerate(M):
        row = row[row >= 0]
        if len(row):
            tok, cnt = Counter(row.tolist()).most_common(1)[0]
            mode[i], out[i] = tok, cnt / len(row)
    return out, mode


def summarize(z: Dict[str, np.ndarray], streams: Sequence[np.ndarray], decode: Callable[[int], str]) -> Dict[str, Any]:
    """一个块的普查数组 → 列式摘要。streams：普查语料各来源的 token 数组（极值位置按 gidx 回查 token）。"""
    hist = z['hist'].astype(np.float64)                          # (F, 65, 4)
    n = float(z['n'])
    sb = strong_bin_index()
    dens = hist[:, sb:, :].sum((1, 2)) / n
    neg = hist[:, :C.N_SIDE - C.N_SIDE // 2, :].sum((1, 2)) / n
    regime = hist.sum(1).argmax(1)
    mean = z['s1'] / n
    std = np.sqrt(np.maximum(z['s2'] / n - mean ** 2, 0))
    tail = z['amax'] / (std + 1e-6)

    def at(gidx, shift):
        src, off = C.split_index(gidx)
        flat = [int(streams[s][o - shift]) if o - shift >= 0 and g >= 0 else -1
                for s, o, g in zip(src.ravel(), off.ravel(), gidx.ravel())]
        return np.array(flat, dtype=np.int64).reshape(gidx.shape)
    pconc, pmode = _concentration(at(z['top_idx'], 0))
    cconc, cmode = _concentration(at(z['top_idx'], 1))
    F = len(dens)
    types = np.full(F, 5, dtype=np.int64)
    for j in range(F):
        if pconc[j] >= RULES['conc_min']:
            types[j] = 0
        elif cconc[j] >= RULES['conc_min']:
            types[j] = 1
        elif dens[j] < RULES['sparse_rate'] and tail[j] >= RULES['tail_min']:
            types[j] = 2
        elif dens[j] >= RULES['dense_rate']:
            types[j] = 3
        elif neg[j] > dens[j]:
            types[j] = 4
    cache: Dict[int, str] = {}

    def dec(i: int) -> str:
        if i < 0:
            return ''
        if i not in cache:
            cache[i] = decode(int(i))
        return cache[i]
    hit = [dec(int(cmode[j] if types[j] == 1 else pmode[j])) for j in range(F)]
    return {'n_units': F, 'n_tokens': int(n),
            'columns': {'dens': _sig(dens, 3), 'neg': _sig(neg, 3), 'amax': _sig(z['amax'], 4), 'std': _sig(std, 3),
                        'regime': regime.astype(int).tolist(), 'type': types.astype(int).tolist(),
                        'pconc': [round(float(v), 2) for v in pconc], 'hit': hit,
                        'ro': [dec(int(z['readout_top'][j, 0])) for j in range(F)]}}


def _sig(values, digits: int) -> List[float]:
    return [float(f'{float(v):.{digits}g}') for v in values]


def code_hash() -> str:
    src = inspect.getsource(summarize) + inspect.getsource(_concentration) + repr(RULES)
    return hashlib.sha256(src.encode('utf-8')).hexdigest()[:12]


def layer_doc(base: Dict[str, Any], *, model_key: str, run: str, block: int, explained: Dict[int, str],
              selectivity: List[Dict[str, Any]]) -> Dict[str, Any]:
    """接口返回的一层：列式摘要 + 解释等级（按单元）+ 各模板选择性前列（按读出位置）。"""
    return {'schema': SCHEMA, 'model_key': model_key, 'run': run, 'block': block, **base,
            'type_names': [f'{a} {b}' for a, b in TYPES], 'regime_names': list(C.REGIMES),
            'explained': {str(u): g for u, g in sorted(explained.items())}, 'selectivity': selectivity,
            'rules': RULES, 'layout': '按真实单元索引排列；分型是描述性分类，不是证据等级'}


def tower_doc(model_key: str, run: str, n_layers: int, d_ff: int, bases: Dict[int, Dict[str, Any]],
              explained: Dict[int, Dict[int, str]], selectivity: List[Dict[str, Any]]) -> Dict[str, Any]:
    """全部块的叠加量：分型计数、强激活率中位数、已解释单元数、各模板的选择性单元数。"""
    rows = []
    for b in range(n_layers):
        base = bases.get(b)
        ex = explained.get(b, {})
        row = {'block': b, 'summary': base is not None,
               'explained': {g: sum(1 for v in ex.values() if v == g) for g in sorted(set(ex.values()))}}
        if base:
            col = base['columns']
            row['types'] = [int(sum(1 for t in col['type'] if t == k)) for k in range(len(TYPES))]
            row['dens_median'] = round(float(np.median(col['dens'])), 6)
        rows.append(row)
    sel = [{k: s.get(k) for k in ('file', 'tm_id', 'version', 'factor', 'hi', 'lo', 'item')} | {'by_readout': s['by_readout']} for s in selectivity]
    return {'schema': 'neuron_tower.v1', 'model_key': model_key, 'run': run, 'n_layers': n_layers, 'd_ff': d_ff,
            'blocks': rows, 'selectivity': sel, 'type_names': [f'{a} {b}' for a, b in TYPES]}
