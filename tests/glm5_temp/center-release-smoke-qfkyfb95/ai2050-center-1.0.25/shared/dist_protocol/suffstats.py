# -*- coding: utf-8 -*-
"""可合并充分统计量（suffstats.v1）与派生分析。

Worker 只上传每个 (split, 条件格, 层, 维) 的 n、Σx、Σx²（float64）。这些量在样本上可加，
因此中心节点把任意多个分片/节点的结果相加，得到与「所有样本在一台机器上计算」逐位等价的
η² 分解、条件均值方向与 RDM（同一桶内：同模板 design_sha × 模型快照 × 精度）。

记号（单层、单维）：格 c 有 n_c、S_c=Σx、Q_c=Σx²；N=Σn_c，总均值 g=ΣS_c/N。
    SS_total = ΣQ_c − N·g²
    SS_cells = Σ n_c (S_c/n_c − g)²              （格间）
    SS_F     = Σ_ℓ n_ℓ (m_ℓ − g)²                （因子 F 的水平 ℓ 边际均值 m_ℓ）
    SS_within = SS_total − SS_cells               （格内，需每格 ≥2 个样本）
    η²_F = SS_F / SS_total；不平衡设计下各因子 SS 不可加，interaction 记为
    max(SS_cells − Σ SS_F, 0) 的「残余」，只作描述。
零假设下 E[η²_F] ≈ (k−1)/(N−1)，报告 η² 超出该基线的部分（excess），不做逐维显著性检验。
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

SCHEMA_VERSION = 'suffstats.v1'
SCHEMA_V2 = 'suffstats.v2'          # v1 + 每个层位标签的读出位置 tag_readout / hidden_states 下标 tag_layer
SUPPORTED_SCHEMAS = (SCHEMA_VERSION, SCHEMA_V2)


def cond_key(cond: Dict[str, Any]) -> str:
    return '|'.join(f'{k}={cond[k]}' for k in sorted(cond))


@dataclass
class SuffStats:
    layer_tags: List[str]
    dim: int
    factors: List[str]
    cells: List[Tuple[str, str]] = field(default_factory=list)   # (split, cond_json)
    n: np.ndarray = None            # (C,)
    s: np.ndarray = None            # (L, C, d)
    q: np.ndarray = None            # (L, C, d)
    tag_readout: Optional[List[str]] = None   # v2：每个标签的读出位置（target/cue/final/probe_answer）
    tag_layer: Optional[List[int]] = None     # v2：每个标签的 hidden_states 下标（-1 = 未知）

    def __post_init__(self):
        L = len(self.layer_tags)
        if self.n is None:
            self.n = np.zeros((len(self.cells),), dtype=np.int64)
        if self.s is None:
            self.s = np.zeros((L, len(self.cells), self.dim), dtype=np.float64)
        if self.q is None:
            self.q = np.zeros((L, len(self.cells), self.dim), dtype=np.float64)
        self._index = {c: i for i, c in enumerate(self.cells)}

    # ---- 累加（runner 与测试用） ----
    def _cell(self, split: str, cond: Dict[str, Any]) -> int:
        key = (split, json.dumps(cond, sort_keys=True, ensure_ascii=False))
        idx = self._index.get(key)
        if idx is None:
            idx = len(self.cells)
            self.cells.append(key)
            self._index[key] = idx
            self.n = np.concatenate([self.n, np.zeros((1,), dtype=np.int64)])
            pad = np.zeros((len(self.layer_tags), 1, self.dim), dtype=np.float64)
            self.s = np.concatenate([self.s, pad], axis=1)
            self.q = np.concatenate([self.q, pad.copy()], axis=1)
        return idx

    def add(self, split: str, cond: Dict[str, Any], vectors: Sequence[np.ndarray]) -> None:
        """vectors: 每个层位一个 (d,) 向量，顺序同 layer_tags。"""
        i = self._cell(split, cond)
        self.n[i] += 1
        for l, v in enumerate(vectors):
            v = np.asarray(v, dtype=np.float64)
            self.s[l, i] += v
            self.q[l, i] += v * v

    # ---- 合并 ----
    def merge_inplace(self, other: 'SuffStats') -> 'SuffStats':
        """与 merge 结果相同，但累加到自身（不复制 s / q）：合并多个分片时内存只占一份。"""
        if list(other.layer_tags) != list(self.layer_tags) or other.dim != self.dim:
            raise ValueError('layer_tags/dim 不一致，不能合并（不同模型或不同采集口径）')
        if list(other.factors) != list(self.factors):
            raise ValueError('factors 不一致，不能合并')
        self.tag_readout = self.tag_readout or other.tag_readout
        self.tag_layer = self.tag_layer or other.tag_layer
        for j, key in enumerate(other.cells):
            i = self._cell(key[0], json.loads(key[1]))
            self.n[i] += other.n[j]
            self.s[:, i] += other.s[:, j]
            self.q[:, i] += other.q[:, j]
        return self

    def merge(self, other: 'SuffStats') -> 'SuffStats':
        if list(other.layer_tags) != list(self.layer_tags) or other.dim != self.dim:
            raise ValueError('layer_tags/dim 不一致，不能合并（不同模型或不同采集口径）')
        if list(other.factors) != list(self.factors):
            raise ValueError('factors 不一致，不能合并')
        out = SuffStats(list(self.layer_tags), self.dim, list(self.factors),
                        list(self.cells), self.n.copy(), self.s.copy(), self.q.copy(),
                        self.tag_readout or other.tag_readout, self.tag_layer or other.tag_layer)
        for j, key in enumerate(other.cells):
            i = out._cell(key[0], json.loads(key[1]))
            out.n[i] += other.n[j]
            out.s[:, i] += other.s[:, j]
            out.q[:, i] += other.q[:, j]
        return out

    # ---- 序列化 ----
    def to_npz_bytes(self) -> bytes:
        """有标签元数据时写 v2，否则写 v1（与旧读取方兼容）。"""
        buf = io.BytesIO()
        extra = {}
        if self.tag_readout is not None and self.tag_layer is not None:
            extra = {'tag_readout': np.array([str(x) for x in self.tag_readout]),
                     'tag_layer': np.array(self.tag_layer, dtype=np.int32)}
        np.savez_compressed(
            buf, schema=np.array(SCHEMA_V2 if extra else SCHEMA_VERSION),
            layer_tags=np.array([str(t) for t in self.layer_tags]),
            factors=np.array(json.dumps(self.factors, ensure_ascii=False)),
            cell_split=np.array([c[0] for c in self.cells]),
            cell_cond=np.array([c[1] for c in self.cells]),
            n=self.n, sum=self.s, sumsq=self.q, **extra)
        return buf.getvalue()

    @classmethod
    def from_npz_bytes(cls, data: bytes) -> 'SuffStats':
        z = np.load(io.BytesIO(data), allow_pickle=False)
        if str(z['schema']) not in SUPPORTED_SCHEMAS:
            raise ValueError(f'unsupported suffstats schema: {z["schema"]}')
        cells = list(zip([str(x) for x in z['cell_split']], [str(x) for x in z['cell_cond']]))
        s = np.asarray(z['sum'], dtype=np.float64)
        has_meta = 'tag_readout' in z.files and 'tag_layer' in z.files
        return cls([str(t) for t in z['layer_tags']], int(s.shape[2]), json.loads(str(z['factors'])),
                   cells, np.asarray(z['n'], dtype=np.int64), s, np.asarray(z['sumsq'], dtype=np.float64),
                   [str(x) for x in z['tag_readout']] if has_meta else None,
                   [int(x) for x in z['tag_layer']] if has_meta else None)

    # ---- 视图 ----
    @property
    def splits(self) -> List[str]:
        return sorted({c[0] for c in self.cells if self.n[self._index[c]] > 0})

    def collapse(self, splits: Iterable[str], exclude_conds: Optional[set] = None):
        """选定 splits 内按条件合并：返回 (conds list[dict], n (C,), s (L,C,d), q (L,C,d))。"""
        want = set(splits)
        groups: Dict[str, List[int]] = {}
        for i, (sp, cj) in enumerate(self.cells):
            if sp in want and self.n[i] > 0:
                if exclude_conds and cond_key(json.loads(cj)) in exclude_conds:
                    continue
                groups.setdefault(cj, []).append(i)
        keys = sorted(groups)
        conds = [json.loads(k) for k in keys]
        n = np.array([self.n[groups[k]].sum() for k in keys], dtype=np.int64)
        s = np.stack([self.s[:, groups[k]].sum(axis=1) for k in keys], axis=1) if keys else \
            np.zeros((len(self.layer_tags), 0, self.dim))
        q = np.stack([self.q[:, groups[k]].sum(axis=1) for k in keys], axis=1) if keys else \
            np.zeros((len(self.layer_tags), 0, self.dim))
        return conds, n, s, q


def decompose(conds: List[Dict[str, Any]], n: np.ndarray, s: np.ndarray, q: np.ndarray,
              factors: List[str]) -> Dict[str, Any]:
    """单层分解。s/q: (C, d)。返回逐维数组与汇总；样本不足时返回 None 字段。"""
    N = int(n.sum())
    C = len(conds)
    if N < 2 or C < 2:
        return {'N': N, 'C': C, 'insufficient': True}
    g = s.sum(axis=0) / N
    ss_total = np.maximum(q.sum(axis=0) - N * g * g, 1e-12)
    means = s / n[:, None]
    ss_cells = (n[:, None] * (means - g) ** 2).sum(axis=0)
    out: Dict[str, Any] = {'N': N, 'C': C, 'factors': {}, 'grand': g}
    ss_main_sum = np.zeros_like(g)
    for f in factors:
        levels: Dict[str, List[int]] = {}
        for i, c in enumerate(conds):
            levels.setdefault(str(c[f]), []).append(i)
        k = len(levels)
        ss_f = np.zeros_like(g)
        dirs = {}
        for lvl, idx in levels.items():
            nl = n[idx].sum()
            ml = s[idx].sum(axis=0) / nl
            ss_f += nl * (ml - g) ** 2
            dirs[lvl] = ml - g
        ss_main_sum += ss_f
        eta = ss_f / ss_total
        out['factors'][f] = {'k': k, 'eta2': eta, 'null': (k - 1) / max(N - 1, 1), 'directions': dirs}
    out['eta2_cells'] = ss_cells / ss_total
    out['eta2_interaction_residual'] = np.maximum(ss_cells - ss_main_sum, 0.0) / ss_total if len(factors) > 1 else None
    out['eta2_within'] = np.maximum(ss_total - ss_cells, 0.0) / ss_total if N > C else None
    out['cell_means_centered'] = means - g
    return out


def _cos(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return None
    return float(a @ b / (na * nb))


def cos_matrix(vectors: np.ndarray) -> np.ndarray:
    nrm = np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    v = vectors / nrm
    return v @ v.T


def _upper(m: np.ndarray) -> np.ndarray:
    i, j = np.triu_indices(m.shape[0], k=1)
    return m[i, j]


def _pearson(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if a.size < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _profile(eta: np.ndarray, top: int, bins: int) -> Dict[str, Any]:
    d = eta.shape[0]
    order = np.argsort(eta)[::-1][:top]
    edges = np.linspace(0, d, num=min(bins, d) + 1).astype(int)
    bmax = [float(eta[a:b].max()) for a, b in zip(edges[:-1], edges[1:])]
    bmean = [float(eta[a:b].mean()) for a, b in zip(edges[:-1], edges[1:])]
    return {'mean': float(eta.mean()), 'median': float(np.median(eta)),
            'p99': float(np.quantile(eta, 0.99)),
            'top': [[int(i), float(eta[i])] for i in order],
            'bins': {'edges': [int(e) for e in edges], 'max': bmax, 'mean': bmean}}


def summarize(ss: SuffStats, *, top: int = 20, bins: int = 64,
              exclude_conds: Optional[set] = None, include_full: Optional[Tuple[str, str]] = None,
              base_split: str = 'discovery') -> Dict[str, Any]:
    """中心节点 AGG-v2 载荷：逐层 × 逐划分的 η² 分解、跨划分复现、RDM 一致性与格 cos 矩阵。

    include_full=(layer_tag, factor)：额外返回该层该因子在 base_split 上的逐维 η²（完整，不分箱）。
    """
    splits = [sp for sp in ('discovery', 'confirm', 'lockbox', 'lockbox_frame') if sp in ss.splits]
    payload: Dict[str, Any] = {'schema': 'agg.v2', 'layers': [], 'splits': splits,
                               'factors': list(ss.factors), 'dim': ss.dim,
                               'cells_by_split': {}}
    for sp in splits:
        conds, n, _, _ = ss.collapse([sp], exclude_conds)
        payload['cells_by_split'][sp] = {'cells': len(conds), 'N': int(n.sum()),
                                         'min_cell_n': int(n.min()) if len(n) else 0}
    for l, tag in enumerate(ss.layer_tags):
        layer: Dict[str, Any] = {'tag': tag, 'by_split': {}, 'replication': {}, 'rdm_agreement': {}}
        decs: Dict[str, Dict[str, Any]] = {}
        cond_lists: Dict[str, List[Dict[str, Any]]] = {}
        for sp in splits:
            conds, n, s, q = ss.collapse([sp], exclude_conds)
            dec = decompose(conds, n, s[l], q[l], ss.factors)
            decs[sp], cond_lists[sp] = dec, conds
            if dec.get('insufficient'):
                layer['by_split'][sp] = {'N': dec['N'], 'C': dec['C'], 'insufficient': True}
                continue
            entry = {'N': dec['N'], 'C': dec['C'], 'factors': {}}
            for f, fd in dec['factors'].items():
                prof = _profile(fd['eta2'], top, bins)
                prof['k'] = fd['k']
                prof['null'] = fd['null']
                prof['excess'] = prof['mean'] - fd['null']
                entry['factors'][f] = prof
            entry['cells'] = float(dec['eta2_cells'].mean())
            if dec['eta2_interaction_residual'] is not None:
                entry['interaction_residual'] = float(dec['eta2_interaction_residual'].mean())
            if dec['eta2_within'] is not None:
                entry['within'] = float(dec['eta2_within'].mean())
            layer['by_split'][sp] = entry
            if include_full and include_full[0] == tag and sp == base_split and include_full[1] in dec['factors']:
                layer['full_eta2'] = {'split': sp, 'factor': include_full[1],
                                      'values': [float(x) for x in dec['factors'][include_full[1]]['eta2']]}
        base = decs.get(base_split)
        if base and not base.get('insufficient'):
            # 跨划分复现：discovery 的因子水平方向 vs 其他划分（未见词表 / 未见句式）的同一水平方向
            for sp in splits:
                if sp == base_split or decs[sp].get('insufficient'):
                    continue
                per_factor = {}
                for f, fd in base['factors'].items():
                    other = decs[sp]['factors'].get(f)
                    if not other:
                        continue
                    levels = {}
                    for lvl, v in fd['directions'].items():
                        if lvl in other['directions']:
                            levels[lvl] = _cos(v, other['directions'][lvl])
                    vals = [v for v in levels.values() if v is not None]
                    per_factor[f] = {'levels': levels, 'mean_cos': float(np.mean(vals)) if vals else None}
                layer['replication'][sp] = per_factor
                # RDM 一致性：共享条件格的格均值 cos 结构相关
                a_keys = [cond_key(c) for c in cond_lists[base_split]]
                b_keys = [cond_key(c) for c in cond_lists[sp]]
                shared = [k for k in a_keys if k in set(b_keys)]
                if len(shared) >= 3:
                    ia = [a_keys.index(k) for k in shared]
                    ib = [b_keys.index(k) for k in shared]
                    ma = cos_matrix(base['cell_means_centered'][ia])
                    mb = cos_matrix(decs[sp]['cell_means_centered'][ib])
                    layer['rdm_agreement'][sp] = {'cells': len(shared), 'pearson': _pearson(_upper(ma), _upper(mb))}
            m = cos_matrix(base['cell_means_centered'])
            layer['cell_cos'] = {'split': base_split, 'keys': [cond_key(c) for c in cond_lists[base_split]],
                                 'matrix': [[round(float(x), 5) for x in row] for row in m]}
        payload['layers'].append(layer)
    return payload


def replicate_consistency(a: SuffStats, b: SuffStats) -> Optional[float]:
    """同一分片被两个节点各跑一次：返回 Σx 的相对差（越小越一致）。格集合不同返回 None。"""
    if a.layer_tags != b.layer_tags or a.dim != b.dim or set(a.cells) != set(b.cells):
        return None
    order = [b._index[c] for c in a.cells]
    num = float(np.linalg.norm(a.s - b.s[:, order]))
    den = max(float(np.linalg.norm(a.s)), 1e-12)
    return num / den


def scoreboard_rows(payload: Dict[str, Any], prereg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """把 agg.v2 载荷压成预测记分板行：每层 × 因子的未见词表/未见句式复现分数与预注册门。"""
    rows = []
    threshold = prereg.get('min_lockbox_cos')
    for layer in payload.get('layers', []):
        for sp, per_factor in layer.get('replication', {}).items():
            if sp not in ('lockbox', 'lockbox_frame', 'confirm'):
                continue
            for f, rec in per_factor.items():
                score = rec.get('mean_cos')
                passed = None
                if threshold is not None and score is not None and sp != 'confirm':
                    passed = score >= float(threshold)
                rows.append({'layer': layer['tag'], 'factor': f, 'split': sp, 'score': score,
                             'rdm_pearson': (layer.get('rdm_agreement', {}).get(sp) or {}).get('pearson'),
                             'threshold': threshold if sp != 'confirm' else None, 'passed': passed})
    return rows
