# -*- coding: utf-8 -*-
"""模式族规律分析（analysis.v1）：在逐条向量（items.npz）上回答「哪些特征稳定」。

对每个层位、每个因子计算：
  1. 迹 η²（基底无关）：Σ_水平 n‖m_ℓ − g‖² / Σ‖x − g‖²；另算「词身份额外解释」= η²(填充) − η²(主因子)。
  2. 白化空间：discovery 中心化后做 PCA，保留 r 个主成分并按特征值收缩白化；方向与探针都在此空间计算，
     避免少数大幅值维度主导 cos。
  3. 方向复现：discovery 水平方向 vs confirm / lockbox / lockbox_frame 同一水平方向的平均 cos
     （原始空间与白化空间各一份）。
  4. 置换零分布：按因子的取值单位（填充级因子按填充、句式级因子按句式、否则逐条）打乱 discovery 标签，
     重算复现 cos 与探针准确率，给出单侧 p 值。
  5. 留出探针：白化空间最近质心分类器，discovery 训练，在留出划分上测准确率（机会水平 = 1/k）。
  6. 句式不变性：主因子方向在另一因子各水平内分别计算，两两 cos 矩阵。
  7. 组合可加性：格均值 vs μ + Δ_a + Δ_b 的残差比。
  8. 长度干扰：token 数与各水平方向投影的相关。
  9. 3D 布局：主因子 / 次因子的 LDA 轴（白化空间）与原始 PCA；层堆叠视图用逐层 2D LDA + Procrustes 对齐。
 10. 规律陈述：按固定判据（CRITERIA，写在代码里、随结果返回）生成，带证据数字与适用边界。

所有判据都是描述性的工程阈值，不是统计显著性的替代；最终证据等级仍由 Research OS 登记。
"""
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from itertools import permutations
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

SCHEMA = 'analysis.v1'
CRITERIA = {
    'probe_margin': 0.25,        # 留出探针准确率需比机会水平高出该值（绝对值）才算「可读出」
    'perm_alpha': 0.01,          # 置换 p 值阈值
    'invariance_cos': 0.6,       # 句式间方向平均 cos ≥ 该值视为跨句式稳定
    'additive_ratio': 0.25,      # 组合残差比 ≤ 该值视为近似可加
    'lexical_excess': 0.15,      # 词身份额外解释方差 ≥ 该值视为词身份主导
    'length_corr': 0.3,          # |r| ≥ 该值提示长度干扰
    'n_perm': 200,
    'whiten_rank': 64,           # 白化保留的主成分上限（另受 discovery 样本数 / 3 限制）
    'whiten_shrink': 1.0,        # 特征值收缩：var + shrink·mean(var)，避免放大噪声维
}
SPLITS_OUT = ('confirm', 'lockbox', 'lockbox_frame')
MAX_POINTS_3D = 4000


@dataclass
class ItemSet:
    layer_tags: List[str]
    space: str
    X: np.ndarray            # (L, N, k) float32
    ids: List[str]
    split: np.ndarray
    cond: List[Dict[str, Any]]
    frame: np.ndarray
    filler: np.ndarray
    n_tok: np.ndarray
    correct: np.ndarray


def load_items(blobs: Sequence[bytes]) -> Optional[ItemSet]:
    parts = []
    for b in blobs:
        z = np.load(io.BytesIO(b), allow_pickle=False)
        if str(z['schema']) != 'items.v1':
            raise ValueError('unsupported items schema')
        parts.append(z)
    if not parts:
        return None
    tags = [str(t) for t in parts[0]['layer_tags']]
    space = str(parts[0]['space'])
    for z in parts[1:]:
        if [str(t) for t in z['layer_tags']] != tags or str(z['space']) != space:
            raise ValueError('items 分片的层位或向量空间不一致，不能合并')
    seen, keep_parts = set(), []
    for z in parts:
        ids = [str(i) for i in z['ids']]
        mask = np.array([i not in seen for i in ids])
        seen.update(ids)
        keep_parts.append((z, mask))
    cat = lambda key: np.concatenate([np.asarray(z[key])[m] for z, m in keep_parts])  # noqa: E731
    X = np.concatenate([np.asarray(z['X'], dtype=np.float32)[:, m] for z, m in keep_parts], axis=1)
    return ItemSet(tags, space, X, [str(i) for i in cat('ids')], cat('split').astype(str),
                   [json.loads(str(c)) for c in cat('cond')], cat('frame').astype(str), cat('filler').astype(str),
                   cat('n_tok').astype(int), cat('correct').astype(int))


def cond_key(cond: Dict[str, Any]) -> str:
    return '|'.join(f'{k}={cond[k]}' for k in sorted(cond))


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return None if na < 1e-9 or nb < 1e-9 else float(a @ b / (na * nb))


def _trace_eta(X: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """校正后的迹 η²（ε²）：(SS_between − (k−1)·MS_within) / SS_total。
    分组越多，未校正 η² 的机会水平越高（≈(k−1)/(N−1)）；校正后无效应时期望≈0。"""
    N = len(X)
    levels = np.unique(labels)
    k = len(levels)
    if N <= k or k < 2:
        return None
    g = X.mean(0)
    tot = float(((X - g) ** 2).sum())
    if tot < 1e-12:
        return None
    ssb = 0.0
    for lv in levels:
        idx = labels == lv
        ssb += idx.sum() * float(((X[idx].mean(0) - g) ** 2).sum())
    msw = (tot - ssb) / (N - k)
    return (ssb - (k - 1) * msw) / tot


def _directions(X: np.ndarray, labels: np.ndarray, levels: Sequence[str]) -> Dict[str, np.ndarray]:
    g = X.mean(0)
    return {lv: X[labels == lv].mean(0) - g for lv in levels if (labels == lv).any()}


def _mean_rep(d1: Dict[str, np.ndarray], d2: Dict[str, np.ndarray]) -> Optional[float]:
    vals = [_cos(d1[k], d2[k]) for k in d1 if k in d2]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def _centroid_acc(Ztr, ytr, Zte, yte, levels) -> Optional[float]:
    cents = {lv: Ztr[ytr == lv].mean(0) for lv in levels if (ytr == lv).any()}
    if len(cents) < 2 or len(Zte) == 0:
        return None
    keys = list(cents)
    C = np.stack([cents[k] for k in keys])
    d = ((Zte[:, None, :] - C[None]) ** 2).sum(-1)
    pred = np.array(keys)[d.argmin(1)]
    return float((pred == yte).mean())


def _unit_of(factor: str, items: ItemSet, idx: np.ndarray) -> str:
    vals = np.array([str(items.cond[i][factor]) for i in idx])
    fill, frame = items.filler[idx], items.frame[idx]
    by_fill = {}
    for f, v in zip(fill, vals):
        by_fill.setdefault(f, set()).add(v)
    if all(len(s) == 1 for s in by_fill.values()) and len(by_fill) > len(set(vals)):
        return 'filler'
    by_frame = {}
    for f, v in zip(frame, vals):
        by_frame.setdefault(f, set()).add(v)
    if all(len(s) == 1 for s in by_frame.values()):
        return 'frame'
    return 'item'


def _permuted_labels(rng, labels, units, unit_kind, frame_perms):
    """返回一次置换后的标签：在取值单位（填充/句式）层面打乱，保持单位内一致。"""
    if unit_kind == 'item':
        return rng.permutation(labels)
    uniq = sorted(set(units))
    lab_of = {u: labels[np.where(units == u)[0][0]] for u in uniq}
    if unit_kind == 'frame' and frame_perms is not None:
        perm = frame_perms[rng.integers(len(frame_perms))]
    else:
        perm = rng.permutation(len(uniq))
    shuffled = {u: lab_of[uniq[p]] for u, p in zip(uniq, perm)}
    return np.array([shuffled[u] for u in units])


def _whiten(Xd: np.ndarray, rank: int, shrink: float):
    mu = Xd.mean(0)
    U, s, Vt = np.linalg.svd(Xd - mu, full_matrices=False)
    r = int(min(rank, max(1, len(Xd) // 3), max(1, (s > 1e-6 * s.max()).sum())))
    var = (s[:r] ** 2) / max(len(Xd) - 1, 1)
    var = var + shrink * var.mean()
    W = Vt[:r].T / np.sqrt(var)
    return mu, W, Vt[:3].T


def _lda_axes(Z: np.ndarray, labels: np.ndarray, n: int = 3) -> np.ndarray:
    """白化空间中，组质心散度矩阵的前 n 个特征向量（不足 n 维时用残差 PCA 补齐）。"""
    g = Z.mean(0)
    levels = np.unique(labels)
    M = np.stack([Z[labels == lv].mean(0) - g for lv in levels])
    _, s, Vt = np.linalg.svd(M, full_matrices=False)
    k = int(min(n, (s > 1e-9).sum()))
    axes = list(Vt[:k])
    if k < n:
        R = (Z - g) - (Z - g) @ np.array(axes).T @ np.array(axes) if axes else (Z - g)
        _, _, Vr = np.linalg.svd(R, full_matrices=False)
        axes += list(Vr[:n - k])
    return np.array(axes).T


def _additivity(Xc, D, filler, labels, levels, a, b):
    """分半可靠的交互比例：按填充词把 discovery 分成两半，各自算格均值与加法预测的残差 I；
    比例 = Σ_c ⟨I_A, I_B⟩ / Σ_c ⟨M_A, M_B⟩（M = 中心化格均值）。两半噪声独立，内积只保留可复现部分；
    0 ≈ 完全可加，越大交互越强。另给未校正的单次残差比作参考。"""
    fills = sorted(set(filler[D]))
    half = {f: i % 2 for i, f in enumerate(sorted(fills, key=lambda x: hashlib.sha256(x.encode('utf-8')).hexdigest()))}

    def parts(sel_idx):
        g0 = Xc[sel_idx].mean(0)
        dA = _directions(Xc[sel_idx], labels[a][sel_idx], levels[a])
        dB = _directions(Xc[sel_idx], labels[b][sel_idx], levels[b])
        out = {}
        for va in levels[a]:
            for vb in levels[b]:
                sel = sel_idx[(labels[a][sel_idx] == va) & (labels[b][sel_idx] == vb)]
                if len(sel) == 0 or va not in dA or vb not in dB:
                    continue
                m = Xc[sel].mean(0) - g0
                out[(va, vb)] = (m, m - dA[va] - dB[vb], len(sel))
        return out

    full = parts(D)
    num = sum(n * float(r @ r) for _, r, n in full.values())
    den = sum(n * float(m @ m) for m, _, n in full.values())
    A = parts(D[np.array([half[f] == 0 for f in filler[D]])])
    B = parts(D[np.array([half[f] == 1 for f in filler[D]])])
    shared = [k for k in A if k in B]
    ni = sum(float(A[k][1] @ B[k][1]) for k in shared)
    nm = sum(float(A[k][0] @ B[k][0]) for k in shared)
    return {'factors': [a, b], 'residual_ratio': num / den if den > 0 else None,
            'reliable_interaction': (ni / nm) if nm > 1e-12 and shared else None, 'cells': len(shared)}


def _procrustes(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """把 B（n×2）旋转/反射到最接近 A 的方向，返回旋转矩阵。"""
    U, _, Vt = np.linalg.svd(B.T @ A)
    return U @ Vt


def analyze(items: ItemSet, factors: List[str], *, texts: Optional[Dict[str, str]] = None,
            exclude_cells: Optional[set] = None, seed: int = 0, n_perm: Optional[int] = None) -> Dict[str, Any]:
    n_perm = CRITERIA['n_perm'] if n_perm is None else n_perm
    keep = np.array([cond_key(c) not in (exclude_cells or set()) for c in items.cond])
    idx_all = np.where(keep)[0]
    split = items.split
    D = idx_all[split[idx_all] == 'discovery']
    main_f = factors[0]
    other_fs = factors[1:]
    levels = {f: sorted({str(items.cond[i][f]) for i in idx_all}) for f in factors}
    labels = {f: np.array([str(c[f]) for c in items.cond]) for f in factors}
    units = {f: _unit_of(f, items, D) for f in factors}
    rng = np.random.default_rng(seed)
    frame_perms = {}
    for f in factors:
        if units[f] == 'frame':
            nf = len(set(items.frame[D]))
            frame_perms[f] = [p for p in permutations(range(nf))][1:] if nf <= 6 else None
    pts = idx_all if len(idx_all) <= MAX_POINTS_3D else np.sort(rng.choice(idx_all, MAX_POINTS_3D, replace=False))
    out_layers = []
    coords: Dict[str, Any] = {}
    stack2d = []
    for li, tag in enumerate(items.layer_tags):
        X = items.X[li].astype(np.float64)
        if len(D) < 4:
            out_layers.append({'tag': tag, 'insufficient': True}); continue
        mu, W, pca3 = _whiten(X[D], CRITERIA['whiten_rank'], CRITERIA['whiten_shrink'])
        Z = (X - mu) @ W
        Xc = X - mu
        layer: Dict[str, Any] = {'tag': tag, 'n_discovery': int(len(D)), 'whiten_rank': int(W.shape[1]), 'factors': {}}
        # 迹 η²
        layer['trace_eta2'] = {f: _trace_eta(Xc[D], labels[f][D]) for f in factors}
        fill_eta = _trace_eta(Xc[D], items.filler[D])
        layer['trace_eta2']['filler'] = fill_eta
        if fill_eta is not None and layer['trace_eta2'][main_f] is not None and units[main_f] == 'filler':
            layer['lexical_excess'] = fill_eta - layer['trace_eta2'][main_f]
        for f in factors:
            lv = levels[f]
            dD_raw = _directions(Xc[D], labels[f][D], lv)
            dD_w = _directions(Z[D], labels[f][D], lv)
            rec: Dict[str, Any] = {'k': len(lv), 'unit': units[f], 'chance': 1.0 / max(len(lv), 1), 'splits': {}}
            for sp in SPLITS_OUT:
                S = idx_all[split[idx_all] == sp]
                if len(S) < 2 or len(set(labels[f][S])) < 2:
                    continue
                rep_raw = _mean_rep(dD_raw, _directions(Xc[S], labels[f][S], lv))
                dS_w = _directions(Z[S], labels[f][S], lv)
                rep_w = _mean_rep(dD_w, dS_w)
                acc = _centroid_acc(Z[D], labels[f][D], Z[S], labels[f][S], lv)
                rec['splits'][sp] = {'n': int(len(S)), 'rep_cos_raw': rep_raw, 'rep_cos_white': rep_w, 'probe_acc': acc}
            # 置换零分布（lockbox 优先，否则 confirm）
            target = 'lockbox' if 'lockbox' in rec['splits'] else ('confirm' if 'confirm' in rec['splits'] else None)
            if target and n_perm > 0:
                S = idx_all[split[idx_all] == target]
                dS_w = _directions(Z[S], labels[f][S], lv)
                obs_rep = rec['splits'][target]['rep_cos_white']
                obs_acc = rec['splits'][target]['probe_acc']
                unit_vals = items.filler[D] if units[f] == 'filler' else (items.frame[D] if units[f] == 'frame' else np.arange(len(D)))
                null_rep, null_acc = [], []
                for _ in range(n_perm):
                    pl = _permuted_labels(rng, labels[f][D], unit_vals, units[f], frame_perms.get(f))
                    null_rep.append(_mean_rep(_directions(Z[D], pl, lv), dS_w) or 0.0)
                    null_acc.append(_centroid_acc(Z[D], pl, Z[S], labels[f][S], lv) or 0.0)
                null_rep, null_acc = np.array(null_rep), np.array(null_acc)
                rec['perm'] = {
                    'split': target, 'n_perm': n_perm,
                    'rep_p': float((1 + (null_rep >= (obs_rep if obs_rep is not None else np.inf)).sum()) / (1 + n_perm)),
                    'acc_p': float((1 + (null_acc >= (obs_acc if obs_acc is not None else np.inf)).sum()) / (1 + n_perm)),
                    'rep_null_q95': float(np.quantile(null_rep, 0.95)), 'acc_null_q95': float(np.quantile(null_acc, 0.95)),
                    'rep_null_hist': np.histogram(null_rep, bins=20, range=(-1, 1))[0].tolist(),
                    'acc_null_hist': np.histogram(null_acc, bins=20, range=(0, 1))[0].tolist(),
                }
            # 长度干扰
            # 格内偏相关：去掉条件格均值后，方向投影与 token 数是否仍相关（句式本身决定长度，不能直接相关）
            nt = items.n_tok[D].astype(float)
            cells = np.array([cond_key(items.cond[i]) for i in D])
            nt_r = nt.copy()
            for ck in np.unique(cells):
                m = cells == ck
                nt_r[m] -= nt[m].mean()
            if np.std(nt_r) > 0:
                rs = []
                for lvl, dv in dD_w.items():
                    proj = Z[D] @ (dv / (np.linalg.norm(dv) + 1e-12))
                    for ck in np.unique(cells):
                        m = cells == ck
                        proj[m] -= proj[m].mean()
                    if np.std(proj) > 0:
                        rs.append(abs(float(np.corrcoef(proj, nt_r)[0, 1])))
                rec['length_corr_max'] = max(rs) if rs else None
            layer['factors'][f] = rec
        # 句式不变性：主因子方向在其他因子各水平内分别计算
        for g in other_fs:
            mats = {}
            gl = levels[g]
            dirs, dirs_raw = {}, {}
            for gv in gl:
                sel = D[labels[g][D] == gv]
                if len(sel) >= 4 and len(set(labels[main_f][sel])) >= 2:
                    dirs[gv] = _directions(Z[sel], labels[main_f][sel], levels[main_f])
                    dirs_raw[gv] = _directions(Xc[sel], labels[main_f][sel], levels[main_f])
            keys = sorted(dirs)
            M = [[(_mean_rep(dirs[a], dirs[b]) if a != b else 1.0) for b in keys] for a in keys]
            off = [M[i][j] for i in range(len(keys)) for j in range(len(keys)) if i != j and M[i][j] is not None]
            off_raw = [_mean_rep(dirs_raw[a], dirs_raw[b]) for a in keys for b in keys if a != b]
            off_raw = [v for v in off_raw if v is not None]
            mats = {'factor': g, 'levels': keys, 'matrix': M, 'mean_offdiag': float(np.mean(off)) if off else None,
                    'mean_offdiag_raw': float(np.mean(off_raw)) if off_raw else None}
            layer.setdefault('invariance', []).append(mats)
        # 组合可加性
        if len(factors) >= 2:
            layer['additivity'] = _additivity(Xc, D, items.filler, labels, levels, factors[0], factors[1])
        # 3D 坐标
        layer_coords = {}
        for f in factors[:2]:
            A = _lda_axes(Z[D], labels[f][D], 3)
            P = Z[pts] @ A
            scale = np.sqrt((P[np.isin(pts, D)] ** 2).sum(1).mean()) or 1.0
            layer_coords[f'lda:{f}'] = np.round(P / scale, 4).tolist()
        P = Xc[pts] @ pca3
        scale = np.sqrt((P ** 2).sum(1).mean()) or 1.0
        layer_coords['pca'] = np.round(P / scale, 4).tolist()
        coords[tag] = layer_coords
        # 层堆叠：主因子 2D LDA 质心
        A2 = _lda_axes(Z[D], labels[main_f][D], 2)
        cents = np.stack([Z[D][labels[main_f][D] == lv].mean(0) @ A2 for lv in levels[main_f]])
        sc = np.sqrt((cents ** 2).sum(1).mean()) or 1.0
        stack2d.append({'tag': tag, 'A2': A2, 'scale': sc, 'cents': cents / sc, 'Z': Z, 'pts': pts})
        out_layers.append(layer)
    # Procrustes 对齐的层堆叠
    stack = None
    if stack2d:
        R = np.eye(2)
        aligned = []
        prev = None
        for st in stack2d:
            c = st['cents']
            if prev is not None:
                R = _procrustes(prev, c)
            c2 = c @ R
            pp = (st['Z'][st['pts']] @ st['A2']) / st['scale'] @ R
            aligned.append({'tag': st['tag'], 'centroids': np.round(c2, 4).tolist(), 'points': np.round(pp, 4).tolist()})
            prev = c2
        stack = {'factor': main_f, 'levels': levels[main_f], 'layers': aligned,
                 'note': '每层主因子 2D LDA（白化空间），相邻层用 Procrustes 旋转对齐质心；z 轴为层序。只保留相对结构，不是同一坐标基底。'}
    show = pts
    point_meta = [{'id': items.ids[i], 'split': str(split[i]), 'cond': items.cond[i], 'frame': str(items.frame[i]),
                   'filler': str(items.filler[i]), 'correct': int(items.correct[i]), 'n_tok': int(items.n_tok[i]),
                   'text': (texts or {}).get(items.ids[i], '')} for i in show]
    payload = {
        'schema': SCHEMA, 'criteria': CRITERIA, 'space': items.space, 'factors': factors, 'levels': levels,
        'units': units, 'n_items': int(len(idx_all)), 'n_excluded': int((~keep).sum()),
        'layers': out_layers, 'points': point_meta, 'coords': coords, 'stack': stack,
    }
    payload['findings'] = findings(payload)
    return payload


def _fmt(v, d=3):
    return '—' if v is None else f'{v:.{d}f}'


def findings(p: Dict[str, Any]) -> List[Dict[str, Any]]:
    """按固定判据把数字翻译成规律陈述。kind: supported / negative / warning / info。"""
    C = p['criteria']
    out: List[Dict[str, Any]] = []
    layers = [l for l in p['layers'] if not l.get('insufficient')]
    if not layers:
        return [{'id': 'none', 'kind': 'info', 'title': '样本不足', 'text': 'discovery 样本不足，无法分析。', 'evidence': {}}]
    factors = p['factors']
    main_f = factors[0]

    def readable(layer, f, sp):
        r = layer['factors'][f]
        s = r['splits'].get(sp)
        perm = r.get('perm') or {}
        if not s or s.get('probe_acc') is None:
            return False
        ok = s['probe_acc'] >= r['chance'] + C['probe_margin']
        if perm.get('split') == sp:
            ok = ok and perm['acc_p'] <= C['perm_alpha']
        return ok

    for f in factors:
        best = max(layers, key=lambda l: ((l['factors'][f]['splits'].get('lockbox') or {}).get('probe_acc') or -1))
        r = best['factors'][f]
        lb = r['splits'].get('lockbox') or {}
        perm = r.get('perm') or {}
        traj = [{'layer': l['tag'], 'acc': (l['factors'][f]['splits'].get('lockbox') or {}).get('probe_acc'),
                 'rep': (l['factors'][f]['splits'].get('lockbox') or {}).get('rep_cos_white')} for l in layers]
        if lb.get('probe_acc') is None:
            continue
        ok_layers = [l['tag'] for l in layers if readable(l, f, 'lockbox')]
        if ok_layers:
            out.append({
                'id': f'readable-{f}', 'kind': 'supported', 'title': f'「{f}」在未见词表上可线性读出',
                'text': (f'{f} 在 {"、".join(ok_layers)} 层可读出；最强在 {best["tag"]} 层：lockbox 探针准确率 {_fmt(lb["probe_acc"])}'
                         f'（机会 {_fmt(r["chance"])}，置换 p={_fmt(perm.get("acc_p"), 4)}），白化方向复现 cos {_fmt(lb.get("rep_cos_white"))}'
                         f'（置换 p={_fmt(perm.get("rep_p"), 4)}）。'),
                'evidence': {'trajectory': traj, 'unit': r['unit']},
                'scope': f'取值单位={r["unit"]}；lockbox 中的{"填充词" if r["unit"] == "filler" else "样本"}未参与发现。',
                'level': '候选稳定观测（未见词表预测）'})
        else:
            out.append({'id': f'readable-{f}', 'kind': 'negative', 'title': f'「{f}」未达到可读出判据',
                        'text': f'各层 lockbox 探针准确率最高 {_fmt(lb["probe_acc"])}（{best["tag"]} 层，机会 {_fmt(r["chance"])}），未同时满足高于机会 {C["probe_margin"]} 且置换 p ≤ {C["perm_alpha"]}。',
                        'evidence': {'trajectory': traj}, 'level': '阴性结果'})
        lf = r['splits'].get('lockbox_frame')
        if lf and lf.get('probe_acc') is not None:
            good = lf['probe_acc'] >= r['chance'] + C['probe_margin']
            out.append({'id': f'frame-gen-{f}', 'kind': 'supported' if good else 'negative',
                        'title': f'「{f}」{"可" if good else "未能"}推广到未见句式',
                        'text': f'{best["tag"]} 层在留出句式上的探针准确率 {_fmt(lf["probe_acc"])}（机会 {_fmt(r["chance"])}），方向复现 cos {_fmt(lf.get("rep_cos_white"))}。',
                        'evidence': {}, 'level': '候选稳定观测（未见句式）' if good else '阴性结果'})
    # 层间分工：哪一层由哪个因子主导
    rows = []
    for l in layers:
        te = l['trace_eta2']
        dom = max(factors, key=lambda f: te.get(f) or -1)
        rows.append({'layer': l['tag'], **{f: te.get(f) for f in factors}, 'filler': te.get('filler'), 'dominant': dom})
    seq = ' → '.join(f'{r["layer"]}:{r["dominant"]}' for r in rows)
    out.append({'id': 'layer-division', 'kind': 'info', 'title': '层间分工（迹 η²，discovery）',
                'text': '各层解释方差最大的因子：' + seq + '。' + '；'.join(
                    f'{r["layer"]} ' + ' / '.join(f'{f} {_fmt(r[f], 2)}' for f in factors) + f' / 词身份 {_fmt(r["filler"], 2)}' for r in rows),
                'evidence': {'rows': rows}, 'level': '描述'})
    # 词身份
    lex = [(l['tag'], l.get('lexical_excess')) for l in layers if l.get('lexical_excess') is not None]
    if lex:
        heavy = [t for t, v in lex if v >= C['lexical_excess']]
        out.append({'id': 'lexical', 'kind': 'warning' if heavy else 'info',
                    'title': '词身份 vs 类别' if not heavy else f'{"、".join(heavy)} 层以词身份为主',
                    'text': '同一类别内不同词之间的额外方差（迹 η²(填充) − η²(' + main_f + ')）：' + '，'.join(f'{t} {_fmt(v, 2)}' for t, v in lex) +
                            ('。这些层的差异主要来自具体词，不宜当作类别特征。' if heavy else '。'),
                    'evidence': {'by_layer': lex}, 'level': '对照'})
    # 句式不变性
    for l in layers:
        for inv in l.get('invariance', []):
            v = inv.get('mean_offdiag')
            if v is None:
                continue
            if l is max(layers, key=lambda x: ((x['factors'][main_f]['splits'].get('lockbox') or {}).get('probe_acc') or -1)):
                stable = v >= C['invariance_cos']
                out.append({'id': f'invariance-{inv["factor"]}', 'kind': 'supported' if stable else 'negative',
                            'title': f'{main_f} 方向{"跨" if stable else "不跨"}{inv["factor"]}稳定',
                            'text': f'{l["tag"]} 层，{main_f} 方向在 {inv["factor"]} 各水平内分别计算，两两平均 cos {_fmt(v)}（白化）/ {_fmt(inv.get("mean_offdiag_raw"))}（原始）；判据 ≥ {C["invariance_cos"]}。',
                            'evidence': {'levels': inv['levels'], 'matrix': inv['matrix']}, 'level': '不变性检查'})
    # 可加性
    adds = [(l['tag'], (l.get('additivity') or {}).get('reliable_interaction')) for l in layers if l.get('additivity')]
    if adds:
        non = [t for t, v in adds if v is not None and v > C['additive_ratio']]
        out.append({'id': 'additivity', 'kind': 'info',
                    'title': '组合近似可加' if not non else f'{"、".join(non)} 层存在可复现的交互',
                    'text': f'分半可靠的交互比例（格均值偏离「μ + Δ{factors[0]} + Δ{factors[1] if len(factors) > 1 else ""}」且两半填充词一致的部分）：' + '，'.join(f'{t} {_fmt(v)}' for t, v in adds) + f'（≤ {C["additive_ratio"]} 视为近似可加）。',
                    'evidence': {'by_layer': adds}, 'level': '组合检查'})
    # 长度干扰
    for l in layers:
        for f, r in l['factors'].items():
            if (r.get('length_corr_max') or 0) >= C['length_corr']:
                out.append({'id': f'length-{l["tag"]}-{f}', 'kind': 'warning', 'title': f'{l["tag"]} 层 {f} 方向与 token 数相关',
                            'text': f'方向投影与 token 数的最大 |r| = {_fmt(r["length_corr_max"])}，需用长度匹配材料复核。',
                            'evidence': {}, 'level': '混杂提示'})
    if p.get('n_excluded'):
        out.append({'id': 'gate', 'kind': 'info', 'title': '行为门排除',
                    'text': f'{p["n_excluded"]} 条样本所在条件格未通过行为门，已排除在全部分析之外。', 'evidence': {}, 'level': '数据'})
    if str(p.get('space', 'full')).startswith('jl'):
        out.append({'id': 'space', 'kind': 'info', 'title': '随机投影空间',
                    'text': f'逐条向量体积超预算，使用固定种子随机投影（{p["space"]}）；方向与距离近似保持，坐标不再对应原生维度。',
                    'evidence': {}, 'level': '数据'})
    return out
