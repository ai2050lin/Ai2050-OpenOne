# -*- coding: utf-8 -*-
"""实体编码图谱（entity_atlas.v1）：G1 层级与差分、G2 共享属性算子。纯 numpy；判据固定在 CRITERIA，随结果返回。

输入：codes[lang][l] = [n_entity, d]，每个实体在中性句式里最后一个 token 的残差（多个句式取平均），l 为 hs 层号；
      labels：super（上位大类）、cat（类别）、color、size；split：discovery / heldout（按实体哈希，确定性）。

G1 层级与差分（苹果 ≈ 物体 + Δ大类 + Δ类别 + δ个体）
  - 嵌套方差分解：总方差 = 大类间 + 类别间（大类内）+ 个体差分 δ（类别内）。
  - 留出预测：只用 discovery 实体的类别 / 大类中心，对 heldout 实体做最近中心分类。
  - 层级正交（Park et al. 2024）：子差 μ_cat − μ_super 与父差 μ_super − μ_all 的 |cos|，对照：大类内打乱类别标签。
  - δ 的有效维度：类别内残差协方差的参与比 PR = (Σλ)² / Σλ²，与全体编码的 PR 对比。
G2 共享属性算子（颜色 ≈ A_color · e）
  - 算子 = 岭回归的线性映射（one-hot 回归，argmax 读出），只在 discovery 实体上拟合；层由 discovery 内 5 折交叉验证选定，
    选定后在 heldout 上只评估一次。
  - 对照：类别多数基线（只知道类别时的最好猜测）与 hs[0] 基线（token 身份）。
  - δ 上的算子：去掉类别中心后再拟合，检验属性是否落在个体差分里（超出层级的信息）。
  - 跨语言复用：中文编码上拟合的算子直接用于英文编码（反之亦然）。
  - 低秩：把编码投影到 discovery 主成分前 r 维再拟合，报告达到满秩准确率 90% 的最小 r。
相对编码：中英两套编码的实体间余弦相似矩阵的 Spearman 相关（RSA），对照为打乱英文实体顺序。
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Sequence

import numpy as np

SCHEMA = 'entity_atlas.v1'
CRITERIA = {
    'schema': SCHEMA,
    'split': 'sha256("EA-v1|" + entity_id)[0] < 154 → discovery（约 60%），否则 heldout',
    'ridge_lambda_rel': 1e-2,                 # λ = 1e-2 · tr(XᵀX)/d
    'cv_folds': 5,
    'min_class_n': 6,                         # 过行为门实体数少于此的颜色不进入颜色分析
    'G1_category_acc_min': 0.6,               # heldout 类别最近中心准确率（机会 1/7）
    'G1_over_hs0_min': 0.15,                  # 且比 hs[0] 高这么多
    'G1_orth_abs_cos_max': 0.2,               # 层级正交：子差与父差的平均 |cos| ≤ 此值
    'G2_margin_min': 0.10,                    # 算子的 heldout 平衡准确率须同时超过类别多数基线与 hs[0] 基线这么多
    'G2_delta_margin_min': 0.10,              # δ 上的算子须超过类别多数基线这么多（属性落在个体差分里）
    'G2_xlang_ratio_min': 0.8,                # 跨语言迁移准确率 ≥ 同语言准确率 × 此值
    'lowrank_frac': 0.9,
    'rsa_perm': 200,
    'seed': 0,
}


def split_of(entity_id: str, salt: str = 'EA-v1') -> str:
    return 'discovery' if hashlib.sha256(f'{salt}|{entity_id}'.encode('utf-8')).digest()[0] < 154 else 'heldout'


# ---------------- 基础 ----------------

def _onehot(y: np.ndarray, classes: Sequence) -> np.ndarray:
    return (np.asarray(y)[:, None] == np.asarray(classes)[None, :]).astype(np.float64)


def ridge_fit(X: np.ndarray, Y: np.ndarray, lam_rel: float):
    mu = X.mean(0)
    Xc = X - mu
    d = Xc.shape[1]
    lam = lam_rel * float((Xc ** 2).sum()) / max(d, 1)
    # 对偶形式（n < d 时更省）：W = Xcᵀ (Xc Xcᵀ + λI)⁻¹ (Y − Ȳ)
    yb = Y.mean(0)
    K = Xc @ Xc.T
    alpha = np.linalg.solve(K + lam * np.eye(K.shape[0]), Y - yb)
    return {'mu': mu, 'W': Xc.T @ alpha, 'b': yb}


def ridge_predict(m, X: np.ndarray) -> np.ndarray:
    return (X - m['mu']) @ m['W'] + m['b']


def balanced_acc(y_true: Sequence, y_pred: Sequence) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    accs = [float(np.mean(y_pred[y_true == c] == c)) for c in np.unique(y_true)]
    return float(np.mean(accs)) if accs else float('nan')


def participation_ratio(X: np.ndarray) -> float:
    Xc = X - X.mean(0)
    s = np.linalg.svd(Xc, compute_uv=False) ** 2
    return float(s.sum() ** 2 / max((s ** 2).sum(), 1e-30))


def _centroids(X, y):
    cls = sorted(set(y))
    return cls, np.stack([X[np.asarray(y) == c].mean(0) for c in cls])


def nearest_centroid_acc(Xtr, ytr, Xte, yte) -> float:
    cls, C = _centroids(Xtr, ytr)
    d = ((Xte[:, None, :] - C[None]) ** 2).sum(-1)
    pred = np.asarray(cls)[d.argmin(1)]
    return float(np.mean(pred == np.asarray(yte)))


# ---------------- G1 ----------------

def hierarchy_variance(X: np.ndarray, sup: Sequence, cat: Sequence) -> Dict[str, float]:
    sup, cat = np.asarray(sup), np.asarray(cat)
    mu = X.mean(0)
    tot = float(((X - mu) ** 2).sum())
    b_sup = sum(float((sup == s).sum() * ((X[sup == s].mean(0) - mu) ** 2).sum()) for s in np.unique(sup))
    b_cat = 0.0
    for c in np.unique(cat):
        m = cat == c
        s = sup[m][0]
        b_cat += float(m.sum() * ((X[m].mean(0) - X[sup == s].mean(0)) ** 2).sum())
    return {'super': b_sup / tot, 'cat_within_super': b_cat / tot, 'delta': 1 - (b_sup + b_cat) / tot}


def hierarchy_orthogonality(X: np.ndarray, sup: Sequence, cat: Sequence, n_shuffle: int = 200, seed: int = 0) -> Dict[str, float]:
    sup, cat = np.asarray(sup), np.asarray(cat)

    def stat(c_labels):
        mu = X.mean(0)
        vals = []
        for c in np.unique(c_labels):
            m = c_labels == c
            s = sup[m][0]
            child = X[m].mean(0) - X[sup == s].mean(0)
            parent = X[sup == s].mean(0) - mu
            vals.append(abs(float(child @ parent / (np.linalg.norm(child) * np.linalg.norm(parent) + 1e-12))))
        return float(np.mean(vals))
    real = stat(cat)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_shuffle):
        sh = cat.copy()
        for s in np.unique(sup):                    # 只在同一大类内打乱，父差不变
            idx = np.where(sup == s)[0]
            sh[idx] = sh[rng.permutation(idx)]
        null.append(stat(sh))
    return {'mean_abs_cos': real, 'shuffled_mean': float(np.mean(null)), 'shuffled_p05': float(np.quantile(null, 0.05))}


def g1_layer(X, sup, cat, split) -> Dict[str, float]:
    split = np.asarray(split)
    d, h = split == 'discovery', split == 'heldout'
    sup, cat = np.asarray(sup), np.asarray(cat)
    resid = X - np.stack([X[cat == c].mean(0) for c in cat])
    return {
        'cat_acc_heldout': nearest_centroid_acc(X[d], cat[d], X[h], cat[h]),
        'super_acc_heldout': nearest_centroid_acc(X[d], sup[d], X[h], sup[h]),
        **{f'var_{k}': v for k, v in hierarchy_variance(X, sup, cat).items()},
        'pr_all': participation_ratio(X), 'pr_delta': participation_ratio(resid),
    }


# ---------------- G2 ----------------

def _cv_acc(X, y, classes, lam, folds, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    parts = np.array_split(idx, folds)
    pred = np.empty(len(y), dtype=object)
    for k in range(folds):
        te = parts[k]
        tr = np.setdiff1d(idx, te)
        m = ridge_fit(X[tr], _onehot(y[tr], classes), lam)
        pred[te] = np.asarray(classes)[ridge_predict(m, X[te]).argmax(1)]
    return balanced_acc(y, pred)


def operator_eval(X_by_layer: List[np.ndarray], y: Sequence, split: Sequence, cat: Sequence, *, crit=CRITERIA,
                  X_transfer: List[np.ndarray] = None, delta: bool = False) -> Dict[str, object]:
    """层由 discovery 交叉验证选定，heldout 评估一次。delta=True 时先去掉类别中心（只用 discovery 实体估计中心）。"""
    y, split, cat = np.asarray(y), np.asarray(split), np.asarray(cat)
    d, h = split == 'discovery', split == 'heldout'
    classes = sorted(set(y[d]))

    def prep(X):
        if not delta:
            return X
        cents = {c: X[d & (cat == c)].mean(0) for c in set(cat)}
        return X - np.stack([cents[c] for c in cat])
    curve = [_cv_acc(prep(X)[d], y[d], classes, crit['ridge_lambda_rel'], crit['cv_folds'], crit['seed']) for X in X_by_layer]
    best = int(np.nanargmax(curve))
    Xb = prep(X_by_layer[best])
    m = ridge_fit(Xb[d], _onehot(y[d], classes), crit['ridge_lambda_rel'])
    acc = balanced_acc(y[h], np.asarray(classes)[ridge_predict(m, Xb[h]).argmax(1)])
    out = {'layer': best, 'cv_curve': curve, 'heldout_bal_acc': acc, 'classes': classes}
    if X_transfer is not None:
        Xt = X_transfer[best]
        if delta:
            Xt = prep(Xt)
        out['transfer_bal_acc'] = balanced_acc(y[h], np.asarray(classes)[ridge_predict(m, Xt[h]).argmax(1)])
    return out


def category_majority_acc(y, split, cat) -> float:
    y, split, cat = np.asarray(y), np.asarray(split), np.asarray(cat)
    d, h = split == 'discovery', split == 'heldout'
    maj = {}
    for c in set(cat):
        vals, cnt = np.unique(y[d & (cat == c)], return_counts=True)
        maj[c] = vals[cnt.argmax()] if len(vals) else None
    return balanced_acc(y[h], np.array([maj.get(c) for c in cat[h]], dtype=object))


def lowrank_curve(X, y, split, ranks, crit=CRITERIA) -> Dict[int, float]:
    y, split = np.asarray(y), np.asarray(split)
    d, h = split == 'discovery', split == 'heldout'
    classes = sorted(set(y[d]))
    mu = X[d].mean(0)
    _, _, Vt = np.linalg.svd(X[d] - mu, full_matrices=False)
    out = {}
    for r in ranks:
        P = Vt[:r].T
        Z = (X - mu) @ P
        m = ridge_fit(Z[d], _onehot(y[d], classes), crit['ridge_lambda_rel'])
        out[int(r)] = balanced_acc(y[h], np.asarray(classes)[ridge_predict(m, Z[h]).argmax(1)])
    return out


def rsa(A: np.ndarray, B: np.ndarray, n_perm: int = 200, seed: int = 0) -> Dict[str, float]:
    def sim(X):
        Xc = X - X.mean(0)
        Xn = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)
        return Xn @ Xn.T
    iu = np.triu_indices(len(A), 1)
    a, b = sim(A)[iu], sim(B)[iu]
    rank = lambda v: np.argsort(np.argsort(v))
    real = float(np.corrcoef(rank(a), rank(b))[0, 1])
    rng = np.random.default_rng(seed)
    SB = sim(B)
    null = []
    for _ in range(n_perm):
        p = rng.permutation(len(B))
        null.append(float(np.corrcoef(rank(a), rank(SB[p][:, p][iu]))[0, 1]))
    return {'spearman': real, 'null_p95': float(np.quantile(null, 0.95))}
