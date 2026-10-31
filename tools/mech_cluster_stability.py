# -*- coding: utf-8 -*-
"""C19 是不是一个「真结构」？——种子稳定性 + 窗口自助重采样。

如果换 KMeans 初值、或把语料窗口重采样 80% 重做一遍，同一批单元仍然聚成一个组
（Jaccard 高），那 C19 是数据里的结构；否则它只是某一次切分的偶然产物。

产出 tempdata/cluster_stability.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

from server import mech_pair_service as mpair

ME = Path('data/mech')
OUT = Path('tempdata/cluster_stability.json')
BLOCK = 24
K = 24


def build(win, n_unit, n_win, col_mask=None):
    rows = np.repeat(np.arange(n_unit, dtype=np.int64), win.shape[1])
    cols = win.ravel()
    X = csr_matrix((np.ones(rows.size, dtype=np.float32), (rows, cols)), shape=(n_unit, n_win))
    X.sum_duplicates()
    X.data[:] = 1.0
    df = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    idf = np.log((1.0 + n_win) / (1.0 + df)) + 1.0
    X = csr_matrix((X.data * np.repeat(idf, np.diff(X.indptr)), X.indices, X.indptr), shape=X.shape)
    nrm = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    nrm[nrm == 0] = 1.0
    X = csr_matrix((X.data / np.repeat(nrm, np.diff(X.indptr)), X.indices, X.indptr), shape=X.shape)
    return X


def embed(X, seed=20261030):
    svd = TruncatedSVD(n_components=32, random_state=seed)
    U = svd.fit_transform(X)
    return U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-9)


def best_match(lab_new, target_idx):
    best = (0.0, -1)
    for c in np.unique(lab_new):
        idx = np.flatnonzero(lab_new == c)
        inter = np.intersect1d(idx, target_idx, assume_unique=True).size
        j = inter / max(idx.size + target_idx.size - inter, 1)
        if j > best[0]:
            best = (j, int(c))
    return best


def main():
    run = mpair._pick_run(36)
    with np.load(mpair._layer_npz(run, BLOCK)) as z:
        top_idx = z['top_idx'].astype(np.int64)
    n_unit = top_idx.shape[0]
    win = mpair._window_ids(top_idx)
    uniq, inv = np.unique(win.ravel(), return_inverse=True)
    win = inv.reshape(win.shape).astype(np.int64)
    n_win = int(uniq.size)
    lab0 = np.load(mpair.CLUST_NPZ)['lab_%d' % BLOCK].astype(np.int64)
    c19 = np.flatnonzero(lab0 == 19)
    print('L%d  n_win %d  C19 size %d' % (BLOCK, n_win, c19.size), flush=True)

    X = build(win, n_unit, n_win)
    res = {'block': BLOCK, 'k': K, 'c19_size': int(c19.size), 'runs': []}

    # ① 只换 KMeans 初值
    for seed in (1, 2, 3):
        Ue = embed(X)
        lab = KMeans(n_clusters=K, n_init=10, random_state=seed).fit_predict(Ue)
        j, c = best_match(lab, c19)
        res['runs'].append({'kind': 'kmeans_seed', 'seed': seed, 'jaccard': round(float(j), 3),
                            'matched_cluster': c, 'size': int((lab == c).sum())})
        print('  kmeans seed %d → 最匹配簇 C%-2d Jaccard %.3f (size %d)'
              % (seed, c, j, (lab == c).sum()), flush=True)

    # ② 窗口自助重采样 80%（结构变了才叫不稳定）
    rng = np.random.default_rng(20261031)
    for t in range(3):
        keep = rng.choice(n_win, size=int(n_win * 0.8), replace=False)
        keep.sort()
        remap = -np.ones(n_win, dtype=np.int64)
        remap[keep] = np.arange(keep.size)
        win2 = remap[win]
        bad = win2 < 0
        if bad.any():                     # 被丢掉的窗口：这些命中直接剔除，用自环占位
            win2 = np.where(bad, np.zeros_like(win2), win2)
        X2 = build(win2, n_unit, int(keep.size))
        Ue = embed(X2, seed=20261030 + t)
        lab = KMeans(n_clusters=K, n_init=10, random_state=20261030 + t).fit_predict(Ue)
        j, c = best_match(lab, c19)
        res['runs'].append({'kind': 'bootstrap80', 'trial': t, 'jaccard': round(float(j), 3),
                            'matched_cluster': c, 'size': int((lab == c).sum())})
        print('  bootstrap %d → 最匹配簇 C%-2d Jaccard %.3f (size %d)'
              % (t, c, j, (lab == c).sum()), flush=True)

    js = [r['jaccard'] for r in res['runs']]
    res['jaccard_median'] = round(float(np.median(js)), 3)
    res['jaccard_min'] = round(float(np.min(js)), 3)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print('median %.3f min %.3f → %s' % (res['jaccard_median'], res['jaccard_min'], OUT), flush=True)


if __name__ == '__main__':
    main()
