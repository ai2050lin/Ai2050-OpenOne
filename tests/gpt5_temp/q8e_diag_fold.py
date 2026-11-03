# -*- coding: utf-8 -*-
"""因子分离：折构造（array_split vs perm%5）× 代码路径（cv_fast vs 手写）对 r2_raw 的影响。"""
import os, sys, json
os.environ.setdefault('HF_HUB_OFFLINE','1'); os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import numpy as np
from pathlib import Path
sys.path.insert(0, 'tools')
import pat_q8b as B

B.MODEL_PATH = 'D:/develop/model/Qwen/Qwen3-14B'; B.SMOKE = False
B.RNG = np.random.default_rng(20261101)
tok, E, _ = B.load_E()
pools = B.build_pools(tok, E)
ic, zc = pools['char']; iw, zw = pools['word']
idx = np.concatenate([ic, iw]); z = np.concatenate([zc, zw])
mu = E.mean(0); X = (E[idx]-mu).astype(np.float32)
freq = np.power(10.0, z-9.0); yf = freq - freq.mean()
n = len(idx)

def manual(X, yf, folds):
    r2s = []
    d = X.shape[1]
    for f in folds:
        te = np.zeros(n, bool); te[f] = True; tr = ~te
        Gt = X[tr].T.astype(np.float64) @ X[tr].astype(np.float64)
        bt = X[tr].T.astype(np.float64) @ yf[tr]
        w = np.linalg.solve(Gt + 1e-3*np.trace(Gt)/d*np.eye(d), bt)
        pred = X[te] @ w
        ss = ((yf[te]-pred)**2).sum(); st_ = ((yf[te]-yf[tr].mean())**2).sum()
        r2s.append(1-ss/st_)
    return float(np.mean(r2s)), [float(x) for x in r2s]

perm = np.random.default_rng(12345).permutation(n)
f_mod = [np.where(perm % 5 == k)[0] for k in range(5)]
B.RNG = np.random.default_rng(12345)
f_arr = B.folds_of(n)

print('perm%%5 + cv_fast :', B.cv_fast(X, yf, f_mod, ns='dx|mod')[0])
print('perm%%5 + manual  :', manual(X, yf, f_mod))
print('arrsplit + cv_fast:', B.cv_fast(X, yf, f_arr, ns='dx|arr')[0])
print('arrsplit + manual :', manual(X, yf, f_arr))
