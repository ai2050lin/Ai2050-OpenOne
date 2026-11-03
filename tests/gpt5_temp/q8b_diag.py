# -*- coding: utf-8 -*-
"""Q8-B 诊断：为什么 cv_fast 的全维 held-out R² 比 lstsq 低 0.06。

同一 X/y/folds，分别用
  A. lstsq（无 ridge，rcond=None）
  B. solve + ridge λ = 1e-3·tr(G)/d   （主脚本口径）
  C. solve + ridge λ = 1e-6·tr(G)/d
  D. solve + ridge λ = 0
比较 held-out R²，定位差异来源。
"""
import sys
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
from transformers import AutoTokenizer
from wordfreq import zipf_frequency as zf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))

MP = 'D:/develop/model/Qwen/Qwen3-4B'
tok = AutoTokenizer.from_pretrained(MP)
p = Path(MP)
idxf = p / 'model.safetensors.index.json'
mp = json.loads(idxf.read_text(encoding='utf-8'))['weight_map']
key = next(k for k in mp if k.endswith('embed_tokens.weight'))
with safe_open(str(p / mp[key]), framework='pt', device='cpu') as f:
    E = f.get_tensor(key).float().numpy()
mu = E.mean(0)


def cjk1(s):
    return len(s) == 1 and ('\u3400' <= s <= '\u9fff' or '\uf900' <= s <= '\ufaff')


dec = [tok.decode([i]) for i in range(E.shape[0])]
idx, zs = [], []
for i, s in enumerate(dec):
    if (not s) or ('\ufffd' in s):
        continue
    if cjk1(s) or (len(s) >= 2 and all(cjk1(c) for c in s)):
        z = zf(s, 'zh')
        if z > 0:
            idx.append(i)
            zs.append(z)
idx = np.array(idx)
zs = np.array(zs, float)
X = (E[idx] - mu).astype(np.float32)
y = zs - zs.mean()
print('n=%d d=%d' % X.shape)

rng = np.random.default_rng(20261101)
folds = np.array_split(rng.permutation(len(y)), 5)
G_all = (X.T @ X).astype(np.float64)
b_all = (X.T @ np.asarray(y, np.float32)).astype(np.float64)
print('G trace/d = %.4f  min eig = %.3e  max eig = %.3e'
      % (np.trace(G_all) / G_all.shape[0], np.linalg.eigvalsh(G_all)[0], np.linalg.eigvalsh(G_all)[-1]))


def r2(a, b):
    return 1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum()


def run(mode, lam_rel):
    pred = np.zeros(len(y))
    for j, f in enumerate(folds):
        tr = np.ones(len(y), bool)
        tr[f] = False
        Gf = (X[f].T @ X[f]).astype(np.float64)
        bf = (X[f].T @ np.asarray(y[f], np.float32)).astype(np.float64)
        Gt, bt = G_all - Gf, b_all - bf
        if mode == 'lstsq':
            A = np.column_stack([np.ones(int(tr.sum())), X[tr].astype(np.float64)])
            coef, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
            pred[f] = coef[0] + X[f].astype(np.float64) @ coef[1:]
        else:
            d = Gt.shape[0]
            lam = lam_rel * np.trace(Gt) / d
            w = np.linalg.solve(Gt + lam * np.eye(d), bt)
            pred[f] = X[f] @ w
    return r2(y, pred)


print('A lstsq            %.4f' % run('lstsq', 0))
for lr in (0.0, 1e-6, 1e-5, 1e-4, 1e-3):
    print('B ridge %.0e        %.4f' % (lr, run('solve', lr)))

# ---- 直接调用主脚本的 cv_fast，复现 0.776
import pat_q8b as B
print('cv_fast(main)      %.4f' % B.cv_fast(X, y, folds, 'main')[0])
print('cv_fast(fresh ns)  %.4f' % B.cv_fast(X, y, folds, 'zz')[0])
Gchk = B.gcache(('all', 'main'), X)
print('cache dtype', Gchk.dtype, 'trace/d %.4f' % (np.trace(Gchk) / Gchk.shape[0]))
print('my  trace/d %.4f' % (np.trace(G_all) / G_all.shape[0]))
print('max|G-cache - G-mine| = %.3e' % np.abs(Gchk - G_all).max())
bchk = B.gmat(X, y)
print('max|b-cache - b-mine| = %.3e' % np.abs(bchk - b_all).max())

# ---- 用主脚本的池顺序（先全部 CHAR 再全部 WORD）重建
ic, zc, iw, zw = [], [], [], []
for i, s in enumerate(dec):
    if (not s) or ('\ufffd' in s):
        continue
    if cjk1(s):
        z = zf(s, 'zh')
        if z > 0:
            ic.append(i); zc.append(z)
    elif len(s) >= 2 and all(cjk1(c) for c in s):
        z = zf(s, 'zh')
        if z > 0:
            iw.append(i); zw.append(z)
idx2 = np.concatenate([ic, iw])
y2 = np.concatenate([zc, zw]).astype(float)
X2 = (E[idx2] - mu).astype(np.float32)
y2 = y2 - y2.mean()
print('pool2 n=%d (char %d + word %d)' % (len(idx2), len(ic), len(iw)))
rng2 = np.random.default_rng(20261101)
folds2 = np.array_split(rng2.permutation(len(y2)), 5)
print('cv_fast(order=char+word) %.4f' % B.cv_fast(X2, y2, folds2, 'm2')[0])

# 同一池但按 zipf 排序，看折内分布是否影响
o = np.argsort(y2)
print('cv_fast(zipf-sorted)     %.4f' % B.cv_fast(X2[o], y2[o], np.array_split(rng2.permutation(len(y2)), 5), 'm3')[0])
