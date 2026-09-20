# -*- coding: utf-8 -*-
"""diag_2912a.py -- why is the permutation zigzag null shifted?

Compare, over many iid 57x10 sign matrices:
  (a) zigzag_k of the raw gap sequence (unpermuted class split)
  (b) zigzag_k after one per-column independent permutation
If (b) systematically exceeds (a), the permutation itself changes
the zigzag law and the calibration failure is a property of the
null construction, not of the test idea.  Also print a worked
example gap sequences.
"""
import numpy as np

rng = np.random.default_rng(12345)
N = 5000
n, w = 57, 10
n0 = 22

za, zb, ka, kb = [], [], [], []
for _ in range(N):
    Sm = np.sign(rng.normal(size=(n, w)))
    Sm[Sm == 0] = 1.0
    # (a) unpermuted split: first n0 rows = class 0
    pf0 = (Sm[:n0] > 0).mean(axis=0)
    pf1 = (Sm[n0:] > 0).mean(axis=0)
    ga = np.abs(pf0 - pf1)
    da = np.diff(ga)
    za.append(int(np.sum(da[:-1] * da[1:] < 0)))
    # (b) one per-column independent permutation
    idx = np.argsort(rng.random((n, w)), axis=0)
    Sp = np.take_along_axis(Sm, idx, axis=0)
    q0 = (Sp[:n0] > 0).mean(axis=0)
    q1 = (Sp[n0:] > 0).mean(axis=0)
    gb = np.abs(q0 - q1)
    db = np.diff(gb)
    zb.append(int(np.sum(db[:-1] * db[1:] < 0)))
    # (a') full-row permutation (all layers share one perm)
    perm = rng.permutation(n)
    Sq = Sm[perm]
    r0 = (Sq[:n0] > 0).mean(axis=0)
    r1 = (Sq[n0:] > 0).mean(axis=0)
    gc = np.abs(r0 - r1)
    dc = np.diff(gc)
    ka.append(int(np.sum(dc[:-1] * dc[1:] < 0)))
    # (b') no-split ab check: gap from two RANDOM halves of the
    # column WITHOUT class structure, per-column independent
    idx2 = np.argsort(rng.random((n, w)), axis=0)
    S2 = np.take_along_axis(Sm, idx2, axis=0)
    h0 = (S2[:n0] > 0).mean(axis=0)
    h1 = (S2[n0:] > 0).mean(axis=0)
    gd = np.abs(h0 - h1)
    dd = np.diff(gd)
    kb.append(int(np.sum(dd[:-1] * dd[1:] < 0)))

za, zb, ka, kb = map(np.asarray, (za, zb, ka, kb))
lines = [
    'iid 57x10 sign matrices, %d draws' % N,
    '(a) unpermuted split zigzag   mean %.3f' % za.mean(),
    '(b) per-column perm zigzag    mean %.3f' % zb.mean(),
    '(c) global-row perm zigzag    mean %.3f' % ka.mean(),
    '(d) per-column perm (same as b, fresh) mean %.3f'
    % kb.mean(),
    'binomial(8,0.5) mean 4.0',
    '(a) hist %s' % np.bincount(za, minlength=9).tolist(),
    '(b) hist %s' % np.bincount(zb, minlength=9).tolist(),
]
# worked example
Sm = np.sign(rng.normal(size=(n, w)))
Sm[Sm == 0] = 1.0
pf0 = (Sm[:n0] > 0).mean(axis=0)
pf1 = (Sm[n0:] > 0).mean(axis=0)
ga = np.abs(pf0 - pf1)
idx = np.argsort(rng.random((n, w)), axis=0)
Sp = np.take_along_axis(Sm, idx, axis=0)
q0 = (Sp[:n0] > 0).mean(axis=0)
q1 = (Sp[n0:] > 0).mean(axis=0)
gb = np.abs(q0 - q1)
lines.append('example raw  gap: %s' % np.round(ga, 3).tolist())
lines.append('example perm  gap: %s' % np.round(gb, 3).tolist())
# lag-1 autocorr of diff(gap) within sequence, pooled
def pooled_rho(gaps):
    xs, ys = [], []
    for g in gaps:
        d = np.diff(g)
        xs.append(d[:-1])
        ys.append(d[1:])
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])
ra, rb = [], []
for _ in range(2000):
    Sm = np.sign(rng.normal(size=(n, w)))
    Sm[Sm == 0] = 1.0
    pf0 = (Sm[:n0] > 0).mean(axis=0)
    pf1 = (Sm[n0:] > 0).mean(axis=0)
    ra.append(np.abs(pf0 - pf1))
    idx = np.argsort(rng.random((n, w)), axis=0)
    Sp = np.take_along_axis(Sm, idx, axis=0)
    q0 = (Sp[:n0] > 0).mean(axis=0)
    q1 = (Sp[n0:] > 0).mean(axis=0)
    rb.append(np.abs(q0 - q1))
lines.append('pooled lag-1 autocorr of diff(gap): raw %.4f / '
             'perm %.4f' % (pooled_rho(ra), pooled_rho(rb)))
with open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\diag_2912a.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('OK diag_2912a')
