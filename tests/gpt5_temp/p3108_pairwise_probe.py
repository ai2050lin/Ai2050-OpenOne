# -*- coding: utf-8 -*-
"""Phase 3108 probe: pairwise cosine structure of bootstrap
ridge solutions for C1 (3105 last|L8).

M2 used top-8 subspace overlap as a proxy.  The floor being
~0.02 is surprising - if the 16 bootstrap w's share a stable
readout direction, floor should be near 1.  This probe measures
the PRIMARY quantity directly: pairwise |cos| among bootstrap
solutions, and |cos| of each bootstrap solution with the
full-train solution.  Diagnostic only (feeds 3109 design);
does not modify the frozen 3108 result.json.
"""
import io
import json
import os

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R13 = (ROOT + r'\tests\glm5\result'
       r'\rdc_query_construction_20260913')
C5 = (R13 + r'\phase3105'
      r'\omega_p103_incontext_truth_consistency')
OUTF = (ROOT + r'\tests\gpt5_temp'
        r'\p3108_pairwise_probe_out.txt')

SEED = 31080
N_BOOT = 16
lam = json.load(io.open(
    R13 + r'\phase3105'
    r'\omega_p103_incontext_truth_consistency'
    r'\result.json', encoding='utf-8'))[
        'T2_ctx']['last|L8']['lam']

z = np.load(os.path.join(C5, 'capture.npz'),
            allow_pickle=False)
POSITIONS = ['pos0', 'crit_pred', 'crit_obj',
             'query_pred', 'query_obj', 'last']
X = z['X'][:, POSITIONS.index('last'), 8, :] \
    .astype(np.float32)
y = z['truth'].astype(np.int64)
tag = [x.decode() if isinstance(x, bytes) else x
       for x in z['tag']]
keep = np.array([t == 'main' for t in tag])
X, y = X[keep], y[keep]
sp = [x.decode() if isinstance(x, bytes) else x
      for x in z['split'][keep]]
tr = np.where(np.array(sp) == 'train')[0]
mu = X[tr].mean(0)
sd = X[tr].std(0) + 1e-6
Z = (X - mu) / sd
Yv = (y * 2.0 - 1.0).astype(np.float32)


def ridge_fit(Zm, yv, lamv):
    n, d = Zm.shape
    A = (Zm.T @ Zm) / n \
        + lamv * np.eye(d, dtype=np.float32)
    return np.linalg.solve(A, (Zm.T @ yv) / n) \
        .astype(np.float32)


w_full = ridge_fit(Z[tr], Yv[tr], lam)
rng = np.random.RandomState(SEED + 500 + 0)
ws = []
for b in range(N_BOOT):
    idx = rng.choice(tr, size=len(tr), replace=True)
    ws.append(ridge_fit(Z[idx], Yv[idx], lam))
W = np.stack(ws, axis=1)  # (2560, 16)
Wn = W / (np.linalg.norm(W, axis=0, keepdims=True)
          + 1e-12)
C = np.abs(Wn.T @ Wn)  # (16, 16) pairwise |cos|
iu = np.triu_indices(N_BOOT, 1)
off = C[iu]
wn = w_full / (np.linalg.norm(w_full) + 1e-12)
cos_full = np.abs(Wn.T @ wn)

lines = []
lines.append('C1 3105 last|L8, lam=%g, n_train=%d, '
             'boot=%d' % (lam, len(tr), N_BOOT))
lines.append('pairwise |cos| among 16 bootstrap w: '
             'n=%d pairs' % len(off))
lines.append('  min=%.4f p25=%.4f median=%.4f '
             'p75=%.4f max=%.4f mean=%.4f'
             % (off.min(), np.percentile(off, 25),
                np.median(off), np.percentile(off, 75),
                off.max(), off.mean()))
lines.append('|cos| bootstrap w vs full-train w: '
             'min=%.4f median=%.4f max=%.4f '
             'mean=%.4f'
             % (cos_full.min(), np.median(cos_full),
                cos_full.max(), cos_full.mean()))
# reference: random 2560-d vectors |cos| ~ 1/sqrt(2560)
lines.append('chance |cos| for random 2560-d: ~%.4f'
             % (1.0 / np.sqrt(2560)))
# effective dimension of the bootstrap solution family:
# participation ratio of the Gram eigenvalues
G = C.copy()
ev = np.linalg.eigvalsh(G)[::-1]
ev = np.clip(ev, 0, None)
pr = float(ev.sum() ** 2 / (ev ** 2).sum())
lines.append('participation ratio of bootstrap '
             'solution family (16 dirs): %.2f / 16'
             % pr)
io.open(OUTF, 'w', encoding='utf-8').write(
    '\n'.join(lines) + '\n')
print('probe done')
