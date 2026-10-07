# -*- coding: utf-8 -*-
"""Diagnose T1 perm anomaly: dump projections + perm dist."""
import io
import json

import numpy as np

R = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase2995'
     r'\omega_f1_glm4_panel')
z = np.load(R + r'\omega_f1_glm4_panel.npz',
            allow_pickle=True)
words = [str(w) for w in z['words']]
proj = z['proj_top'].astype(np.float64)
r = json.load(io.open(R + r'\result.json', encoding='utf-8'))
o = []

# rebuild F/C index from words composite fmt 'F:en:w'
iF = [i for i, w in enumerate(words)
      if w.startswith('F:en:')]
iC = [i for i, w in enumerate(words)
      if w.startswith('C:en:')]
pool = proj[iF + iC]
lab = np.array([1] * len(iF) + [0] * len(iC))
o.append('nF=%d nC=%d' % (len(iF), len(iC)))
o.append('F vals=%s' % np.round(pool[:len(iF)], 2).tolist())
o.append('C vals=%s' % np.round(pool[len(iF):], 2).tolist())
stat = float(np.median(pool[lab == 1])
             - np.median(pool[lab == 0]))
o.append('obs stat=%.4f' % stat)
rng = np.random.default_rng(2995 + 10)
ds = np.empty(4999)
for k in range(4999):
    pm = rng.permutation(pool.size)
    lb = lab[pm]
    ds[k] = np.median(pool[pm][lb == 1]) \
        - np.median(pool[pm][lb == 0])
o.append('perm |d|: min=%.4f p1=%.4f med=%.4f max=%.4f'
         % (np.abs(ds).min(), np.quantile(np.abs(ds), .01),
            np.median(np.abs(ds)), np.abs(ds).max()))
o.append('n_perm |d|>=|obs|: %d' % int((np.abs(ds)
                                        >= abs(stat)).sum()))
# median of 15 samples is the 8th order stat; check gap
sv = np.sort(pool)
o.append('sorted pool=%s' % np.round(sv, 2).tolist())
io.open(r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\tmp_diag2995.txt',
        'w', encoding='utf-8').write('\n'.join(o))
print('done')
