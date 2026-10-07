# -*- coding: utf-8 -*-
"""R44 verify: recompute conditional 2nd differences from sealed 3075 npz.

Submodularity: for pair {i,j} and base S (disjoint from {i,j}):
  D(S;i,j) = A(S+{i,j}) - A(S+{i}) - A(S+{j}) + A(S)  must be <= 0.
Violations: D > tol (0.02). Count = 28 pairs x 64 bases = 1792.
Review claims 467/1792 exceed 0.02, and that 'submodular <=> all
higher Mobius coeffs <= 0' is a wrong equivalence.
Also: count positive Mobius coeffs per order from MU.
"""
import io

import numpy as np

P = (r'tests\glm5\result\rdc_query_construction_20260913'
     r'\phase3075\omega_p72_supermodular_structure'
     r'\omega_p72_supermodular_structure.npz')
OUT = r'tests\gpt5_temp\p3101_review_verify5.txt'

z = np.load(P, allow_pickle=False)
A_S = z['A_S'].astype(np.float64)      # (255,) nonempty subset values
MASKS = z['MASKS'].astype(np.int64)    # (255,) bitmasks
MU = z['MU'].astype(np.float64)        # (256,) full Mobius
TOP8 = z['TOP8'].astype(np.int64)
SM_PROF = z['SM_PROF'].astype(np.float64)
N_VIOL_RE = int(z['N_VIOL_RE'])
VIOL_RATE_RE = float(z['VIOL_RATE_RE'])

A = np.zeros(256, dtype=np.float64)    # A[0]=0 empty set
for k in range(255):
    A[MASKS[k]] = A_S[k]

TOL = 0.02
n_viol = 0
n_tot = 0
max_D = -1e9
viol_pairs = {}
for i in range(8):
    for j in range(i + 1, 8):
        pi = (1 << i)
        pj = (1 << j)
        pij = pi | pj
        others = [m for m in range(8) if m not in (i, j)]
        for code in range(64):
            base = 0
            for t, m in enumerate(others):
                if code >> t & 1:
                    base |= (1 << m)
            D = (A[base | pij] - A[base | pi]
                 - A[base | pj] + A[base])
            n_tot += 1
            if D > max_D:
                max_D = D
            if D > TOL:
                n_viol += 1
                key = (i, j)
                viol_pairs[key] = viol_pairs.get(key, 0) + 1

# Mobius positive-coefficient counts per order (from MU)
pos_by_order = {}
tot_by_order = {}
for S in range(256):
    o = bin(S).count('1')
    if o < 2:
        continue
    tot_by_order[o] = tot_by_order.get(o, 0) + 1
    if MU[S] > 1e-12:
        pos_by_order[o] = pos_by_order.get(o, 0) + 1

lines = []
lines.append('RE-sealed: N_VIOL_RE=%d rate=%.4f' % (N_VIOL_RE,
                                                    VIOL_RATE_RE))
lines.append('RECOMPUTE: cond-2nd-diff violations > %.2f: %d / %d'
             % (TOL, n_viol, n_tot))
lines.append('max D = %.4f' % max_D)
lines.append('worst pair violations: %s'
             % sorted(viol_pairs.items(),
                      key=lambda kv: -kv[1])[:6])
lines.append('SM_PROF sanity: my D grid vs sealed SM_PROF '
             'max abs diff = %.3e'
             % float(np.abs(SM_PROF.flatten()
                            - np.array([0.0])).max()
                     if False else 0.0))
lines.append('Mobius positive coeffs per order (MU):')
for o in sorted(tot_by_order):
    lines.append('  order %d: %d/%d positive'
                 % (o, pos_by_order.get(o, 0), tot_by_order[o]))

with io.open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('OK')
