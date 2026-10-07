# -*- coding: utf-8 -*-
"""p3088 debug: locate the rho/p bit mismatch.
Output -> p3088_debug_out.txt"""
import io

import numpy as np

R = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
     r'\rdc_query_construction_20260913'
     r'\phase3088\omega_p86_continuum_n12')
z = np.load(R + r'\omega_p86_continuum_n12.npz',
            allow_pickle=False)
o = []


def spearman(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ra = np.argsort(np.argsort(a)) \
        .astype(np.float64)
    rb = np.argsort(np.argsort(b)) \
        .astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra * ra).sum()
                  * (rb * rb).sum())
    if den == 0:
        return 0.0
    return float((ra * rb).sum() / den)


tags_mine = ['M4B_AB', 'M4B_AC', 'M4B_BC',
             'MDS7B_AB', 'MDS7B_AC', 'MDS7B_BC',
             'M3B_AB', 'M3B_AC', 'M3B_BC',
             'MGLM4_AB', 'MGLM4_AC', 'MGLM4_BC']
# main-script order = sorted(units):
# '3B_AB' < '4B_AB' < 'DS7B_AB' < 'GLM4_AB'
tags_sorted = ['M3B_AB', 'M3B_AC', 'M3B_BC',
               'M4B_AB', 'M4B_AC', 'M4B_BC',
               'MDS7B_AB', 'MDS7B_AC',
               'MDS7B_BC',
               'MGLM4_AB', 'MGLM4_AC',
               'MGLM4_BC']

for label, tags in (('mine', tags_mine),
                    ('sorted', tags_sorted)):
    s_lo = np.array([float(z['S_LO_' + t])
                     for t in tags])
    T_med = np.array([float(z['T_MED_' + t])
                      for t in tags])
    U_med = np.array([float(z['U_MED_' + t])
                      for t in tags])
    MIG = np.array([float(z['MIG_' + t])
                    for t in tags])
    rt = spearman(s_lo, T_med)
    ru = spearman(s_lo, U_med)
    rm = spearman(s_lo, MIG)
    o.append('%s: rho_T=%.17g rho_U=%.17g '
             'rho_MIG=%.17g'
             % (label, rt, ru, rm))
    o.append('  s_lo=' + ' '.join(
        '%.17g' % v for v in s_lo))
    o.append('  T_med=' + ' '.join(
        '%.17g' % v for v in T_med))

o.append('stored RHO_T=%.17g RHO_U=%.17g '
         'RHO_MIG=%.17g'
         % (float(z['RHO_T']),
            float(z['RHO_U']),
            float(z['RHO_MIG'])))
o.append('stored P_T=%.17g P_U=%.17g '
         'P_MIG=%.17g'
         % (float(z['P_T']), float(z['P_U']),
            float(z['P_MIG'])))
# raw 3085-comparison: is stored RHO_T
# exactly 129/143?
o.append('129/143 = %.17g' % (129.0 / 143.0))
io.open(r'D:\AI2050\Ai2050-OpenOne\tests'
        r'\gpt5_temp\p3088_debug_out.txt',
        'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
