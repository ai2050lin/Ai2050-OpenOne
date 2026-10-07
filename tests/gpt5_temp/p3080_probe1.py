# -*- coding: utf-8 -*-
"""Probe for Phase 3080: confirm frozen npz keys
(3078 DM_L34, 3079 T/U/F arrays) and exact values."""
import io
import json

import numpy as np

RDIR = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
z8 = np.load(RDIR + (r'\phase3078'
                     r'\omega_p75_routing_timing'
                     r'\omega_p75_routing_timing.npz'))
z9 = np.load(RDIR + (r'\phase3079'
                     r'\omega_p76_migration_lock'
                     r'\omega_p76_migration_lock.npz'))
z76 = np.load(RDIR + (r'\phase3076'
                      r'\omega_p73_cross_prompt_family'
                      r'\omega_p73_cross_prompt_family'
                      r'.npz'))
o = []
o.append('z8 LYRS=%s' % [int(v) for v in z8['LYRS']])
o.append('z8 keys with DM_L34: %s'
         % [k for k in z8.files if 'L34' in k])
o.append('z8 DM_L34_A shape=%s dtype=%s'
         % (z8['DM_L34_A'].shape,
            z8['DM_L34_A'].dtype))
o.append('z9 keys sample: %s'
         % [k for k in z9.files if
            k in ('T_AB', 'U_AB', 'F1_STT_AB',
                  'F2_CTT_AB', 'F3_SLGP_AB',
                  'F4_SLGB_AB', 'F5_AMP_AB',
                  'E3_F2_CTT_T_AC',
                  'E3P_F2_CTT_T_AC',
                  'SP_R1_REPLAY_AB',
                  'TT_MED_AB')])
res8 = json.load(io.open(
    RDIR + (r'\phase3078\omega_p75_routing_timing'
            r'\result.json'), encoding='utf-8'))
o.append('z8 result sp_cross[7]=%r'
         % (res8['stats']['sp_cross'][7],))

def spearman(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    if den == 0:
        return 0.0
    return float((ra * rb).sum() / den)


re8 = [spearman(np.abs(z8['DM_L34_' + f]),
                np.abs(z8['DM_L34_' + g]))
       for f, g in (('A', 'B'), ('A', 'C'),
                    ('B', 'C'))]
o.append('DM_L34 cross replay=%r' % (re8,))
o.append('replay bit vs result: %s'
         % (float(np.max(np.abs(
             np.array(re8)
             - np.array(res8['stats']
                        ['sp_cross'][7])))) == 0.0,))
n = {f: np.linalg.norm(
    z76['TT_' + f].astype(np.float64), axis=1)
    for f in ('A', 'B', 'C')}
o.append('TT norms med A/B/C = %.6f %.6f %.6f'
         % (float(np.median(n['A'])),
            float(np.median(n['B'])),
            float(np.median(n['C']))))
o.append('z9 F2_CTT_AB[:4]=%s'
         % np.array2string(z9['F2_CTT_AB'][:4],
                           precision=6))
o.append('z9 T_AB[:4]=%s'
         % np.array2string(z9['T_AB'][:4],
                           precision=6))
io.open(r'D:\AI2050\Ai2050-OpenOne\tests'
        r'\gpt5_temp\p3080_probe1.txt', 'w',
        encoding='utf-8').write('\n'.join(o) + '\n')
print('PROBE_DONE')
