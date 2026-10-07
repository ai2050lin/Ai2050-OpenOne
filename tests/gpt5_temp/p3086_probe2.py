# -*- coding: utf-8 -*-
"""Probe2: read 3082 INPUT_SHA values + recompute
sha8 of the three source npz files; read 3079/
3081/3085 T/U medians + spec values. Report ->
p3086_probe2.txt"""
import hashlib
import io

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
        r'\result'
        r'\rdc_query_construction_20260913')
N79 = (BASE + r'\phase3079'
       r'\omega_p76_migration_lock'
       r'\omega_p76_migration_lock.npz')
N81 = (BASE + r'\phase3081'
       r'\omega_p78_ds7b_crossmodel'
       r'\omega_p78_ds7b_crossmodel.npz')
N82 = (BASE + r'\phase3082'
       r'\omega_p79_ds7b_negative_anatomy'
       r'\omega_p79_ds7b_negative_anatomy.npz')
N85 = (BASE + r'\phase3085'
       r'\omega_p82_l34_full_arbitration'
       r'\omega_p82_l34_full_arbitration.npz')
o = []


def sha8(p):
    with io.open(p, 'rb') as f:
        return hashlib.sha256(
            f.read()).hexdigest()[:8]


z82 = np.load(N82, allow_pickle=False)
o.append('INPUT_SHA76=%s recompute76=%s match=%s'
         % (str(z82['INPUT_SHA76']), sha8(N79),
            str(z82['INPUT_SHA76']) == sha8(N79)))
o.append('INPUT_SHA79=%s recompute79=%s match=%s'
         % (str(z82['INPUT_SHA79']), sha8(N81),
            str(z82['INPUT_SHA79']) == sha8(N81)))
o.append('INPUT_SHA81=%s recompute81=%s match=%s'
         % (str(z82['INPUT_SHA81']), sha8(N82),
            str(z82['INPUT_SHA81']) == sha8(N82)))

z79 = np.load(N79, allow_pickle=False)
z81 = np.load(N81, allow_pickle=False)
z85 = np.load(N85, allow_pickle=False)

for nm, z, spref in (('4B', z79, 'SP_R1_REPLAY'),
                     ('DS7B', z81, 'MIG'),
                     ('3B', z85, 'MIG')):
    for p in ('AB', 'AC', 'BC'):
        t = float(np.median(z['T_' + p]))
        u = float(np.median(z['U_' + p]))
        m = float(z[spref + '_' + p])
        o.append('%s %s T_med=%.4f U_med=%.4f '
                 'mig=%.4f' % (nm, p, t, u, m))

for nm, z, pfk in (('4B', z82, 'E3_TOP3_CS_4B_'),
                   ('DS7B', z82,
                    'E3_TOP3_CS_DS7B_'),
                   ('3B', z85, 'E3_TOP3_CS_')):
    v = [float(z[pfk + fk]) for fk in 'ABC']
    pr = [float(z[pfk.replace('TOP3', 'PR')
                + fk]) for fk in 'ABC']
    o.append('%s top3 %s PR %s'
             % (nm,
                ' '.join('%.4f' % x for x in v),
                ' '.join('%.2f' % x for x in pr)))

for nm, z in (('4B', z79), ('DS7B', z81),
              ('3B', z85)):
    f2 = [float(z['F2_CTT_' + p].mean())
          for p in ('AB', 'AC', 'BC')]
    o.append('%s F2_CTT mean %s'
             % (nm, ' '.join('%.4f' % x
                             for x in f2)))
o.append('3079 VERDICT=%s' % str(z79['VERDICT']))
o.append('3081 VERDICT=%s'
         % str(z81['VERDICT']))
o.append('3081 FORWARDS=%d 3085 FORWARDS=%d'
         % (int(z81['FORWARDS']),
            int(z85['FORWARDS'])))
with io.open(r'D:\AI2050\Ai2050-OpenOne'
             r'\tests\gpt5_temp'
             r'\p3086_probe2.txt', 'w',
             encoding='utf-8') as fh:
    fh.write('\n'.join(o) + '\n')
print('PROBE2_DONE')
