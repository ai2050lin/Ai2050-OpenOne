# -*- coding: utf-8 -*-
"""p3089 probe: A1 layer-scan npz keys at L38
(repro-anchor material for the L38 replica).
Output -> tests/gpt5_temp/p3089_probe_out.txt"""
import io

import numpy as np

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
     r'\rdc_query_construction_20260913'
     r'\phase3087\omega_p84_glm4_layer_scan'
     r'\omega_p84_glm4_layer_scan.npz')
z = np.load(P, allow_pickle=False)
keys = sorted(z.files)
o = ['N_KEYS=%d' % len(keys)]
o.append('L38 KEYS: ' + ' '.join(
    k for k in keys if 'L38' in k))
o.append('SCALAR KEYS: ' + ' '.join(
    k for k in keys
    if not any(s in k for s in
               ('L31', 'L34', 'L37', 'L38'))))
for k in ('VERDICT', 'SEED', 'FORWARDS',
          'SMOKE', 'SETUP_OK', 'L_SEL',
          'L_INJ', 'SPEC_CLASS'):
    if k in keys:
        o.append('%s=%s' % (k, z[k]))
for fk in 'ABC':
    o.append('N_NEG_L38_%s=%s' % (
        fk, int(z['N_NEG_L38_' + fk])))
    o.append('MED_C_L38_%s=%.9f' % (
        fk, float(z['MED_C_L38_' + fk])))
    o.append('R_ALL_L38_%s=%.9f' % (
        fk, float(z['R_ALL_L38_' + fk])))
    t8 = z.get('TOP8_L38_' + fk)
    if t8 is not None:
        o.append('TOP8_L38_%s=%s' % (
            fk, ' '.join(str(int(v))
                         for v in t8)))
    c = z.get('CS1H_L38_' + fk)
    if c is not None:
        o.append('CS1H_L38_%s present len=%d '
                 'top1=%.6f' % (
                     fk, len(c),
                     float(np.max(np.abs(c)))))
for fk in 'ABC':
    o.append('N_NEG_L37_%s=%d MED_C_L37_%s='
             '%.9f R_ALL_L37_%s=%.9f'
             % (fk, int(z['N_NEG_L37_' + fk]),
                fk, float(z['MED_C_L37_' + fk]),
                fk, float(z['R_ALL_L37_' + fk])))
io.open(r'D:\AI2050\Ai2050-OpenOne\tests'
        r'\gpt5_temp\p3089_probe_out.txt',
        'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
