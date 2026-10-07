# -*- coding: utf-8 -*-
"""Probe: bit-compare 3083 (L33, fp64 TT path)
vs 3084 (L33 replay, fp32-roundtrip TT path).
Writes report to p3085_probe.txt."""
import io

import numpy as np

R3 = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
      r'\rdc_query_construction_20260913'
      r'\phase3083'
      r'\omega_p80_third_model_arbitration'
      r'\omega_p80_third_model_arbitration.npz')
R4 = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
      r'\rdc_query_construction_20260913'
      r'\phase3084'
      r'\omega_p81_3b_layer_scan'
      r'\omega_p81_3b_layer_scan.npz')
z3 = np.load(R3, allow_pickle=False)
z4 = np.load(R4, allow_pickle=False)
o = []
for fk in ('A', 'B', 'C'):
    m3 = float(z3['MED_C_' + fk])
    m4 = float(z4['MED_C_L33_' + fk])
    d_mc = abs(m3 - m4)
    n3 = int(z3['N_NEG_' + fk])
    n4 = int(z4['N_NEG_L33_' + fk])
    t3 = [int(h) for h in z3['TOP8_' + fk]]
    t4 = [int(h) for h in z4['TOP8_L33_' + fk]]
    c3 = np.asarray(z3['CS1H_' + fk])
    c4 = np.asarray(z4['CS1H_L33_' + fk])
    d_cs = (float(np.max(np.abs(c3 - c4)))
            if c3.shape == c4.shape
            else float('inf'))
    r3 = float(z3['R_ALL_' + fk])
    r4 = float(z4['R_ALL_L33_' + fk])
    d_ra = abs(r3 - r4)
    o.append('%s med_c %.17g vs %.17g diff=%g '
             'nneg %d/%d top8_eq=%s cs1h_diff=%g '
             'r_all %.17g vs %.17g diff=%g'
             % (fk, m3, m4, d_mc, n3, n4,
                t3 == t4, d_cs, r3, r4, d_ra))
    o.append('   top8_3=%s top8_4=%s'
             % (t3, t4))
io.open(r'D:\AI2050\Ai2050-OpenOne\tests'
        r'\gpt5_temp\p3085_probe.txt', 'w'
        ).write('\n'.join(o) + '\n')
print('PROBE_DONE')
