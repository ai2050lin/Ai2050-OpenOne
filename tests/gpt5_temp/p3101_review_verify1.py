# -*- coding: utf-8 -*-
"""Verify review claims R53-magnitude / R51 / R55 against sealed npz.

Reads:
  P98 (3100): RESD_A/B/C  -> relative L2 error of the first-order
             increment prediction (median / quartiles), ACTC pooled.
  P99z (3099): Jaccard gate value; Top256 increment share if present.
Writes a report; no GPU work.
"""
import io
import json

import numpy as np

R98 = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
       r'\rdc_query_construction_20260913\phase3100'
       r'\omega_p98_upstream_predict'
       r'\omega_p98_upstream_predict.npz')
R99 = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
       r'\rdc_query_construction_20260913\phase3099'
       r'\omega_p97_mlp_neuron_anatomy'
       r'\omega_p97_mlp_neuron_anatomy.npz')
OUT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\p3101_review_verify1.txt')

lines = []
z = np.load(R98, allow_pickle=False)
keys = list(z.keys())
lines.append('P98 keys: %d' % len(keys))
for fk in ('A', 'B', 'C'):
    k = 'RESD_' + fk
    if k in keys:
        v = z[k]
        lines.append(
            '%s n=%d med=%.4f q25=%.4f q75=%.4f '
            'min=%.4f max=%.4f'
            % (k, v.size, float(np.median(v)),
               float(np.percentile(v, 25)),
               float(np.percentile(v, 75)),
               float(v.min()), float(v.max())))
for k in ('PRED_POOLED_A', 'PRED_POOLED_B',
          'PRED_POOLED_C', 'H_F1', 'H_F2',
          'H_F3', 'VERDICT'):
    if k in keys:
        lines.append('%s = %s'
                     % (k, str(z[k].tolist())
                        if z[k].ndim else str(z[k])))

try:
    z9 = np.load(R99, allow_pickle=False)
    k9 = list(z9.keys())
    lines.append('P99 keys: %d' % len(k9))
    for k in k9:
        if ('JAC' in k.upper() or 'GATE' in k.upper()
                or 'TOP' in k.upper() or 'SHARE' in k.upper()
                or 'VERDICT' in k.upper()
                or 'THRESH' in k.upper()):
            v = z9[k]
            lines.append('%s = %s'
                         % (k, str(v.tolist())
                            if v.ndim and v.size < 32
                            else str(v)))
except Exception as e:
    lines.append('P99 load err %r' % e)

with io.open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('OK')
