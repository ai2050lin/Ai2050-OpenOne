# -*- coding: utf-8 -*-
"""Phase 3034 prereg probe: check 3032 npz head-energy
storage and 3027 consumer-head profile storage."""
import io
import json
import os

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
out = []

z32 = np.load(os.path.join(
    BASE, 'phase3032',
    'omega_p2z_deep_peak_anatomy_qwen',
    'omega_p2z_deep_peak_anatomy_qwen.npz'),
    allow_pickle=True)
out.append('=== 3032 keys ===')
for k in z32.files:
    a = z32[k]
    try:
        out.append(' %-28s %s %s'
                   % (k, a.shape, a.dtype))
    except Exception:
        out.append(' %-28s 0-d' % k)

r27 = os.path.join(
    BASE, 'phase3027',
    'omega_p2u_consumer_heads_qwen')
out.append('=== 3027 dir ===')
out.append(str(os.listdir(r27)))
z27 = np.load(os.path.join(
    r27, 'omega_p2u_consumer_heads_qwen.npz'),
    allow_pickle=True)
out.append('=== 3027 keys ===')
for k in z27.files:
    a = z27[k]
    try:
        out.append(' %-28s %s %s'
                   % (k, a.shape, a.dtype))
    except Exception:
        out.append(' %-28s 0-d' % k)

# peek head-energy-like arrays in 3032
for k in ('E_rows', 'head_energy', 'E_all',
          'head8_list', 'head_top8', 'incl_rows'):
    if k in z32.files:
        a = z32[k]
        out.append('peek %s: shape=%s sample=%s'
                   % (k, a.shape,
                      np.round(
                          np.asarray(a, dtype=float)
                          .ravel()[:8], 4).tolist()))

io.open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
        r'\phase3034_probe.txt', 'w',
        encoding='utf-8').write('\n'.join(out))
print('ok')
