# -*- coding: utf-8 -*-
"""p3156 debug: per-layer norms of A0 / A128 / B128 (zh) to resolve ctx_effect anomaly"""
import numpy as np
import json

NPZ = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913\phase3156\g3p1_position_shift_family\qwen3-4b\collect.npz'
z = np.load(NPZ)
H = z['H'].astype(np.float32)
langs = [s.decode() if isinstance(s, bytes) else str(s) for s in z['lang']]
arms = [s.decode() if isinstance(s, bytes) else str(s) for s in z['arm']]
ks = [int(v) for v in z['k']]
ntg = [int(v) for v in z['n_tgt']]
IDX = {(langs[i], arms[i], ks[i]): i for i in range(len(langs))}
lines = ['H=%s' % (H.shape,)]
ia0 = IDX[('zh', 'A', 0)]
ia = IDX[('zh', 'A', 128)]
ib = IDX[('zh', 'B', 128)]
n = ntg[ia0]
lines.append('ia0=%d ia=%d ib=%d n=%d' % (ia0, ia, ib, n))
for l in (0, 6, 12, 18, 24, 30, 35):
    a0 = H[ia0, :n, l]
    a = H[ia, :n, l]
    b = H[ib, :n, l]
    lines.append('L%02d |A0|=%.3f |A128|=%.3f |B128|=%.3f | d(A128,A0)=%.4f d(B128,A0)=%.4f d(A128,B128)=%.4f | maxabs A0=%.3f B128=%.3f' % (
        l, float(np.linalg.norm(a0)), float(np.linalg.norm(a)), float(np.linalg.norm(b)),
        float(np.linalg.norm(a - a0)), float(np.linalg.norm(b - a0)), float(np.linalg.norm(a - b)),
        float(np.abs(a0).max()), float(np.abs(b).max())))
# fin: token0 readout row values
lines.append('A0 fin token readout head: %s' % np.array2string(H[ia0, 0, 35, :6], precision=3))
lines.append('B128 fin token readout head: %s' % np.array2string(H[ib, 0, 35, :6], precision=3))
lines.append('A0 mid L18 token0 head: %s' % np.array2string(H[ia0, 0, 18, :6], precision=3))
lines.append('B128 mid L18 token0 head: %s' % np.array2string(H[ib, 0, 18, :6], precision=3))
open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3156_diag2.txt', 'w', encoding='utf-8').write(chr(10).join(lines))
print('written')
