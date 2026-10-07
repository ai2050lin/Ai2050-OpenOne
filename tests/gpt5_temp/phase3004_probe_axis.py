# -*- coding: utf-8 -*-
"""Probe: verify the 3001/3002 T3 axis hypothesis.

Hypothesis: in 3001/3002 forward_trk returns per-layer
arrays of shape (1, n_cells, hid) (np.stack of a single
batch capture), and T3 computed
  norm(Dl, axis=1)  -> norms ACROSS CELLS per hidden dim
  sum(Dl*xdir, axis=1) -> sums ACROSS CELLS
i.e. the stored trk_ratio / xdir_proj are NOT the declared
per-cell quantities.

Test on 3004 run3 stored Dl (real per-cell (L,57,2560)):
  variant A (3002 formula, Dl reshaped (1,57,2560)):
      expect bit-level match vs 3002 stored profiles
  variant B (declared per-cell quantity):
      the corrected T3 values.
"""
import io
import json

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
NPZ4 = (BASE + r'\phase3004\omega_p4_operator_'
        r'structure_qwen\omega_p4_operator_'
        r'structure_qwen.npz')
R02 = (BASE + r'\phase3002\omega_g2_robustness_'
       r'source_qwen\result.json')

z = np.load(NPZ4, allow_pickle=True)
r02 = json.load(io.open(R02, encoding='utf-8'))
xdir = z['xdir']            # (57, 2560)
out = []
for key in ('L4', 'L17'):
    D = z['Dl_' + key]       # (L, 57, 2560) real
    li_inj = int(key[1:])
    ref = r02['T3']['trk'][key]['profile']
    out.append('== %s (D %s) ==' % (key, D.shape))
    dA = []
    dB = []
    for k1 in range(D.shape[0]):
        li = li_inj + k1
        Dl = D[k1]
        # variant A: 3002 formula on (1,n,hid)
        Dl3 = Dl[None]
        nrA = float(np.median(
            np.linalg.norm(Dl3, axis=1))) / 2.0
        pjA = float(np.median(
            np.sum(Dl3 * xdir, axis=1) / 2.0))
        # variant B: declared per-cell quantity
        nrB = float(np.median(
            np.linalg.norm(Dl, axis=1))) / 2.0
        pjB = float(np.median(
            np.sum(Dl * xdir, axis=1) / 2.0))
        rp = ref[str(li)]
        dA.append(max(abs(nrA - rp['trk_ratio']),
                      abs(pjA - rp['xdir_proj'])))
        dB.append(max(abs(nrB - rp['trk_ratio']),
                      abs(pjB - rp['xdir_proj'])))
        if li in (li_inj, li_inj + 1, li_inj + 2,
                  NL_SENTINEL := (li_inj + 14)):
            out.append(
                ' l=%d A: trk=%.4f proj=%.4f | '
                'B: trk=%.4f proj=%.4f | ref: '
                'trk=%.4f proj=%.4f'
                % (li, nrA, pjA, nrB, pjB,
                   rp['trk_ratio'],
                   rp['xdir_proj']))
    out.append(' maxdiff A vs ref = %.3e '
               '(B vs ref = %.3e)'
               % (max(dA), max(dB)))

# med ||xdir|| check
mx = np.linalg.norm(xdir, axis=1)
out.append('med||xdir||=%.4f  med||xdir||^2=%.2f'
           % (float(np.median(mx)),
              float(np.median(mx ** 2))))
io.open(r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\tmp_probe.txt', 'w',
        encoding='utf-8').write('\n'.join(out) + '\n')
print('ok')
