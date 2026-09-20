# -*- coding: utf-8 -*-
"""Phase 2934 seal: anatomy forensics + SHA registration."""
import hashlib
import json
import os

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase2934', 'loadband_anatomy')
SRC_2931 = os.path.join(BASE, 'phase2931', 'skeleton_overlap_null',
                        'skeleton_overlap_null.npz')
SRC_2933 = os.path.join(BASE, 'phase2933', 'full_atlas_ci',
                        'full_atlas_ci.npz')
SURV = [(1, 6), (5, 6), (7, 19), (8, 2), (14, 9), (20, 8),
        (21, 6)]
CFGS = ('pos0', 'pos1', 'pos01')
CONDS = ('same', 'func', 'null')
NH = 32

out = []
rj = json.load(open(os.path.join(OUT, 'result.json'),
                    encoding='utf-8'))
z = np.load(os.path.join(OUT, 'loadband_anatomy.npz'),
            allow_pickle=True)
cells = [tuple(int(v) for v in c) for c in z['cells']]
ci = z['ci_rel'].astype(np.float64)
ix = {(cfg, cn): k for k, (cfg, cn) in
      enumerate((c, n) for c in CFGS for n in CONDS)}
cim = {(cfg, cn): dict(zip(cells, ci[ix[(cfg, cn)]]))
       for cfg in CFGS for cn in CONDS}

out.append('verdict: %s' % rj['final_verdict'])
out.append('runtime: %.1f s' % rj['runtime_s'])
out.append('anchors: a1 %s a2 %s a6 %s | sep same %.2f '
           'func %.2f null %.2f'
           % (rj['anchors']['a1_diff'],
              rj['anchors']['a2_rel'],
              rj['anchors']['a6_diff'],
              rj['anchors']['sep_same'],
              rj['anchors']['a5_sep'],
              rj['anchors']['sep_null']))
out.append('correction_note in prereg: %s'
           % bool(rj['prereg'].get('correction_note')))

# 1. config x cond grand medians (whole-grid)
out.append('')
out.append('== grand median CI_rel (1120 cells) ==')
for cfg in CFGS:
    row = []
    for cn in CONDS:
        v = np.median(list(cim[(cfg, cn)].values()))
        row.append('%s %.6f' % (cn, v))
    out.append('%s: %s' % (cfg, ' | '.join(row)))
g_pos0 = np.median(list(cim[('pos0', 'func')].values()))
g_pos1 = np.median(list(cim[('pos1', 'func')].values()))
g_pos01 = np.median(list(cim[('pos01', 'func')].values()))
out.append('pos01/pos1 ratio %.3f | pos0/pos1 %.3f '
           '(subadditivity check: pos01 vs pos0+pos1 '
           '%.3f vs %.3f)'
           % (g_pos01 / g_pos1, g_pos0 / g_pos1,
              g_pos01, g_pos0 + g_pos1))

# 2. per-layer median profile extremes per config (func)
out.append('')
out.append('== per-layer median CI (func): top-5 layers '
           'per config ==')
for cfg in CFGS:
    med = []
    for li in range(1, 36):
        med.append((float(np.median(
            [cim[(cfg, 'func')][(h, li)]
             for h in range(NH)])), li))
    med.sort(reverse=True)
    out.append('%s: %s' % (cfg, ', '.join(
        'L%d %.6f' % (li, v) for v, li in med[:5])))

# 3. null-condition anomaly forensics
out.append('')
out.append('== null vs func CI ratio per band (pos1) ==')
load = list(range(6, 13))
deep = list(range(28, 36))
for band, name in ((load, 'LOAD'), (deep, 'DEEP')):
    vf, vn = [], []
    for li in band:
        for h in range(NH):
            c = (h, li)
            vf.append(cim[('pos1', 'func')][c])
            vn.append(cim[('pos1', 'null')][c])
    out.append('%s: func %.6f null %.6f ratio %.3f'
               % (name, float(np.median(vf)),
                  float(np.median(vn)),
                  float(np.median(vn)) / float(np.median(vf))))

# 4. survivor + skeleton membership
z31 = np.load(SRC_2931, allow_pickle=True)
shared = (z31['S29_corrected'].astype(bool)
          & z31['Smir_corrected'].astype(bool))
# shared (32,36) -> flat in cells order (li outer, h inner, L0 excl)
sh = dict(zip(cells, shared.T[1:].flatten()))
out.append('')
out.append('== survivor 7 (func) ==')
for c in SURV:
    out.append('%s pos0 %.6f pos1 %.6f pos01 %.6f shared=%s'
               % (c, cim[('pos0', 'func')][c],
                  cim[('pos1', 'func')][c],
                  cim[('pos01', 'func')][c],
                  bool(sh[c])))
skv = np.array([cim[('pos1', 'func')][c]
                for c in cells if sh[c]])
rstv = np.array([cim[('pos1', 'func')][c]
                 for c in cells if not sh[c]])
out.append('skeleton median %.6f vs rest %.6f (x%.3f)'
           % (float(np.median(skv)), float(np.median(rstv)),
              float(np.median(skv)) / float(np.median(rstv))))

# 5. cross-phase: 2934 pos1/func vs 2933 top cells agree?
z33 = np.load(SRC_2933, allow_pickle=True)
ci33 = dict(zip(
    [tuple(int(v) for v in c) for c in z33['cells']],
    z33['ci_rel'].astype(np.float64)))
out.append('')
out.append('== top-5 pos1/func cells (2934 vs 2933 CI) ==')
top = sorted(cells, key=lambda c: -cim[('pos1', 'func')][c])[:5]
for c in top:
    out.append('%s 2934 %.6f 2933 %.6f'
               % (c, cim[('pos1', 'func')][c], ci33[c]))

out.append('')
out.append('== SHA256-8 ==')


def sha8(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


for f in ('execution.json', 'result.json',
          'loadband_anatomy.npz'):
    out.append('%s %s' % (f, sha8(os.path.join(OUT, f))))
for f in (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
          r'\phase2934_loadband_anatomy.py',):
    out.append('script %s' % sha8(f))

rep = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\phase2934_seal_report.txt')
open(rep, 'w', encoding='utf-8').write('\n'.join(out) + '\n')
print('seal ok')
