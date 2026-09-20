# -*- coding: utf-8 -*-
"""Phase 2935 seal: amplification forensics + SHA registration."""
import hashlib
import json
import os

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase2935', 'null_amp_anatomy')
SRC_2931 = os.path.join(BASE, 'phase2931', 'skeleton_overlap_null',
                        'skeleton_overlap_null.npz')
NH = 32

out = []
rj = json.load(open(os.path.join(OUT, 'result.json'),
                    encoding='utf-8'))
z = np.load(os.path.join(OUT, 'null_amp_anatomy.npz'),
            allow_pickle=True)
cells = [tuple(int(v) for v in c) for c in z['cells']]
amp = z['amp'].astype(np.float64)  # (4, 1120)
amp_names = [str(x) for x in z['amp_names']]
ci = z['ci_rel'].astype(np.float64)  # (5, 1120)
cond_names = [str(x) for x in z['cond_names']]
pos_of = {c: i for i, c in enumerate(cells)}

out.append('verdict: %s' % rj['final_verdict'])
out.append('runtime: %.1f s' % rj['runtime_s'])
out.append('anchors ok: %s (a6 %s bit-exact)'
           % (rj['anchors']['ok'], rj['anchors']['a6_diff']))
out.append('P2 amp-ratio pairwise median: %s'
           % rj['P2']['median'])
out.append('P3 raw-CI pairwise median: %s' % rj['P3']['median'])

# 1. per-cell amp dispersion across the 4 sets
amp_std = amp.std(axis=0)
amp_med = amp.mean(axis=0)
out.append('')
out.append('== per-cell amp dispersion across 4 sets ==')
out.append('cross-set amp std: median %.4f p90 %.4f max %.4f '
           '(cell %s)'
           % (float(np.median(amp_std)),
              float(np.percentile(amp_std, 90)),
              float(amp_std.max()),
              cells[int(amp_std.argmax())]))
hi_var = sorted(cells, key=lambda c: -amp_std[pos_of[c]])[:5]
out.append('top-5 variance cells: %s'
           % [(list(c), round(float(amp_std[pos_of[c]]), 4),
               round(float(amp_med[pos_of[c]]), 3))
              for c in hi_var])

# 2. which layers amplify most (median over heads x sets)
out.append('')
out.append('== per-layer median amp (over 4 sets) ==')
lay_amp = []
for li in range(1, 36):
    vals = []
    for h in range(NH):
        c = (h, li)
        vals.append(float(np.median(amp[:, pos_of[c]])))
    lay_amp.append((float(np.median(vals)), li))
lay_amp.sort(reverse=True)
out.append('top-5 amplifying layers: %s'
           % [('L%d' % li, round(v, 3)) for v, li
              in lay_amp[:5]])
out.append('bottom-5 amplifying layers: %s'
           % [('L%d' % li, round(v, 3)) for v, li
              in lay_amp[-5:]])

# 3. skeleton membership vs amplification
z31 = np.load(SRC_2931, allow_pickle=True)
shared = (z31['S29_corrected'].astype(bool)
          & z31['Smir_corrected'].astype(bool))
sh = dict(zip(cells, shared.T[1:].flatten()))
sk_amp = [amp_med[pos_of[c]] for c in cells if sh[c]]
rst_amp = [amp_med[pos_of[c]] for c in cells if not sh[c]]
out.append('')
out.append('== skeleton vs rest amplification ==')
out.append('skeleton median amp %.4f (n=%d) vs rest %.4f '
           '(n=%d)'
           % (float(np.median(sk_amp)), len(sk_amp),
              float(np.median(rst_amp)), len(rst_amp)))

# 4. CI level comparison func vs null sets (raw medians)
out.append('')
out.append('== raw CI grand medians ==')
for i, cn in enumerate(cond_names):
    out.append('%s: %.6f' % (cn, float(np.median(ci[i]))))

# 5. survivor amp table
out.append('')
out.append('== survivor amp (mean over 4 sets) ==')
SURV = [(1, 6), (5, 6), (7, 19), (8, 2), (14, 9), (20, 8),
        (21, 6)]
for c in SURV:
    out.append('%s amp %.3f (func CI %.6f)'
               % (c, float(amp_med[pos_of[c]]),
                  float(ci[0][pos_of[c]])))

out.append('')
out.append('== SHA256-8 ==')


def sha8(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


for f in ('execution.json', 'result.json',
          'null_amp_anatomy.npz'):
    out.append('%s %s' % (f, sha8(os.path.join(OUT, f))))
out.append('script %s' % sha8(
    r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
    r'\phase2935_null_amp_anatomy.py'))

rep = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\phase2935_seal_report.txt')
open(rep, 'w', encoding='utf-8').write('\n'.join(out) + '\n')
print('seal ok')
