# -*- coding: utf-8 -*-
"""phase2920 seal + zero-forward power diagnostic (descriptive).
Diagnostic: per-axis observed top margins vs the axis's own maxT
null max distribution - separates 'true absence' (observed max
well below null tail, like concept) from 'small-n granularity
power failure' (observed max large but matched by permuted max,
plausible for 15-21 word pools).
Output: tests/gpt5_temp/phase2920_seal_report.txt
"""
import hashlib
import json
import os

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase2920', 'multiaxis_word_atlas')
REP = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\phase2920_seal_report.txt')
L = []


def w(s):
    L.append(str(s))


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


with open(os.path.join(OUT, 'execution.json'),
          encoding='utf-8') as f:
    ex = json.load(f)
w('== seal ==')
w('created: %s' % ex['created'])
w('script_sha256_8: %s' % sha8(
    r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
    r'\phase2920_multiaxis_word_atlas.py'))
w('execution.json: %s' % sha8(
    os.path.join(OUT, 'execution.json')))
w('result.json: %s' % sha8(
    os.path.join(OUT, 'result.json')))
w('multiaxis_word_atlas.npz: %s' % sha8(
    os.path.join(OUT, 'multiaxis_word_atlas.npz')))

z = np.load(os.path.join(OUT, 'multiaxis_word_atlas.npz'),
            allow_pickle=True)
sign = z['sign_M'].astype(np.float64)
pmax = z['p_maxT'].astype(np.float64)
mp = z['max_perm'].astype(np.float64)
axes = [str(x) for x in z['axis_names']]
w('== power diagnostic (descriptive, post-hoc) ==')
for ai, ax in enumerate(axes):
    sm = sign[ai]
    flat = [(float(sm[h, li]), int(h), int(li))
            for h in range(32) for li in range(36)]
    flat.sort(key=lambda t: -t[0])
    top3 = flat[:3]
    m = mp[ai]
    top_m = top3[0][0]
    n_ge = int(np.sum(m >= top_m))
    w('%s:' % ax)
    w('  top3 margins: %s'
      % [(round(v, 4), h, li) for (v, h, li) in top3])
    w('  max_perm: min=%.4f p50=%.4f p95=%.4f max=%.4f'
      % (float(m.min()), float(np.percentile(m, 50)),
         float(np.percentile(m, 95)), float(m.max())))
    w('  perms with max >= top obs: %d/200 '
      '(p_maxT_topcell=%.6f)' % (n_ge, float(
          pmax[ai][top3[0][1], top3[0][2]])))
    vals = sorted(set(round(v, 6)
                      for (v, _, _) in flat[:40]), reverse=True)
    w('  distinct margin values in top-40 cells: %d %s'
      % (len(vals), [round(v, 3) for v in vals[:8]]))
w('== transfer curve peaks (|cos|) ==')
cc = np.abs(z['cos_curve'].astype(np.float64))
for ai, ax in enumerate(['lang', 'speed', 'size', 'moist']):
    li = int(np.argmax(cc[ai, 1:36])) + 1
    w('%s: peak |cos| %.4f @L%d'
      % (ax, float(cc[ai, li]), li))

with open(REP, 'w', encoding='utf-8') as f:
    f.write('\n'.join(L) + '\n')
print('OK seal 2920')
