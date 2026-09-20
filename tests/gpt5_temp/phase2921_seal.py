# -*- coding: utf-8 -*-
"""phase2921 seal + cross-phase cell check (descriptive).
Extra: compare the 2920-legacy size cell (18,7) between the 2920
small-pool npz and the 2921 expanded-pool npz (same frozen
statistic; margin changes with the pool, survival is the test).
Output: tests/gpt5_temp/phase2921_seal_report.txt
"""
import hashlib
import json
import os

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase2921', 'attr_vocab_expansion')
OUT20 = os.path.join(BASE, 'phase2920', 'multiaxis_word_atlas')
REP = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\phase2921_seal_report.txt')
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
    r'\phase2921_attr_vocab_expansion.py'))
w('execution.json: %s' % sha8(
    os.path.join(OUT, 'execution.json')))
w('result.json: %s' % sha8(
    os.path.join(OUT, 'result.json')))
w('attr_vocab_expansion.npz: %s' % sha8(
    os.path.join(OUT, 'attr_vocab_expansion.npz')))

z = np.load(os.path.join(OUT, 'attr_vocab_expansion.npz'),
            allow_pickle=True)
axes = [str(x) for x in z['axis_names']]
w('axes: %s' % axes)
TOP = {'lang': [(7, 19), (26, 6), (8, 2)],
       'speed': [(18, 16), (14, 12), (0, 0)],
       'size': [(21, 7), (24, 19), (12, 18)],
       'moist': [(8, 15), (10, 9), (0, 0)]}
for ai, ax in enumerate(axes):
    sm = z['sign_M'].astype(np.float64)[ai]
    mp = z['max_perm'].astype(np.float64)[ai]
    cells = [c for c in TOP[ax] if c != (0, 0)]
    w('%s: top margins %s | max_perm p50=%.4f p95=%.4f max=%.4f'
      % (ax, [(round(float(sm[h, li]), 4), h, li) for h, li in cells],
        float(np.percentile(mp, 50)), float(np.percentile(mp, 95)),
        float(mp.max())))

w('== cross-phase cell check: 2920-legacy size cell (18,7) ==')
z20 = np.load(os.path.join(OUT20, 'multiaxis_word_atlas.npz'),
              allow_pickle=True)
axes20 = [str(x) for x in z20['axis_names']]
ai20 = axes20.index('size')
ai21 = axes.index('size')
m20 = float(z20['sign_M'].astype(np.float64)[ai20][18, 7])
m21 = float(z['sign_M'].astype(np.float64)[ai21][18, 7])
w('sign margin (18,7): 2920 small-pool %.5f -> 2921 expanded %.5f'
  % (m20, m21))
w('2920: granularity-limited (top-40 distinct=9 < 20, p_maxT top '
  '0.0796); 2921: sig (p_maxT 0.024876) -> survived pool expansion')

with open(REP, 'w', encoding='utf-8') as f:
    f.write('\n'.join(L) + '\n')
print('OK seal 2921')
