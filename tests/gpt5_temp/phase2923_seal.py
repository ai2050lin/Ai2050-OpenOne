# -*- coding: utf-8 -*-
"""phase2923 seal + zero-forward polarity sign diagnostic.
Output: tests/gpt5_temp/phase2923_seal_report.txt
"""
import hashlib
import json
import os

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase2923', 'polarity_sign_anatomy')
REP = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\phase2923_seal_report.txt')
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
    r'\phase2923_polarity_sign_anatomy.py'))
w('execution.json: %s' % sha8(
    os.path.join(OUT, 'execution.json')))
w('result.json: %s' % sha8(os.path.join(OUT, 'result.json')))
w('polarity_sign_anatomy.npz: %s' % sha8(
    os.path.join(OUT, 'polarity_sign_anatomy.npz')))

with open(os.path.join(OUT, 'result.json'),
          encoding='utf-8') as f:
    res = json.load(f)
w('verdict: %s' % res['final_verdict'])
w('anchors: a1 %s a2 %s a3 %s'
  % (res['anchors']['a1_ok'], res['anchors']['a2_ok'],
     res['anchors']['a3_ok']))
w('== diagnostic (descriptive) ==')
p1 = res['P1']
w('P1 main: T5 %d/13 p %.4f | T10 %d/13 p %.4f'
  % (p1['t5_obs'], p1['p5_perm'], p1['t10_obs'],
     p1['p10_perm']))
w('P2: %s' % json.dumps(res['P2']['fisher_table']))
w('P2 fisher_p %.4f r_pb %.3f median_peak %.1f'
  % (res['P2']['fisher_p'], res['P2']['point_biserial'],
     res['P2']['median_peak']))
w('P3 contrast/magnitude: 13/0')
w('P5: k_match %d/4 binom p %.4f'
  % (res['P5']['k_match'], res['P5']['exact_binomial_p']))
z = np.load(os.path.join(OUT, 'polarity_sign_anatomy.npz'),
            allow_pickle=True)
t5 = z['t5_perm']
w('t5_perm: p50 %.1f p95 %.1f max %d (obs %d)'
  % (float(np.percentile(t5, 50)), float(np.percentile(t5, 95)),
     int(t5.max()), p1['t5_obs']))

with open(REP, 'w', encoding='utf-8') as f:
    f.write('\n'.join(L) + '\n')
print('OK seal 2923')
