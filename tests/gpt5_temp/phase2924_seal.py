# -*- coding: utf-8 -*-
"""phase2924 seal + zero-forward gradient diagnostic.
Output: tests/gpt5_temp/phase2924_seal_report.txt
"""
import hashlib
import json
import os

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase2924', 'depth_polarity_gradient')
REP = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\phase2924_seal_report.txt')
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
    r'\phase2924_depth_polarity_gradient.py'))
w('execution.json: %s' % sha8(
    os.path.join(OUT, 'execution.json')))
w('result.json: %s' % sha8(os.path.join(OUT, 'result.json')))
w('depth_polarity_gradient.npz: %s' % sha8(
    os.path.join(OUT, 'depth_polarity_gradient.npz')))

with open(os.path.join(OUT, 'result.json'),
          encoding='utf-8') as f:
    res = json.load(f)
w('verdict: %s (n_axes_pass %s)'
  % (res['final_verdict'], res['n_axes_pass']))
w('anchors: a1 %s a2 %s (max_diff %.1e) a3 %s'
  % (res['anchors']['a1_ok'], res['anchors']['a2_ok'],
     res['anchors']['a2_max_diff'], res['anchors']['a3_ok']))
w('== diagnostic (descriptive) ==')
for ax in ('speed', 'size', 'moist'):
    e = res['P1'][ax]
    w('%s: rho %.4f p_axis %.4f | sig layers %s | c<0 %d c>0 %d '
      '| shallow12 %.2e deep12 %.2e'
      % (ax, e['rho'], e['p_axis'], e['sig_layers'],
         e['c_neg_layers'], e['c_pos_layers'],
         e['c_first12_mean'], e['c_last12_mean']))
w('P2 lang: %s' % json.dumps(res['P2']))
w('P3: %s' % json.dumps(res['P3']))
w('P4: %s' % json.dumps(res['P4']))
z = np.load(os.path.join(OUT, 'depth_polarity_gradient.npz'),
            allow_pickle=True)
for ax_i, ax in enumerate(['speed', 'size', 'moist']):
    c = z['c_%s' % ax]
    w('c_%s quartiles: %s' % (ax, [round(float(x), 6) for x in
                                   np.percentile(c, [0, 25, 50,
                                                     75, 100])]))

with open(REP, 'w', encoding='utf-8') as f:
    f.write('\n'.join(L) + '\n')
print('OK seal 2924')
