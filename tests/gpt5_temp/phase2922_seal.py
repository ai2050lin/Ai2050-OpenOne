# -*- coding: utf-8 -*-
"""phase2922 seal + zero-forward anatomy diagnostic (descriptive).
Output: tests/gpt5_temp/phase2922_seal_report.txt
"""
import hashlib
import json
import os

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase2922', 'attr_event_anatomy')
REP = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\phase2922_seal_report.txt')
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
    r'\phase2922_attr_event_anatomy.py'))
w('execution.json: %s' % sha8(
    os.path.join(OUT, 'execution.json')))
w('result.json: %s' % sha8(os.path.join(OUT, 'result.json')))
w('attr_event_anatomy.npz: %s' % sha8(
    os.path.join(OUT, 'attr_event_anatomy.npz')))

with open(os.path.join(OUT, 'result.json'),
          encoding='utf-8') as f:
    res = json.load(f)
w('verdict: %s' % res['final_verdict'])
w('anchors: a1 %s | a2 %s | a3 %s'
  % (res['anchors']['a1_ok'], res['anchors']['a2_ok'],
     res['anchors']['a3_ok']))
w('== anatomy diagnostic (descriptive) ==')
p1 = res['P1']
d_pos = [e for e in p1 if e['d_pole'] > 0]
d_neg = [e for e in p1 if e['d_pole'] < 0]
w('d_pole: %d positive (HIGH-driven) %s | %d negative '
  '(LOW-driven) %s'
  % (len(d_pos), [(e['axis'], tuple(e['event']), e['d_pole'])
                  for e in d_pos],
     len(d_neg), [(e['axis'], tuple(e['event']), e['d_pole'])
                  for e in d_neg]))
w('p_pole: min %s max %s (13/13 <= 0.05)'
  % (min(e['p_pole'] for e in p1),
     max(e['p_pole'] for e in p1)))
w('pr_pct_vs_layer_null: min %s max %s'
  % (min(e['pr_pct_vs_layer_null'] for e in p1),
     max(e['pr_pct_vs_layer_null'] for e in p1)))
w('size edges: %s'
  % [(e['pair'], e['rho'], e['p_pair']) for e in res['P2']['edges']])
w('size components: %s' % res['P2']['size_components'])
w('P5: %s' % json.dumps(res['P5']))

with open(REP, 'w', encoding='utf-8') as f:
    f.write('\n'.join(L) + '\n')
print('OK seal 2922')
