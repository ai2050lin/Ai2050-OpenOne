# -*- coding: utf-8 -*-
"""phase2925 post-hoc diagnostic: |phi| percentile recheck.
The preregistered phi test used SIGNED phi with a one-sided
percentile - mathematically unable to reach the verdict
criterion for LOW-driven cells (their phi is the most EXTREME
NEGATIVE). This probe recomputes the same percentiles on |phi|
(descriptive, post-hoc, NOT the frozen verdict).
Output: appended facts to phase2925_seal_report.txt
"""
import hashlib
import json
import os

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC21 = os.path.join(BASE, 'phase2921', 'attr_vocab_expansion',
                     'attr_vocab_expansion.npz')
OUT = os.path.join(BASE, 'phase2925', 'event_selection_anatomy')
REP = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\phase2925_seal_report.txt')
FDR_Q = 0.05
NH, NL = 32, 36

EVENTS = {
    'speed': [(18, 16), (14, 12)],
    'size': [(21, 7), (24, 19), (12, 18), (26, 21), (25, 15),
             (11, 23), (11, 27), (22, 2), (18, 7)],
    'moist': [(8, 15), (10, 9)],
}


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


def phi_of(B, h, l, hi, lo):
    r = B[h, :, l]
    a = int(np.sum(hi & (r >= 0)))
    b = int(np.sum(hi & (r < 0)))
    c = int(np.sum(lo & (r >= 0)))
    d = int(np.sum(lo & (r < 0)))
    den = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    return (a * d - b * c) / max(den, 1e-30)


z = np.load(SRC21, allow_pickle=True)
B = {ax: z['B_%s' % ax].astype(np.float64)
     for ax in ('speed', 'size', 'moist')}
pole = {ax: np.asarray(z['labels_%s' % ax]).astype(int)
        for ax in ('speed', 'size', 'moist')}

L = ['== post-hoc |phi| diagnostic (NOT the frozen verdict) ==']
pcts = []
for ax in ('speed', 'size', 'moist'):
    lab = pole[ax]
    hi = lab == 1
    lo = lab == 0
    sig_set = set((h, li) for h in range(NH)
                  for li in range(NL)
                  if False)  # placeholder; sig via p_maxT below
z_p = np.load(os.path.join(OUT, 'result.json'),
              encoding='utf-8') if False else None
# recompute sig sets from 2921 p_maxT
p_maxT = z['p_maxT'].astype(np.float64)
AX_AI = {'speed': 1, 'size': 2, 'moist': 3}
for ax in ('speed', 'size', 'moist'):
    lab = pole[ax]
    hi = lab == 1
    lo = lab == 0
    sig_set = set((h, li) for h in range(NH)
                  for li in range(NL)
                  if p_maxT[AX_AI[ax]][h, li] <= FDR_Q)
    for (h, li) in EVENTS[ax]:
        v = abs(phi_of(B[ax], h, li, hi, lo))
        pool = [v]
        for hh in range(NH):
            if (hh, li) in sig_set:
                continue
            pool.append(abs(phi_of(B[ax], hh, li, hi, lo)))
        pool = np.array(pool)
        pcts.append(float(np.mean(pool <= v)))
L.append('|phi| pcts (13 events, axis order speed/size/moist): '
         '%s' % [round(x, 4) for x in pcts])
L.append('median %.4f | n>0.5 %d/13 | n==1.0 %d/13'
         % (float(np.median(pcts)),
            int(np.sum(np.array(pcts) > 0.5)),
            int(np.sum(np.array(pcts) >= 0.9999))))
from math import comb
n_gt = int(np.sum(np.array(pcts) > 0.5))
L.append('sign test p (one-sided) %.6e'
         % (sum(comb(13, x) for x in range(n_gt, 14))
            / 2.0 ** 13))
with open(os.path.join(OUT, 'execution.json'),
          encoding='utf-8') as f:
    ex = json.load(f)
L.append('')
L.append('== seal ==')
L.append('created: %s' % ex['created'])
L.append('script_sha256_8: %s' % sha8(
    r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
    r'\phase2925_event_selection_anatomy.py'))
L.append('execution.json: %s' % sha8(
    os.path.join(OUT, 'execution.json')))
L.append('result.json: %s' % sha8(
    os.path.join(OUT, 'result.json')))
L.append('event_selection_anatomy.npz: %s' % sha8(
    os.path.join(OUT, 'event_selection_anatomy.npz')))
with open(os.path.join(OUT, 'result.json'),
          encoding='utf-8') as f:
    res = json.load(f)
L.append('verdict: %s' % res['final_verdict'])
L.append('anchors: a1 %s a2 %s (max %.1e) a3 %s'
         % (res['anchors']['a1_ok'], res['anchors']['a2_ok'],
            res['anchors']['a2_max_diff'],
            res['anchors']['a3_ok']))
with open(REP, 'w', encoding='utf-8') as f:
    f.write('\n'.join(L) + '\n')
print('OK seal 2925')
