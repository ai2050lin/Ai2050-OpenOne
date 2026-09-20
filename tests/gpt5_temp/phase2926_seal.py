# -*- coding: utf-8 -*-
"""Phase 2926 seal: SHA registry + P3 rank-2 forensics + P2 sanity."""
import hashlib
import json
import os

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase2926', 'event_selection_polar_fix')
SRC21 = os.path.join(BASE, 'phase2921', 'attr_vocab_expansion',
                     'attr_vocab_expansion.npz')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2926_seal_report.txt')


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


lines = []
z = np.load(SRC21, allow_pickle=True)
B = {ax: z['B_%s' % ax].astype(np.float64)
     for ax in ('speed', 'size', 'moist')}
sign_M = z['sign_M'].astype(np.float64)
pole = {ax: np.asarray(z['labels_%s' % ax]).astype(int)
        for ax in ('speed', 'size', 'moist')}


def phi_cell(r, hi, lo):
    a = int(np.sum(hi & (r >= 0)))
    b = int(np.sum(hi & (r < 0)))
    c = int(np.sum(lo & (r >= 0)))
    d = int(np.sum(lo & (r < 0)))
    den = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    return (a * d - b * c) / max(den, 1e-30)


# ---- P3 rank-2 forensics: (18,7) vs (21,7) ----
lab = pole['size']
hi = lab == 1
lo = lab == 0
g = np.zeros((32, 36))
for h in range(32):
    for l in range(36):
        g[h, l] = abs(float(phi_cell(B['size'][h, :, l], hi, lo)))
p18 = abs(float(phi_cell(B['size'][18, :, 7], hi, lo)))
p21 = abs(float(phi_cell(B['size'][21, :, 7], hi, lo)))
col7 = g[:, 7]
top3 = np.argsort(col7)[::-1][:3]
lines.append('P3 forensics L7: |phi|(18,7)=%.6f |phi|(21,7)=%.6f '
             '=> 21 wins: %s' % (p18, p21, bool(p21 > p18)))
lines.append('L7 top-3 |phi| heads: %s'
             % [(int(hh), round(float(col7[hh]), 6))
                for hh in top3])
lines.append('L7 margin: (21,7)=%.6f (18,7)=%.6f'
             % (float(sign_M[2][21, 7]), float(sign_M[2][18, 7])))

# ---- P2 null sanity ----
z26 = np.load(os.path.join(OUT, 'event_selection_polar_fix.npz'),
              allow_pickle=True)
null = z26['p2_null']
lines.append('P2 null: n=%d max|rho|=%.4f p95=%.4f '
             'n>=0.978: %d' % (len(null),
                               float(np.abs(null).max()),
                               float(np.percentile(np.abs(null),
                                                   95)),
                               int(np.sum(np.abs(null) >= 0.978))))

# ---- P1 phi_abs pcts vs 2925 seal (13/13) cross-check ----
res = json.load(open(os.path.join(OUT, 'result.json'),
                     encoding='utf-8'))
pcts = res['P1']['features']['phi_abs']['pcts']
lines.append('P1 phi_abs pcts: %s (all 1.0: %s)'
             % (pcts, all(p == 1.0 for p in pcts)))
lines.append('created=%s' % res['prereg'].get('mode', '')[:0]
             + json.load(open(os.path.join(OUT, 'execution.json'),
                              encoding='utf-8'))['created'])

# ---- SHA registry ----
for name in ('execution.json', 'result.json',
             'event_selection_polar_fix.npz'):
    p = os.path.join(OUT, name)
    lines.append('%s sha256_8 %s' % (name, sha8(p)))
lines.append('script sha256_8 %s'
             % sha8(r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
                    r'\phase2926_event_selection_polar_fix.py'))

with open(REPORT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('OK seal report written')
