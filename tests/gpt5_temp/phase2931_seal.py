# -*- coding: utf-8 -*-
"""Phase 2931 seal probe: forensics + SHA registration.

S1  null jaccard distribution detail + excess decomposition
S2  corrected-skeleton layer profile (both calibers) +
    per-layer intersection vs independence expectation
S3  worst layers (negative pairing) forensics: L32 cell-level
    structure vs lin_r nonlinearity profile
S4  survivor-core double membership + skeleton-exclusive
    survivor accounting
S5  SHA256-8 registration
Report: tests/gpt5_temp/phase2931_seal_report.txt
"""
import hashlib
import json
import os

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase2931', 'skeleton_overlap_null')
SRC2927 = os.path.join(BASE, 'phase2927', 'probe_relativity',
                       'probe_relativity.npz')
SRC2930 = os.path.join(BASE, 'phase2930',
                       'direction_flip_control',
                       'direction_flip_control.npz')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2931_seal_report.txt')
NH, NL = 32, 36
SURV = [(1, 6), (5, 6), (7, 19), (8, 2), (14, 9), (20, 8),
        (21, 6)]


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


lines = []
z = np.load(os.path.join(OUT, 'skeleton_overlap_null.npz'),
            allow_pickle=True)
j_null = z['null_jacc']
S29 = z['S29_corrected'].astype(bool)
Smir = z['Smir_corrected'].astype(bool)
gate = z['gate'].astype(bool)
z30 = np.load(SRC2930, allow_pickle=True)
lin_r = z30['lin_r_profile']

# S1 null distribution detail
qs = [round(float(np.percentile(j_null, q)), 4)
      for q in (1, 5, 25, 50, 75, 95, 99)]
lines.append('S1 null jacc percentiles 1/5/25/50/75/95/99: %s'
             % qs)
inter = int((S29 & Smir).sum())
n29 = int(S29.sum())
nmir = int(Smir.sum())
lines.append('S1 corrected sizes %d/%d inter %d obs_jacc '
             '%.4f | excess vs independence %.1f cells'
             % (n29, nmir, inter, inter / (n29 + nmir
                                           - inter),
                inter - n29 * nmir / (NH * NL)))

# S2 layer profile
lines.append('S2 layer profile (l, |S29|, |Smir|, inter, '
             'lin_r):')
for li in range(NL):
    a = int(S29[:, li].sum())
    b = int(Smir[:, li].sum())
    c = int((S29[:, li] & Smir[:, li]).sum())
    lines.append('    (%2d, %2d, %2d, %2d, %.2f)'
                 % (li, a, b, c, float(lin_r[li])))

# S3 worst layers vs nonlinearity
worst = sorted(range(2, NL),
               key=lambda l: -abs(lin_r[l]))[:6]
lines.append('S3 top lin_r layers: %s' % worst)

# S4 survivor accounting
lines.append('S4 survivor-core corrected-skeleton membership:')
for (h, li) in SURV:
    lines.append('    (%d,%2d) S29=%s Smir=%s gate=%s'
                 % (h, li, bool(S29[h, li]),
                    bool(Smir[h, li]), bool(gate[h, li])))
n_both = sum(1 for (h, li) in SURV
             if S29[h, li] and Smir[h, li])
lines.append('S4 survivors in BOTH corrected skeletons: '
             '%d/7' % n_both)

# S5 SHA
sha = {'execution.json': sha8(os.path.join(OUT,
                                           'execution.json')),
       'result.json': sha8(os.path.join(OUT, 'result.json')),
       'skeleton_overlap_null.npz':
           sha8(os.path.join(OUT,
                             'skeleton_overlap_null.npz')),
       'src2927': sha8(SRC2927),
       'src2930': sha8(SRC2930),
       'script_phase2931': sha8(
           r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
           r'\phase2931_skeleton_overlap_null.py')}
lines.append('S5 SHA256-8: %s' % json.dumps(sha, indent=1))

with open(REPORT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('seal OK')
