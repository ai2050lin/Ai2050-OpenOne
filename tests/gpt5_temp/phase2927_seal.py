# -*- coding: utf-8 -*-
"""Phase 2927 seal: SHA registry + survivor-core forensics."""
import hashlib
import json
import os

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase2927', 'probe_relativity')
SRC2917 = os.path.join(BASE, 'phase2917', 'event_atlas',
                       'event_atlas.npz')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2927_seal_report.txt')


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


lines = []
z = np.load(os.path.join(OUT, 'probe_relativity.npz'),
            allow_pickle=True)
z17 = np.load(SRC2917, allow_pickle=True)
sm_w = z['sign_M_word'].astype(np.float64)
sm_86 = z['sign_M86'].astype(np.float64)
pm_w = z['p_maxT_word'].astype(np.float64)
pm_86 = z['p_maxT86'].astype(np.float64)
sm_17 = z17['sign_M'].astype(np.float64)
E_w = set((h, li) for h in range(32) for li in range(36)
          if pm_w[h, li] <= 0.05)
E_17 = set((h, li) for h in range(32) for li in range(36)
           if pm_86[h, li] <= 0.05)
surv = sorted(E_w & E_17)
lost = sorted(E_17 - E_w)
new = sorted(E_w - E_17)
lines.append('survivor core (%d): %s' % (len(surv), surv))
lines.append('lost under word probe (%d): %s'
             % (len(lost), lost))
lines.append('new under word probe (%d): %s' % (len(new), new))
lines.append('(7,19) margin: sentence %.5f -> word %.5f '
             '(rank in layer: %d -> %d)'
             % (sm_17[7, 19], sm_w[7, 19],
                int(np.sum(sm_17[:, 19] > sm_17[7, 19])) + 1,
                int(np.sum(sm_w[:, 19] > sm_w[7, 19])) + 1))
deep_lost = [e for e in lost if e[1] >= 20]
lines.append('lost deep events (l>=20): %d/%d of lost'
             % (len(deep_lost), len(lost)))
early_new = [e for e in new if e[1] <= 10]
lines.append('new early events (l<=10): %d/%d of new'
             % (len(early_new), len(new)))
# margin rank correlation among the 24 E17 events under both
# probes (do the E17 events keep their relative order?)
pairs = sorted(E_17)
r86 = [sm_86[h, li] for (h, li) in pairs]
rw = [sm_w[h, li] for (h, li) in pairs]
from math import isnan


def spearman(x, y):
    def rk(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for rank, i in enumerate(order):
            r[i] = float(rank)
        return r
    rx, ry = rk(x), rk(y)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx)
           * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den > 0 else float('nan')


lines.append('E17-internal margin rank corr (86 vs word): %.4f'
             % spearman(r86, rw))

for name in ('execution.json', 'result.json',
             'probe_relativity.npz'):
    p = os.path.join(OUT, name)
    lines.append('%s sha256_8 %s' % (name, sha8(p)))
lines.append('script sha256_8 %s'
             % sha8(r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
                    r'\phase2927_probe_relativity.py'))
lines.append('created=%s'
             % json.load(open(os.path.join(OUT, 'execution.json'),
                              encoding='utf-8'))['created'])

with open(REPORT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('OK seal report written')
