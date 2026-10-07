# -*- coding: utf-8 -*-
"""R53/R51 deep verify:
1. Read RESID_* from P98 npz -> percentiles; compare vs cos residual (1-cos).
   Math: if RESID==1-cos, equal-norm relative L2 = sqrt(2*RESID).
2. Search literal 1.6941 in top-level py/md candidates (R51 Jaccard gate).
"""
import glob
import io
import os

import numpy as np

P98 = (r'tests\glm5\result\rdc_query_construction_20260913'
       r'\phase3100\omega_p98_upstream_predict'
       r'\omega_p98_upstream_predict.npz')
OUT = r'tests\gpt5_temp\p3101_review_verify3.txt'
lines = []

z = np.load(P98, allow_pickle=False)
for fk in ('A', 'B', 'C'):
    for m in ('4B', '14B'):
        r = z['RESID_%s_%s' % (m, fk)]
        c = z['PRED_COS_%s_%s' % (m, fk)]
        pc = z['XPCOS_%s_%s' % (m, fk)]
        resid_p50 = float(np.percentile(r, 50))
        l2_eq = float(np.sqrt(2.0 * np.median(1.0 - c)))
        lines.append(
            '%s %s: RESID med=%.4f q25=%.4f q75=%.4f | '
            '1-cos med=%.4f -> eqnorm relL2=%.4f | '
            'XPCOS med=%.4f'
            % (m, fk, resid_p50,
               float(np.percentile(r, 25)),
               float(np.percentile(r, 75)),
               float(np.median(1.0 - c)), l2_eq,
               float(np.median(pc))))

# search 1.6941 in candidate text files (top-level only, fast)
cands = []
for pat in (r'tests\glm5\*.py', r'tests\gpt5\*.py',
            r'tests\gpt5_temp\*.py', r'gpt5_temp\*.py',
            r'research\gpt5\docs\*.md', r'research\glm5\docs\*.md'):
    cands.extend(glob.glob(pat))
hits = []
for p in cands:
    try:
        with io.open(p, encoding='utf-8', errors='ignore') as f:
            t = f.read()
    except Exception:
        continue
    if '1.6941' in t:
        i = t.find('1.6941')
        hits.append(os.path.basename(p))
        lines.append('HIT 1.6941 in %s (ctx): ...%s...'
                     % (p, t[max(0, i - 200):i + 200].replace('\n', ' ')))
if not hits:
    lines.append('1.6941: NOT FOUND in %d candidate files' % len(cands))

with io.open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('OK cands=%d' % len(cands))
