"""Phase 2862 prep probe: inspect npz keys/shapes + 2846 thresholds."""
import glob
import json
import os

import numpy as np

R = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result' \
    r'\rdc_query_construction_20260913'
OUT = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\_p2862_prep_report.txt'

lines = []


def w(s):
    lines.append(str(s))


for tag, pat in (
        ('2846', os.path.join(R, 'phase2846', '*', '*.npz')),
        ('2859', os.path.join(R, 'phase2859', '*', '*.npz'))):
    for p in sorted(glob.glob(pat)):
        z = np.load(p)
        w('[%s] %s' % (tag, os.path.relpath(p, R)))
        for k in sorted(z.files):
            w('    %-22s %s %s' % (k, z[k].shape, z[k].dtype))

for tag, pat in (
        ('2846', os.path.join(R, 'phase2846', '*', 'result.json')),
        ('2859', os.path.join(R, 'phase2859', '*', 'result.json'))):
    for p in sorted(glob.glob(pat)):
        d = json.load(open(p, encoding='utf-8'))
        w('[%s] %s' % (tag, os.path.relpath(p, R)))
        v = d.get('verdict', d)
        for k in sorted(v.keys()):
            s = json.dumps(v[k])
            w('    %-26s %s' % (k, s[:160]))

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('WROTE', OUT)
