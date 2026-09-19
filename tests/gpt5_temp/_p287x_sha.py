# -*- coding: utf-8 -*-
import hashlib
import io
import os

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
TARGETS = ['phase2875\\attr_census_v2', 'phase2876\\growth_axis2_v2',
           'phase2877\\attr_mlp_spectrum']
out = io.open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\_p287x_sha_report.txt',
              'w', encoding='utf-8')
for t in TARGETS:
    root = os.path.join(BASE, t)
    out.write('== %s\n' % t)
    for f in sorted(os.listdir(root)):
        p = os.path.join(root, f)
        h = hashlib.sha256()
        with open(p, 'rb') as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b''):
                h.update(chunk)
        out.write('   %s  %s  %d bytes\n'
                  % (h.hexdigest()[:8], f, os.path.getsize(p)))
out.close()
print('OK')
