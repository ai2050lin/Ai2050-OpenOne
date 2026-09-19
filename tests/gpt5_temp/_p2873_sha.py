# -*- coding: utf-8 -*-
import hashlib
import io
import os

ROOT = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913\phase2873\growth_axis2')
out = io.open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\_p2873_sha_report.txt',
              'w', encoding='utf-8')
total = 0
for root, dirs, files in os.walk(ROOT):
    for f in sorted(files):
        p = os.path.join(root, f)
        h = hashlib.sha256()
        with open(p, 'rb') as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b''):
                h.update(chunk)
        out.write('%s  %s  %d bytes\n'
                  % (h.hexdigest()[:8], os.path.relpath(p, ROOT),
                     os.path.getsize(p)))
        total += 1
out.write('files=%d\n' % total)
out.close()
print('OK %d' % total)
