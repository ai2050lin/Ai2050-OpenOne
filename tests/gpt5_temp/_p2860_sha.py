"""Phase 2860 artifact probe: os.walk + SHA256 -> report file."""
import hashlib
import os

BASE = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result' \
       r'\rdc_query_construction_20260913\phase2860'
OUT = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\_p2860_sha_report.txt'


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


rows = []
for root, dirs, files in os.walk(BASE):
    for fn in sorted(files):
        p = os.path.join(root, fn)
        rows.append('%s  %d B  %s' % (
            sha256(p), os.path.getsize(p), os.path.relpath(p, BASE)))

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('PHASE2860 ARTIFACTS (%d files)\n' % len(rows))
    f.write('\n'.join(rows) + '\n')
print('WROTE', OUT, len(rows))
