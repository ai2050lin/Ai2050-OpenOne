import hashlib
import os

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913\phase2865\l13h30_trace')
OUTP = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\_p2865_sha_report.txt'
rows = []
for root, dirs, files in os.walk(os.path.dirname(BASE)):
    for f in sorted(files):
        p = os.path.join(root, f)
        h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
        rows.append('%s  %d  %s' % (h[:8], os.path.getsize(p), p))
open(OUTP, 'w', encoding='utf-8').write('\n'.join(rows) + '\n')
print('OK %d files' % len(rows))
