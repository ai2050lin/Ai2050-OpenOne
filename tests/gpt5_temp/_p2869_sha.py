import hashlib
import io
import os

BASE = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913\phase2869'
TMP = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\_p2869_sha_report.txt'

lines = []
for root, dirs, files in os.walk(BASE):
    for f in sorted(files):
        p = os.path.join(root, f)
        h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
        rel = os.path.relpath(p, BASE)
        lines.append('%s  %s  %d bytes' % (h[:8], rel, os.path.getsize(p)))

out = io.open(TMP, 'w', encoding='utf-8')
out.write('\n'.join(lines) + '\n')
out.close()
print('OK %d files' % len(lines))
