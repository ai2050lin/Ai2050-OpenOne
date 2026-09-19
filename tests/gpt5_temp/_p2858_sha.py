import hashlib, os
base = r'D:/AI2050/Ai2050-OpenOne/tests/glm5/result/rdc_query_construction_20260913/phase2858'
paths = [
    r'D:/AI2050/Ai2050-OpenOne/tests/glm5/phase2858_g_criteria.py',
    os.path.join(base, 'g_criteria', 'execution.json'),
    os.path.join(base, 'g_criteria', 'result.json'),
    os.path.join(base, 'g_criteria', 'g_criteria.npz'),
]
out = ['--- dir listing %s ---' % base]
for root, dirs, files in os.walk(base):
    for f in files:
        out.append('  %s\\%s  %d bytes' % (os.path.relpath(root, base), f,
                                           os.path.getsize(os.path.join(root, f))))
out.append('--- sha256 ---')
for p in paths:
    h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
    out.append('%s  %s  %d bytes' % (h[:16], os.path.basename(p), os.path.getsize(p)))
rep = r'D:/AI2050/Ai2050-OpenOne/tests/glm5/result/_p2858_sha.txt'
open(rep, 'w', encoding='utf-8').write('\n'.join(out) + '\n')
print('sha done')
