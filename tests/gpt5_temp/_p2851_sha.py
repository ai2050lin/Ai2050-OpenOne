import hashlib, os, json
paths = [
    r'D:/AI2050/Ai2050-OpenOne/tests/glm5/phase2851_emergence_anatomy.py',
    r'D:/AI2050/Ai2050-OpenOne/tests/glm5/result/rdc_query_construction_20260913/phase2851/emergence_anatomy/execution.json',
    r'D:/AI2050/Ai2050-OpenOne/tests/glm5/result/rdc_query_construction_20260913/phase2851/emergence_anatomy/result.json',
    r'D:/AI2050/Ai2050-OpenOne/tests/glm5/result/rdc_query_construction_20260913/phase2851/emergence_anatomy/emergence.npz',
]
out = []
for p in paths:
    h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
    out.append('%s  %s  %d bytes' % (h[:16], os.path.basename(p), os.path.getsize(p)))
rep = r'D:/AI2050/Ai2050-OpenOne/tests/glm5/result/_p2851_sha.txt'
open(rep, 'w', encoding='utf-8').write('\n'.join(out) + '\n')
print('sha done')
