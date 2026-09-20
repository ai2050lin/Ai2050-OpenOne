# -*- coding: utf-8 -*-
"""Phase 2942 seal: artifact hashes + execution created."""
import hashlib
import json
import os

OUT = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
       r'\rdc_query_construction_20260913\phase2942'
       r'\u8_joint_injection')
R = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
     r'\phase2942_seal_report.txt')


def sha8(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()[:8]


lines = []
for fn in ['execution.json', 'result.json',
           'u8_joint_injection.npz']:
    p = os.path.join(OUT, fn)
    lines.append('%s: %s (%d bytes)'
                 % (fn, sha8(p), os.path.getsize(p)))
script = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
          r'\phase2942_u8_joint_injection.py')
lines.append('script: %s' % sha8(script))
ex = json.load(open(os.path.join(OUT, 'execution.json'),
                    encoding='utf-8'))
lines.append('created: %s' % ex['created'])
rj = json.load(open(os.path.join(OUT, 'result.json'),
                    encoding='utf-8'))
lines.append('final_verdict: %s' % rj['final_verdict'])
lines.append('runtime_s: %s' % rj['runtime_s'])
txt = '\n'.join(lines) + '\n'
open(R, 'w', encoding='utf-8').write(txt)
print(txt)
