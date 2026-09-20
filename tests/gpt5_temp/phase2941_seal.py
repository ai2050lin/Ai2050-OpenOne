# -*- coding: utf-8 -*-
"""Phase 2941 seal: artifact hashes + execution created stamp."""
import hashlib
import json
import os

OUT = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
       r'\rdc_query_construction_20260913\phase2941'
       r'\v3_causal_injection')
R = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
     r'\phase2941_seal_report.txt')


def sha8(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()[:8]


files = ['execution.json', 'result.json',
         'v3_causal_injection.npz']
script = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
          r'\phase2941_v3_causal_injection.py')
lines = []
for fn in files:
    p = os.path.join(OUT, fn)
    lines.append('%s: %s (%d bytes)'
                 % (fn, sha8(p), os.path.getsize(p)))
lines.append('script: %s' % sha8(script))
with open(os.path.join(OUT, 'execution.json'),
          encoding='utf-8') as f:
    ex = json.load(f)
lines.append('created: %s' % ex['created'])
lines.append('script_sha_in_execution: %s'
              % ex['script_sha256_8'])
with open(os.path.join(OUT, 'result.json'),
          encoding='utf-8') as f:
    rj = json.load(f)
lines.append('final_verdict: %s' % rj['final_verdict'])
lines.append('runtime_s: %s' % rj['runtime_s'])
txt = '\n'.join(lines) + '\n'
with open(R, 'w', encoding='utf-8') as f:
    f.write(txt)
print(txt)
