# -*- coding: utf-8 -*-
"""SHA registration for Phase 2907/2908 outputs."""
import hashlib
import json
import os

OUT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\sha_2908.txt')
BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


d8 = os.path.join(BASE, 'phase2908',
                  'qwen_attn_boundary_precision')
rep = {}
for fn in ('execution.json', 'result.json',
           'qwen_attn_boundary_precision.npz'):
    p = os.path.join(d8, fn)
    rep[fn] = {'sha8': sha8(p), 'bytes': os.path.getsize(p)}
scr = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
       r'\phase2908_qwen_attn_boundary_precision.py')
rep['SCRIPT phase2908'] = {'sha8': sha8(scr),
                           'bytes': os.path.getsize(scr)}
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(rep, f, indent=1)
print('OK', OUT)
