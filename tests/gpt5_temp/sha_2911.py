# -*- coding: utf-8 -*-
"""SHA registration for 2911."""
import hashlib
import json
import os

OUT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\sha_2911.txt')
BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


d = os.path.join(BASE, 'phase2911', 'alternation_structure')
rep = {}
for fn in ('execution.json', 'result.json',
           'alternation_structure.npz'):
    p = os.path.join(d, fn)
    rep[fn] = {'sha8': sha8(p), 'bytes': os.path.getsize(p)}
scr = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
       r'\phase2911_alternation_structure.py')
rep['SCRIPT phase2911'] = {'sha8': sha8(scr),
                           'bytes': os.path.getsize(scr)}
rep['execution_created'] = json.load(
    open(os.path.join(d, 'execution.json'),
         encoding='utf-8'))['created']
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(rep, f, indent=1)
print('OK', OUT)
