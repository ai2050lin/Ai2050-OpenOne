# -*- coding: utf-8 -*-
"""SHA registration for 2910."""
import hashlib
import json
import os

OUT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\sha_2910.txt')
BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


d = os.path.join(BASE, 'phase2910', 'cross_layer_coherence')
rep = {}
for fn in ('execution.json', 'result.json',
           'cross_layer_coherence.npz'):
    p = os.path.join(d, fn)
    rep[fn] = {'sha8': sha8(p), 'bytes': os.path.getsize(p)}
scr = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
       r'\phase2910_cross_layer_coherence.py')
rep['SCRIPT phase2910'] = {'sha8': sha8(scr),
                           'bytes': os.path.getsize(scr)}
rep['execution_created'] = json.load(
    open(os.path.join(d, 'execution.json'),
         encoding='utf-8'))['created']
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(rep, f, indent=1)
print('OK', OUT)
