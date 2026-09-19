# -*- coding: utf-8 -*-
"""SHA registration for 2909 + negatives format probe."""
import hashlib
import json
import os

OUT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\sha_2909.txt')
BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
LED = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
       r'\atlas_ledger.json')


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


d9 = os.path.join(BASE, 'phase2909', 'p_axis_stability')
rep = {}
for fn in ('execution.json', 'result.json',
           'p_axis_stability.npz'):
    p = os.path.join(d9, fn)
    rep[fn] = {'sha8': sha8(p), 'bytes': os.path.getsize(p)}
scr = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
       r'\phase2909_p_axis_stability.py')
rep['SCRIPT phase2909'] = {'sha8': sha8(scr),
                           'bytes': os.path.getsize(scr)}
rep['execution_created'] = json.load(
    open(os.path.join(d9, 'execution.json'),
         encoding='utf-8'))['created']

led = json.load(open(LED, encoding='utf-8'))
neg = led.get('negatives', [])
rep['negatives_n'] = len(neg)
rep['negatives_last2'] = neg[-2:]

with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(rep, f, indent=1, ensure_ascii=False)
print('OK', OUT)
