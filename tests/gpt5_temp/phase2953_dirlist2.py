# -*- coding: utf-8 -*-
import hashlib
import os

OUT = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
       r'\rdc_query_construction_20260913\phase2953'
       r'\a11_s_response')
OUTTXT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2953_dirlist2.txt')


def sha8(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()[:8]


with open(OUTTXT, 'w', encoding='utf-8') as f:
    for fn in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, fn)
        try:
            f.write('%s %d %s\n' % (fn, os.path.getsize(p),
                                    sha8(p)))
        except Exception as e:
            f.write('%s ERR %r\n' % (fn, e))
print('ok')
