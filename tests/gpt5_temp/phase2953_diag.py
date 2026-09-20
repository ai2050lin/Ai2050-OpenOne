# -*- coding: utf-8 -*-
import os
import time

D = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase2953'
     r'\a11_s_response')
P = os.path.join(D, 'a11_s_response.npz')
P2 = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
      r'\rdc_query_construction_20260913\phase2953'
      r'\a11_s_response.npz')
OUTTXT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2953_diag.txt')
lines = []
lines.append('exists(join)=%r' % os.path.exists(P))
lines.append('exists(raw)=%r' % os.path.exists(P2))
lines.append('listdir=%r' % os.listdir(D))
try:
    with open(P, 'rb') as f:
        head = f.read(16)
    lines.append('open(join) ok head=%r' % head)
except Exception as e:
    lines.append('open(join) ERR %r' % e)
try:
    with open(P2, 'rb') as f:
        head = f.read(16)
    lines.append('open(raw) ok head=%r' % head)
except Exception as e:
    lines.append('open(raw) ERR %r' % e)
time.sleep(1.0)
lines.append('after sleep exists=%r' % os.path.exists(P))
with open(OUTTXT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('ok')
