# -*- coding: utf-8 -*-
# Phase 2964 patch 2: fix T3 head_stats slice (2D).
import ast
import io
import shutil
import os

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase2964_carrier_anatomy.py')
t = io.open(P, encoding='utf-8').read()

old = "                hs = head_stats(C_all[li_top][:, r, :])\n"
new = "                hs = head_stats(C_all[li_top][r, :])\n"
assert old in t, 'T3 slice not found'
t = t.replace(old, new, 1)
io.open(P, 'w', encoding='utf-8').write(t)

t2 = io.open(P, encoding='utf-8').read()
try:
    ast.parse(t2)
    r = 'syntax OK'
except SyntaxError as e:
    r = 'SYNTAX ERROR line %s: %s' % (e.lineno, e.msg)
d = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase2964')
if os.path.exists(d):
    shutil.rmtree(d)
out = io.open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
              r'\p2964_patch2.txt', 'w', encoding='utf-8')
out.write('%s\ncleaned: %s\n'
          % (r, not os.path.exists(d)))
out.close()
print('patched')
