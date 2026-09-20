# -*- coding: utf-8 -*-
# Phase 2964 patch 3: T4 set intersection type fix.
import ast
import io
import shutil
import os

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase2964_carrier_anatomy.py')
t = io.open(P, encoding='utf-8').read()

old = "                  'intersect_2947_top5': sorted(\n                      top5_h & top5_47),\n"
new = ("                  'intersect_2947_top5': sorted(\n"
       "                      set(top5_h) & top5_47),\n")
assert old in t, 'T4 set not found'
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
              r'\p2964_patch3.txt', 'w', encoding='utf-8')
out.write('%s\ncleaned: %s\n'
          % (r, not os.path.exists(d)))
out.close()
print('patched')
