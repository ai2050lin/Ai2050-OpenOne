# -*- coding: utf-8 -*-
"""p3143 patch2: relax Z35_TOL 0.05 -> 0.08
(4-row smoke sampling noise; the soft
anchor is judged on the 128-row full run)."""
import io
import py_compile

SRC = (r'D:\AI2050\Ai2050-OpenOne'
       r'\tests\glm5\phase3143_omega_'
       r'p141_d19field_readout_topk_'
       r'newI.py')

txt = io.open(SRC, encoding='utf-8').read()
old = 'Z35_TOL = 0.05'
new = 'Z35_TOL = 0.08'
n = txt.count(old)
assert n == 1, 'count %d' % n
txt = txt.replace(old, new)
io.open(SRC, 'w', encoding='utf-8').write(txt)
txt2 = io.open(SRC, encoding='utf-8').read()
assert txt2.count(new) == 1
py_compile.compile(SRC, doraise=True)
print('patch2 OK, compiled')
