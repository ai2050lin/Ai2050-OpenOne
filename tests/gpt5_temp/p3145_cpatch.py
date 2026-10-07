# -*- coding: utf-8 -*-
"""p3145 closeout patch: remove dead
p_log first-assignment block."""
import io
import py_compile

FP = (r'D:\AI2050\Ai2050-OpenOne\tests'
      r'\gpt5_temp\p3145_closeout.py')
t = io.open(FP, encoding='utf-8').read()
BS = chr(92)
# dead block uses r'\\.workbuddy\\memory' etc
old = ("p_log = (ROOT + r'" + BS + BS
       + ".workbuddy" + BS + BS
       + "memory'\n")
c1 = t.count(old)
old2 = ("         r'" + BS + BS
        + "2026-09-30.md'.replace(\n")
c2 = t.count(old2)
old3 = ("             '" + BS + BS * 2
        + "', '" + BS + "'))\n")
c3 = t.count(old3)
assert (c1, c2, c3) == (1, 1, 1), \
    (c1, c2, c3)
t = t.replace(old, '').replace(old2, '') \
     .replace(old3, '')
io.open(FP, 'w', encoding='utf-8').write(t)
chk = io.open(FP, encoding='utf-8').read()
assert chk.count('p_log = (ROOT') == 1
assert '.replace(' not in chk.split(
    'workspace daily log')[1].split(
    'add =')[0]
py_compile.compile(FP, doraise=True)
print('CLEAN OK')
