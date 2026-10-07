# -*- coding: utf-8 -*-
"""p3145 closeout patch v2: remove dead
p_log first-assignment block (3 lines)."""
import io
import py_compile

FP = (r'D:\AI2050\Ai2050-OpenOne\tests'
      r'\gpt5_temp\p3145_closeout.py')
t = io.open(FP, encoding='utf-8').read()
BS = chr(92)
lines = t.split(chr(10))
# find the dead block: line starting
# "p_log = (ROOT + r'" + BS*2
idx = None
for i, l in enumerate(lines):
    if l.startswith("p_log = (ROOT + r'"
                   + BS + BS):
        idx = i
        break
assert idx is not None, 'dead block not found'
# dead block = 3 lines: idx, idx+1, idx+2
assert lines[idx + 1].strip().startswith(
    "r'" + BS + BS + '2026'), lines[idx + 1]
assert lines[idx + 2].strip().endswith(
    "'))"), lines[idx + 2]
assert lines[idx + 3].startswith(
    "p_log = (ROOT + '" + BS), lines[idx + 3]
del lines[idx:idx + 3]
t2 = chr(10).join(lines)
io.open(FP, 'w', encoding='utf-8').write(t2)
chk = io.open(FP, encoding='utf-8').read()
assert chk.count('p_log = (ROOT') == 1
py_compile.compile(FP, doraise=True)
print('CLEAN OK (removed 3 lines at %d)'
      % idx)
