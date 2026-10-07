# -*- coding: utf-8 -*-
"""Probe 2: print the concatenated entry
format string around index 878 and all
percent positions."""
import ast
import io

SRC = (r'D:\AI2050\Ai2050-OpenOne'
       r'\tests\gpt5_temp'
       r'\phase3095_closeout.py')
OUT = (r'D:\AI2050\Ai2050-OpenOne'
       r'\tests\gpt5_temp'
       r'\probe_pct2_3095.txt')
src = io.open(SRC, encoding='utf-8').read()
tree = ast.parse(src)
fmt = None
for node in ast.walk(tree):
    if (isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Mod)
            and isinstance(node.left,
                           ast.Constant)
            and isinstance(node.left.value,
                           str)
            and 'Phase 3095' in
            node.left.value):
        fmt = node.left.value
rep = ''
if fmt is None:
    rep += 'fmt not found\n'
else:
    rep += 'len=%d\n' % len(fmt)
    rep += 'seg[840:920]=%r\n' % fmt[840:920]
    rep += 'pct positions: '
    rep += ' '.join(str(i) for i, c in
                    enumerate(fmt)
                    if c == '%') + '\n'
with io.open(OUT, 'w',
             encoding='utf-8') as f:
    f.write(rep)
print('PROBE2_DONE')
