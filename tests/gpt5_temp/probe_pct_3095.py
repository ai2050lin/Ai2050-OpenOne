# -*- coding: utf-8 -*-
"""Probe: locate the bad % in the
concatenated entry format string of
phase3095_closeout.py via ast."""
import ast
import io

SRC = (r'D:\AI2050\Ai2050-OpenOne'
       r'\tests\gpt5_temp'
       r'\phase3095_closeout.py')
OUT = (r'D:\AI2050\Ai2050-OpenOne'
       r'\tests\gpt5_temp'
       r'\probe_pct_3095.txt')
src = io.open(SRC, encoding='utf-8').read()
tree = ast.parse(src)
found = []
for node in ast.walk(tree):
    if (isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Mod)
            and isinstance(node.left,
                           ast.Constant)
            and isinstance(node.left.value,
                           str)):
        fmt = node.left.value
        for i, ch in enumerate(fmt):
            if ch == '%':
                nxt = (fmt[i + 1]
                       if i + 1 < len(fmt)
                       else '<end>')
                ok = nxt in 'sdfregxXoce%+- 0123456789.#('
                if not ok:
                    found.append(
                        (i, repr(fmt[max(0, i-40):i+12])))
rep = 'bad_count=%d\n' % len(found)
for pos, ctx in found:
    rep += 'idx=%d ctx=%s\n' % (pos, ctx)
with io.open(OUT, 'w',
             encoding='utf-8') as f:
    f.write(rep)
print('PROBE_DONE')
