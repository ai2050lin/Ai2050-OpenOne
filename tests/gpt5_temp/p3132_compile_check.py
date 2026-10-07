# -*- coding: utf-8 -*-
import io
import os
import py_compile
import tokenize

SRC = (r'D:\AI2050\Ai2050-OpenOne\tests'
       r'\glm5\phase3132_omega_p130_'
       r'forkcausal_single256.py')
OUTF = (r'D:\AI2050\Ai2050-OpenOne'
        r'\tests\gpt5_temp'
        r'\p3132_compile_out.txt')
lines = []
ok = True
try:
    py_compile.compile(SRC, doraise=True)
    lines.append('py_compile: OK')
except Exception as e:
    ok = False
    lines.append('py_compile FAIL: %r' % e)
try:
    depth = 0
    mind = 0
    with open(SRC, 'rb') as f:
        toks = list(tokenize.tokenize(
            f.readline))
    for t in toks:
        if t.type == tokenize.OP:
            if t.string in '([{':
                depth += 1
            elif t.string in ')]}':
                depth -= 1
                if depth < 0:
                    mind = min(mind, depth)
    lines.append('final depth=%d min=%d'
                 % (depth, mind))
    if depth != 0 or mind < 0:
        ok = False
        lines.append('BRACKET IMBALANCE')
except Exception as e:
    ok = False
    lines.append('tokenize FAIL: %r' % e)
lines.append('RESULT: %s'
             % ('GREEN' if ok else 'RED'))
with io.open(OUTF, 'w',
             encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('done')
