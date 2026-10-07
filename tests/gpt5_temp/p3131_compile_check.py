# -*- coding: utf-8 -*-
"""Compile check for phase3131 main script:
py_compile + tokenize bracket-depth profile."""
import io
import py_compile
import tokenize

SRC = (r'D:\AI2050\Ai2050-OpenOne\tests'
       r'\glm5\phase3131_omega_p129_'
       r'layerwindow_single40_a1gen_'
       r'forklayer.py')
OUTP = (r'D:\AI2050\Ai2050-OpenOne\tests'
        r'\gpt5_temp\p3131_compile_out.txt')

lines = []
try:
    py_compile.compile(SRC, doraise=True)
    lines.append('PY_COMPILE: OK')
except Exception as e:
    lines.append('PY_COMPILE: FAIL')
    lines.append(repr(e))

try:
    txt = io.open(SRC, encoding='utf-8').read()
    n = len(txt.splitlines())
    lines.append('LINES: %d' % n)
    bad = []
    depth = 0
    stack = []
    toks = None
    try:
        toks = list(tokenize.generate_tokens(
            io.StringIO(txt).readline))
    except Exception as e:
        bad.append('tokenize error: %r' % (e,))
    if toks is not None:
        for tk in toks:
            if tk.type == tokenize.OP:
                if tk.string in '([{':
                    depth += 1
                    stack.append((tk.string, tk.start))
                elif tk.string in ')]}':
                    depth -= 1
                    if stack:
                        stack.pop()
                    if depth < 0:
                        bad.append('negative depth at %r' % (tk.start,))
                        depth = 0
        if depth != 0:
            bad.append('final depth %d unmatched %r'
                       % (depth, stack[:6]))
    lines.append('BRACKET: %s'
                 % ('OK' if not bad else 'ISSUES'))
    for b in bad:
        lines.append('  ' + b)
except Exception as e:
    lines.append('READ FAIL: %r' % (e,))

with io.open(OUTP, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print('done')
