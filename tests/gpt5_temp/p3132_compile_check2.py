# -*- coding: utf-8 -*-
import io
import py_compile
import tokenize

FILES = [
    (r'D:\AI2050\Ai2050-OpenOne\tests'
     r'\glm5\phase3132_omega_p130_'
     r'forkcausal_single256.py',
     'main'),
    (r'D:\AI2050\Ai2050-OpenOne\tests'
     r'\gpt5_temp\phase3132_closeout.py',
     'closeout'),
    (r'D:\AI2050\Ai2050-OpenOne\tests'
     r'\gpt5_temp\p3132_disk_verify.py',
     'verify'),
]
OUTF = (r'D:\AI2050\Ai2050-OpenOne'
        r'\tests\gpt5_temp'
        r'\p3132_compile_out2.txt')
lines = []
allok = True
for SRC, tag in FILES:
    ok = True
    try:
        py_compile.compile(SRC,
                           doraise=True)
        lines.append('%s py_compile: OK'
                     % tag)
    except Exception as e:
        ok = False
        lines.append('%s py_compile '
                     'FAIL: %r' % (tag, e))
    try:
        depth = 0
        mind = 0
        with open(SRC, 'rb') as f:
            toks = list(
                tokenize.tokenize(
                    f.readline))
        for t in toks:
            if t.type == tokenize.OP:
                if t.string in '([{':
                    depth += 1
                elif t.string in ')]}':
                    depth -= 1
                    if depth < 0:
                        mind = min(mind,
                                   depth)
        lines.append('%s depth=%d min=%d'
                     % (tag, depth, mind))
        if depth != 0 or mind < 0:
            ok = False
            lines.append('%s BRACKET '
                         'IMBALANCE' % tag)
    except Exception as e:
        ok = False
        lines.append('%s tokenize FAIL: '
                     '%r' % (tag, e))
    allok = allok and ok
lines.append('RESULT: %s'
             % ('GREEN' if allok
                else 'RED'))
with io.open(OUTF, 'w',
             encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('done')
