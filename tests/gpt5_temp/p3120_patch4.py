# -*- coding: utf-8 -*-
"""p3120_patch4: fix illegal line break at closeout
lines 76-77 (assert continuation without enclosing
parens). Compile-check then write. Idempotent."""
import io

P = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
     r'\phase3120_closeout.py')
LOG = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\p3120_patch4_log.txt')
o = []
src = io.open(P, encoding='utf-8').read()
old = ("assert pa['subtag_table']['A1']"
       "['context_other']\n"
       "    ['n'] == 2570\n")
new = ("assert pa['subtag_table']['A1']"
       "['context_other']['n'] == 2570\n")
c = src.count(old)
o.append('old count=%d' % c)
if c == 1:
    src = src.replace(old, new)
    compile(src, P, 'exec')
    o.append('compile ok')
    with io.open(P, 'w', encoding='utf-8') as f:
        f.write(src)
    o.append('written')
elif c == 0 and ("['context_other']['n'] == 2570"
                 in src):
    o.append('already patched')
else:
    o.append('UNEXPECTED: count=%d' % c)
# verify on-disk after write
back = io.open(P, encoding='utf-8').read()
o.append('disk has one-line=%s'
         % ("['context_other']['n'] == 2570" in back))
o.append('disk still broken=%s'
         % (old in back))
io.open(LOG, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
