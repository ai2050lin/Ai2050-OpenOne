# -*- coding: utf-8 -*-
"""p3147 patch4 v2: fix closeout marker
+ strip literal backslash-n in daily."""
import io

FP = (r'D:\AI2050\Ai2050-OpenOne\tests'
      r'\gpt5_temp\p3147_closeout.py')
t = io.open(FP, encoding='utf-8').read()

old1 = "if 'p145_tbsym' not in tl:"
new1 = "if 'P145\uff09\u95ed\u73af' not in tl:"
c1 = t.count(old1)
assert c1 == 1, ('fix1', c1)
t = t.replace(old1, new1)

old2 = "assert 'p145_tbsym' in tl2"
new2 = "assert 'P145\uff09\u95ed\u73af' in tl2"
c2 = t.count(old2)
assert c2 == 1, ('fix2', c2)
t = t.replace(old2, new2)

io.open(FP, 'w',
        encoding='utf-8').write(t)
chk = io.open(FP, encoding='utf-8').read()
assert "'p145_tbsym'" not in chk
print('PATCH4A OK')

DP = (r'D:\AI2050\Ai2050-OpenOne'
      r'\.workbuddy\memory'
      r'\2026-09-30.md')
td = io.open(DP, encoding='utf-8').read()
BS = chr(92)
BAD = BS + 'n'
td2 = td.rstrip()
had = td2.endswith(BAD)
if had:
    td2 = td2[:-2] + chr(10)
io.open(DP, 'w',
        encoding='utf-8').write(td2)
td3 = io.open(DP, encoding='utf-8').read()
assert not td3.rstrip().endswith(BAD)
assert 'P145\uff09\u95ed\u73af' in td3
print('PATCH4B OK (tail_fixed=%s)'
      % had)
