# -*- coding: utf-8 -*-
import io
F = (r'D:\AI2050\Ai2050-OpenOne\tests'
     r'\gpt5_temp\p3139_closeout.py')
s = io.open(F, encoding='utf-8').read()
BT = chr(96)  # backtick, avoid inline bash traps
old1 = ('**判决**：' + BT + '%s' + BT
        + '（正式跑 2700.3s 一次通过，xphase=1.0，'
          'bank 8 分片直接复用 3138 免重跑 capture；'
          'result sha8=%s，ledger n=276 sha8=%s）')
new1 = ('**判决**：' + BT + '__VERDICT__' + BT
        + '（正式跑 2700.3s 一次通过，xphase=1.0，'
          'bank 8 分片直接复用 3138 免重跑 capture；'
          'result sha8=__RESSHA__，ledger n=276 '
          'sha8=__LEDSHA__）')
assert s.count(old1) == 1, s.count(old1)
s = s.replace(old1, new1)
old2 = '" % (hhmm, VERDICT, res_sha8, led_sha)'
assert s.count(old2) == 1, s.count(old2)
s = s.replace(old2, ')')
# hhmm placeholder in the heading
old3 = '（T4 第22 Phase）[%s]'
assert s.count(old3) == 1
s = s.replace(old3, '（T4 第22 Phase）[__HHMM__]')
io.open(F, 'w', encoding='utf-8').write(s)
chk = io.open(F, encoding='utf-8').read()
for frag in ('__VERDICT__', '__RESSHA__',
             '__LEDSHA__', '__HHMM__]'):
    assert frag in chk, frag
assert '% (hhmm, VERDICT' not in chk
# no bare %s left inside the sec literal
a = chk.index('    sec = (')
b = chk.index("# ---------- 3. wlog ----------")
secseg = chk[a:b]
assert '%s' not in secseg.replace(
    "print", ""), 'bare %s remains in sec'
print('FIX_OK')
