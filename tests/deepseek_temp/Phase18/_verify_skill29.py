# -*- coding: utf-8 -*-
import io
p = r'C:\Users\Admin\.workbuddy\skills\rdc-phase-closeout\SKILL.md'
s = io.open(p, encoding='utf-8').read()
BT = chr(96)
for t in ['逐位复现', '先重算、再落笔', '不得写死「预期成功」', '13 个假 FAIL', '(f)', '(e)']:
    print('%-24s %s' % (t, 'OK' if t in s else 'MISS'))
print('lines', len(s.splitlines()), 'bytes', len(s.encode('utf-8')))
print('旧错误句残留  REACH 3.00 :', ('REACH' + BT + ' 3.00') in s)
print('旧错误句残留  两种支撑上都不复现 :', '两种支撑上都不复现' in s)
