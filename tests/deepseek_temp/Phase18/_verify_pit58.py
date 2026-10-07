# -*- coding: utf-8 -*-
import io
p = r'C:\Users\Admin\.workbuddy\skills\rdc-main-axis-probe\SKILL.md'
s = io.open(p, encoding='utf-8').read().splitlines()
L = s[177]
print('len(line178) =', len(L))
print('has 4:', '\u56db' in L, '| has 5 item:', '\u4e94\u6761' in L)
for t in ['\u56db\u6761\u300d', '\u4e94\u6761\u300d', '\u5904\u7f6e**\uff1a**\u4e0d\u8ffd\u6539\u5224\u636e', '\u9519\u4e86\uff0c']:
    print('%-24s %s' % (t, t in L))
print(L[-160:])
print()
print('lines', len(s), 'bytes', len(io.open(p, encoding='utf-8').read().encode('utf-8')))
