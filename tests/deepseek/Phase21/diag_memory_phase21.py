# -*- coding: utf-8 -*-
"""诊断：以真实磁盘为准，dump MEMORY.md 中含目标锚点的行（repr）。"""
import os
import io

SRC = r'D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\MEMORY.md'
OUT = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek_temp\Phase21\_diag_memory.txt'
o = []


def w(s=''):
    o.append(str(s))


b = open(SRC, 'rb').read()
t = b.decode('utf-8-sig')
lines = t.split('\n')
w('bytes=%d lines=%d bom=%s' % (len(b), len(lines), b[:3] == b'\xef\xbb\xbf'))
KEYS = [u'收尾链', u'Ledger n=', u'Phase 21 =', u'rdc-main-axis-probe', u'Phase 22', u'n=**30']
for i, l in enumerate(lines):
    for k in KEYS:
        if k in l:
            w('')
            w('--- line %d (len=%d) ---' % (i + 1, len(l)))
            w(repr(l))
            break
io.open(OUT, 'w', encoding='utf-8', newline='\n').write('\n'.join(o) + '\n')
print('DIAG OK')
