# -*- coding: utf-8 -*-
import os

root = r'D:\AI2050\Ai2050-OpenOne'
dst = os.path.join(root, '.workbuddy', 'memory', '2026-10-01.md')
src = os.path.join(root, 'gpt5_temp', 'wlog_n2h1.md')
chk = os.path.join(root, 'gpt5_temp', 'verify_wlog_n2h1.txt')

add = open(src, encoding='utf-8').read()
if not add.startswith('\n'):
    add = '\n' + add
if not add.endswith('\n'):
    add += '\n'
b0 = os.path.getsize(dst)
with open(dst, 'ab') as f:
    f.write(add.encode('utf-8'))
b1 = os.path.getsize(dst)
T = open(dst, encoding='utf-8').read()

out = []
out.append('wlog before %d after %d delta %d' % (b0, b1, b1 - b0))
out.append('has_section %s' % ('## N2-h1 置换向量消融' in T))
out.append('tail_ok %s' % T.rstrip().endswith('末层接口修正后重测。'))
out.append('tail200 %r' % T[-200:])
open(chk, 'w', encoding='utf-8').write('\n'.join(out))
print('\n'.join(out[:3]))
