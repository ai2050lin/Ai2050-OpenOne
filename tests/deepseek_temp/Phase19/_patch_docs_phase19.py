# -*- coding: utf-8 -*-
"""补丁 closeout_docs_phase19.py：
  (1) 勘误行没有 % 运算符，`%%` 会原样渲染 -> 改回 `%`；
  (2) 记录行里 n_forwards 的三元表达式写反 -> 直接用 R['arms'][A0n]['n_forwards']。
"""
import io
import os

P = os.path.join(r'D:\AI2050\Ai2050-OpenOne', 'tests', 'deepseek', 'Phase19', 'closeout_docs_phase19.py')
s = io.open(P, encoding='utf-8').read()

old1 = 'segfault（~19%% 权重）'
new1 = 'segfault（~19% 权重）'
assert s.count(old1) == 1, 'old1 count=%d' % s.count(old1)
s = s.replace(old1, new1)

old2 = ("     sum(int(R['arms'][a]['n_forwards']) for a in ARMS), "
        "V[A0n].get('n_forwards', 0) if 'n_forwards' not in V[A0n] else R['arms'][A0n]['n_forwards'],")
new2 = "     sum(int(R['arms'][a]['n_forwards']) for a in ARMS), R['arms'][A0n]['n_forwards'],"
assert s.count(old2) == 1, 'old2 count=%d' % s.count(old2)
s = s.replace(old2, new2)

io.open(P, 'w', encoding='utf-8', newline='\n').write(s)

s2 = io.open(P, encoding='utf-8').read()
assert s2.count(new1) == 1 and s2.count(new2) == 1, 'patch 未落盘'
assert '~19%%' not in s2, '残留 %%'
print('PATCH OK  bytes=%d' % len(s2.encode('utf-8')))
