# -*- coding: utf-8 -*-
"""补丁2：gen_memo_phase19.py 里 CALIB_TOL_COMV 用 0 位小数渲染成 '0'，改为 %g（0.001）。"""
import io
import os

P = os.path.join(r'D:\AI2050\Ai2050-OpenOne', 'tests', 'deepseek', 'Phase19', 'gen_memo_phase19.py')
s = io.open(P, encoding='utf-8').read()

old = "f(FL['CALIB_TOL_COMV'], 0)"
new = "('%g' % FL['CALIB_TOL_COMV'])"
assert s.count(old) == 1, 'old count=%d' % s.count(old)
s = s.replace(old, new)
io.open(P, 'w', encoding='utf-8', newline='\n').write(s)

s2 = io.open(P, encoding='utf-8').read()
assert s2.count(new) == 1, 'patch 未落盘'
print('PATCH2 OK  bytes=%d' % len(s2.encode('utf-8')))
