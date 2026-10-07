# -*- coding: utf-8 -*-
"""补丁：gen_memo_phase19.py 里 ARMS[a]['quant'] 的键名错误（result 实际键为 'scheme'）。"""
import io
import os

P = os.path.join(r'D:\AI2050\Ai2050-OpenOne', 'tests', 'deepseek', 'Phase19', 'gen_memo_phase19.py')
s = io.open(P, encoding='utf-8').read()

old = "ARMS[a]['quant']"
new = "ARMS[a]['scheme']"
assert s.count(old) == 1, 'old count=%d' % s.count(old)
s = s.replace(old, new)

io.open(P, 'w', encoding='utf-8', newline='\n').write(s)

# 回读复核
s2 = io.open(P, encoding='utf-8').read()
assert s2.count("ARMS[a]['scheme']") == 1, 'patch 未落盘'
assert "'quant'" not in s2, '残留 quant 键'
print('PATCH OK  bytes=%d' % len(s2.encode('utf-8')))
