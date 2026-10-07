# -*- coding: utf-8 -*-
"""Phase18 探针勘误 P-A：d_mlp 的受体项取错 capture（CAP[dw][2] -> CAP[rw][2]）。"""
import io

P = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek\Phase18\probe_feasibility_phase18.py'
s = io.open(P, encoding='utf-8').read()

old = "            OR_, MR = CAP[rw][1], CAP[dw][2]\n            OD, MD = CAP[dw][1], CAP[dw][2]\n"
new = "            OR_, MR = CAP[rw][1], CAP[rw][2]\n            OD, MD = CAP[dw][1], CAP[dw][2]\n"
assert s.count(old) == 1, 'anchor count=%d' % s.count(old)
s = s.replace(old, new)
io.open(P, 'w', encoding='utf-8', newline='\n').write(s)

t = io.open(P, encoding='utf-8').read()
assert "OR_, MR = CAP[rw][1], CAP[rw][2]" in t, 'not landed'
assert "OR_, MR = CAP[rw][1], CAP[dw][2]" not in t, 'old still present'
print('PATCH OK: MR now from CAP[rw][2]')
