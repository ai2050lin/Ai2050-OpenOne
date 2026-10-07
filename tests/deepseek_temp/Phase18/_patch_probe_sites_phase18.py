# -*- coding: utf-8 -*-
"""Phase18 探针勘误 P-C：位点过滤从「REACH 内」放宽为「1..L-1」，以支持全域质量支撑。"""
import io

P = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek\Phase18\probe_feasibility_phase18.py'
s = io.open(P, encoding='utf-8').read()

old = "SITES = [s for s in SITES if s in REACH]\nw('probe sites = %s (of REACH)' % SITES)\n"
new = ("SITES = [s for s in SITES if 1 <= s <= L - 1]\n"
       "w('probe sites = %s (n=%d, 域 1..L-1)' % (SITES, len(SITES)))\n")
assert s.count(old) == 1, 'anchor count=%d' % s.count(old)
s = s.replace(old, new)
io.open(P, 'w', encoding='utf-8', newline='\n').write(s)

t = io.open(P, encoding='utf-8').read()
assert '1 <= s <= L - 1' in t and '(of REACH)' not in t
print('PATCH P-C OK (site filter widened)')
