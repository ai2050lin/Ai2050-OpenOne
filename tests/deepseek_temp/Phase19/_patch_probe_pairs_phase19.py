# -*- coding: utf-8 -*-
"""补丁：探针误把 discovery(2元 [word,sup]) 当成 5 元 pair -> IndexError。改为与 P17 同口径。"""
import io

p = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek\Phase19\probe_feasibility_phase19.py'
s = io.open(p, encoding='utf-8').read()

old1 = "INST_ALL = [tuple(x) for x in EX['instances_all']]\nDISC = [tuple(x) for x in EX['discovery']]"
new1 = ("INST_ALL = [tuple(x) for x in EX['instances_all']]\n"
        "PAIRS_ALL = [tuple(x) for x in EX['pairs_all']]\n"
        "DISC = [tuple(x) for x in EX['discovery']]")
assert s.count(old1) == 1, 'old1 count=%d' % s.count(old1)
s = s.replace(old1, new1)

old2 = "    pairs = [tuple(x) for x in DISC[:NPAIR]]"
new2 = ("    DISC_W = set(x[0] for x in DISC)\n"
        "    pairs = [p for p in PAIRS_ALL if p[0] in DISC_W and p[2] in DISC_W][:NPAIR]")
assert s.count(old2) == 1, 'old2 count=%d' % s.count(old2)
s = s.replace(old2, new2)

io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('patched OK')
print('has PAIRS_ALL  :', 'PAIRS_ALL = [tuple' in s)
print('has DISC_W filt:', 'DISC_W = set(x[0] for x in DISC)' in s)
print('n_pairs expr   :', 'pairs = [p for p in PAIRS_ALL' in s)
