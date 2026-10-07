# -*- coding: utf-8 -*-
"""补丁 4：配对过滤与 P17 逐字一致 —— P17 只要求 donor 属于 discovery（p[0] in DISC_W），
   并要求 p[0]/p[2] 在实例集内（INST_ALL 全覆盖）。我误加 p[2] in DISC_W => 24 对降为 17 对
   => nf4 com_V 26.1956 != P17 锚 26.1501（差 0.0455）。"""
import io

p = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek\Phase19\probe_feasibility_phase19.py'
s = io.open(p, encoding='utf-8').read()

old1 = ("    DISC_W = set(x[0] for x in DISC)\n"
        "    pairs = [p for p in PAIRS_ALL if p[0] in DISC_W and p[2] in DISC_W]\n"
        "    if NPAIR > 0:\n"
        "        pairs = pairs[:NPAIR]")
new1 = ("    DISC_W = set(x[0] for x in DISC)\n"
        "    # 与 P17 逐字同口径：disc_pairs = [p for p in PAIRS_ALL if p[0] in DISC_W and p[0]/p[2] in CAP]\n"
        "    # （INST_ALL 覆盖全部实例，故此处只需 p[0] in DISC_W）。\n"
        "    pairs = [p for p in PAIRS_ALL if p[0] in DISC_W]\n"
        "    if NPAIR > 0:\n"
        "        pairs = pairs[:NPAIR]")
assert s.count(old1) == 1, 'old1 count=%d' % s.count(old1)
s = s.replace(old1, new1)

io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('patched OK')
print('has P17 filter:', 'pairs = [p for p in PAIRS_ALL if p[0] in DISC_W]' in s)
