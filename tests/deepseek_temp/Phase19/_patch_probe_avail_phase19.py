# -*- coding: utf-8 -*-
"""补丁 2：探针 (a) AVAIL 基于 CAP 覆盖；(b) NPAIR=0 表示全量（全 41 实例估 U_l，与 P17 同口径）。"""
import io

p = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek\Phase19\probe_feasibility_phase19.py'
s = io.open(p, encoding='utf-8').read()

# (a) NPAIR 默认 0 = 全量
old0 = "NPAIR = int(os.environ.get('NPAIR', '4'))"
new0 = "NPAIR = int(os.environ.get('NPAIR', '0'))   # 0 = 全量（全 41 实例估 U_l，与 P17 同口径）"
assert s.count(old0) == 1, 'old0'
s = s.replace(old0, new0)

# (b) pairs 全量支持
old1 = ("    DISC_W = set(x[0] for x in DISC)\n"
        "    pairs = [p for p in PAIRS_ALL if p[0] in DISC_W and p[2] in DISC_W][:NPAIR]")
new1 = ("    DISC_W = set(x[0] for x in DISC)\n"
        "    pairs = [p for p in PAIRS_ALL if p[0] in DISC_W and p[2] in DISC_W]\n"
        "    if NPAIR > 0:\n"
        "        pairs = pairs[:NPAIR]")
assert s.count(old1) == 1, 'old1'
s = s.replace(old1, new1)

# (c) inst 全量支持
old2 = ("    keep = []\n"
        "    for p in pairs:\n"
        "        for x in (p[0], p[2]):\n"
        "            if x not in keep:\n"
        "                keep.append(x)\n"
        "    inst = [t for t in INST_ALL if t[0] in keep]\n"
        "    if len(inst) < 4:\n"
        "        inst = INST_ALL[:8]")
new2 = ("    if NPAIR > 0:\n"
        "        keep = []\n"
        "        for p in pairs:\n"
        "            for x in (p[0], p[2]):\n"
        "                if x not in keep:\n"
        "                    keep.append(x)\n"
        "        inst = [t for t in INST_ALL if t[0] in keep]\n"
        "        if len(inst) < 4:\n"
        "            inst = INST_ALL[:8]\n"
        "    else:\n"
        "        inst = list(INST_ALL)")
assert s.count(old2) == 1, 'old2'
s = s.replace(old2, new2)

# (d) AVAIL 基于 CAP 覆盖
old3 = ("    by = {}\n"
        "    for wd, sup in INST_ALL:\n"
        "        by.setdefault(sup, []).append(wd)\n"
        "    AVAIL = [s for s in SUPS if s in by and all(x in CAP for x in by[s])]\n"
        "    rk = max(len(AVAIL) - 1, 1)")
new3 = ("    by = {}\n"
        "    for wd, sup in INST_ALL:\n"
        "        if wd in CAP:\n"
        "            by.setdefault(sup, []).append(wd)\n"
        "    AVAIL = [s for s in SUPS if s in by]\n"
        "    if len(AVAIL) < 2:\n"
        "        raise RuntimeError('AVAIL=%s <2：实例集不足以估计 U_l' % AVAIL)\n"
        "    rk = max(len(AVAIL) - 1, 1)")
assert s.count(old3) == 1, 'old3'
s = s.replace(old3, new3)

io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('patched OK')
print('NPAIR default 0 :', "NPAIR = int(os.environ.get('NPAIR', '0'))" in s)
print('pairs full supp :', 'if NPAIR > 0:' in s)
print('AVAIL CAP-based :', 'if wd in CAP:' in s)
print('raise on AVAIL<2:', 'AVAIL=%s <2' in s)
