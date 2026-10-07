# -*- coding: utf-8 -*-
"""补丁 3：gen_present_phase20.py —— seal.predictions 实为 list（元素键 id/name/criterion/falsified_by）。
修：建 id->dict 索引；B 节 nb 显示表达式改直取。"""
import io
import os

P = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek\Phase20\gen_present_phase20.py'
s = io.open(P, encoding='utf-8').read()
n = 0

old1 = "SEAL = json.load(io.open(SEALP, encoding='utf-8'))\n"
new1 = ("SEAL = json.load(io.open(SEALP, encoding='utf-8'))\n"
        "SEALP_D = {x['id']: x for x in SEAL['predictions']}\n")
assert s.count(old1) == 1, 'old1 %d' % s.count(old1)
s = s.replace(old1, new1); n += 1

old2 = "    claim = str(SEAL['predictions'][k].get('claim', '')).replace('\\n', ' ')[:170]\n"
new2 = "    claim = str(SEALP_D.get(k, {}).get('criterion', '')).replace('\\n', ' ')[:170]\n"
assert s.count(old2) == 1, 'old2 %d' % s.count(old2)
s = s.replace(old2, new2); n += 1

old3 = "         % (str(V[ARMS[0]]['null_comB_inc'] and R['arms'][ARMS[0]]['E10_summary']['nb']),\n"
new3 = "         % (str(R['arms'][ARMS[0]]['E10_summary']['nb']),\n"
assert s.count(old3) == 1, 'old3 %d' % s.count(old3)
s = s.replace(old3, new3); n += 1

io.open(P, 'w', encoding='utf-8', newline='\n').write(s)
print('PATCHED %d spots' % n)
for probe in ('SEALP_D = {x[', "SEALP_D.get(k, {})", "str(R['arms'][ARMS[0]]['E10_summary']['nb']),\n"):
    print('  chk %-40s -> %d' % (probe[:40], s.count(probe)))
