# -*- coding: utf-8 -*-
import io
import json
import re
import os

P = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek_temp\Phase20\N2h1a13_design_seal.json'
s = io.open(P, encoding='utf-8').read()
o = []
for m in re.finditer(r'[^"]{0,220}(?:xhalf|XH_FAITHFUL|xh_frac|\u534a\u9971\u548c|\u8df3\u53d8|REACH)[^"]{0,220}', s):
    o.append(m.group(0))
    o.append('-' * 80)
io.open(r'D:\AI2050\Ai2050-OpenOne\tests\deepseek_temp\_infra\probe_seal_xhalf.txt',
        'w', encoding='utf-8', newline='\n').write('\n'.join(o) + '\n')
print('blocks=', len(o) // 2)
