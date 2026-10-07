# -*- coding: utf-8 -*-
import io
P = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek\Phase17\gen_memo_phase17.py'
s = io.open(P, encoding='utf-8').read()
OLD = "  % jd({sh(a): RES['arms'][a]['E5_com_V']['neighbourhood'] for a in ARMS}))\n"
NEW = "  % json.dumps({sh(a): RES['arms'][a]['E5_com_V']['neighbourhood'] for a in ARMS}, ensure_ascii=False))\n"
assert s.count(OLD) == 1, s.count(OLD)
io.open(P, 'w', encoding='utf-8', newline='\n').write(s.replace(OLD, NEW))
t = io.open(P, encoding='utf-8').read()
assert 'jd({' not in t
print('FIX OK')
