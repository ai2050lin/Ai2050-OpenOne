# -*- coding: utf-8 -*-
import io
t = io.open(r'D:/AI2050/Ai2050-OpenOne/tests/glm5/phase3201_g5e3_eline_audit.py', encoding='utf-8').read()
import re
for m in re.finditer(r'Phase 3201', t):
    print(t[max(0,m.start()-80):m.end()+40].replace('\n', '|'))
print('---phase 3201 literal---')
for m in re.finditer(r"'phase': 3201", t):
    print(t[max(0,m.start()-60):m.end()+20].replace('\n', '|'))
