# -*- coding: utf-8 -*-
import io
t = io.open(r'D:/AI2050/Ai2050-OpenOne/tests/glm5/phase3201_g5e3_eline_audit.py', encoding='utf-8').read()
lines = t.split('\n')
for i, l in enumerate(lines, 1):
    if 'reg24' in l and ('deepcopy' in l or "version'] == '1.23'" in l or 'registry_v123' in l or 'upgrade_log' in l and 'json.dumps' in l):
        print(i, l)
