# -*- coding: utf-8 -*-
import io
t = io.open(r'D:/AI2050/Ai2050-OpenOne/tests/glm5/phase3200_g5e2_ftr23_heldout_upgrade.py', encoding='utf-8').read()
lines = t.split('\n')
for i, l in enumerate(lines, 1):
    if 'G_MED' in l:
        print(i, l)
