# -*- coding: utf-8 -*-
import io
F = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3137_omega_p135_modecoop_'
     r'k0anat_coorddecomp_l35recheck.py')
src = io.open(F, encoding='utf-8').read()
old = """    gen_672 = []
    for b0 in range(0, NP, GEN_BATCH):
        batch = pids_P_all[b0:b0 + GEN_BATCH]"""
new = """    gen_672 = []
    _N672 = SCAN_N if SMOKE else NP
    for b0 in range(0, _N672, GEN_BATCH):
        batch = pids_P_all[b0:b0 + GEN_BATCH]"""
assert src.count(old) == 1, \
    'patch count=%d' % src.count(old)
src = src.replace(old, new)
io.open(F, 'w', encoding='utf-8').write(src)
chk = io.open(F, encoding='utf-8').read()
assert chk.count('_N672') == 2
print('PATCH OK rev3137b')
