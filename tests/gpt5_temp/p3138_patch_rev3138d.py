# -*- coding: utf-8 -*-
import io
F = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3138_omega_p136_statebank_'
     r'bicdecomp_probecontrast.py')
src = io.open(F, encoding='utf-8').read()
old = ("z36 = np.load(\n"
       "    D36 + r'\\p134_readout.npz',\n"
       "    allow_pickle=False)\n"
       "f_full = z36['fstep_w8full'].astype(np.int64)\n"
       "f_nok0 = z36['fstep_w8_nok0'].astype(np.int64)")
new = ("f_full = z37['fstep_w8full'] \\\n"
       "    .astype(np.int64)\n"
       "f_nok0 = z37['fstep_w8_nok0'] \\\n"
       "    .astype(np.int64)")
assert src.count(old) == 1, \
    'p1 %d' % src.count(old)
src = src.replace(old, new)
io.open(F, 'w', encoding='utf-8').write(src)
chk = io.open(F, encoding='utf-8').read()
assert chk.count("z37['fstep_w8full']") == 1
assert chk.count('z36') == 0
print('PATCH OK rev3138d')
