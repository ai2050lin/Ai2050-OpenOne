# -*- coding: utf-8 -*-
import io
F = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3138_omega_p136_statebank_'
     r'bicdecomp_probecontrast.py')
src = io.open(F, encoding='utf-8').read()
old = ("    comps = {\n"
       "        'raw': X,\n"
       "        'BI': Hhat,\n"
       "        'I': (B[None, None, None, :]\n"
       "              + I[:, None, :, :])}")
new = ("    I_full = np.broadcast_to(\n"
       "        B[None, None, None, :]\n"
       "        + I[:, None, :, :],\n"
       "        (2, N_T, BANK_N, HIDG))\n"
       "    comps = {\n"
       "        'raw': X,\n"
       "        'BI': Hhat,\n"
       "        'I': I_full}")
assert src.count(old) == 1, \
    'p1 %d' % src.count(old)
src = src.replace(old, new)
io.open(F, 'w', encoding='utf-8').write(src)
chk = io.open(F, encoding='utf-8').read()
assert chk.count('I_full = np.broadcast_to') == 1
assert chk.count("'I': I_full}") == 1
print('PATCH OK rev3138c')
