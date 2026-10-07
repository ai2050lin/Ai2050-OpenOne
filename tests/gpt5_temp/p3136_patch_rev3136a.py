# -*- coding: utf-8 -*-
import io
F = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\phase3136_omega_p134_conddose_crossmatrix_w8drop.py'
src = io.open(F, encoding='utf-8').read()
old = """        n_cap = dvec[il].shape[0]
        h_inj = capture_states_inject(
            rows_cap[:n_cap], ALL_L, il,
            dvec[il], dose * DOSE_COND)"""
new = """        n_cap = min(NP_CAP,
                    dvec[il].shape[0])
        h_inj = capture_states_inject(
            rows_cap[:n_cap], ALL_L, il,
            dvec[il][:n_cap],
            dose * DOSE_COND)"""
assert src.count(old) == 1, 'patch1 count=%d' % src.count(old)
src = src.replace(old, new)
io.open(F, 'w', encoding='utf-8').write(src)
# verify on disk
chk = io.open(F, encoding='utf-8').read()
assert chk.count(new) == 1, 'verify failed'
assert chk.count(old) == 0, 'old remains'
print('PATCH OK rev3136a')
