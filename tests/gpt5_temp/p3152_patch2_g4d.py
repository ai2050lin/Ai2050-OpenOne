# -*- coding: utf-8 -*-
# p3152 patch2: glm4k1 D from npz (glm4-9b hidden=4096, not 5120)
import io, os
P = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\phase3152_g1p2_tri_model_k1.py'
src = io.open(P, encoding='utf-8').read()
old = """    NL = 40
    D = 5120
    KSTAR = int(round(KSTAR_FRAC * NL))"""
new = """    NL = 40
    _z0 = np.load(NPZ)
    D = int(_z0['H'].shape[3])
    del _z0
    KSTAR = int(round(KSTAR_FRAC * NL))"""
assert src.count(old) == 1, ('old count', src.count(old))
src = src.replace(old, new)
io.open(P, 'w', encoding='utf-8', newline='').write(src)

exe = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913\phase3152\g1p2_tri_model_k1\glm4k1\execution.json'
if os.path.exists(exe):
    os.remove(exe)
    print('removed stale execution.json')
print('patched ok')
