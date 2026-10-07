# -*- coding: utf-8 -*-
import io
F = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3137_omega_p135_modecoop_'
     r'k0anat_coorddecomp_l35recheck.py')
src = io.open(F, encoding='utf-8').read()
a = src.index('# 3134 prompt-only anchor')
b = src.index("log('A hard asserts ok")
new_seg = """# 3134 prompt-only anchor: locate the
# scale-2.0 cell by the four MEMORY
# anchor values (d2 chg L17 0.2560 /
# L29 0.0625 / L33 0.0491 / L38 0.0893)
b34 = res34['part_b']
cm34 = b34['chg_matrix']
CAND34 = [0.25595238095238093,
          0.0625,
          0.049107142857142905,
          0.0892857142857143]
i2 = None
for _i in range(3):
    _vals = [float(cm34[l][_i])
             for l in ('17', '29', '33',
                       '38')]
    if all(abs(x - y) < 1e-9
           for x, y in zip(_vals,
                           CAND34)):
        i2 = _i
        break
assert i2 is not None, 'no dose-2 cell'
got38 = float(cm34['38'][i2])
assert abs(got38 - L38_PO_672_34) < 1e-9, got38
"""
src2 = src[:a] + new_seg + src[b:]
io.open(F, 'w', encoding='utf-8').write(src2)
chk = io.open(F, encoding='utf-8').read()
assert chk.count('CAND34') == 2
assert "b34.get('b_dose')" not in chk
assert chk.count('no dose-2 cell') == 1
print('PATCH OK rev3137a2')
