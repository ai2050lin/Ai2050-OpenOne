# -*- coding: utf-8 -*-
import io
import py_compile

src = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
       r'\phase3142_omega_p140_l19anat_'
       r'blockmat_quartile_multitpl.py')
txt = io.open(src, encoding='utf-8').read()

old = (
    "if not SMOKE:\n"
    "    got = E_res['c_all_l23_d2.0']['chg']\n"
    "    m = abs(got - ALL_D2_41['23']) < 1e-9\n"
    "    n_bitC += int(m)\n"
    "    bit_anchors_C['c_all_l23_d2.0'] = {\n"
    "        'got': got, 'want': ALL_D2_41['23'],\n"
    "        'match': bool(m)}\n")
new = (
    "if not SMOKE:\n"
    "    got = E_res['c_all_l19_d2.0']['chg']\n"
    "    m = abs(got - ALL_D2_41['19']) < 1e-9\n"
    "    n_bitC += int(m)\n"
    "    bit_anchors_C['c_all_l19_d2.0'] = {\n"
    "        'got': got, 'want': ALL_D2_41['19'],\n"
    "        'match': bool(m)}\n")
n = txt.count(old)
assert n == 1, 'count %d' % n
txt = txt.replace(old, new)

# seal anchor name update is NOT needed:
# seal does not reference the anchor
# trial name, only values.
io.open(src, 'w', encoding='utf-8').write(txt)
txt2 = io.open(src, encoding='utf-8').read()
assert txt2.count(new) == 1
assert "c_all_l23_d2.0" not in txt2
py_compile.compile(src, doraise=True)
print('patch OK, compiled')
