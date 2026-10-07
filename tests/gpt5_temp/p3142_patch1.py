# -*- coding: utf-8 -*-
import io
import py_compile

src = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
       r'\phase3142_omega_p140_l19anat_'
       r'blockmat_quartile_multitpl.py')
txt = io.open(src, encoding='utf-8').read()

# clean 1: dvec19 batch construction
old1 = (
    "_dv_d19 = np.repeat(\n"
    "    dvec19[:1], 1, axis=0)\n"
    "if _N19 >= SCAN_N:\n"
    "    _dv_d19 = dvec19[:SCAN_N].copy()\n"
    "else:\n"
    "    _dv_d19 = np.zeros((SCAN_N, HIDG),\n"
    "                       dtype=np.float32)\n"
    "    _dv_d19[:_N19] = dvec19\n"
    "    # rows beyond captured range: reuse\n"
    "    # row 0 pattern is WRONG; assert cap\n"
    "    assert SCAN_N <= _N19 or SMOKE, \\\n"
    "        'dvec19 rows insufficient'\n"
    "    if SMOKE:\n"
    "        for j in range(SCAN_N):\n"
    "            _dv_d19[j] = dvec19[j % _N19]\n")
new1 = (
    "assert _N19 >= SCAN_N, \\\n"
    "    'dvec19 rows insufficient'\n"
    "_dv_d19 = dvec19[:SCAN_N].copy()\n")
n1 = txt.count(old1)
assert n1 == 1, 'old1 count %d' % n1
txt = txt.replace(old1, new1)

# clean 2: f3 first-pass dead code
old2 = (
    "    vkeys = ('V0', 'V1x', 'V2', 'V3')\n"
    "    full3 = []\n"
    "    for i in range(XMAT_N):\n"
    "        seq = []\n"
    "        for vk in vkeys:\n"
    "            seq += rows_var[int(\n"
    "                vk[1] if vk[1:].isdigit()\n"
    "                else vk[1])]\n"
    "        full3.append(seq)\n"
    "    # rebuild with per-row correct variant\n"
    "    # rows (above slicing is wrong; redo)\n"
    "    full3 = []\n"
    "    vidx = {'V0': 0, 'V1x': 1, 'V2': 2,\n"
    "            'V3': 3}\n")
new2 = (
    "    vkeys = ('V0', 'V1x', 'V2', 'V3')\n"
    "    vidx = {'V0': 0, 'V1x': 1, 'V2': 2,\n"
    "            'V3': 3}\n"
    "    full3 = []\n")
n2 = txt.count(old2)
assert n2 == 1, 'old2 count %d' % n2
txt = txt.replace(old2, new2)

io.open(src, 'w', encoding='utf-8').write(txt)
txt2 = io.open(src, encoding='utf-8').read()
assert txt2.count(new1) == 1
assert txt2.count(new2) == 1
assert 'rebuild with per-row' not in txt2
assert 'row 0 pattern is WRONG' not in txt2
py_compile.compile(src, doraise=True)
lines = txt2.splitlines()
print('cleaned + compiled OK, lines = %d'
      % len(lines))
