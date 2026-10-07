# -*- coding: utf-8 -*-
import io

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3043_omega_p40_field_variance_qwen.py')
s = io.open(P, encoding='utf-8').read()

# 1) PREREG anchors text: a98 description
old_a = ("0.0, 8/8; a98 full '\n"
         "               'old-bank anchor vs 3042 npz: "
         "V3 AND '\n"
         "               'V20 bit-identical over the 32 "
         "old '\n"
         "               'prompts (same assembly order); "
         "a99 '")
new_a = ("0.0, 8/8; a98 '\n"
         "               'displacement-chain anchor vs "
         "3042 npz: '\n"
         "               'D3 AND D20 bit-identical over "
         "the 24 '\n"
         "               'old displacements (same build "
         "order); a99 '")
assert s.count(old_a) == 1, ('a', s.count(old_a))
s = s.replace(old_a, new_a)

# 2) PREREG corrections
old_b = "    'corrections': 'none (first run)',"
new_b = ("    'corrections': 'run1 crashed PRE-verdict '\n"
         "                   '(anchor stage only, no '\n"
         "                   'test statistic observed): '\n"
         "                   'a98 referenced V3/V20 keys '\n"
         "                   'absent from the 3042 npz '\n"
         "                   '(which stores displacement '\n"
         "                   'arrays D3/D20); corrected to '\n"
         "                   'a displacement-chain anchor '\n"
         "                   'comparing our D3/D20 vs the '\n"
         "                   '3042 npz D3/D20 bit-exact '\n"
         "                   '(stronger derived-quantity '\n"
         "                   'chain check); verdict tree '\n"
         "                   'unchanged',")
assert s.count(old_b) == 1, ('b', s.count(old_b))
s = s.replace(old_b, new_b)

# 3) remove the broken a98 block (keep z42 load)
old_c = ("V3_42 = z42['V3']\n"
         "V20_42 = z42['V20']\n"
         "a98_shape_ok = bool(V3_42.shape == (n_old, "
         "HDIM))\n"
         "a98_v3 = float(np.max(np.abs(\n"
         "    V3o[:n_old] - V3_42))) if a98_shape_ok \\\n"
         "    else float('nan')\n"
         "a98_v20 = float(np.max(np.abs(\n"
         "    V20o[:n_old] - V20_42))) if a98_shape_ok \\\n"
         "    else float('nan')\n"
         "a98_ok = bool(a98_shape_ok and a98_v3 == 0.0\n"
         "              and a98_v20 == 0.0)\n"
         "log('a98 shape_ok=%s dV3=%.3e dV20=%.3e'\n"
         "    % (a98_shape_ok, a98_v3, a98_v20))\n")
new_c = ""
assert s.count(old_c) == 1, ('c', s.count(old_c))
s = s.replace(old_c, new_c)

# 4) insert a98 computation after D20_old built
old_d = ("ok20_old = np.linalg.norm(D20_old, axis=1) \\\n"
         "    >= DEGEN_NORM\n")
new_d = ("a98_shape_ok = bool(\n"
         "    z42['D3'].shape == D3_old.shape\n"
         "    and z42['D20'].shape == D20_old.shape)\n"
         "a98_d3 = float(np.max(np.abs(\n"
         "    D3_old - z42['D3']))) if a98_shape_ok \\\n"
         "    else float('nan')\n"
         "a98_d20 = float(np.max(np.abs(\n"
         "    D20_old - z42['D20']))) if a98_shape_ok \\\n"
         "    else float('nan')\n"
         "a98_ok = bool(a98_shape_ok and a98_d3 == 0.0\n"
         "              and a98_d20 == 0.0)\n"
         "log('a98 shape_ok=%s dD3=%.3e dD20=%.3e'\n"
         "    % (a98_shape_ok, a98_d3, a98_d20))\n"
         "ok20_old = np.linalg.norm(D20_old, axis=1) \\\n"
         "    >= DEGEN_NORM\n")
assert s.count(old_d) == 1, ('d', s.count(old_d))
s = s.replace(old_d, new_d)

# 5) verdict log line: rename refs
old_e = "'a97_ok=%s (%.3e, %d) a98_ok=%s (%.1e/%.1e) '"
assert s.count(old_e) == 1, ('e', s.count(old_e))
s = s.replace(old_e, old_e)  # format string unchanged

old_f = ("       a98_ok, a98_v3, a98_v20, a99_ok,")
new_f = ("       a98_ok, a98_d3, a98_d20, a99_ok,")
assert s.count(old_f) == 1, ('f', s.count(old_f))
s = s.replace(old_f, new_f)

# 6) npz keys
old_g = ("    a98_v3=np.float64(a98_v3),\n"
         "    a98_v20=np.float64(a98_v20),")
new_g = ("    a98_d3=np.float64(a98_d3),\n"
         "    a98_d20=np.float64(a98_d20),")
assert s.count(old_g) == 1, ('g', s.count(old_g))
s = s.replace(old_g, new_g)

# 7) result anchors keys
old_h = ("        'a98_v3_bit': a98_v3,\n"
         "        'a98_v20_bit': a98_v20,")
new_h = ("        'a98_d3_bit': a98_d3,\n"
         "        'a98_d20_bit': a98_d20,")
assert s.count(old_h) == 1, ('h', s.count(old_h))
s = s.replace(old_h, new_h)

io.open(P, 'w', encoding='utf-8').write(s)

import py_compile
py_compile.compile(P, doraise=True)
with open(r'D:\AI2050\Ai2050-OpenOne\gpt5_temp'
          r'\patch3043_result.txt', 'w',
          encoding='utf-8') as f:
    f.write('patch ok, compile ok')
print('done')
