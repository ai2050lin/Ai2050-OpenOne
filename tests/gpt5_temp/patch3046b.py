# -*- coding: utf-8 -*-
import io
import py_compile

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3046_omega_p43_kfield_injection_qwen.py')
s = io.open(P, encoding='utf-8').read()

old = ("        dlg_a, kinj, dpre_ok = forward_inj(\n"
       "            ids, li, pos, g * axis)")
new = ("        dlg_a, kinj, _ = forward_inj(\n"
       "            ids, li, pos, g * axis)")
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new)
io.open(P, 'w', encoding='utf-8').write(s)

py_compile.compile(P, doraise=True)
print('ok')
