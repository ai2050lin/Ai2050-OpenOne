# -*- coding: utf-8 -*-
import io
import py_compile

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3046_omega_p43_kfield_injection_qwen.py')
s = io.open(P, encoding='utf-8').read()

old = "    n_integ_fail=np.int64(n_integ_fail),"
new = ("    rec_kind=np.array(rec_kind),\n"
       "    rec_body=np.array(rec_body),\n"
       "    rec_dlg=np.array(rec_dlg),\n"
       "    n_integ_fail=np.int64(n_integ_fail),")
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new)
io.open(P, 'w', encoding='utf-8').write(s)

py_compile.compile(P, doraise=True)
io.open(r'D:\AI2050\Ai2050-OpenOne\gpt5_temp'
        r'\compile3046.txt', 'w',
        encoding='utf-8').write('compile OK\n')
print('ok')
