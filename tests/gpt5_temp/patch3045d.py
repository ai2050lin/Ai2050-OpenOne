# -*- coding: utf-8 -*-
import io
import py_compile

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3045_omega_p42_l20_axis_anatomy_'
     r'qwen.py')
s = io.open(P, encoding='utf-8').read()

old = "                   -> per-coordinate norms ~0.001, '\n"
new = "                   '-> per-coordinate norms ~0.001, '\n"
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new)

io.open(P, 'w', encoding='utf-8').write(s)
py_compile.compile(P, doraise=True)

out = ('patch3045d ok: quote fixed, compile zero errors')
io.open(r'D:\AI2050\Ai2050-OpenOne\gpt5_temp'
        r'\patch3045d_result.txt', 'w',
        encoding='utf-8').write(out)
print(out)
