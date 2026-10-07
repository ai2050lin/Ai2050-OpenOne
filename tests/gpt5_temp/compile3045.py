# -*- coding: utf-8 -*-
import io
import py_compile

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3045_omega_p42_l20_axis_anatomy_'
     r'qwen.py')
py_compile.compile(P, doraise=True)
s = io.open(P, encoding='utf-8').read()
assert 'if False' not in s
assert 'GBAR20_new' not in s
out = 'compile zero errors; len=%d' % len(s)
io.open(r'D:\AI2050\Ai2050-OpenOne\gpt5_temp'
        r'\compile3045.txt', 'w',
        encoding='utf-8').write(out)
print(out)
