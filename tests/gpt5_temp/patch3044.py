# -*- coding: utf-8 -*-
import io
import py_compile

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3044_omega_p41_field_axis_injection_'
     r'qwen.py')
s = io.open(P, encoding='utf-8').read()

old = ("                lg_i, v3p, v20p = forward_inj(\n"
       "                    ids, 3, pos, delta,\n"
       "                    full=(s == 1.0\n"
       "                          and probe_pending()))\n")
new = ("                lg_i, v3p, v20p = forward_inj(\n"
       "                    ids, 3, pos, delta)\n")
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new)

assert 'probe_pending' not in s, 'probe_pending left'
assert 'if False' not in s, 'if False left'
assert 'None  #' not in s, 'placeholder left'
assert '__wrapped__' not in s, 'wrapped left'

io.open(P, 'w', encoding='utf-8').write(s)
py_compile.compile(P, doraise=True)

out = ('patched ok; probe_pending removed; compile '
       'zero errors; len=%d' % len(s))
io.open(r'D:\AI2050\Ai2050-OpenOne\gpt5_temp'
        r'\patch3044_result.txt', 'w',
        encoding='utf-8').write(out)
print(out)
