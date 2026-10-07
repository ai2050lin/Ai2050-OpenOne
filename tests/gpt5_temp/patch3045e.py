# -*- coding: utf-8 -*-
import io
import py_compile

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3045_omega_p42_l20_axis_anatomy_'
     r'qwen.py')
s = io.open(P, encoding='utf-8').read()

fixes = [
    ("                   inflated ratios 66/31 violating '\n",
     "                   'inflated ratios 66/31 violating '\n"),
    ("                   the sigma_max bound); T3a fixed '\n",
     "                   'the sigma_max bound); T3a fixed '\n"),
    ("                   to axis=0 and fully rerun as '\n",
     "                   'to axis=0 and fully rerun as '\n"),
]
for old, new in fixes:
    assert s.count(old) == 1, (old[:40], s.count(old))
    s = s.replace(old, new)

io.open(P, 'w', encoding='utf-8').write(s)
py_compile.compile(P, doraise=True)

out = 'patch3045e ok: quotes fixed, compile zero errors'
io.open(r'D:\AI2050\Ai2050-OpenOne\gpt5_temp'
        r'\patch3045e_result.txt', 'w',
        encoding='utf-8').write(out)
print(out)
