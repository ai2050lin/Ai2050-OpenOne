# -*- coding: utf-8 -*-
"""p3147 patch3: co50ex size = 50
(co50 and co36 are disjoint, 3135)."""
import io

FP = (r'D:\AI2050\Ai2050-OpenOne\tests'
      r'\glm5\phase3147_omega_p145_'
      r'tbsym_co50exk_negmech_v1micro.py')
t = io.open(FP, encoding='utf-8').read()

old1 = """assert co50ex.ndim == 1
assert len(co50ex) == 25"""
new1 = """assert co50ex.ndim == 1
assert len(co50ex) == 50, len(co50ex)
assert len(set(co50ex.tolist())
           & set(co36.tolist())) == 0, \\
    'co50ex must be disjoint from co36' \\
    ' (3135)'"""
c1 = t.count(old1)
assert c1 == 1, ('fix1', c1)
t = t.replace(old1, new1)

io.open(FP, 'w', encoding='utf-8').write(t)
chk = io.open(FP, encoding='utf-8').read()
assert 'assert len(co50ex) == 50' in chk
assert 'must be disjoint from co36' in chk
import py_compile
py_compile.compile(FP, doraise=True)
print('PATCH3 OK')
