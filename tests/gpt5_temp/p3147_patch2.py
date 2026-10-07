# -*- coding: utf-8 -*-
"""p3147 patch2: co50ex list -> ndarray."""
import io

FP = (r'D:\AI2050\Ai2050-OpenOne\tests'
      r'\glm5\phase3147_omega_p145_'
      r'tbsym_co50exk_negmech_v1micro.py')
t = io.open(FP, encoding='utf-8').read()

old1 = """co50 = CO_SETS['co50']
co50ex = sorted(set(co50.tolist())
                - set(co36.tolist()))
score_ex = np.mean(fr[:, co50ex], axis=0)"""
new1 = """co50 = CO_SETS['co50']
co50ex = np.asarray(
    sorted(set(co50.tolist())
           - set(co36.tolist())),
    dtype=np.int64)
assert co50ex.ndim == 1
assert len(co50ex) == 25
score_ex = np.mean(fr[:, co50ex], axis=0)"""
c1 = t.count(old1)
assert c1 == 1, ('fix1', c1)
t = t.replace(old1, new1)

io.open(FP, 'w', encoding='utf-8').write(t)
chk = io.open(FP, encoding='utf-8').read()
assert 'co50ex = np.asarray(' in chk
assert 'np.argsort(-score_ex)' in chk
import py_compile
py_compile.compile(FP, doraise=True)
print('PATCH2 OK')
