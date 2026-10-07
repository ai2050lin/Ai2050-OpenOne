# -*- coding: utf-8 -*-
import io
import py_compile

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3046_omega_p43_kfield_injection_qwen.py')
s = io.open(P, encoding='utf-8').read()

old1 = ("        dlg_a, _, _ = forward_inj(\n"
        "            ids, li, pos, nrm * axis)")
new1 = ("        dlg_a, _ = forward_inj(\n"
        "            ids, li, pos, nrm * axis)")
assert s.count(old1) == 1, ('old1', s.count(old1))
s = s.replace(old1, new1)

old2 = ("            dlg_r, _, _ = forward_inj(\n"
        "                ids, li, pos, nrm * R[rd])")
new2 = ("            dlg_r, _ = forward_inj(\n"
        "                ids, li, pos, nrm * R[rd])")
assert s.count(old2) == 1, ('old2', s.count(old2))
s = s.replace(old2, new2)

old3 = "    'run': 'run4 authoritative (pre-norm redesign '"
new3 = "    'run': 'run5 authoritative (pre-norm redesign '"
assert s.count(old3) == 1, ('old3', s.count(old3))
s = s.replace(old3, new3)

old4 = "'corrections': 'run1 crashed pre-anchor at '"
new4 = ("'corrections': 'run4 crashed in T5: "
        "forward_inj '\n                   'was reduced to a "
        "2-value return in the '\n                   'run4 rewrite "
        "but the T5 unpacks kept 3 '\n                   'values; "
        "T1pre/T1post/T2/T3a/T3b/T4 were '\n                   '"
        "observed in run4 and reproduce under '\n                   '"
        "frozen seeds; unpacks fixed; run5 '\n                   '"
        "authoritative; run1 crashed pre-anchor at '")
assert s.count(old4) == 1, ('old4', s.count(old4))
s = s.replace(old4, new4)

io.open(P, 'w', encoding='utf-8').write(s)
py_compile.compile(P, doraise=True)
print('ok')
