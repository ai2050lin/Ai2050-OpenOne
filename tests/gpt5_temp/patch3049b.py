# -*- coding: utf-8 -*-
import io
import py_compile

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3049_omega_p46_kvload_localization_'
     r'qwen.py')
R = (r'D:\AI2050\Ai2050-OpenOne\gpt5_temp'
     r'\patch3049b_result.txt')
s = io.open(P, encoding='utf-8').read()

# 1) n_pr assert: 3049 assembles only the 32
#    old-body prompts (no NEW_BODIES bank)
old = 'assert n_old == 32 and n_pr == 48'
new = ('assert n_old == 32 and n_pr == 32  '
       '# 3049 drops the NEW_BODIES bank '
       '(old bodies only)')
assert s.count(old) == 1, ('a1', s.count(old))
s = s.replace(old, new)

# 2) a132 sampling: keep indices inside the
#    32 old-body rows of the z48 bank
old2 = 'for si in (0, 9, 31, 47):'
new2 = 'for si in (0, 9, 17, 31):'
assert s.count(old2) == 1, ('a2', s.count(old2))
s = s.replace(old2, new2)

# 3) corrections: register the run1 crash
old3 = "    'corrections': 'none (run1 authoritative)',"
new3 = ("    'corrections': 'run1 crashed pre-anchor "
        "on the assembly assertion: this phase "
        "assembles only the 32 old-body prompts "
        "(the NEW_BODIES bank of 3043-3048 is not "
        "used here) but the copied assertion still "
        "required n_pr==48, and one a132 sampling "
        "index pointed at a new-body row of the "
        "z48 bank; assertion corrected to n_pr==32 "
        "and sampling restricted to the old-body "
        "rows; run2 authoritative',")
assert s.count(old3) == 1, ('a3', s.count(old3))
s = s.replace(old3, new3)

# 4) run label
old4 = ("          'run': 'run1 authoritative (fp32; "
        "capture '")
new4 = ("          'run': 'run2 authoritative (fp32; "
        "run1 crashed pre-anchor on an assembly "
        "assertion, see corrections; capture '")
assert s.count(old4) == 1, ('a4', s.count(old4))
s = s.replace(old4, new4)

io.open(P, 'w', encoding='utf-8').write(s)
py_compile.compile(P, doraise=True)
io.open(R, 'w', encoding='utf-8').write('ok\n')
print('ok')
