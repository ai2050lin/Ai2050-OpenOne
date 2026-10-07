# -*- coding: utf-8 -*-
import io
import py_compile

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3046_omega_p43_kfield_injection_qwen.py')
s = io.open(P, encoding='utf-8').read()

old = ("                   'before the run1 crash',\n"
       "                   'assembly: the NEW-bodies loop else-'\n"
       "                   'branch used BODIES[bi] (transcription '\n"
       "                   'slip; 3045 line 409 correctly reads '\n"
       "                   'NEW_BODIES[bi]); NEW b3 cond0 target '\n"
       "                   'although had count 0 in the wrong '\n"
       "                   'sentence; fixed to NEW_BODIES[bi]; no '\n"
       "                   'anchor or statistic observed before '\n"
       "                   'the crash; run2 authoritative '\n"
       "                   'candidate',\n"
       "}")
new = "                   'before the run1 crash',\n}"
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new)
io.open(P, 'w', encoding='utf-8').write(s)

py_compile.compile(P, doraise=True)
print('ok')
