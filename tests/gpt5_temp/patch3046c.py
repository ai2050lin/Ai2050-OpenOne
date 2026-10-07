# -*- coding: utf-8 -*-
import io
import py_compile

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3046_omega_p43_kfield_injection_qwen.py')
s = io.open(P, encoding='utf-8').read()

old = ("        s = (PREFIXES[ci] + ' ' + NEW_BODIES[bi]) \\\n"
       "            if PREFIXES[ci] else BODIES[bi]")
new = ("        s = (PREFIXES[ci] + ' ' + NEW_BODIES[bi]) \\\n"
       "            if PREFIXES[ci] else NEW_BODIES[bi]")
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new)

old2 = "    'corrections': 'none (run1 authoritative '\n                   'candidate)',"
new2 = ("    'corrections': 'run1 crashed pre-anchor "
        "at prompt '\n                   'assembly: the NEW-bodies loop "
        "else-'\n                   'branch used BODIES[bi] "
        "(transcription '\n                   'slip; 3045 line 409 "
        "correctly reads '\n                   'NEW_BODIES[bi]); NEW b3 "
        "cond0 target '\n                   'although had count 0 in "
        "the wrong '\n                   'sentence; fixed to "
        "NEW_BODIES[bi]; no '\n                   'anchor or statistic "
        "observed before '\n                   'the crash; run2 "
        "authoritative '\n                   'candidate',")
assert s.count(old2) == 1, s.count(old2)
s = s.replace(old2, new2)
io.open(P, 'w', encoding='utf-8').write(s)

py_compile.compile(P, doraise=True)
print('ok')
