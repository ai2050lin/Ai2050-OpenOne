# -*- coding: utf-8 -*-
import io
import re
import py_compile

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3045_omega_p42_l20_axis_anatomy_'
     r'qwen.py')
s = io.open(P, encoding='utf-8').read()

pat = re.compile(
    r"    'corrections': 'run1 \(bf16\) crashed.*?"
    r"documented as-is per prereg',\n",
    re.S)
new = (
    "    'corrections': 'run1 (bf16) crashed pre-'\n"
    "                   'verdict on RND indexing AND '\n"
    "                   'failed a112; probe '\n"
    "                   'established the bf16 flip-'\n"
    "                   'noise floor and the 3044 T5 '\n"
    "                   'scale mismatch; run2 = fp32 '\n"
    "                   'redesign; run2 completed but '\n"
    "                   'T3a had a reduction-axis bug '\n"
    "                   '(norm over the wrong axis of '\n"
    "                   'the (NVOC,24) matrix -> per-'\n"
    "                   'coordinate norms ~0.001, '\n"
    "                   'inflated ratios 66/31 that '\n"
    "                   'violate the sigma_max bound); '\n"
    "                   'T3a fixed to axis=0 and fully '\n"
    "                   'rerun as run3; T1/T2/T3b/T3c/'\n"
    "                   'T4/T5 unaffected (reproduced '\n"
    "                   'identically); a111 (0.99879 < '\n"
    "                   '0.999) and the a113 med_col '\n"
    "                   'gate (body0 L20 med_col 0.0150 '\n"
    "                   'is REAL linear signal at the '\n"
    "                   'measured L20 gain 0.052 per '\n"
    "                   'unit, threshold miscalibrated) '\n"
    "                   'documented as-is per prereg',\n")
s2, n = pat.subn(new, s)
assert n == 1, n
io.open(P, 'w', encoding='utf-8').write(s2)
py_compile.compile(P, doraise=True)

out = 'patch3045f ok: corrections block rewritten, compile zero errors'
io.open(r'D:\AI2050\Ai2050-OpenOne\gpt5_temp'
        r'\patch3045f_result.txt', 'w',
        encoding='utf-8').write(out)
print(out)
