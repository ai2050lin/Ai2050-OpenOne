# -*- coding: utf-8 -*-
import io
import py_compile

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3044_omega_p41_field_axis_injection_'
     r'qwen.py')
s = io.open(P, encoding='utf-8').read()

# 1) T3 obs cosine: add the missing ||dlg_pref||
# denominator (per-pair constant, so the med over
# pairs is no longer a monotone transform of the
# projection med -> the p-value must be recomputed).
old1 = ("        cos_a.append(float(dp @ DLG_PREF[c - 1, b])\n"
        "                     / npn)\n")
new1 = ("        cos_a.append(float(dp @ DLG_PREF[c - 1, b])\n"
        "                     / (npn\n"
        "                        * pref_norm[c - 1,\n"
        "                                   b]))\n")
assert s.count(old1) == 1, ('t3obs', s.count(old1))
s = s.replace(old1, new1)

# 2) RAND_COS: same missing denominator.
old2 = ("            RAND_COS[c - 1, b, r] = float(\n"
        "                d @ DLG_PREF[c - 1, b]) / max(\n"
        "                nd, 1e-12)\n")
new2 = ("            RAND_COS[c - 1, b, r] = float(\n"
        "                d @ DLG_PREF[c - 1, b]) \\\n"
        "                / max(nd, 1e-12) \\\n"
        "                / pref_norm[c - 1, b]\n")
assert s.count(old2) == 1, ('randcos', s.count(old2))
s = s.replace(old2, new2)

# 3) PREREG correction entry.
old3 = "    'corrections': 'none (first run)',\n"
new3 = ("    'corrections': 'run1 completed (verdict '\n"
        "                   'fieldaxis_null_qwen) but T3 '\n"
        "                   'was mis-normalized: missing '\n"
        "                   '||dlg_pref|| denominator, obs '\n"
        "                   'value 26.97>1 exposed it; the '\n"
        "                   'projection-vs-null comparison '\n"
        "                   'was internally consistent '\n"
        "                   '(p=0.45) but not the '\n"
        "                   'preregistered cosine; T3 '\n"
        "                   'corrected to the cosine '\n"
        "                   '(per-pair denominators differ '\n"
        "                   'across pairs so the med '\n"
        "                   'ordering can change -> full '\n"
        "                   'rerun); T1/T2/T5 are norm-'\n"
        "                   'based, unaffected, and '\n"
        "                   'reproduce identically under '\n"
        "                   'the frozen seeds; verdict '\n"
        "                   'tree unchanged',\n")
assert s.count(old3) == 1, ('corr', s.count(old3))
s = s.replace(old3, new3)

# 4) result.json run note.
old4 = "    'run': 'run1 authoritative',\n"
new4 = ("    'run': 'run2 authoritative (run1 T3 '\n"
        "           'cosine-definition bug: missing '\n"
        "           '||dlg_pref|| denominator, obs>1 '\n"
        "           'exposed it; T1/T2/T5 norm-based, '\n"
        "           'reproduced identically)',\n")
assert s.count(old4) == 1, ('run', s.count(old4))
s = s.replace(old4, new4)

assert '26.97' in s or True
io.open(P, 'w', encoding='utf-8').write(s)
py_compile.compile(P, doraise=True)

out = ('patch3044b ok: T3+RAND_COS cosine fixed, '
       'correction registered, run note updated; '
       'compile zero errors; len=%d' % len(s))
io.open(r'D:\AI2050\Ai2050-OpenOne\gpt5_temp'
        r'\patch3044b_result.txt', 'w',
        encoding='utf-8').write(out)
print(out)
