# -*- coding: utf-8 -*-
import io
import py_compile

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3045_omega_p42_l20_axis_anatomy_'
     r'qwen.py')
s = io.open(P, encoding='utf-8').read()

# 1) T3a eff_rnd: reduce over the NVOC axis (axis=0),
# not the direction axis (axis=1). run2 produced
# per-coordinate norms (~0.001) -> inflated ratios 66/31.
old1 = ("        eff_ax = float(np.linalg.norm(J @ axis))\n"
        "        eff_rnd = np.linalg.norm(J @ rnds.T,\n"
        "                                 axis=1)\n")
new1 = ("        eff_ax = float(np.linalg.norm(J @ axis))\n"
        "        eff_rnd = np.linalg.norm(J @ rnds.T,\n"
        "                                 axis=0)\n")
assert s.count(old1) == 1, ('t3a', s.count(old1))
s = s.replace(old1, new1)

# 2) register the correction.
old2 = ("                   'redefined same-scale; verdict '\n"
        "                   'tree restructured (T1 now the '\n"
        "                   'primary spec test at natural '\n"
        "                   'scale)',\n")
new2 = ("                   'redefined same-scale; verdict '\n"
        "                   'tree restructured (T1 now the '\n"
        "                   'primary spec test at natural '\n"
        "                   'scale); run2 completed but T3a '\n"
        "                   'had a reduction-axis bug '\n"
        "                   '(norm over axis=1 of (NVOC,24) '\n"
        "                   -> per-coordinate norms ~0.001, '\n"
        "                   inflated ratios 66/31 violating '\n"
        "                   the sigma_max bound); T3a fixed '\n"
        "                   to axis=0 and fully rerun as '\n"
        "                   run3; T1/T2/T3b/T3c/T4/T5 '\n"
        "                   unaffected (reproduced '\n"
        "                   identically); a111 (0.99879 < '\n"
        "                   0.999) and a113 med_col gate '\n"
        "                   (body0 L20 med_col 0.0150 is '\n"
        "                   REAL linear signal at the '\n"
        "                   measured L20 gain 0.052/unit, '\n"
        "                   threshold miscalibrated) '\n"
        "                   documented as-is per prereg',\n")
assert s.count(old2) == 1, ('corr', s.count(old2))
s = s.replace(old2, new2)

# 3) run note.
old3 = ("    'run': 'run2 authoritative (fp32 redesign; run1 '\n"
        "           'bf16 crashed pre-verdict: RND indexing '\n"
        "           'bug + a112 FAIL exposing the 3044 T5 '\n"
        "           'scale mismatch and the bf16 flip-noise '\n"
        "           'floor, see run1_findings)',\n")
new3 = ("    'run': 'run3 authoritative (fp32; run2 T3a '\n"
        "           'reduction-axis bug fixed; T1/T2/T3b/'\n"
        "           'T3c/T4/T5 reproduced identically; '\n"
        "           'run1 bf16 crashed pre-verdict, see '\n"
        "           'run1_findings)',\n")
assert s.count(old3) == 1, ('run', s.count(old3))
s = s.replace(old3, new3)

io.open(P, 'w', encoding='utf-8').write(s)
py_compile.compile(P, doraise=True)
out = ('patch3045c ok: T3a axis=0 fixed, correction '
       'registered, run note updated; compile zero '
       'errors')
io.open(r'D:\AI2050\Ai2050-OpenOne\gpt5_temp'
        r'\patch3045c_result.txt', 'w',
        encoding='utf-8').write(out)
print(out)
