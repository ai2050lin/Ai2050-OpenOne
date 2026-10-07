# -*- coding: utf-8 -*-
"""Patch 3065 run3: select the same subset on both
sides of the geometry correlation."""
import io
import py_compile

P = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\phase3065_omega_p62_v_sign_orchestration.py"

with io.open(P, "r", encoding="utf-8", newline="") as f:
    raw = f.read()

old = (
    "        r_g, p_g = pearson_perm(\n"
    "            med_cV[sel], med_c, N_PERM_R, 3021)\n"
)
if old not in raw:
    old = old.replace("\n", "\r\n")

new = old.replace("med_cV[sel], med_c,",
                  "med_cV[sel], med_c[sel],")

n = raw.count(old)
print("count old =", n)
assert n == 1, "expected 1, got %d" % n

patched = raw.replace(old, new)
assert patched.count("med_c[sel]") == 1

with io.open(P, "w", encoding="utf-8", newline="") as f:
    f.write(patched)

py_compile.compile(P, doraise=True)
print("PATCH_OK COMPILE_OK")
