# -*- coding: utf-8 -*-
"""Patch 3064 run2c: fix off-by-one in S4 ANOVA prefix index.
cidx holds 1-based condition IDs (1,2,3); C_c is stacked in
order c=(1,2,3) -> positional index must be cidx-1.
Replaces both occurrences; count==2 asserted.
"""
import io
import py_compile

P = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\phase3064_omega_p61_ds7b_chain_replication.py"

with io.open(P, "r", encoding="utf-8", newline="") as f:
    raw = f.read()

n = raw.count("C_c[cidx]")
print("count C_c[cidx] =", n)
assert n == 2, "expected 2 occurrences, got %d" % n

patched = raw.replace("C_c[cidx]", "C_c[cidx - 1]")
assert patched.count("C_c[cidx - 1]") == 2
assert "C_c[cidx]" not in patched

with io.open(P, "w", encoding="utf-8", newline="") as f:
    f.write(patched)

py_compile.compile(P, doraise=True)
print("PATCH_OK COMPILE_OK")
