# -*- coding: utf-8 -*-
"""3153 patch1: M1 数值锚加 SMOKE 守卫（SMOKE 下 ALS iters 与 3152 smoke 不同）。"""
import io

P = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\phase3153_g1p3_failure_mode_anatomy.py"
s = io.open(P, encoding="utf-8").read()

old1 = "    if mk.get('margins') and mk['margins'][0] is not None:\n"
new1 = "    if (not SMOKE) and mk.get('margins') and mk['margins'][0] is not None:\n"
assert s.count(old1) == 1, ("anchor1 count", s.count(old1))
s = s.replace(old1, new1)

old2 = "    if g3152 is not None:\n"
new2 = "    if (not SMOKE) and g3152 is not None:\n"
assert s.count(old2) == 1, ("anchor2 count", s.count(old2))
s = s.replace(old2, new2)

io.open(P, "w", encoding="utf-8", newline="").write(s)
print("patched 2 guards")
