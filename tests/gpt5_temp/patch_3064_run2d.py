# -*- coding: utf-8 -*-
"""Patch 3064 run2d: move `del wud` after its last use.
Gmet = wud.T @ wud (line ~934) still needs wud; the del was
wrongly placed between project() calls and the G-metric.
"""
import io
import py_compile

P = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\phase3064_omega_p61_ds7b_chain_replication.py"

with io.open(P, "r", encoding="utf-8", newline="") as f:
    raw = f.read()

EOL = "\r\n" if "\r\n" in raw else "\n"


def L(*parts):
    return EOL.join(parts)


old1 = L(
    "Y_KP = project(D_KP)",
    "del wud  # free the 4.4 GB fp64 copy",
)
new1 = L(
    "Y_KP = project(D_KP)",
)

old2 = L(
    "Gmet = wud.T @ wud  # (HID, HID) fp64",
    "rng_ax = np.random.default_rng(9999)",
)
new2 = L(
    "Gmet = wud.T @ wud  # (HID, HID) fp64",
    "del wud  # free the 4.4 GB fp64 copy (last use)",
    "rng_ax = np.random.default_rng(9999)",
)

n1 = raw.count(old1)
n2 = raw.count(old2)
print("count old1 =", n1)
print("count old2 =", n2)
assert n1 == 1, "old1 count != 1: %d" % n1
assert n2 == 1, "old2 count != 1: %d" % n2

patched = raw.replace(old1, new1).replace(old2, new2)
assert "del wud" in patched
assert patched.count("del wud") == 1, \
    "del wud must appear exactly once"

with io.open(P, "w", encoding="utf-8", newline="") as f:
    f.write(patched)

py_compile.compile(P, doraise=True)
print("PATCH_OK COMPILE_OK")
