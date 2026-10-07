# -*- coding: utf-8 -*-
"""Patch 3064 run2b: fix DynamicCache access for transformers 5.14.1.
Only viable path: cache.layers[li].keys  (probe confirmed).
Both b5 blocks (out_r / out_p) replaced; count==1 asserted each.
"""
import io
import py_compile

P = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\phase3064_omega_p61_ds7b_chain_replication.py"

with io.open(P, "r", encoding="utf-8", newline="") as f:
    raw = f.read()

EOL = "\r\n" if "\r\n" in raw else "\n"


def L(*parts):
    return EOL.join(parts)


old_r = L(
    "_kc = getattr(out_r.past_key_values,",
    "              'key_cache', None)",
    "_kl = _kc[L_TGT] if _kc is not None \\",
    "    else out_r.past_key_values[L_TGT][0]",
    "pk_r = _kl[0, :, rows5[0]].double() \\",
)
new_r = L(
    "# transformers 5.x: only cache.layers[li].keys"
    " (probe 2026-09-21)",
    "_kl = out_r.past_key_values.layers[L_TGT].keys",
    "pk_r = _kl[0, :, rows5[0]].double() \\",
)

old_p = L(
    "_kc = getattr(out_p.past_key_values,",
    "              'key_cache', None)",
    "_kl = _kc[L_TGT] if _kc is not None \\",
    "    else out_p.past_key_values[L_TGT][0]",
    "pk_p = _kl[0, :, off5 + rows5[0]] \\",
)
new_p = L(
    "_kl = out_p.past_key_values.layers[L_TGT].keys",
    "pk_p = _kl[0, :, off5 + rows5[0]] \\",
)

n_r = raw.count(old_r)
n_p = raw.count(old_p)
print("count old_r =", n_r)
print("count old_p =", n_p)
assert n_r == 1, "old_r count != 1: %d" % n_r
assert n_p == 1, "old_p count != 1: %d" % n_p

patched = raw.replace(old_r, new_r).replace(old_p, new_p)
assert patched.count("layers[L_TGT].keys") == 2, \
    "expected exactly 2 new access lines"
assert "key_cache" not in patched, \
    "stale key_cache reference remains"

with io.open(P, "w", encoding="utf-8", newline="") as f:
    f.write(patched)

py_compile.compile(P, doraise=True)
print("PATCH_OK COMPILE_OK EOL=%r" % EOL)
