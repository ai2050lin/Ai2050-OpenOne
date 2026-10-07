# -*- coding: utf-8 -*-
"""Patch 3065 run2: detach tk_rows indexing of the
grad-requiring lm_head weight."""
import io
import py_compile

P = r"D:\AI2050\Ai2050-OpenOne\tests\glm5\phase3065_omega_p62_v_sign_orchestration.py"

with io.open(P, "r", encoding="utf-8", newline="") as f:
    raw = f.read()

old = (
    "    tk_rows = Wemb[torch.tensor(\n"
    "        tgt_ids, device='cuda')].double() \\\n"
    "        .cpu().numpy()\n"
)
# tolerate CRLF
if old not in raw:
    old = old.replace("\n", "\r\n")

new = old.replace(
    "device='cuda')].double()",
    "device='cuda')].detach().double()")

n = raw.count(old)
print("count old =", n)
assert n == 1, "expected 1 occurrence, got %d" % n

patched = raw.replace(old, new)
assert patched.count("].detach().double()") == 1

with io.open(P, "w", encoding="utf-8", newline="") as f:
    f.write(patched)

py_compile.compile(P, doraise=True)
print("PATCH_OK COMPILE_OK")
