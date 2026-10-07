# -*- coding: utf-8 -*-
"""3062 run2 fix: T3 row indexing (c-major,
c_idx in 0..2) + correction + run marker."""
import io

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3062_omega_p59_body_identity_decode_'
     r'qwen.py')
src = io.open(P, encoding='utf-8').read()

old = ("        rows = [c * 8 + b2 for c in (1, 2, 3)]\n")
new = ("        rows = [ci * 8 + b2\n"
       "                for ci in range(3)]\n")
assert src.count(old) == 1, src.count(old)
src = src.replace(old, new)

old2 = ("                   'wud. run2 '\n"
        "                   'authoritative if '\n"
        "                   'anchors pass.',\n")
new2 = ("                   'wud. run2 '\n"
        "                   '(34s) crashed pre-'\n"
        "                   'verdict at T3: row '\n"
        "                   'indexing - D_TAN/'\n"
        "                   'Vmat rows are c-'\n"
        "                   'major with c_idx '\n"
        "                   '0..2 (k = c_idx*8 '\n"
        "                   '+ b), the loop '\n"
        "                   'used c in 1..3 '\n"
        "                   'giving k up to 31 '\n"
        "                   '(IndexError 24); '\n"
        "                   'rows for body b '\n"
        "                   'are b, 8+b, 16+b.'\n"
        "                   ' Fix applied. '\n"
        "                   'run3 authoritative '\n"
        "                   'if anchors pass.',\n")
assert src.count(old2) == 1, src.count(old2)
src = src.replace(old2, new2)

old3 = "'run': 'run2 authoritative (fp32 '"
new3 = "'run': 'run3 authoritative (fp32 '"
assert src.count(old3) == 1, src.count(old3)
src = src.replace(old3, new3)

io.open(P, 'w', encoding='utf-8').write(src)
print('patched ok')
