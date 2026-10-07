# -*- coding: utf-8 -*-
"""3062 run1 fix: re-insert the d_bar/raw decode
block that was lost to a phantom Edit, and
register the correction + run marker."""
import io

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3062_omega_p59_body_identity_decode_'
     r'qwen.py')
src = io.open(P, encoding='utf-8').read()

old = (
    "    self_u[lo:lo + ROT_CHUNK] = "
    "LU[:, tok_ids]\n"
    "del wud  # free the 3.1 GB fp64 copy\n")
new = (
    "    self_u[lo:lo + ROT_CHUNK] = "
    "LU[:, tok_ids]\n"
    "# descriptive decode of d_bar and raw B_b\n"
    "# (shared-component controls, BEFORE del)\n"
    "lg_dbar = wud @ d_bar\n"
    "top8_dbar = np.argsort(lg_dbar)[::-1][:8]\n"
    "logits_raw = (wud @ B_b.T).T  # (8, V) fp64\n"
    "top8_raw = np.argsort(logits_raw,\n"
    "                      axis=1)[:, ::-1][:, :8]\n"
    "del wud  # free the 3.1 GB fp64 copy\n")
assert src.count(old) == 1, src.count(old)
src = src.replace(old, new)

old2 = ("    'corrections': 'none (run1 "
        "authoritative if '\n"
        "                   'anchors pass)',\n")
new2 = ("    'corrections': 'run1 (35s) crashed '\n"
        "                   'pre-verdict at the '\n"
        "                   'decode report: the '\n"
        "                   'd_bar/raw B_b '\n"
        "                   'descriptive decode '\n"
        "                   'block was lost to a '\n"
        "                   'phantom Edit (report '\n"
        "                   'section landed, the '\n"
        "                   'computing block did '\n"
        "                   'not - local phantom-'\n"
        "                   'edit defect); '\n"
        "                   'NameError top8_dbar '\n"
        "                   'after T2 was logged '\n"
        "                   '(count_margin=0, '\n"
        "                   'count_self=1, T3-T5 '\n"
        "                   'unreached). Fix: '\n"
        "                   'block re-inserted '\n"
        "                   'via a Python patch '\n"
        "                   'with a count==1 '\n"
        "                   'assert before del '\n"
        "                   'wud. run2 '\n"
        "                   'authoritative if '\n"
        "                   'anchors pass.',\n")
assert src.count(old2) == 1, src.count(old2)
src = src.replace(old2, new2)

old3 = "'run': 'run1 authoritative (fp32 '"
new3 = "'run': 'run2 authoritative (fp32 '"
assert src.count(old3) == 1, src.count(old3)
src = src.replace(old3, new3)

io.open(P, 'w', encoding='utf-8').write(src)
print('patched ok')
