# -*- coding: utf-8 -*-
import io

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3001_omega_g1_robustness_source_glm4.py')
t = io.open(P, encoding='utf-8').read()

a1 = ("        t2 = {}\n"
      "        for li in SEL_LAYERS:")
b1 = ("        t2 = {}\n"
      "        raw_x = {}\n"
      "        for li in SEL_LAYERS:")
assert a1 in t, 'a1'
t = t.replace(a1, b1, 1)

a2 = ("            t2['L%d' % li] = {'xdir': {\n"
      "                'sep': round(sep_x, 2),\n"
      "                'ratio': round(ratio_x, 4)}}")
b2 = ("            raw_x[li] = ratio_x\n"
      "            t2['L%d' % li] = {'xdir': {\n"
      "                'sep': round(sep_x, 2),\n"
      "                'ratio': round(ratio_x, 4)}}")
assert a2 in t, 'a2'
t = t.replace(a2, b2, 1)

a3 = ("        a9_parts = {\n"
      "            'ratio19': abs(\n"
      "                t2['L19']['xdir']['ratio']\n"
      "                - REF_2999['ratio19']),")
b3 = ("        a9_parts = {\n"
      "            'ratio19': abs(\n"
      "                raw_x[REF_LAYER]\n"
      "                - REF_2999['ratio19']),")
assert a3 in t, 'a3'
t = t.replace(a3, b3, 1)

a4 = "        'correction_note': 'first run',"
b4 = ("        'correction_note':\n"
      "            'run1: crashed at xdir_coords matmul '\n"
      "            '(xdir is per-cell (98,4096)); prereg T1 '\n"
      "            'redesigned BEFORE any verdict: strictly-'\n"
      "            'opposite pairing made the ctx-label sep '\n"
      "            'vacuously -sep_D (always-true identity, '\n"
      "            'banned); replaced with seeded random '\n"
      "            'partner + additive 2x2 decomposition; '\n"
      "            'run2: SyntaxError paren mismatch, never '\n"
      "            'executed; run3: stale xdir_coords left '\n"
      "            'on disk by a phantom edit, crashed '\n"
      "            'post-T1; run4: full pass but a9 '\n"
      "            'ratio19 compared the 4dp-rounded scan '\n"
      "            'value against the full-precision 2999 '\n"
      "            'reference (diff 4.66e-5 = pure '\n"
      "            'rounding vs gate 1e-6) - anchor '\n"
      "            'comparison bug, verdict void by '\n"
      "            'protocol; run5: raw ratio retained '\n"
      "            'for a9; authoritative',")
assert a4 in t, 'a4'
t = t.replace(a4, b4, 1)

io.open(P, 'w', encoding='utf-8').write(t)
t2 = io.open(P, encoding='utf-8').read()
assert 'raw_x[REF_LAYER]' in t2
assert 'raw_x[li] = ratio_x' in t2
assert 'run5: raw ratio' in t2
print('patched ok')
