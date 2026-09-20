# -*- coding: utf-8 -*-
"""Phase 2970 patch4: correct C_flat transpose + promote a15."""
import ast
import io
import shutil
import os

P = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\phase2970_delay_carrier_localization.py'
t = io.open(P, encoding='utf-8').read()

old = """        C_flat = C_all.transpose(1, 0, 3, 2).reshape(
            NL * NH, nS, n_words)"""
new = """        C_flat = C_all.transpose(1, 3, 0, 2).reshape(
            NL * NH, nS, n_words)"""
assert old in t, 'transpose not found'
t = t.replace(old, new, 1)

# promote d3415 identity to anchor a15 (gates the whole T1)
old2 = """        d3415_diff = abs(float(d3415) - float(
            r69['T2_paired_lang']['mean_d_L_minus_en']))"""
new2 = """        d3415_diff = abs(float(d3415) - float(
            r69['T2_paired_lang']['mean_d_L_minus_en']))
        a15_ok = bool(d3415_diff < 5.01e-4)
        log('a15 d[34,15] identity vs 2969 %.4f diff %.2e '
            'ok=%s' % (d3415, d3415_diff, a15_ok), lines)
        if not a15_ok:
            verdict = 'anchor_fail_all_void'
            t1 = {'invalidated': 'a15 transpose identity '
                                 'failed - head-level data '
                                 'corrupt'}
            t2 = dict(t2, invalidated=True)"""
assert old2 in t, 'a15 block not found'
t = t.replace(old2, new2, 1)

io.open(P, 'w', encoding='utf-8').write(t)
t2 = io.open(P, encoding='utf-8').read()
ast.parse(t2)
res = ['transpose fixed: %s' % ('transpose(1, 3, 0, 2)' in t2),
       'a15 gate landed: %s' % ('a15_ok' in t2)]
d = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result' \
    r'\rdc_query_construction_20260913\phase2970'
if os.path.exists(d):
    shutil.rmtree(d)
res.append('phase2970 cleaned: %s' % (not os.path.exists(d)))
io.open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
        r'\p2970_fix4.txt', 'w',
        encoding='utf-8').write('\n'.join(res) + '\n')
print('patched')
