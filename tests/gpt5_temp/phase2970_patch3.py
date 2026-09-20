# -*- coding: utf-8 -*-
"""Phase 2970 patch3: fix C_flat transpose (h before w)."""
import ast
import io
import shutil
import os

P = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\phase2970_delay_carrier_localization.py'
t = io.open(P, encoding='utf-8').read()

old = """        # ---------- T1 head level ----------
        # per (l,h) word curves; flatten to (nlh, nS, n_words)
        C_flat = C_all.transpose(1, 0, 2, 3).reshape(
            NL * NH, nS, n_words)"""
new = """        # ---------- T1 head level ----------
        # per (l,h) word curves; k = l*NH + h rows are (nS,
        # n_words) matrices -> transpose must put NH before
        # n_words (2969 identity: d[34,15] must reproduce
        # +0.5132; a wrong reshape folds the NH axis into the
        # element stream and yields noise)
        C_flat = C_all.transpose(1, 0, 3, 2).reshape(
            NL * NH, nS, n_words)"""
assert old in t, 'transpose block not found'
t = t.replace(old, new, 1)
io.open(P, 'w', encoding='utf-8').write(t)
t2 = io.open(P, encoding='utf-8').read()
ast.parse(t2)
res = ['transpose fixed: %s' % ('transpose(1, 0, 3, 2)' in t2)]
d = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result' \
    r'\rdc_query_construction_20260913\phase2970'
if os.path.exists(d):
    shutil.rmtree(d)
res.append('phase2970 cleaned: %s' % (not os.path.exists(d)))
io.open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
        r'\p2970_fix3.txt', 'w',
        encoding='utf-8').write('\n'.join(res) + '\n')
print('patched')
