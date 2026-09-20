# -*- coding: utf-8 -*-
"""Phase 2970 patch2: fix T1 grouped maxT broadcast."""
import ast
import io
import shutil
import os

P = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\phase2970_delay_carrier_localization.py'
t = io.open(P, encoding='utf-8').read()

old = """                for L, ks in groups.items():
                    D = np.stack([d_arr[k] for k in ks])
                    means = np.abs(
                        (sg[:, :L] * D[None, :, :])
                        .mean(axis=2))
                    worst = np.maximum(worst,
                                       means.max(axis=1))"""
new = """                for L, ks in groups.items():
                    D = np.stack([d_arr[k] for k in ks])
                    means = np.abs(
                        (sg[:, None, :L] * D[None, :, :])
                        .mean(axis=2))
                    worst = np.maximum(worst,
                                       means.max(axis=1))"""
assert old in t, 'broadcast block not found'
t = t.replace(old, new, 1)
io.open(P, 'w', encoding='utf-8').write(t)
t2 = io.open(P, encoding='utf-8').read()
ast.parse(t2)
res = ['fixed: %s' % ('sg[:, None, :L]' in t2)]
d = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result' \
    r'\rdc_query_construction_20260913\phase2970'
if os.path.exists(d):
    shutil.rmtree(d)
res.append('phase2970 cleaned: %s' % (not os.path.exists(d)))
io.open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
        r'\p2970_fix2.txt', 'w',
        encoding='utf-8').write('\n'.join(res) + '\n')
print('patched')
