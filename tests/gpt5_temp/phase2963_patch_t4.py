# -*- coding: utf-8 -*-
# Phase 2963 patch: T4 prediction on raw-tid scale.
import ast
import io

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase2963_frequency_controlled_band.py')
t = io.open(P, encoding='utf-8').read()

old = (
    "        xr = rankdata(tid_test[ix_c].astype(np.float64))\n"
    "        xr = (xr - xr.mean()) / xr.std()\n"
    "        beta_c, *_ = np.linalg.lstsq(\n"
    "            np.stack([np.ones(len(xr)), xr], axis=1),\n"
    "            B_all[ix_c], rcond=None)\n"
    "        xf = rankdata(tid_test[mF].astype(np.float64))\n"
    "        xf = (xf - np.mean(rankdata(\n"
    "            tid_test.astype(np.float64)))) \\\n"
    "            / np.std(rankdata(\n"
    "                tid_test.astype(np.float64)))\n"
    "        pred_F = beta_c[0] + beta_c[1] * xf\n")
new = (
    "        xc = tid_test[ix_c].astype(np.float64)\n"
    "        xc = (xc - xc.mean()) / xc.std()\n"
    "        beta_c, *_ = np.linalg.lstsq(\n"
    "            np.stack([np.ones(len(xc)), xc], axis=1),\n"
    "            B_all[ix_c], rcond=None)\n"
    "        xfp = (tid_test[mF].astype(np.float64)\n"
    "               - tid_test[ix_c].mean()) \\\n"
    "            / tid_test[ix_c].std()\n"
    "        pred_F = beta_c[0] + beta_c[1] * xfp\n")

assert old in t, 'T4 block not found'
t = t.replace(old, new, 1)
io.open(P, 'w', encoding='utf-8').write(t)

t2 = io.open(P, encoding='utf-8').read()
try:
    ast.parse(t2)
    r = 'syntax OK'
except SyntaxError as e:
    r = 'SYNTAX ERROR line %s' % e.lineno
out = io.open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
              r'\p2963_syntax.txt', 'w', encoding='utf-8')
out.write('patch applied: %s\n%s\nrank-xf leftover: %d\n'
          % ('xfp' in t2, r,
             t2.count('rankdata(tid_test[mF]')))
out.close()
print('patched')
