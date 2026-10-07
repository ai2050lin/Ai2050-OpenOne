# -*- coding: utf-8 -*-
"""Probe for phase 3098 sealed npz:
is KL[:, no_mlp] == KL[:, skip] exactly?
And dump GMED group medians for the
memo table.  Writes report to txt."""
import io
import json

import numpy as np

OD = (r'D:\AI2050\Ai2050-OpenOne'
      r'\tests\glm5\result'
      r'\rdc_query_construction_20260913'
      r'\phase3098'
      r'\omega_p96_lastblock_substep_'
      r'anatomy')
z = np.load(OD + r'\omega_p96_lastblock_'
            r'substep_anatomy.npz',
            allow_pickle=False)
o = []
o.append('verdict=%s' % str(z['VERDICT']))
FKEYS = ('A', 'B', 'C')
for side in ('4B', '14B'):
    for fa in FKEYS:
        kl = z['KL_%s_%s' % (side, fa)]
        t1 = z['T1_%s_%s' % (side, fa)]
        d12 = float(np.max(np.abs(
            kl[:, 1] - kl[:, 2])))
        o.append('%s %s: max|KL[no_mlp]'
                 '-KL[skip]|=%.3e '
                 't1eq=%s'
                 % (side, fa, d12,
                    str(bool(np.all(
                        t1[:, 1]
                        == t1[:, 2])))))
    for key in ('AB', 'AC', 'BC'):
        f2i = z['F2I_%s_%s' % (side, key)]
        d12 = float(np.max(np.abs(
            f2i[:, 1] - f2i[:, 2])))
        o.append('%s %s: max|F2I[no_mlp]'
                 '-F2I[skip]|=%.3e'
                 % (side, key, d12))
o.append('')
o.append('GMED [pre, att, post] per group')
for side in ('4B', '14B'):
    for key in ('AB', 'AC', 'BC'):
        for ci in (1, 2, 3):
            g = z['GMED_%s_%s_ci%d'
                  % (side, key, ci)]
            o.append('%s %s_ci%d: '
                     'pre=%.3f att=%.3f '
                     'post=%.3f '
                     '(attn %+.3f mlp %+.3f)'
                     % (side, key, ci,
                        g[0], g[1], g[2],
                        g[1] - g[0],
                        g[2] - g[1]))
o.append('')
o.append('F2S substep f2 raw check '
         '(first 3 k per key, 14B)')
for key in ('AB', 'AC', 'BC'):
    f2s = z['F2S_14B_%s' % key]
    for k in (0, 8, 16):
        o.append('14B %s k=%d: %.3f %.3f '
                 '%.3f' % (key, k,
                           f2s[k, 0],
                           f2s[k, 1],
                           f2s[k, 2]))
with io.open(OD + r'\probe_report.txt',
             'w', encoding='utf-8') as f:
    f.write('\n'.join(o) + '\n')
print('PROBE_DONE')
