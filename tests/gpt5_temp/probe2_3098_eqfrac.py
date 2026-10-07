# -*- coding: utf-8 -*-
"""Probe 2 for phase 3098: count
exact-equal fractions KL[:,1]==KL[:,2]
and F2I[:,1]==F2I[:,2], plus which k
differ.  Report to txt."""
import io

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
for side in ('4B', '14B'):
    tot_eq = 0
    tot = 0
    for fa in ('A', 'B', 'C'):
        kl = z['KL_%s_%s' % (side, fa)]
        eq = (kl[:, 1] == kl[:, 2])
        tot_eq += int(eq.sum())
        tot += len(eq)
        o.append('%s %s: KL eq %d/%d '
                 '(frac %.3f)'
                 % (side, fa,
                    int(eq.sum()),
                    len(eq),
                    float(eq.mean())))
    o.append('%s KL total eq frac: '
             '%.4f' % (side,
                       tot_eq / tot))
    for key in ('AB', 'AC', 'BC'):
        f2i = z['F2I_%s_%s'
                % (side, key)]
        d = np.abs(f2i[:, 1]
                   - f2i[:, 2])
        ne = (d > 0)
        o.append('%s %s: F2I diff>0 '
                 'count %d/24, diff vals '
                 '%s'
                 % (side, key,
                    int(ne.sum()),
                    ' '.join(
                        '%.3f' % v
                        for v in d[ne])))
with io.open(OD + r'\probe2_report.txt',
             'w', encoding='utf-8') as f:
    f.write('\n'.join(o) + '\n')
print('PROBE2_DONE')
