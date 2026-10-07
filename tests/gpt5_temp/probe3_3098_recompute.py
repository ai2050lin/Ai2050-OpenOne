# -*- coding: utf-8 -*-
"""Probe 3: recompute KL medians and
T1 means from sealed npz.  -> txt"""
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
    for i, nm in enumerate(
            ('no_attn', 'no_mlp',
             'skip')):
        c = np.concatenate(
            [z['KL_%s_%s' % (side, fa)][:, i]
             for fa in 'ABC'])
        t = np.concatenate(
            [z['T1_%s_%s' % (side, fa)][:, i]
             for fa in 'ABC']).mean()
        o.append('%s %s: med=%.17g '
                 't1mean=%.6f'
                 % (side, nm,
                    float(np.median(c)),
                    float(t)))
with io.open(OD + r'\probe3_report.txt',
             'w', encoding='utf-8') as f:
    f.write('\n'.join(o) + '\n')
print('PROBE3_DONE')
