# -*- coding: utf-8 -*-
"""Debug: compare 3132 matg span_idx vs
z26 frozen span_idx. Find diff pattern."""
import io
import json
import os
import random as _rnd
import zlib

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913')
D05 = os.path.join(RDIR, 'phase3105',
                   'omega_p103_incontext_truth_'
                   'consistency') \
    if False else (RDIR + r'\phase3105'
                   r'\omega_p103_incontext_'
                   r'truth_consistency')
D13 = (RDIR + r'\phase3113'
       r'\omega_p111_artifact_writein')
D26 = (RDIR + r'\phase3126'
       r'\omega_p124_glm4_anchoredlast_'
       r'regen_writechain')
OUTF = (ROOT + r'\tests\gpt5_temp'
        r'\p3132_dbg_out.txt')
lines = []

mat5 = json.load(io.open(
    os.path.join(D05, 'material.json'),
    encoding='utf-8'))
capb = np.load(os.path.join(D13,
                            'capture_b.npz'),
               allow_pickle=False)
pkB = capb['pk']
condB = capb['cond']
z26 = np.load(D26 + r'\p124_readout.npz',
              allow_pickle=False)
p2r = mat5['pair2rel']
NP = 672
hP = {}
for i in range(len(pkB)):
    pk = str(pkB[i])
    if str(condB[i]) == 'P':
        hP[pk] = i
pks = sorted(hP.keys())
lines.append('len pks=%d' % len(pks))
lines.append('pks[:5]=%s' % pks[:5])
lines.append('pks[-3:]=%s' % pks[-3:])
sP = z26['span_idx_P']
lines.append('z26 span_idx_P shape=%s'
             % (sP.shape,))
nz = int((sP[:, 0] >= 0).sum())
lines.append('z26 spans found=%d/672' % nz)

# check: how many pk keys in pair2rel
lines.append('pair2rel keys=%d'
             % len(p2r))
inset = sum(1 for pk in pks
            if pk in p2r)
lines.append('pks in pair2rel=%d' % inset)
# first 5 span rows of z26
lines.append('z26 span rows 0-4: %s'
             % (sP[:5].tolist(),))
# material sanity for pks[0]
pk0 = pks[0]
lines.append('pk0=%s rel=%s false=%s'
             % (pk0, p2r[pk0],
                mat5['false_rels'][pk0]))

with io.open(OUTF, 'w',
             encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('done')
