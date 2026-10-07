# -*- coding: utf-8 -*-
"""Debug2: exec factory slice of 3132 main
script, build materials with tok_g, compare
span_idx vs z26 frozen."""
import io
import json
import os
import random as _rnd
import zlib

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913')
D05 = (RDIR + r'\phase3105'
       r'\omega_p103_incontext_'
       r'truth_consistency')
D13 = (RDIR + r'\phase3113'
       r'\omega_p111_artifact_writein')
D26 = (RDIR + r'\phase3126'
       r'\omega_p124_glm4_anchoredlast_'
       r'regen_writechain')
MDIR_G = os.path.join(ROOT, 'models', 'hf',
                      'glm4-9b-chat-hf')
MAIN = (ROOT + r'\tests\glm5'
        r'\phase3132_omega_p130_'
        r'forkcausal_single256.py')
OUTF = (ROOT + r'\tests\gpt5_temp'
        r'\p3132_dbg2_out.txt')
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
NP = 672
N_NEW = 12
DIRS = ('P', 'A1')
hP = {}
for i in range(len(pkB)):
    pk = str(pkB[i])
    if str(condB[i]) == 'P':
        hP[pk] = i
pks = sorted(hP.keys())

src = io.open(MAIN, encoding='utf-8').read()
i0 = src.index("p2r = mat5['pair2rel']")
i1 = src.index("log('factory defined')")
fs = src[i0:i1]
g = {'mat5': mat5, 'pkB': pkB,
     'condB': condB, 'DIRS': DIRS,
     'NP': NP, 'N_NEW': N_NEW, 'pks': pks,
     '_rnd': _rnd, 'zlib': zlib,
     'np': np}
exec(compile(fs, '<factory>', 'exec'), g)
lines.append('factory exec ok')

from transformers import AutoTokenizer
tok_g = AutoTokenizer.from_pretrained(
    MDIR_G, trust_remote_code=True)
gens = {'P': z26['gen_base_P'],
        'A1': z26['gen_base_A1']}
mm = g['make_materials'](tok_g, gens)
lines.append('materials built')
sp = mm['span_idx']['P']
zP = z26['span_idx_P']
nd = int(((sp[:, 0] != zP[:, 0])
          | (sp[:, 1] != zP[:, 1])).sum())
lines.append('P span diffs=%d/672' % nd)
nf = int((sp[:, 0] < 0).sum())
lines.append('my spans not-found=%d' % nf)
if nd > 0:
    didx = np.where(
        (sp[:, 0] != zP[:, 0])
        | (sp[:, 1] != zP[:, 1]))[0][:10]
    for j in didx:
        lines.append(
            'j=%d pk=%s mine=%s z26=%s'
            % (j, pks[j], sp[j].tolist(),
               zP[j].tolist()))
sa = mm['span_idx']['A1']
zA = z26['span_idx_A1']
ndA = int(((sa[:, 0] != zA[:, 0])
           | (sa[:, 1] != zA[:, 1])).sum())
lines.append('A1 span diffs=%d/672' % ndA)
# token length check on j=0
t0 = mm['texts']['P'][pks[0]]
enc = tok_g(t0, add_special_tokens=False)
lines.append('len text0 ids=%d z26 PID=%d'
             % (len(enc['input_ids']),
                len(mm['PID_T']['P'][0])))
with io.open(OUTF, 'w',
             encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('done')
