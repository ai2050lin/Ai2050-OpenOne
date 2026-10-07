# -*- coding: utf-8 -*-
"""Debug3: build texts via 3132 factory AND
via 3126 source slice, compare per-pk text
+ span. Locate first divergence."""
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
M32 = (ROOT + r'\tests\glm5'
       r'\phase3132_omega_p130_'
       r'forkcausal_single256.py')
M26 = (ROOT + r'\tests\glm5'
       r'\phase3126_omega_p124_glm4_'
       r'anchoredlast_regen_'
       r'writechain.py')
OUTF = (ROOT + r'\tests\gpt5_temp'
        r'\p3132_dbg3_out.txt')
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
NB = len(pkB)
DIRS = ('P', 'A1')
hP = {}
for i in range(NB):
    pk = str(pkB[i])
    if str(condB[i]) == 'P':
        hP[pk] = i
pks = sorted(hP.keys())

# ---- 3132 factory ----
s32 = io.open(M32, encoding='utf-8').read()
a0 = s32.index("p2r = mat5['pair2rel']")
a1 = s32.index("log('factory defined')")
g32 = {'mat5': mat5, 'DIRS': DIRS,
       'NP': NP, 'N_NEW': N_NEW, 'pks': pks,
       '_rnd': _rnd, 'zlib': zlib,
       'np': np}
exec(compile(s32[a0:a1], '<f32>', 'exec'),
     g32)
mm32 = g32['make_materials'](
    *([None] * 0)) if False else None

# ---- 3126 texts slice (no model) ----
s26 = io.open(M26, encoding='utf-8').read()
b0 = s26.index("p2r = mat5['pair2rel']")
b1 = s26.index("import torch  # noqa: E402")
g26 = {'mat5': mat5, 'pkB': pkB,
       'condB': condB, 'NB': NB,
       'NP': NP, 'DIRS': ('P', 'A1'),
       'pks': pks, '_rnd': _rnd,
       'zlib': zlib}
exec(compile(s26[b0:b1], '<f26>', 'exec'),
     g26)
texts26 = g26['texts']
lines.append('3126 texts built')

# span calc slice from 3126
c0 = s26.index('def find_span_gq')
c1 = s26.index('def line_tokens_g')
g26['N_NEW'] = N_NEW
exec(compile(s26[c0:c1], '<fs26>', 'exec'),
     g26)
fsg = g26['find_span_gq']

from transformers import AutoTokenizer
tok_g = AutoTokenizer.from_pretrained(
    MDIR_G, trust_remote_code=True)
p2r = mat5['pair2rel']
nd = 0
for j in range(NP):
    pk = pks[j]
    t32 = g32['texts']['P'][pk] \
        if False else None
# build 3132 texts via its factory funcs
build_prompt32 = g32['build_prompt']
frel = mat5['false_rels']
nd_txt = 0
first = []
for j in range(NP):
    pk = pks[j]
    (s, o) = (int(v)
              for v in pk.split('_'))
    r = p2r[pk]
    ri1, ri2 = frel[pk]
    t32 = build_prompt32(mat5, s, o, r, r)
    t26 = texts26['P'][pk]
    if t32 != t26:
        nd_txt += 1
        if len(first) < 3:
            k = 0
            while k < min(len(t32),
                          len(t26)) \
                    and t32[k] == t26[k]:
                k += 1
            first.append((pk, k,
                          t32[max(0, k - 25):
                              k + 25],
                          t26[max(0, k - 25):
                              k + 25]))
lines.append('P text diffs=%d/672' % nd_txt)
for (pk, k, x32, x26) in first:
    lines.append('pk=%s diff@%d' % (pk, k))
    lines.append('  32: ...%s...' % x32)
    lines.append('  26: ...%s...' % x26)
# span recompute on 3126 texts
sp = mm32 if False else None
span_check = []
for j in (0, 1, 2):
    pk = pks[j]
    (s, o) = (int(v)
              for v in pk.split('_'))
    t26 = texts26['P'][pk]
    enc = tok_g(t26,
                add_special_tokens=False,
                return_offsets_mapping=True)
    offs = [tuple(v) for v in
            enc['offset_mapping']]
    s26j = fsg(t26, offs, s, o,
               p2r[pk])
    span_check.append((pk, s26j,
                       z26['span_idx_P'][j]
                       .tolist()))
lines.append('span recompute (3126 texts):'
             ' %s' % span_check)
with io.open(OUTF, 'w',
             encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('done')
