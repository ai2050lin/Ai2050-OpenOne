# -*- coding: utf-8 -*-
"""Debug4: full-track margin for j=0 with
3131 forward_track_g replica; compare ALL
positions vs z26 mlg_s0_P[0] to find the
true position mapping of index 0..12."""
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
OUTF = (ROOT + r'\tests\gpt5_temp'
        r'\p3132_dbg4_out.txt')
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

s32 = io.open(M32, encoding='utf-8').read()
a0 = s32.index("p2r = mat5['pair2rel']")
a1 = s32.index("log('factory defined')")
g = {'mat5': mat5, 'pkB': pkB,
     'condB': condB, 'DIRS': DIRS,
     'NP': NP, 'N_NEW': N_NEW, 'pks': pks,
     '_rnd': _rnd, 'zlib': zlib,
     'np': np}
exec(compile(s32[a0:a1], '<f>', 'exec'), g)

import torch
from transformers import \
    AutoModelForCausalLM, AutoTokenizer
tok_g = AutoTokenizer.from_pretrained(
    MDIR_G, trust_remote_code=True)
model_g = AutoModelForCausalLM\
    .from_pretrained(
        MDIR_G, torch_dtype=torch.bfloat16,
        attn_implementation='eager',
        trust_remote_code=True).to('cuda')\
    .eval()
NLG = len(model_g.model.layers)
WUG = model_g.lm_head.weight.detach()
YES_G = int(tok_g(' yes',
                  add_special_tokens=False)
            ['input_ids'][0])
NO_G = int(tok_g(' no',
                 add_special_tokens=False)
           ['input_ids'][0])
w_dn = (WUG[YES_G] - WUG[NO_G]) \
    .float().cpu().numpy()
norm_g = model_g.model.norm

gens = {'P': z26['gen_base_P'],
        'A1': z26['gen_base_A1']}
matg = g['make_materials'](tok_g, gens)
pids0 = matg['pids_all']['P'][0]['s0']
traj = matg['traj_tokens']('P', 0)[2]

# 3131 forward_track_g replica
ids = list(pids0) + [int(x) for x in traj]
t_in = torch.tensor([ids], device='cuda')
pos0 = len(pids0) - 1
npts = len(traj) + 1
feats = []
hooks = []


def _mk():
    def hook(mod, inp, out):
        o2 = out[0] \
            if isinstance(out, tuple) \
            else out
        feats.append(o2.detach())
    return hook


for lyr in model_g.model.layers:
    hooks.append(
        lyr.register_forward_hook(_mk()))
with torch.inference_mode():
    model_g(t_in, use_cache=False)
    for hk in hooks:
        hk.remove()
    seq = [model_g.model.embed_tokens(t_in)] \
        + feats
    ml = np.zeros((NLG + 1, npts),
                  dtype=np.float64)
    for L in range(NLG + 1):
        h = norm_g(seq[L][0]).float() \
            .cpu().numpy()
        for k in range(npts):
            ml[L, k] = float(
                h[pos0 + k] @ w_dn)
ref = z26['mlg_s0_P'][0].astype(np.float64)
lines.append('npts=%d pos0=%d len_ids=%d'
             % (npts, pos0, len(ids)))
d_all = np.abs(ml - ref).max()
lines.append('max|ml - ref| over ALL '
             '(L,k) = %.6f' % d_all)
for k in range(npts):
    dk = float(np.abs(ml[:, k]
                      - ref[:, k]).max())
    lines.append('k=%d maxd=%.6f '
                 'mine[L40]=%.4f ref[L40]=%.4f'
                 % (k, dk, ml[NLG, k],
                    ref[NLG, k]))
with io.open(OUTF, 'w',
             encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('done')
