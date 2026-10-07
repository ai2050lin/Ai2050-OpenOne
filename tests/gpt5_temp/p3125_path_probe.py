# -*- coding: utf-8 -*-
"""Probe: why lens margin vs lm logits margin diverge
on qwen3-4b (transformers 5.14). One sample, several
candidate paths + position alignment checks."""
import json
import io
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, \
    AutoTokenizer

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913')
D05 = os.path.join(RDIR, 'phase3105',
                   'omega_p103_incontext_truth_'
                   'consistency')
D13 = os.path.join(RDIR, 'phase3113',
                   'omega_p111_artifact_writein')
D18 = os.path.join(RDIR, 'phase3118',
                   'omega_p116_autoregressive_margin_'
                   'trajectory')
MDIR = os.path.join(ROOT, 'models', 'hf', 'qwen3-4b')
OUTP = (ROOT + r'\tests\gpt5_temp'
        r'\p3125_path_probe_out.txt')

mat5 = json.load(io.open(
    os.path.join(D05, 'material.json'),
    encoding='utf-8'))
capb = np.load(os.path.join(D13, 'capture_b.npz'),
               allow_pickle=False)
pkB = capb['pk']
condB = capb['cond']
z18 = np.load(os.path.join(D18, 'traj_readout.npz'),
              allow_pickle=False)

p2r = mat5['pair2rel']
frel = mat5['false_rels']
ents_all = mat5['entities']
PREDS_all = mat5['predicates']
import random as _rnd
import zlib


def build_prompt(mat, s, o, lrel, qrel):
    ents = mat['entities']
    PREDS = mat['predicates']
    D = [tuple(d) for d in
         mat['distractors']['%d_%d' % (s, o)]]
    k = mat['kline']['%d_%d' % (s, o)]
    lines = [(s, lrel, o)] + list(D)
    rng2 = _rnd.Random(zlib.crc32(
        ('%d_%d_ord5' % (s, o)).encode('ascii')))
    order = list(range(8))
    rng2.shuffle(order)
    lines = [lines[i] for i in order]
    ci = lines.index((s, lrel, o))
    lines[ci], lines[k] = lines[k], lines[ci]
    text = 'Facts:'
    for (ls, lr, lo) in lines:
        text += ' The %s %s the %s.' % (
            ents[ls], PREDS[lr], ents[lo])
    text += (' Query: The %s %s the %s. Is this '
             'query true? Answer:'
             % (ents[s], PREDS[qrel], ents[o]))
    return text


hP = {}
for i in range(len(pkB)):
    pk = str(pkB[i])
    if str(condB[i]) == 'P':
        hP[pk] = i
pks = sorted(hP.keys())
pk0 = pks[0]
(s0, o0) = (int(v) for v in pk0.split('_'))
r0 = p2r[pk0]
ptxt = build_prompt(mat5, s0, o0, r0, r0)
gen0 = [int(t) for t in z18['gen_clean__P'][0]]

tok = AutoTokenizer.from_pretrained(MDIR)
model = AutoModelForCausalLM.from_pretrained(
    MDIR, dtype=torch.bfloat16,
    attn_implementation='eager').to('cuda').eval()
NL = len(model.model.layers)
WU = model.lm_head.weight.detach()
YES_ID = int(mat5['yes_id'])
NO_ID = int(mat5['no_id'])
w_dn = (WU[YES_ID] - WU[NO_ID]) \
    .float().cpu().numpy()
norm_mod = model.model.norm

pid = tok(ptxt, add_special_tokens=False)[
    'input_ids']
ids = list(pid) + gen0
t_in = torch.tensor([ids], device='cuda')
pos0 = len(pid) - 1
npts = len(gen0) + 1
o = []
with torch.inference_mode():
    out = model(t_in, output_hidden_states=True,
                use_cache=False)
    hs = out.hidden_states
    o.append('len(hidden_states) = %d (NL=%d)'
             % (len(hs), NL))
    o.append('hs[0] shape %s; hs[-1] shape %s'
             % (tuple(hs[0].shape),
                tuple(hs[-1].shape)))
    o.append('tie_word_embeddings = %s'
             % getattr(model.config,
                       'tie_word_embeddings',
                       'n/a'))
    lg = out.logits[0].float().cpu().numpy()
    lmv = lg[:, YES_ID] - lg[:, NO_ID]
    # candidate lens margins at last state index
    cands = {}
    for li, tag in ((NL, 'hs[%d]' % NL),
                    (NL + 1 if len(hs) > NL + 1
                     else NL, 'hs[last]'),
                    (0, 'hs[0]')):
        h = norm_mod(hs[li][0]).float() \
            .cpu().numpy()
        cands['norm(%s)' % tag] = np.array(
            [float(h[pos0 + k] @ w_dn)
             for k in range(npts)])
        h2 = hs[li][0].float().cpu().numpy()
        cands['raw(%s)' % tag] = np.array(
            [float(h2[pos0 + k] @ w_dn)
             for k in range(npts)])
    # logits-shifted candidates
    for sh in (-1, 0, 1):
        seg = lg[pos0 + sh:pos0 + sh + npts]
        if seg.shape[0] == npts:
            cands['logitdiff shift%+d' % sh] = \
                seg[:, YES_ID] - seg[:, NO_ID]
    del out

ref = cands['logitdiff shift+0']
o.append('')
o.append('logitdiff shift0 (13 pts): %s'
         % np.array2string(ref, precision=4,
                           max_line_width=200))
for tag, v in cands.items():
    if 'logitdiff' in tag:
        continue
    r = float(np.corrcoef(v, ref)[0, 1])
    md = float(np.max(np.abs(v - ref)))
    o.append('%-14s vs logitdiff: r=%.6f '
             'maxdiff=%.4e  first3=%s'
             % (tag, r, md,
                np.array2string(
                    v[:3], precision=4)))

with io.open(OUTP, 'w', encoding='utf-8') as f:
    f.write('\n'.join(o) + '\n')
print('PROBE_OK')
