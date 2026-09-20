# -*- coding: utf-8 -*-
"""Phase 2942 reproducibility probe: is the s=2 injection
forward deterministic across repeats and across forward
history? Decides between (a) CUDA nondeterminism in injected
forwards, (b) grid-composition state dependence.

Reads Vt8/dcks directly from 2939 npz (bit-level identical to
in-run rebuild per a3=0.00), no pass1 needed."""
import json
import os
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC_2887 = os.path.join(BASE, 'phase2887', 'language_axis_mlp',
                        'language_axis_mlp.npz')
SRC_2927 = os.path.join(BASE, 'phase2927', 'probe_relativity',
                        'probe_relativity.npz')
SRC_2939 = os.path.join(BASE, 'phase2939', 'rotation_target',
                        'rotation_target.npz')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2942_probe_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
L_INJ = 16
S_IDX = (0, 1, 4)

lines = []


def log(m):
    lines.append(m)
    print(m, flush=True)


z87 = np.load(SRC_2887, allow_pickle=True)
words = [tuple(str(w).split(':')) for w in z87['words']]
z27 = np.load(SRC_2927, allow_pickle=True)
z39 = np.load(SRC_2939, allow_pickle=True)
Vt8 = z39['Vt8'].astype(np.float64)
coords = z39['coords'].astype(np.float64)
conds = [str(s) for s in z39['cond_names']]
dcks = coords[conds.index('null0')] - coords[conds.index('func')]
dcks_S = dcks[:, list(S_IDX)]
xdir = dcks_S @ Vt8[list(S_IDX)]
log('xdir built, norm med %.2f' % float(np.median(
    np.linalg.norm(xdir, axis=1))))

import torch
import sys
sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
from phase2662_symmetric_mapping_contract import load_native
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(
    MD, local_files_only=True, trust_remote_code=True,
    use_fast=True)
tc = {}


def tid(t):
    if t not in tc:
        ids = tok(' ' + t, add_special_tokens=False)[
            'input_ids']
        if len(ids) != 1:
            ids = tok(t, add_special_tokens=False)[
                'input_ids']
        assert len(ids) == 1
        tc[t] = int(ids[0])
    return tc[t]


tid_map = {}
for lang, ck, w in words:
    tid_map[w] = tid(w)
func_tid = tid('the')
batch = [[func_tid, tid_map[words[i][2]]] for i in range(57)]

model, _ = load_native('qwen4')
model.eval()
layers = model.model.layers
u35 = z27['dirs_word'].astype(np.float64)[35]

inj = {'on': False, 'scale': 0.0, 'vec': None}


def pre_attn(li):
    def h(module, args, kwargs):
        x = args[0] if args else kwargs.get('hidden_states')
        if x is None or x.dim() < 2:
            return
        if li == L_INJ and inj['on']:
            x = x.clone()
            x[:, 1, :] = x[:, 1, :] + inj['scale'] * inj['vec']
            if args:
                return (x,) + tuple(args[1:]), kwargs
            nkw = dict(kwargs)
            nkw['hidden_states'] = x
            return args, nkw
        return None
    return h


h16 = layers[L_INJ].self_attn.register_forward_pre_hook(
    pre_attn(L_INJ), with_kwargs=True)
fin_cap = {}
state = {'on': False}


def pre_norm(module, args, kwargs):
    if state['on']:
        fin_cap['x'] = args[0][:, -1, :].detach() \
            .float().cpu().numpy()


model.model.norm.register_forward_pre_hook(
    pre_norm, with_kwargs=True)

vec_t = torch.tensor(xdir, device='cuda', dtype=torch.bfloat16)


def fwd(scale):
    fin_cap.pop('x', None)
    state['on'] = True
    inj['on'] = scale != 0.0
    inj['scale'] = float(scale)
    inj['vec'] = vec_t
    with torch.no_grad():
        model(torch.tensor(batch, device='cuda'))
    inj['on'] = False
    state['on'] = False
    fin = fin_cap['x'].astype(np.float64)
    proj = fin @ u35
    c8 = fin @ Vt8.T
    cs = c8[:, list(S_IDX)]
    return proj, cs


def stats(tag, proj, cs):
    sep = float(proj[:29].mean() - proj[29:].mean())
    log('%s: proj med %+.4f | sep %+.3f | c_shift_S norm med '
        '%.3f' % (tag, float(np.median(proj)), sep,
                  float(np.median(
                      np.linalg.norm(cs, axis=1)))))


log('--- base repeats ---')
for r in range(3):
    p, c = fwd(0.0)
    stats('base#%d' % (r + 1), p, c)
log('--- s=2 repeats (consecutive) ---')
for r in range(3):
    p, c = fwd(2.0)
    stats('s2#%d' % (r + 1), p, c)
log('--- alternation s=1,s=2,s=1,s=2 ---')
for s in (1.0, 2.0, 1.0, 2.0):
    p, c = fwd(s)
    stats('alt s=%g' % s, p, c)
log('--- s=2.5, s=3 (context) ---')
for s in (2.5, 3.0):
    p, c = fwd(s)
    stats('ctx s=%g' % s, p, c)
log('--- s=2 again after context ---')
p, c = fwd(2.0)
stats('s2-after', p, c)

with open(REPORT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('WROTE', REPORT, flush=True)
