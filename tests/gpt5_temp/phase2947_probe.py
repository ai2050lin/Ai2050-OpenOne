# -*- coding: utf-8 -*-
"""Probe: why are heads 20-31 bit-level zero under ablation?

Check (1) the actual shape of Qwen3Attention output[0],
(2) per-head pos-1 norms of the attention output (pre
o_proj? post o_proj?) at L17 under injection, (3) whether
ablating head 25 changes the final readout at all vs head 3.
"""
import os
import sys

import numpy as np

MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NL = 36
HD = 128
L17 = 17
S = 1.0

lines = []

import torch
sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
from phase2662_symmetric_mapping_contract import load_native
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(
    MD, local_files_only=True, trust_remote_code=True,
    use_fast=True)


def tid(t):
    ids = tok(' ' + t, add_special_tokens=False)['input_ids']
    if len(ids) != 1:
        ids = tok(t, add_special_tokens=False)['input_ids']
    assert len(ids) == 1
    return int(ids[0])


toks = [tid('the'), tid('house')]
model, _ = load_native('qwen4')
model.eval()
layers = model.model.layers

# config facts
cfg = model.config
lines.append('config: hidden=%s heads=%s kv_heads=%s '
             'head_dim=%s' % (cfg.hidden_size,
                              cfg.num_attention_heads,
                              cfg.num_key_value_heads,
                              cfg.head_dim))

cap_out = {}
state = {'on': False}


def post_attn(li):
    def h(module, args, output):
        if state['on'] and li == L17:
            cap_out['o'] = output[0].detach() \
                .float().cpu().numpy()
            cap_out['type'] = type(output)
            cap_out['n'] = len(output) if hasattr(
                output, '__len__') else -1
        return None
    return h


hnd = layers[L17].self_attn.register_forward_hook(
    post_attn(L17))

inj = {'on': False}
ablate = {'h': None}


def post_abl(li):
    def h(module, args, output):
        if ablate['h'] is not None and li == L17:
            o = output[0].clone()
            hi = ablate['h']
            o[:, 1, hi * HD:(hi + 1) * HD] = 0.0
            return (o,) + tuple(output[1:])
        return None
    return h


hnd2 = layers[L17].self_attn.register_forward_hook(
    post_abl(L17))

# run ref with capture
state['on'] = True
with torch.no_grad():
    model(torch.tensor([toks], device='cuda'))
o = cap_out['o']
lines.append('attn output[0] shape: %s | tuple len %s'
             % (o.shape, cap_out['n']))
# per-head pos-1 norm assuming (b, seq, NH*HD)
p1 = o[0, 1]
norms_reshaped = np.linalg.norm(
    p1.reshape(-1, HD), axis=1)
lines.append('per-head pos1 norms (reshape view): %s'
             % np.round(norms_reshaped, 4).tolist())
lines.append('total pos1 norm: %.4f' % float(
    np.linalg.norm(p1)))

state['on'] = False


def run(abl_h):
    ablate['h'] = abl_h
    with torch.no_grad():
        model(torch.tensor([toks], device='cuda'))
    ablate['h'] = None


# reference fin capture via norm hook is complex; instead
# compare hidden at model.model.norm input is overkill.
# Simpler: compare the ATTENTION OUTPUT itself for h=25
# ablation vs no ablation: rerun capture with ablation.
def run_capture(abl_h):
    cap_out.pop('o', None)
    state['on'] = True
    ablate['h'] = abl_h
    with torch.no_grad():
        model(torch.tensor([toks], device='cuda'))
    state['on'] = False
    ablate['h'] = None
    return cap_out['o']


o_ref = o
o_25 = run_capture(25)
o_3 = run_capture(3)
d25 = float(np.abs(o_25 - o_ref).max())
d3 = float(np.abs(o_3 - o_ref).max())
lines.append('attnout max|diff| abl25 vs ref: %.6e' % d25)
lines.append('attnout max|diff| abl3  vs ref: %.6e' % d3)

# norm-input comparison: hook on final norm
fin_cap = {}


def pre_norm(module, args, kwargs):
    fin_cap['x'] = args[0][:, -1, :].detach() \
        .float().cpu().numpy()


hnd3 = model.model.norm.register_forward_pre_hook(
    pre_norm, with_kwargs=True)


def run_fin(abl_h):
    fin_cap.pop('x', None)
    ablate['h'] = abl_h
    with torch.no_grad():
        model(torch.tensor([toks], device='cuda'))
    ablate['h'] = None
    return fin_cap['x'].astype(np.float64)


f_ref = run_fin(None)
f_25 = run_fin(25)
f_3 = run_fin(3)
lines.append('fin max|diff| abl25 vs ref: %.6e' % float(
    np.abs(f_25 - f_ref).max()))
lines.append('fin max|diff| abl3  vs ref: %.6e' % float(
    np.abs(f_3 - f_ref).max()))

open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
     r'\phase2947_probe_report.txt', 'w',
     encoding='utf-8').write('\n'.join(lines) + '\n')
print('probe done')
