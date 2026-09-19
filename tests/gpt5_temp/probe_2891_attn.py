# -*- coding: utf-8 -*-
"""Probe: GLM4 self_attn direct-call feasibility (for phase 2891).

Tests: (1) input_layernorm direct call; (2) self_attn direct call with
explicit position_embeddings (rotary_emb), attention_mask=None;
(3) consistency vs hook-captured baseline attnout (rel err);
(4) perturbation response sanity: dln1 injection produces nonzero
delta aligned with the injected direction.
Writes report to gpt5_temp/probe_2891_attn.txt.
"""
import io
import json
import os
import sys

import numpy as np

sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')

MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\glm4-9b-chat-hf'
REP = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\probe_2891_attn.txt'
L = 40
WIN_LO, WIN_HI = 28, 40
LI = 28

lines = []


def log(s):
    lines.append(str(s))
    print(s, flush=True)


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained(
    MD, local_files_only=True, trust_remote_code=True, use_fast=True)
dm = {'model.embed_tokens': 0, 'model.norm': 0, 'lm_head': 0}
for li_ in range(L):
    dm['model.layers.%d' % li_] = 0 if WIN_LO <= li_ < WIN_HI else 'cpu'
model = AutoModelForCausalLM.from_pretrained(
    MD, local_files_only=True, device_map=dm)
model.eval()
layers = model.model.layers
log('loaded; layers[28] type=%s' % type(layers[LI]).__name__)
log('self_attn type=%s' % type(layers[LI].self_attn).__name__)

import inspect
sig = inspect.signature(layers[LI].self_attn.forward)
log('self_attn.forward params: %s' % list(sig.parameters.keys()))

cap = {}


def pre(name, li):
    def h(module, args, kwargs):
        x = args[0] if args else kwargs.get('hidden_states')
        if x is None or x.dim() < 2:
            return
        cap.setdefault(name, {}).setdefault(li, []).append(
            x.detach().float().cpu().numpy())
    return h


def out(name, li):
    def h(module, args, output):
        o = output[0] if isinstance(output, tuple) else output
        if o.dim() < 2:
            return
        cap.setdefault(name, {}).setdefault(li, []).append(
            o.detach().float().cpu().numpy())
    return h


hs = []
layer = layers[LI]
hs.append(layer.input_layernorm.register_forward_pre_hook(
    pre('ln1in', LI), with_kwargs=True))
hs.append(layer.self_attn.register_forward_pre_hook(
    pre('attnin', LI), with_kwargs=True))
hs.append(layer.self_attn.register_forward_hook(out('attnout', LI)))

with torch.no_grad():
    ids = tok('The cat is black.', return_tensors='pt')[
        'input_ids'].to('cuda')
    model(ids)

ln1in = torch.tensor(cap['ln1in'][LI][0], device='cuda',
                     dtype=torch.bfloat16)
attnin = torch.tensor(cap['attnin'][LI][0], device='cuda',
                      dtype=torch.bfloat16)
attnout_ref = cap['attnout'][LI][0][0, 1]   # batch0, position 1
log('captured shapes: ln1in %s attnin %s attnout_pos1 %s'
    % (ln1in.shape, attnin.shape, attnout_ref.shape))

vdt = torch.bfloat16
dev = next(layers[LI].mlp.parameters()).device

# (1) ln1 direct
with torch.no_grad():
    ln1x = layer.input_layernorm(ln1in)
rel_ln1 = float((ln1x[0, 1].float().cpu().numpy() - attnin[0, 1]
                 .float().cpu().numpy()).max())
log('(1) ln1 direct ok; |ln1(ln1in) - attnin| max = %.6f (pos1)'
    % rel_ln1)

# (2) self_attn direct with position_embeddings
rng = np.random.default_rng(2891)
lang_dir = rng.normal(size=(int(ln1in.shape[-1]))).astype(np.float64)
lang_dir = lang_dir / np.linalg.norm(lang_dir)
ldt = torch.tensor(lang_dir[None, :], device=dev, dtype=torch.bfloat16)
seq_len = attnin.shape[1]
position_ids = torch.arange(seq_len, device=dev).unsqueeze(0)

try:
    rotary = getattr(layer, 'rotary_emb', None)
    if rotary is None:
        rotary = model.model.rotary_emb
        log('(2) layer.rotary_emb absent, using model.model.rotary_emb')
    else:
        log('(2) layer.rotary_emb present')
    with torch.no_grad():
        pos_emb = rotary(attnin, position_ids)
        o = layer.self_attn(attnin, position_embeddings=pos_emb,
                            attention_mask=None,
                            past_key_values=None)
    if isinstance(o, tuple):
        o = o[0]
    got = o[0, 1].detach().float().cpu().numpy()
    den = max(float(np.linalg.norm(attnout_ref)), 1e-30)
    rel = float(np.linalg.norm(got - attnout_ref)) / den
    log('(2) self_attn direct call ok; rel err vs hook pos1 = %.3e'
        % rel)

    # (4) perturbation response sanity
    eps = 1.0
    d = layer.input_layernorm(ln1in + eps * ldt) \
        - layer.input_layernorm(ln1in)
    with torch.no_grad():
        o2 = layer.self_attn(attnin + d, position_embeddings=pos_emb,
                             attention_mask=None,
                             past_key_values=None)
    if isinstance(o2, tuple):
        o2 = o2[0]
    delta = (o2[0, 1] - o[0, 1]).detach().float().cpu().numpy()
    g = float(delta @ lang_dir) / eps
    gnorm = float(np.linalg.norm(delta)) / eps
    log('(4) perturbation response: proj=%.6f |delta|=%.6f '
        'ratio=%.4f' % (g, gnorm, abs(g) / max(gnorm, 1e-30)))
    log('PROBE=OK')
except Exception as e:
    log('(2/4) FAILED: %r' % e)
    log('PROBE=FAIL')

with io.open(REP, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('report written')
