# -*- coding: utf-8 -*-
"""Probe4: bisect the lens corruption. Config A:
prior forward + layer hooks only. Config B: k/v
hooks registered (passive) + layer hooks, fresh.
Config C: prior forward + k/v hooks + layer hooks
(= probe3). Each config reports determinism
(forward logits vs the first clean forward),
norm-input vs cap35 consistency, and the lens
composition diff."""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, \
    AutoTokenizer

MODEL_DIR = (r'D:\AI2050\Ai2050-OpenOne\models'
             r'\hf\qwen3-4b')
o = []
NL = 36
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
torch.manual_seed(3009)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, torch_dtype=torch.float32,
    attn_implementation='eager').to('cuda').eval()
layers = model.model.layers

ids = tok('In a formal style, The weather was '
          'cold, so', add_special_tokens=False)[
    'input_ids']
xin = torch.tensor([ids], device='cuda')

with torch.no_grad():
    out0 = model(xin, use_cache=True)
LG0 = out0.logits.detach()
o.append('reference logits captured')

state = {'capH': {}, 'norm_inp': None}


def mk_layer(li):
    def h(module, inp, out):
        state['capH'][li] = out[0].detach().clone()
    return h


def mk_norm_passive(st, cp):
    def h(module, inp, out):
        if cp['rec']:
            cp['orig'] = out[0].detach().clone()
        if st['repl'] is not None:
            out[0][st['mask']] = st['repl']
        return out
    return h


def mk_v_passive(st, cp):
    def h(module, inp, out):
        if cp['rec']:
            cp['orig'] = out[0].detach().clone()
        if st['repl'] is not None:
            out[0][st['mask']] = st['repl']
        return out
    return h


def mk_kpre_passive(cp):
    def h(module, inp, out):
        if cp['rec']:
            cp['orig'] = out[0].detach().clone()
        return out
    return h


def hnorm_inp(module, inp, out):
    state['norm_inp'] = inp[0].detach().clone()


hndl = []


def register_layers():
    for li in range(NL):
        hndl.append(layers[li]
                    .register_forward_hook(
                        mk_layer(li)))
    hndl.append(model.model.norm
                .register_forward_hook(hnorm_inp))


def register_kv():
    st = {'repl': None, 'mask': None}
    cp = {'rec': False, 'orig': None}
    for li in range(NL):
        hndl.append(
            layers[li].self_attn.k_norm
            .register_forward_hook(
                mk_norm_passive(st, dict(cp))))
        hndl.append(
            layers[li].self_attn.v_proj
            .register_forward_hook(
                mk_v_passive(st, dict(cp))))
        hndl.append(
            layers[li].self_attn.k_proj
            .register_forward_hook(
                mk_kpre_passive(dict(cp))))


def unregister():
    for h_ in hndl:
        h_.remove()
    hndl.clear()
    state['capH'] = {}
    state['norm_inp'] = None


def run(tag):
    with torch.no_grad():
        out = model(xin, use_cache=True)
    d_logit = float((out.logits - LG0)
                    .abs().max())
    cap35 = state['capH'][NL - 1]
    d_pre = float((state['norm_inp'][0] - cap35)
                  .abs().max())
    with torch.no_grad():
        man = model.lm_head(
            model.model.norm(cap35))
    d_lens = float((man[0, -1]
                    - out.logits[0, -1])
                   .abs().max())
    o.append('%s: logit-vs-ref=%.3e | norm-inp vs '
             'cap35=%.3e | lens35-vs-own-logits='
             '%.3e' % (tag, d_logit, d_pre,
                       d_lens))
    return d_lens


# Config B: k/v hooks registered, fresh forward
register_kv()
register_layers()
run('B kv-passive+layer, fresh')
unregister()

# Config A: layer hooks only, with one prior
# forward (the unregister() above leaves model
# warm; the prior forward is any previous run)
register_layers()
run('A layer-only, warm')
unregister()

# Config C: full probe3 replication
register_kv()
register_layers()
run('C kv+layer, warm (=probe3)')
unregister()

with open(r'D:\AI2050\Ai2050-OpenOne\gpt5_temp'
          r'\probe3050d.txt', 'w',
          encoding='utf-8') as f:
    f.write('\n'.join(o) + '\n')
print('probe4 written')
