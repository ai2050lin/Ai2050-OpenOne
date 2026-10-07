# -*- coding: utf-8 -*-
"""3032 preflight GPU probe: validate (1) erase chain js
vs 3022, (2) per-head dnorm vs 3029 npz d1, (3) L3 MLP
SwiGLU attribution s vs 3022 s_relay - all bit-level on
tag P0:5.  Machine copied verbatim from 3030/3022."""
import json
import os
import sys

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
D_3022 = os.path.join(BASE, 'phase3022',
                      'omega_p2p_l3_relay_neurons_qwen')
D_3029 = os.path.join(BASE, 'phase3029',
                      'omega_p2w_recruitment_decomp_qwen')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NL, HID = 36, 2560
L3_GATED, L_RELAY, L_ERR = 3, 3, 4
G7_HEAD, TOPK, HDIM, NHQ = 7, 32, 128, 32
L_MIN = 4
GEN_PROMPTS = (
    'The weather was cold, so',
    'He studied every night because',
    'She wanted to buy the car, but',
    'The experiment failed, therefore',
    'You should take an umbrella if',
    'The meeting was long, and',
    'He missed the train, however',
    'The garden grows quickly while',
    'The price was high, yet',
    'She speaks French, although',
    'The road was closed, thus',
    'We left early because',)
out = []

z22 = np.load(D_3022 + r'\omega_p2p_l3_relay_neurons_'
              r'qwen.npz', allow_pickle=True)
s22 = z22['s_relay'].astype(np.float64)
tags22 = [str(t) for t in z22['tags']]
js22 = z22['js_final_logic'].astype(np.float64)
z29 = np.load(D_3029 + r'\omega_p2w_recruitment_decomp_'
              r'qwen.npz', allow_pickle=True)
out.append('tag P0:5 idx22=%d' % tags22.index('P0:5'))

import torch
sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
from phase2662_symmetric_mapping_contract import load_native
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(
    MD, local_files_only=True, trust_remote_code=True,
    use_fast=True)
model, _ = load_native('qwen4')
model.eval()
layers = model.model.layers

state_r = {'on': False}
rs = {}
state_hcap = {'on': False}
state_o = {'on': False}
o_in = {}
mlp_h = {}
mlp_m = {}
MLP_LAYERS = [L_RELAY] + list(range(21, 35))
handles = []


def pre_layer(li):
    def h(module, args, kwargs):
        if state_r['on']:
            x = args[0] if args \
                else kwargs.get('hidden_states')
            if x is not None and x.dim() >= 2:
                rs.setdefault(li, []).append(
                    x[:, -1, :].detach().float()
                    .cpu().numpy().copy())
        return None
    return h


def pre_oproj(li):
    def h(module, args, kwargs):
        if state_o['on']:
            x = args[0]
            o_in.setdefault(li, []).append(
                x[:, -1, :].detach().float()
                .cpu().numpy().copy())
        return None
    return h


def pre_down(li):
    def h(module, args, kwargs):
        if state_hcap['on']:
            x = args[0]
            mlp_h.setdefault(li, []).append(
                x[:, -1, :].detach().float()
                .cpu().numpy().copy())
        return None
    return h


def post_mlp(li):
    def h(module, args, kwargs, output):
        if state_hcap['on']:
            mlp_m.setdefault(li, []).append(
                output[:, -1, :].detach().float()
                .cpu().numpy().copy())
        return None
    return h


for li in range(NL):
    handles.append(layers[li]
                   .register_forward_pre_hook(
                       pre_layer(li),
                       with_kwargs=True))
    handles.append(layers[li].self_attn.o_proj
                   .register_forward_pre_hook(
                       pre_oproj(li),
                       with_kwargs=True))
for li in MLP_LAYERS:
    handles.append(layers[li].mlp.down_proj
                   .register_forward_pre_hook(
                       pre_down(li),
                       with_kwargs=True))
    handles.append(layers[li].mlp
                   .register_forward_hook(
                       post_mlp(li),
                       with_kwargs=True))


def grab_o():
    return np.stack([o_in[li][-1][0]
                     .astype(np.float64)
                     for li in range(NL)])


def grab_r():
    return np.stack([rs[li][-1][0]
                     .astype(np.float64)
                     for li in range(NL)])


def grab_mlps():
    h = {li: mlp_h[li][-1][0].astype(np.float64)
         for li in MLP_LAYERS}
    m = {li: mlp_m[li][-1][0].astype(np.float64)
         for li in MLP_LAYERS}
    return h, m


def kv_scale_arm(past, p, s, arm, h=None,
                 li=L3_GATED):
    L = past.layers[li]
    if h is None:
        if arm in ('JOINT', 'KONLY'):
            L.keys[:, :, p, :] *= s
    else:
        if arm in ('JOINT', 'KONLY'):
            L.keys[:, h, p, :] *= s


def js_nats(p, q):
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask]
                            * np.log(a[mask]
                                     / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


pi, p_pos = 0, 5
pr = GEN_PROMPTS[pi]
ids = tok(pr, add_special_tokens=False)['input_ids']
Wd3 = layers[L_RELAY].mlp.down_proj.weight


def capture_base():
    o_in.clear()
    rs.clear()
    mlp_h.clear()
    mlp_m.clear()
    state_hcap['on'] = True
    state_o['on'] = True
    with torch.no_grad():
        out = model(torch.tensor([ids],
                                 device='cuda'),
                    use_cache=True)
        past = out.past_key_values
        state_r['on'] = True
        out2 = model(
            input_ids=torch.tensor(
                [[int(ids[-1])]], device='cuda'),
            past_key_values=past,
            use_cache=False)
        state_r['on'] = False
    state_hcap['on'] = False
    state_o['on'] = False
    lg = out2.logits[0, -1].detach() \
        .double().cpu().numpy()
    lg = lg - lg.max()
    p0 = np.exp(lg)
    p0 = p0 / p0.sum()
    return p0, past, grab_o(), grab_r(), grab_mlps()


def run_chain(erase):
    o_in.clear()
    rs.clear()
    mlp_h.clear()
    mlp_m.clear()
    state_o['on'] = True
    state_hcap['on'] = True
    state_r['on'] = True
    with torch.no_grad():
        out = model(torch.tensor([ids],
                                 device='cuda'),
                    use_cache=True)
        past = out.past_key_values
        if erase:
            kv_scale_arm(past, p_pos, 0.0,
                         'KONLY', G7_HEAD)
        out2 = model(
            input_ids=torch.tensor(
                [[int(ids[-1])]], device='cuda'),
            past_key_values=past,
            use_cache=False)
    state_r['on'] = False
    state_o['on'] = False
    state_hcap['on'] = False
    lg = out2.logits[0, -1].detach() \
        .double().cpu().numpy()
    lg = lg - lg.max()
    p_ = np.exp(lg)
    p_ = p_ / p_.sum()
    return p_, grab_o(), grab_r(), grab_mlps()


p0, past, o_b, r_b, (hb, mb) = capture_base()
# NOTE: past reused across arms like 3029/3030
# (kv_scale mutates past in place; erase applies
# 0.0 scale each time - repeated *= 0 is idempotent)

p_e, o_e, r_e, (he, me) = run_chain(True)
out.append('a28 js_erase=%.6f vs 3022 %.6f diff=%.2e'
           % (js_nats(p0, p_e), js22[0],
              abs(js_nats(p0, p_e) - js22[0])))

# dnorm vs 3029 d1[0]
d1m = np.zeros((NL, NHQ))
for li in range(L_MIN, NL):
    db = o_b[li].reshape(NHQ, HDIM)
    num1 = np.linalg.norm(
        o_e[li].reshape(NHQ, HDIM) - db, axis=1)
    den = np.linalg.norm(db, axis=1) + 1e-12
    d1m[li] = num1 / den
d29 = z29['d1'][0]
out.append('a40 dnorm vs 3029 d1[0] max|d|=%.3e'
           % float(np.abs(d1m - d29).max()))

# L3 attribution vs 3022 s_relay[0]
e4 = r_e[L_ERR] - r_b[L_ERR]
dh = he[L_RELAY] - hb[L_RELAY]
dm = me[L_RELAY] - mb[L_RELAY]
ne2 = max(float(e4 @ e4), 1e-30)
Wf = Wd3.detach().float()
u = (torch.tensor(e4, dtype=torch.float32,
                  device='cuda') @ Wf) \
    .cpu().numpy().astype(np.float64)
s = 2.0 * dh * u / ne2
p_m = float(dm @ e4) / ne2
ref = 2.0 * p_m
scale = 2.0 * float(np.linalg.norm(dm)) \
    * float(np.linalg.norm(e4)) / ne2
ident = abs(float(np.sum(s)) - ref) \
    / max(scale, 1e-30)
out.append('a41 s_L3 vs 3022 s_relay[0] max|d|=%.3e '
           'ident=%.3e'
           % (float(np.abs(s - s22[0]).max()), ident))

# deep-layer attribution shape check (L30)
li = 30
e_l = r_e[li + 1] - r_b[li + 1]
dh_l = he[li] - hb[li]
ne2l = max(float(e_l @ e_l), 1e-30)
Wl = layers[li].mlp.down_proj.weight.detach().float()
u_l = (torch.tensor(e_l, dtype=torch.float32,
                    device='cuda') @ Wl) \
    .cpu().numpy().astype(np.float64)
s_l = 2.0 * dh_l * u_l / ne2l
out.append('deep L%d s: nnz=%d top32share=%.4f '
           'maxabs=%.3e'
           % (li, int(np.sum(np.abs(s_l) > 0)),
              float(np.sort(np.abs(s_l))[::-1][:32]
                    .sum()
                    / max(np.abs(s_l).sum(),
                          1e-30)),
              float(np.abs(s_l).max())))

rep = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp' \
      r'\phase3032_probe2.txt'
with open(rep, 'w', encoding='utf-8') as f:
    f.write('\n'.join(out) + '\n')
print('WROTE')
