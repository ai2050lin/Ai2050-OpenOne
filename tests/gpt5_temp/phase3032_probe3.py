# -*- coding: utf-8 -*-
"""3032 ident mini-probe: check W@dh vs dm identity on
tag P0:5 erase arm; locate why ident metric is O(10)
while 3022 ident_vals ~ 1e-4."""
import os
import sys
import traceback

import numpy as np

MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NL, NHQ, HDIM = 36, 32, 128
L3, L_ERR = 3, 4
G7_HEAD = 7
pr = 'The weather was cold, so'
p_pos = 5
out = []

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
state_o = {'on': False}
o_in = {}
state_mcap = {'on': False}
mlp_h = {}
mlp_m = {}


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


def pre_down(li):
    def h(module, args, kwargs):
        if state_mcap['on']:
            mlp_h.setdefault(li, []).append(
                args[0][:, -1, :].detach().float()
                .cpu().numpy().copy())
        return None
    return h


def post_mlp(li):
    def h(module, args, kwargs, output):
        if state_mcap['on']:
            mlp_m.setdefault(li, []).append(
                output[:, -1, :].detach().float()
                .cpu().numpy().copy())
        return None
    return h


for li in range(NL):
    layers[li].register_forward_pre_hook(
        pre_layer(li), with_kwargs=True)
    layers[li].self_attn.o_proj \
        .register_forward_pre_hook(
            pre_oproj_dummy(li), with_kwargs=True) \
        if False else None
for li in (L3,):
    layers[li].mlp.down_proj \
        .register_forward_pre_hook(
            pre_down(li), with_kwargs=True)
    layers[li].mlp.register_forward_hook(
        post_mlp(li), with_kwargs=True)


def grab_r():
    return np.stack([rs[li][-1][0]
                     .astype(np.float64)
                     for li in range(NL)])


ids = tok(pr, add_special_tokens=False)['input_ids']

# base
rs.clear()
mlp_h.clear()
mlp_m.clear()
state_mcap['on'] = True
state_r['on'] = True
with torch.no_grad():
    fwd = model(torch.tensor([ids], device='cuda'),
                use_cache=True)
    past = fwd.past_key_values
    model(input_ids=torch.tensor(
        [[int(ids[-1])]], device='cuda'),
        past_key_values=past, use_cache=False)
state_r['on'] = False
state_mcap['on'] = False
r_b = grab_r()
h_b = mlp_h[3][-1][0].astype(np.float64)
m_b = mlp_m[3][-1][0].astype(np.float64)

# erase arm
rs.clear()
mlp_h.clear()
mlp_m.clear()
state_mcap['on'] = True
state_r['on'] = True
with torch.no_grad():
    fwd = model(torch.tensor([ids], device='cuda'),
                use_cache=True)
    past = fwd.past_key_values
    past.layers[3].keys[:, G7_HEAD, p_pos, :] *= 0.0
    model(input_ids=torch.tensor(
        [[int(ids[-1])]], device='cuda'),
        past_key_values=past, use_cache=False)
state_r['on'] = False
state_mcap['on'] = False
r_e = grab_r()
h_e = mlp_h[3][-1][0].astype(np.float64)
m_e = mlp_m[3][-1][0].astype(np.float64)

e4 = r_e[L_ERR] - r_b[L_ERR]
dh = h_e - h_b
dm = m_e - m_b
ne2 = max(float(e4 @ e4), 1e-30)
Wf = layers[3].mlp.down_proj.weight.detach().float()
u = (torch.tensor(e4, dtype=torch.float32,
                  device='cuda') @ Wf) \
    .cpu().numpy().astype(np.float64)
s = 2.0 * dh * u / ne2
recon = (torch.tensor(dh, dtype=torch.float32,
                      device='cuda') @ Wf.T) \
    .cpu().numpy().astype(np.float64)
out.append('norm_e4=%.4f norm_dm=%.6f norm_dh=%.6f'
           % (float(np.linalg.norm(e4)),
              float(np.linalg.norm(dm)),
              float(np.linalg.norm(dh))))
out.append('sum_s=%.6f  2pm=%.6f'
           % (float(np.sum(s)), 2.0 * float(dm @ e4)
              / ne2))
out.append('recon_err=|W@dh - dm|/|dm| = %.3e'
           % (float(np.linalg.norm(recon - dm))
              / max(float(np.linalg.norm(dm)),
                    1e-30)))
out.append('ident3022=%.3e'
           % (abs(float(np.sum(s))
                  - 2.0 * float(dm @ e4) / ne2 * ne2
                  - 2.0 * float(dm @ e4))
              if False else
              abs(float(np.sum(s))
                  - 2.0 * float(dm @ e4))
              / max(2.0 * float(np.linalg.norm(dm))
                    * float(np.linalg.norm(e4))
                    / ne2, 1e-30)))

rep = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp' \
      r'\phase3032_probe3.txt'
with open(rep, 'w', encoding='utf-8') as f:
    f.write('\n'.join(out) + '\n')
print('WROTE')
