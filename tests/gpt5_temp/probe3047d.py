# -*- coding: utf-8 -*-
# Probe: is the a122 integ failure a subtraction-
# rounding artifact of the CHECK, while the
# injection itself is bit-exact (mod == orig+delta)?
import io
import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

P = r'D:\AI2050\Ai2050-OpenOne\gpt5_temp' \
    r'\probe3047d_result.txt'
f = io.open(P, 'w', encoding='utf-8')


def w(s):
    f.write(str(s) + '\n')
    f.flush()


MODEL_DIR = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
torch.manual_seed(3009)
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, torch_dtype=torch.float32,
    attn_implementation='eager').to('cuda').eval()
layers = model.model.layers
NL = 36
HDIM = 128
KV_HEAD = 7
SL = slice(KV_HEAD * HDIM, (KV_HEAD + 1) * HDIM)

z47 = np.load(
    r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
    r'\rdc_query_construction_20260913\phase3047'
    r'\omega_p44_kv_joint_replay_qwen'
    r'\omega_p44_kv_joint_replay_qwen.npz',
    allow_pickle=True)
bodies = [str(x) for x in z47['bodies']]
dK = z47['DPREK'][0, 3]  # pair 0, layer 3

cap = {'rec': False, 'pos': -1, 'orig': None,
       'mod': None}
st = {'on': False, 'pos': -1, 'delta': None}


def h(module, inp, out):
    if cap['rec']:
        cap['orig'] = out[0, cap['pos'], SL] \
            .detach().clone()
    if st['on']:
        out[0, st['pos'], SL] += st['delta']
    if cap['rec']:
        cap['mod'] = out[0, st['pos'], SL] \
            .detach().clone()
    return out


layers[3].self_attn.k_proj.register_forward_hook(h)

ids = tok(bodies[0], add_special_tokens=False)[
    'input_ids']
pos = len(ids) - 1

# pass 1: base capture
cap.update(rec=True, pos=pos)
with torch.no_grad():
    model(torch.tensor([ids], device='cuda'),
          use_cache=True)
orig = cap['orig'].clone()
cap['rec'] = False

# pass 2: inject delta (fp32 cast, as in the run)
dt = torch.tensor(np.ascontiguousarray(dK),
                  dtype=torch.float32, device='cuda')
st.update(on=True, pos=pos, delta=dt)
cap.update(rec=True, pos=pos)
with torch.no_grad():
    model(torch.tensor([ids], device='cuda'),
          use_cache=True)
mod = cap['mod'].clone()
st['on'] = False
cap['rec'] = False

e_sub = float((mod.double() - orig.double()
               - dt.double()).abs().max())
e_bit = float((mod - (orig + dt)).abs().max())
e_cast = float(np.max(np.abs(
    dt.cpu().numpy() - dK.astype(np.float32))))
rel = e_sub / float(np.abs(dt.cpu().numpy()).max())
w('e_sub (mod-orig-delta) = %.3e' % e_sub)
w('e_bit (mod-(orig+delta)) = %.3e' % e_bit)
w('e_cast (delta cast) = %.3e' % e_cast)
w('rel e_sub/max|delta| = %.3e' % rel)
w('norm delta = %.4f, norm orig-slice = %.4f'
  % (float(dt.norm()), float(orig.norm())))
w('done')
f.close()
print('ok')
