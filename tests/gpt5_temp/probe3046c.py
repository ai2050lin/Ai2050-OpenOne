# -*- coding: utf-8 -*-
import io
import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
NPZ = os.path.join(BASE, 'phase3046',
                   'omega_p43_kfield_injection_qwen',
                   'omega_p43_kfield_injection_qwen.npz')
out = []

z = np.load(NPZ, allow_pickle=True)
out.append('verdict %s' % z['verdict'])
out.append('eff_a3 %s' % np.round(z['eff_a3'], 4))
out.append('med eff_r3 %s' % np.round(
    np.median(z['eff_r3'], axis=1), 4))
out.append('eff_a20 %s' % np.round(z['eff_a20'], 4))
out.append('med eff_r20 %s' % np.round(
    np.median(z['eff_r20'], axis=1), 4))
out.append('GBARK3 %s' % np.round(z['GBARK3'], 3))
out.append('GBARK20 %s' % np.round(z['GBARK20'], 3))
rd = z['rec_dlg']
out.append('rec_dlg: n=%d max=%.3f med=%.3f min=%.3f'
           % (len(rd), rd.max(), np.median(rd),
              rd.min()))
out.append('Kbase3 norms %s' % np.round(
    np.linalg.norm(z['Kbase3'], axis=1), 2))
out.append('Kbase20 norms %s' % np.round(
    np.linalg.norm(z['Kbase20'], axis=1), 2))

tok = AutoTokenizer.from_pretrained(MD)
model = AutoModelForCausalLM.from_pretrained(
    MD, torch_dtype=torch.float32,
    attn_implementation='eager').to('cuda').eval()
layers = model.model.layers
sa = layers[3].self_attn
out.append('has q_norm %s k_norm %s'
           % (hasattr(sa, 'q_norm'), hasattr(sa, 'k_norm')))
if hasattr(sa, 'k_norm'):
    w = sa.k_norm.weight.detach().double().cpu().numpy()
    out.append('k_norm weight: mean=%.4f std=%.4f '
               'min=%.4f max=%.4f'
               % (w.mean(), w.std(), w.min(), w.max()))

# one injection to split the three integrity subchecks
BODIES = ('The weather was cold, so',)
TARGETS = ('so',)
HDIM = 128
KV_HEAD = 7
SL = slice(KV_HEAD * HDIM, (KV_HEAD + 1) * HDIM)
cap = {'orig': None, 'mod': None}


def make_cap():
    def h(module, inp, out):
        cap['orig'] = out[0, POS, SL].detach().clone()
        return out
    return h


def make_inj(delta):
    def h(module, inp, out):
        cap['mod'] = out[0, POS, SL].detach().clone()
        out[0, POS, SL] += delta
        return out
    return h


ids = [int(x) for x in tok(
    BODIES[0], add_special_tokens=False)['input_ids']]
t = tok(' so', add_special_tokens=False)['input_ids'][0]
POS = ids.index(t)

with torch.no_grad():
    o0 = model(torch.tensor([ids], device='cuda'),
               use_cache=True)
kb = o0.past_key_values.layers[3].keys[0, KV_HEAD] \
    .detach().double().cpu().numpy()
out.append('Kbase L3 norm=%.3f'
           % float(np.linalg.norm(kb[POS])))

z46 = z
ub = z46['ubarK3']
g = float(z46['GBARK3'][0])
delta = torch.tensor(g * ub, dtype=torch.float32,
                     device='cuda')
h1 = sa.k_proj.register_forward_hook(make_cap())
h2 = sa.k_proj.register_forward_hook(make_inj(delta))
with torch.no_grad():
    o1 = model(torch.tensor([ids], device='cuda'),
               use_cache=True)
h1.remove()
h2.remove()
kinj = o1.past_key_values.layers[3].keys[0, KV_HEAD] \
    .detach().double().cpu().numpy()
orig = cap['orig'].double().cpu().numpy()
mod = cap['mod'].double().cpu().numpy()
dpre = float(np.max(np.abs(
    mod - orig - g * ub)))
mask = np.ones(kb.shape[0], dtype=bool)
mask[POS] = False
dnt = float(np.max(np.abs(kinj[mask] - kb[mask])))
nd = float(np.linalg.norm(kinj[POS] - kb[POS]))
ndl = float(np.linalg.norm(g * ub))
ratio = nd / ndl
out.append('dpre=%.3e dnt=%.3e ratio=%.6f '
           '(nd=%.4f ndl=%.4f)' % (dpre, dnt, ratio,
                                   nd, ndl))
out.append('pre-norm K norm=%.4f delta norm=%.4f'
           % (float(np.linalg.norm(orig)),
              ndl))
out.append('cos(diff_post, delta)=%.6f'
           % float((kinj[POS] - kb[POS]) @ (g * ub)
                   / (nd * ndl)))

io.open(r'D:\AI2050\Ai2050-OpenOne\gpt5_temp'
        r'\probe3046c_result.txt', 'w',
        encoding='utf-8').write('\n'.join(out))
print('ok')
