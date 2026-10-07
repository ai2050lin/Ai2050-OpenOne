# -*- coding: utf-8 -*-
"""Probe3: replicate the 3050 script state sequence
(forward_cap -> sham forward_run -> forward_lens)
with the exact same four hooks per layer, then
diagnose capH tensor sharing and ll[35] vs fresh
logits."""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, \
    AutoTokenizer

MODEL_DIR = (r'D:\AI2050\Ai2050-OpenOne\models'
             r'\hf\qwen3-4b')
o = []
NL = 36
HDIM = 128

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
torch.manual_seed(3009)
np.random.seed(3009)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, torch_dtype=torch.float32,
    attn_implementation='eager').to('cuda').eval()
layers = model.model.layers

stateKn = {li: {'repl': None, 'mask': None}
           for li in range(NL)}
stateV = {li: {'repl': None, 'mask': None}
          for li in range(NL)}
capKn = {li: {'rec': False, 'orig': None,
              'mod': None} for li in range(NL)}
capV = {li: {'rec': False, 'orig': None,
             'mod': None} for li in range(NL)}
capKpre = {li: {'rec': False, 'orig': None}
           for li in range(NL)}
capH = {li: {'rec': False, 'orig': None}
        for li in range(NL)}


def hook_norm(st, cp):
    def h(module, inp, out):
        if cp['rec']:
            cp['orig'] = out[0].detach().clone()
        if st['repl'] is not None:
            out[0][st['mask']] = st['repl']
        if cp['rec']:
            cp['mod'] = out[0].detach().clone()
        return out
    return h


def hook_v(st, cp):
    def h(module, inp, out):
        if cp['rec']:
            cp['orig'] = out[0].detach().clone()
        if st['repl'] is not None:
            out[0][st['mask']] = st['repl']
        if cp['rec']:
            cp['mod'] = out[0].detach().clone()
        return out
    return h


def hook_kpre(cp):
    def h(module, inp, out):
        if cp['rec']:
            cp['orig'] = out[0].detach().clone()
        return out
    return h


def hook_h(cp):
    def h(module, inp, out):
        if cp['rec']:
            cp['orig'] = out[0].detach().clone()
        return out
    return h


for li in range(NL):
    layers[li].self_attn.k_norm \
        .register_forward_hook(hook_norm(
            stateKn[li], capKn[li]))
    layers[li].self_attn.v_proj \
        .register_forward_hook(hook_v(
            stateV[li], capV[li]))
    layers[li].self_attn.k_proj \
        .register_forward_hook(hook_kpre(
            capKpre[li]))
    layers[li].register_forward_hook(hook_h(
        capH[li]))


def reset_all():
    for li in range(NL):
        stateKn[li]['repl'] = None
        stateKn[li]['mask'] = None
        stateV[li]['repl'] = None
        stateV[li]['mask'] = None
        capKn[li]['rec'] = False
        capV[li]['rec'] = False
        capKpre[li]['rec'] = False
        capH[li]['rec'] = False


ids = tok('In a formal style, The weather was '
          'cold, so', add_special_tokens=False)[
    'input_ids']
n = len(ids)
NVOC = int(model.config.vocab_size)

# step 1: forward_cap-style capture (a139 state)
reset_all()
for li in range(NL):
    capKn[li]['rec'] = True
    capV[li]['rec'] = True
    capKpre[li]['rec'] = True
with torch.no_grad():
    out1 = model(torch.tensor([ids], device='cuda'),
                 use_cache=True)
LG1 = out1.logits[0, -1].detach().double() \
    .cpu().numpy()
reset_all()

# step 2: forward_lens (T5 machinery)
reset_all()
for li in range(NL):
    capH[li]['rec'] = True
with torch.no_grad():
    model(torch.tensor([ids], device='cuda'),
          use_cache=True)
ptrs = [capH[li]['orig'].data_ptr()
        for li in range(NL)]
o.append('unique data_ptr among 36 capH origs: %d'
         % len(set(ptrs)))
dups = []
for li in range(NL):
    for lj in range(li + 1, NL):
        if ptrs[li] == ptrs[lj]:
            dups.append((li, lj))
o.append('duplicate pairs: %s' % dups[:20])
lens = np.zeros((NL, NVOC), dtype=np.float64)
with torch.no_grad():
    for li in range(NL):
        hH = capH[li]['orig']
        if hH.dim() == 3:
            hH = hH[0]
        ln = model.model.norm(hH)
        lg = model.lm_head(ln)
        lens[li] = lg[0, -1].detach().double() \
            .cpu().numpy()
reset_all()
d35 = float(np.max(np.abs(lens[NL - 1] - LG1)))
o.append('ll[35] vs LG1 diff=%.3e' % d35)
# consecutive layer-output diffs
dd = [float((capH[li]['orig'].float()
             - capH[li + 1]['orig'].float())
            .abs().max()) for li in range(NL - 1)]
o.append('consecutive capH maxdiff: %s'
         % np.array2string(
             np.array(dd), precision=3,
             max_line_width=120))
with open(r'D:\AI2050\Ai2050-OpenOne\gpt5_temp'
          r'\probe3050c.txt', 'w',
          encoding='utf-8') as f:
    f.write('\n'.join(o) + '\n')
print('probe3 written')
