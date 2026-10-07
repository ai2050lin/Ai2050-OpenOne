# -*- coding: utf-8 -*-
"""Probe2: isolate where the lens composition
breaks. Steps: (A) clean forward without hooks;
(B) forward WITH layer hooks -> do hooks perturb
the logits? (C) norm-hook input vs layer-35 hook
output; (D) lm_head(norm-input) vs real logits,
per position."""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, \
    AutoTokenizer

MODEL_DIR = (r'D:\AI2050\Ai2050-OpenOne\models'
             r'\hf\qwen3-4b')
o = []
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, torch_dtype=torch.float32,
    attn_implementation='eager').to('cuda').eval()
NL = 36

ids = tok('In a formal style, The weather was '
          'cold, so', add_special_tokens=False)[
    'input_ids']
with torch.no_grad():
    outA = model(torch.tensor([ids], device='cuda'),
                 use_cache=True)
L_A = outA.logits.detach().clone()
o.append('logits%s L_A[0,-1,:5]=%s'
         % (tuple(L_A.shape),
            np.array2string(
                L_A[0, -1, :5].cpu().numpy(),
                precision=4)))

cap = {}
normcap = {'inp': None, 'out': None}
handles = []


def mk(li):
    def h(module, inp, out):
        cap[li] = out[0].detach().clone()
    return h


for li in range(NL):
    handles.append(
        model.model.layers[li]
        .register_forward_hook(mk(li)))


def hnorm(module, inp, out):
    normcap['inp'] = inp[0].detach().clone()
    normcap['out'] = out.detach().clone()


handles.append(
    model.model.norm.register_forward_hook(hnorm))

with torch.no_grad():
    outB = model(torch.tensor([ids], device='cuda'),
                 use_cache=True)
L_B = outB.logits.detach()
d_hooks = float((L_A - L_B).abs().max())
o.append('hooks perturb logits: %.3e' % d_hooks)

cap35 = cap[NL - 1]
o.append('cap35%s norminp%s normout%s'
         % (tuple(cap35.shape),
            tuple(normcap['inp'].shape),
            tuple(normcap['out'].shape)))
d_pre = float((normcap['inp'][0] - cap35)
              .abs().max())
o.append('norm-input vs cap35: %.3e' % d_pre)
with torch.no_grad():
    man = model.lm_head(model.model.norm(cap35))
d_all = (man - L_B[0]).abs().amax(dim=1)
o.append('lm_head(norm(cap35)) vs logits: '
         'max=%.3e' % float(d_all.max()))
o.append('per-position diff: %s'
         % np.array2string(
             d_all.cpu().numpy(), precision=4,
             max_line_width=100))
with torch.no_grad():
    man2 = model.lm_head(normcap['out'])
d_post = float((man2[0, -1] - L_B[0, -1])
               .abs().max())
o.append('lm_head(norm-output) vs logits: %.3e'
         % d_post)
with torch.no_grad():
    man3 = model.lm_head(normcap['inp'])
d_in = float((man3[0, -1] - L_B[0, -1])
             .abs().max())
o.append('lm_head(norm-INPUT raw) vs logits: '
         '%.3e' % d_in)
o.append('man[-1,:5]=%s'
         % np.array2string(
             man[-1, :5].detach().cpu()
             .numpy(), precision=4))
for h_ in handles:
    h_.remove()
with open(r'D:\AI2050\Ai2050-OpenOne\gpt5_temp'
          r'\probe3050b.txt', 'w',
          encoding='utf-8') as f:
    f.write('\n'.join(o) + '\n')
print('probe2 written')
