# -*- coding: utf-8 -*-
"""Probe: where does res_{l+1} = res_l + attn + mlp
break on qwen3-4b?  Single prompt, one chain."""
import io
import os
import sys

import numpy as np

MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NL = 36
out = []
sys.path.insert(
    0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
from phase2662_symmetric_mapping_contract \
    import load_native
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(
    MD, local_files_only=True, trust_remote_code=True,
    use_fast=True)
model, _ = load_native('qwen4')
model.eval()
layers = model.model.layers

ai = {}
ao = {}
mo = {}

def pre_attn(li):
    def h(module, args, kwargs):
        x = args[0] if args \
            else kwargs.get('hidden_states')
        if x is None or x.dim() < 2:
            return None
        ai.setdefault(li, []).append(
            x.detach().float().cpu().numpy().copy())
        return None
    return h

def cap_attn(li):
    def h(module, args, output):
        o0 = output[0] if isinstance(output, tuple) \
            else output
        ao.setdefault(li, []).append(
            o0[:, -1, :].detach().float().cpu()
            .numpy().copy())
        return None
    return h

def cap_mlp(li):
    def h(module, args, output):
        mo.setdefault(li, []).append(
            output[:, -1, :].detach().float().cpu()
            .numpy().copy())
        return None
    return h

hs = []
for li in range(NL):
    hs.append(layers[li].self_attn
              .register_forward_pre_hook(
                  pre_attn(li), with_kwargs=True))
    hs.append(layers[li].self_attn
              .register_forward_hook(cap_attn(li)))
    hs.append(layers[li].mlp
              .register_forward_hook(cap_mlp(li)))

pr = 'The weather was cold, so'
ids = tok(pr, add_special_tokens=False)['input_ids']
with __import__('torch').no_grad():
    model(__import__('torch').tensor([ids],
                                      device='cuda'),
          use_cache=True)

out.append('entries per layer: ai=%d ao=%d mo=%d'
           % (len(ai[0]), len(ao[0]), len(mo[0])))
res = [ai[li][-1][0, -1, :].astype(np.float64)
       for li in range(NL)]
a = [ao[li][-1][0].astype(np.float64)
     for li in range(NL)]
m = [mo[li][-1][0].astype(np.float64)
     for li in range(NL)]
out.append('single-chain recursion check (prefill, '
           'last pos):')
worst = 0.0
for li in range(NL - 1):
    d = res[li + 1] - res[li]
    resid = d - a[li] - m[li]
    r = float(np.max(np.abs(resid))) \
        / max(float(np.median(np.abs(d))), 1e-30)
    worst = max(worst, r)
    if li < 6 or li > 32 or r > 1e-3:
        out.append('l=%2d |d|=%.4f |a|=%.4f |m|=%.4f '
                   'maxresid=%.3e rel=%.2e'
                   % (li, float(np.linalg.norm(d)),
                      float(np.linalg.norm(a[li])),
                      float(np.linalg.norm(m[li])),
                      float(np.max(np.abs(resid))),
                      r))
out.append('worst rel=%.3e' % worst)

# layer module structure check
out.append('layer0 attn type=%s'
           % type(layers[0].self_attn).__name__)
out.append('attn forward sig modules: has o_proj=%s '
           'has q_norm=%s k_norm=%s'
           % (hasattr(layers[0].self_attn, 'o_proj'),
              hasattr(layers[0].self_attn, 'q_norm'),
              hasattr(layers[0].self_attn, 'k_norm')))
io.open(r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\tmp_probe17.txt', 'w',
        encoding='utf-8').write('\n'.join(out))
print('ok')
