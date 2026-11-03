# -*- coding: utf-8 -*-
import os, sys, torch
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT = Path(r'D:/AI2050/Ai2050-OpenOne')
MODEL_PATH = 'D:/develop/model/Qwen/Qwen3-4B'
tok = AutoTokenizer.from_pretrained(MODEL_PATH)
m = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, device_map='cuda:0')
m.eval()
base = m.model
L = len(base.layers)
caps = {'attn': [None]*L, 'mlp': [None]*L}
def ha(i):
    def fn(mod, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        caps['attn'][i] = t.detach()
    return fn
def hm(i):
    def fn(mod, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        caps['mlp'][i] = t.detach()
    return fn
hs_hooks = [base.layers[i].self_attn.register_forward_hook(ha(i)) for i in range(L)]
hs_hooks += [base.layers[i].mlp.register_forward_hook(hm(i)) for i in range(L)]

text = '你知道吧，苹果很甜。'
ids = torch.tensor([tok.encode(text)]).cuda()
with torch.no_grad():
    out = m(input_ids=ids, output_hidden_states=True)
hss = list(out.hidden_states)
T = ids.shape[1]
p = T - 1
print('n hidden:', len(hss), 'shapes ok')
for l in [0, 1, 5, 18, 35]:
    r_in = hss[l][0, p].float()
    r_out = hss[l+1][0, p].float() if l+1 < len(hss) else None
    a = caps['attn'][l][0, p].float()
    mm = caps['mlp'][l][0, p].float()
    print(f'l={l}: |r_in|={r_in.norm():.2f} |attn|={a.norm():.2f} |mlp|={mm.norm():.2f}', end=' ')
    if r_out is not None:
        rec = r_in + a + mm
        print(f'|r_out|={r_out.norm():.2f} |rec-r_out|={(rec-r_out).norm():.4f}')
    else:
        print('no r_out')
