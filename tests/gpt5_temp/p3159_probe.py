# -*- coding: utf-8 -*-
# probe: inspect Qwen3DecoderLayer forward output structure at L_mid-1
import os, sys, json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MDIR = os.path.join(ROOT, 'models', 'hf', 'qwen3-4b')
cfg = json.load(open(os.path.join(MDIR, 'config.json'), encoding='utf-8'))
NL = int(cfg['num_hidden_layers'])
L_MID = int(round(0.5 * NL))
tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MDIR, dtype=torch.bfloat16, trust_remote_code=True).to('cuda').eval()

seen = []
def probe(module, args, output):
    t = type(output).__name__
    if isinstance(output, tuple):
        shapes = [tuple(x.shape) if torch.is_tensor(x) else type(x).__name__ for x in output]
    else:
        shapes = [tuple(output.shape)]
    seen.append((t, shapes))
    return None

model.model.layers[L_MID - 1].register_forward_hook(probe)
ids = tok('我喜欢吃苹果。', add_special_tokens=False)['input_ids']
with torch.no_grad():
    o = model(input_ids=torch.tensor([ids], device='cuda'), output_hidden_states=True)
hs = o.hidden_states
lines = ['seen=%d calls' % len(seen)]
for t, sh in seen[:4]:
    lines.append('%s: %s' % (t, sh))
lines.append('hs slots: %d' % len(hs))
lines.append('hs[L_mid] shape: %s' % (tuple(hs[L_MID].shape),))
lines.append('hs[L_mid-1] shape: %s' % (tuple(hs[L_MID - 1].shape),))
lines.append('logits shape: %s' % (tuple(o.logits.shape),))
lines.append('layer class: %s' % type(model.model.layers[L_MID - 1]).__name__)
open(os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3159_probe.txt'), 'w', encoding='utf-8').write('\n'.join(lines))
print('written')
