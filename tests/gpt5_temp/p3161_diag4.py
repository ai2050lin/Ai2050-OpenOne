# -*- coding: utf-8 -*-
"""diag4: does a layer-level forward hook fire AND does its return value propagate, on tf 4.57?"""
import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MDIR = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
model = AutoModelForCausalLM.from_pretrained(MDIR, dtype=torch.bfloat16).to('cuda').eval()
tok = AutoTokenizer.from_pretrained(MDIR)

calls = []
def observe_hook(module, args, output):
    calls.append((type(output).__name__, getattr(output, 'shape', None) if not isinstance(output, tuple) else output[0].shape))
    return None

def replace_hook(module, args, output):
    out0 = output[0] if isinstance(output, tuple) else output
    new0 = out0.clone()
    new0[:, -1, :] = new0[:, -1, :] + 100.0
    if isinstance(output, tuple):
        return (new0,) + tuple(output[1:])
    return new0

model.model.layers[17].register_forward_hook(observe_hook)
ids = tok('The quick brown fox', return_tensors='pt')['input_ids'].cuda()
with torch.no_grad():
    o_plain = model(input_ids=ids, output_hidden_states=True)
print('observe hook fired:', len(calls), 'first:', calls[0] if calls else None)

# now test replacement propagation
h = model.model.layers[17].register_forward_hook(replace_hook)
with torch.no_grad():
    o_pert = model(input_ids=ids, output_hidden_states=True)
h.remove()
d = (o_pert.hidden_states[18][0, -1] - o_plain.hidden_states[18][0, -1]).abs().max().item()
d19 = (o_pert.hidden_states[19][0, -1] - o_plain.hidden_states[19][0, -1]).abs().max().item()
print('hidden_states[18] (slot L_mid, layer17 out) max|diff| =', d)
print('hidden_states[19] max|diff| =', d19)
print('VERDICT:', 'INJECTION WORKS' if d > 10 else 'INJECTION LOST')
