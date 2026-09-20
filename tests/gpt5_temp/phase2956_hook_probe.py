# -*- coding: utf-8 -*-
"""Probe: which module hooks fire on the load_native model?
Diagnose why input_layernorm pre-hook never fires in 2956.
Writes report to tests/gpt5_temp/phase2956_hook_probe.txt
"""
import sys

import numpy as np

sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
from phase2662_symmetric_mapping_contract import load_native
from transformers import AutoTokenizer

MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
OUT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\phase2956_hook_probe.txt')
lines = []

model, _ = load_native('qwen4')
model.eval()
layers = model.model.layers
l0 = layers[0]
lines.append('layer0 type: %s' % type(l0).__name__)
lines.append('layer0 attrs (norm/mlp/attn related): %s'
             % [n for n in dir(l0)
                if not n.startswith('_')
                and any(k in n.lower()
                        for k in ('norm', 'mlp', 'attn'))])
sa = l0.self_attn
lines.append('self_attn type: %s' % type(sa).__name__)

tok = AutoTokenizer.from_pretrained(
    MD, local_files_only=True, trust_remote_code=True,
    use_fast=True)
ids = tok(' the word', add_special_tokens=False)['input_ids']
toks = [tok(' the', add_special_tokens=False)
        ['input_ids'][0], ids[-1]]

fired = {}


def mk(name):
    def h(module, args, kwargs):
        fired[name] = fired.get(name, 0) + 1
        return None
    return h


h1 = l0.input_layernorm.register_forward_pre_hook(
    mk('iln0_pre'), with_kwargs=True) \
    if hasattr(l0, 'input_layernorm') else None
h2 = l0.mlp.register_forward_pre_hook(
    mk('mlp0_pre'), with_kwargs=True)
h3 = l0.mlp.register_forward_hook(mk('mlp0_post'))
h4 = sa.register_forward_pre_hook(
    mk('sa0_pre'), with_kwargs=True)
h5 = sa.register_forward_hook(mk('sa0_post'))
h6 = sa.o_proj.register_forward_pre_hook(
    mk('oproj0_pre'), with_kwargs=True)
h7 = model.model.norm.register_forward_pre_hook(
    mk('finalnorm_pre'), with_kwargs=True)

import torch
with torch.no_grad():
    model(torch.tensor([toks], device='cuda'))

lines.append('fired: %s' % fired)

# try decoder layer pre-hook capture of residual (banned for
# kwargs-style per lesson 23, but test plain positional)
fired2 = {}


def dl_pre(module, args):
    fired2['dl_args'] = len(args)
    if args:
        x = args[0]
        fired2['dl_shape'] = tuple(x.shape)
    return None


try:
    h8 = l0.register_forward_pre_hook(dl_pre)
    with torch.no_grad():
        model(torch.tensor([toks], device='cuda'))
    lines.append('decoder pre (positional) fired: %s' % fired2)
    h8.remove()
except Exception as e:
    lines.append('decoder pre positional failed: %r' % e)

# check input_layernorm forward signature
try:
    import inspect
    lines.append('iln type: %s'
                 % type(l0.input_layernorm).__name__)
    lines.append('iln forward sig: %s'
                 % str(inspect.signature(
                     l0.input_layernorm.forward)))
except Exception as e:
    lines.append('iln inspect failed: %r' % e)

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('PROBE OK')
