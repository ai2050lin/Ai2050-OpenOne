"""P2836 probe: why does the residual increment identity fail?
For word 'apple', condition 'same', compare hs[l+1]-hs[l] vs
attn_out[l] + mlp_out[l] per layer; print norms to a report file.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, 'D:/AI2050/Ai2050-OpenOne/tests/glm5')

import rdc_construction_common as cc

ROOT = cc.ROOT

lines = []


def say(s):
    lines.append(s)


import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(str(ROOT / 'models' / 'hf' / 'qwen3-4b'),
                                    local_files_only=True,
                                    trust_remote_code=True, use_fast=True)
from phase2662_symmetric_mapping_contract import load_native  # noqa: E402
model, _ = load_native('qwen4')
model.eval()
say('model type: %s' % type(model.model.layers[0]).__name__)
say('layer forward repr keys: %s'
    % [n for n, _ in model.model.layers[0].named_children()])

cap = {'attn': {}, 'mlp': {}}


def make_out_hook(kind, li):
    def hook(module, args, output):
        o = output[0] if isinstance(output, tuple) else output
        cap[kind].setdefault(li, []).append(
            o[0].detach().float().cpu().numpy())
    return hook


handles = []
for li, layer in enumerate(model.model.layers):
    handles.append(layer.self_attn.register_forward_hook(
        make_out_hook('attn', li)))
    handles.append(layer.mlp.register_forward_hook(
        make_out_hook('mlp', li)))


def run(tokens, pos):
    for d in ('attn', 'mlp'):
        for li in cap[d]:
            del cap[d][li][:]
    with torch.no_grad():
        out = model(torch.tensor([tokens], device='cuda'),
                    output_hidden_states=True)
        hs = np.stack([h[0, pos, :].float().cpu().numpy()
                       for h in out.hidden_states])
    attn = np.stack([cap['attn'][li][0][pos] for li in range(36)])
    mlp = np.stack([cap['mlp'][li][0][pos] for li in range(36)])
    return hs, attn, mlp


apple = tok(' apple', add_special_tokens=False)['input_ids']
assert len(apple) == 1
the = tok(' the', add_special_tokens=False)['input_ids']
hs, attn, mlp = run([the[0], apple[0]], 1)
for li in range(36):
    inc = hs[li + 1] - hs[li]
    dev = np.linalg.norm(inc - (attn[li] + mlp[li]))
    say('L%02d ||hs||=%9.3f ||inc||=%9.3f ||attn||=%9.3f ||mlp||=%9.3f '
        'dev=%9.3f' % (li, np.linalg.norm(hs[li]), np.linalg.norm(inc),
                       np.linalg.norm(attn[li]), np.linalg.norm(mlp[li]),
                       dev))

Path('D:/AI2050/Ai2050-OpenOne/tests/gpt5_temp/p2836_probe_report.txt') \
    .write_text('\n'.join(lines), encoding='utf-8')
print('probe done')
