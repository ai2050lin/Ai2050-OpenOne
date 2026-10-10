# -*- coding: utf-8 -*-
# debug: which slots of hidden_states change when injecting at block L_mid-1 output
import os, sys, json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MDIR = os.path.join(ROOT, 'models', 'hf', 'qwen3-4b')
cfg = json.load(open(os.path.join(MDIR, 'config.json'), encoding='utf-8'))
NL = int(cfg['num_hidden_layers'])
L_MID = int(round(0.5 * NL))
tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MDIR, dtype=torch.bfloat16, trust_remote_code=True).to('cuda').eval()

INJ = {'on': False, 'delta': None}
def _inj_hook(module, args, output):
    if INJ['on'] and INJ['delta'] is not None:
        out0 = output[0] if isinstance(output, tuple) else output
        new0 = out0.clone()
        new0[:, -1, :] = new0[:, -1, :] + INJ['delta'].to(new0.dtype)
        if isinstance(output, tuple):
            return (new0,) + tuple(output[1:])
        return new0
    return None

model.model.layers[L_MID - 1].register_forward_hook(_inj_hook)

ids = tok('苹果是水果。', add_special_tokens=False)['input_ids']
def fwd(inj):
    ii = torch.tensor([ids], device='cuda')
    with torch.no_grad():
        o = model(input_ids=ii, output_hidden_states=True)
    hs = np.stack([h[0, -1].float().cpu().numpy() for h in o.hidden_states], 0)
    lg = o.logits[0, -1].float().cpu().numpy().astype(np.float64)
    return lg, hs

lg0, hs0 = fwd(False)
D = hs0.shape[1]
u = np.zeros(D, np.float32); u[0] = 1.0
INJ['on'] = True
INJ['delta'] = torch.from_numpy((np.array([0.5])[:, None] * u[None, :]))
lg1, hs1 = fwd(True)
INJ['on'] = False
dh = hs1 - hs0
norms = np.abs(dh).max(axis=1)
lines = ['L_MID=%d NL=%d' % (L_MID, NL)]
lines.append('dh slot norms (max abs):')
for i in range(NL + 1):
    lines.append('  slot %2d: %.6g' % (i, norms[i]))
lines.append('base hmid norm=%.3f' % float(np.linalg.norm(hs0[L_MID])))
lines.append('layer type returned: tensor' )
open(os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3159_dbg.txt'), 'w', encoding='utf-8').write('\n'.join(lines))
print('written')
