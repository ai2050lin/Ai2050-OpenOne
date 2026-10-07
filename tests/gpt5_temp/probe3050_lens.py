# -*- coding: utf-8 -*-
"""Probe: what does the decoder-layer forward hook
actually capture on this transformers version?
Compare capH[li] against hidden_states[li+1] (the
independent pre-norm layer output) and test the
lens composition at layer 35 vs the real logits."""
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
NL = len(model.model.layers)
o.append('NL=%d norm=%s lm_head=%s tied=%s'
         % (NL, type(model.model.norm).__name__,
            type(model.lm_head).__name__,
            bool(model.config.tie_word_embeddings)))

cap = {}


def mk(li):
    def h(module, inp, out):
        cap[li] = out[0].detach().clone()
    return h


for li in range(NL):
    model.model.layers[li].register_forward_hook(
        mk(li))

ids = tok('In a formal style, The weather was '
          'cold, so', add_special_tokens=False)[
    'input_ids']
with torch.no_grad():
    outm = model(torch.tensor([ids], device='cuda'),
                 use_cache=True,
                 output_hidden_states=True)
hs = outm.hidden_states
o.append('n hidden_states=%d hs[0]%s hs[1]%s '
         'hs[-1]%s' % (len(hs), tuple(hs[0].shape),
                       tuple(hs[1].shape),
                       tuple(hs[-1].shape)))
logits = outm.logits[0, -1].detach()
o.append('cap[0]%s cap[1]%s cap[35]%s'
         % (tuple(cap[0].shape), tuple(cap[1].shape),
            tuple(cap[NL - 1].shape)))
for li in (0, 1, 2, 3, 17, 35):
    d = float((cap[li][0] - hs[li + 1][0])
              .abs().max())
    o.append('cap[L%d] vs hs[L%d+1]: maxdiff=%.3e'
             % (li, li, d))
d01 = float((cap[0] - cap[1]).abs().max())
d12 = float((cap[1] - cap[2]).abs().max())
o.append('cap[0] vs cap[1] maxdiff=%.3e | cap[1] '
         'vs cap[2] maxdiff=%.3e' % (d01, d12))
# lens composition at the last layer
with torch.no_grad():
    hH = cap[NL - 1]
    if hH.dim() == 3:
        hH = hH[0]
    ln = model.model.norm(hH)
    lg = model.lm_head(ln)[0, -1]
    d_lg = float((lg - logits).abs().max())
    # also on the model's own post-norm last hidden
    ln2 = model.model.norm(hs[NL][0])
    lg2 = model.lm_head(ln2)[0, -1]
    d_lg2 = float((lg2 - logits).abs().max())
    # and hs[-1] (= hs[NL]) vs lm_head directly
    lg3 = model.lm_head(hs[NL][0])[0, -1]
    d_lg3 = float((lg3 - logits).abs().max())
o.append('lens(cap[L35]) vs logits: %.3e' % d_lg)
o.append('lens(norm(hs[NL])) vs logits: %.3e'
         % d_lg2)
o.append('lm_head(hs[NL]) vs logits: %.3e' % d_lg3)
o.append('logits[:8]=%s' % np.array2string(
    logits[:8].cpu().numpy(), precision=3))
with open(r'D:\AI2050\Ai2050-OpenOne\gpt5_temp'
          r'\probe3050_lens.txt', 'w',
          encoding='utf-8') as f:
    f.write('\n'.join(o) + '\n')
print('probe written')
