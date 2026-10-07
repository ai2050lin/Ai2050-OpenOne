# -*- coding: utf-8 -*-
"""p3087 struct probe: verify GLM4 module tree
matches the 3084 hook contract (v_proj/o_proj/
mlp/attn shapes), bf16 load."""
import io
import time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MDIR = ROOT + r'\models\hf\glm4-9b-chat-hf'
OUT = ROOT + r'\tests\gpt5_temp\p3087_struct.txt'
o = []


def w(m):
    o.append(str(m))
    io.open(OUT, 'w', encoding='utf-8').write(
        '\n'.join(o) + '\n')


import torch
from transformers import AutoModelForCausalLM, \
    AutoTokenizer

tok = AutoTokenizer.from_pretrained(MDIR)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MDIR, dtype=torch.bfloat16,
    attn_implementation='eager').to('cuda').eval()
w('loaded %.1fs' % (time.time() - t0))
L0 = model.model.layers[0]
w('attn attrs: %s'
  % [x for x in dir(L0.self_attn)
     if x.endswith('_proj')])
w('mlp type: %s' % type(L0.mlp).__name__)
w('layer type: %s' % type(L0).__name__)
w('config: kv=%s nq=%s hid=%s vocab=%s '
  'inter=%s head_dim=%s tied=%s'
  % (model.config.num_key_value_heads,
     model.config.num_attention_heads,
     model.config.hidden_size,
     model.config.vocab_size,
     model.config.intermediate_size,
     getattr(model.config, 'head_dim', None),
     getattr(model.config,
             'tie_word_embeddings', None)))

ids = tok('The apple is a fruit.',
          return_tensors='pt')['input_ids'] \
    .to('cuda')
cap = {}
for key, mod in (('v', L0.self_attn.v_proj),
                 ('oin', L0.self_attn.o_proj),
                 ('attn', L0.self_attn),
                 ('mlp', L0.mlp),
                 ('layer', L0)):

    def mk(k):
        def h(module, args, out):
            if k == 'oin':
                cap[k] = tuple(
                    a.shape for a in args
                    if torch.is_tensor(a))
            else:
                t = out[0] if isinstance(
                    out, tuple) else out
                cap[k] = tuple(t.shape)
        return h
    mod.register_forward_hook(mk(key))
with torch.no_grad():
    model(ids, use_cache=False)
for k in sorted(cap):
    w('cap %s: %s' % (k, cap[k]))

# b7a-style identity check at one layer:
# o_proj input replacement with itself
try:
    w('index_json exists: %s'
      % __import__('os').path.exists(
          MDIR + r'\model.safetensors.index.json'))
except Exception:
    pass
w('STRUCT_DONE')
