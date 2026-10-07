# -*- coding: utf-8 -*-
"""p3087 env probe 3: glm4 dir contents and sizes."""
import io
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
D = os.path.join(ROOT, 'models', 'hf',
                 'glm4-9b-chat-hf')
OUT = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3087_env3.txt'
o = []
if not os.path.isdir(D):
    o.append('DIR MISSING %s' % D)
else:
    total = 0
    for f in sorted(os.listdir(D)):
        p = os.path.join(D, f)
        if os.path.isfile(p):
            sz = os.path.getsize(p)
            total += sz
            o.append('%-42s %12.1f MB'
                     % (f, sz / 1048576.0))
        else:
            o.append('%-42s <DIR>'
                     % f)
    o.append('TOTAL %.2f GB'
             % (total / 1073741824.0))
    cfg = os.path.join(D, 'config.json')
    if os.path.exists(cfg):
        import json
        c = json.load(io.open(
            cfg, encoding='utf-8'))
        o.append('config keys: %s'
                 % sorted(c.keys()))
        for k in ('architectures', 'model_type',
                  'num_hidden_layers', 'hidden_size',
                  'num_attention_heads',
                  'num_key_value_heads',
                  'vocab_size', 'torch_dtype',
                  'torch_dtype '):
            if k in c:
                o.append('  %s = %s'
                         % (k, c[k]))
io.open(OUT, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
