# -*- coding: utf-8 -*-
"""Probe: list models/hf inventory + configs of
candidate third models (Glob unreliable here).
Writes report to file."""
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
HF = ROOT + r'\models\hf'
OUT = (ROOT + r'\tests\gpt5_temp'
       r'\p3083_probe_models.txt')
o = []

names = sorted(os.listdir(HF))
o.append('models/hf entries (%d):' % len(names))
for n in names:
    p = os.path.join(HF, n)
    if os.path.isdir(p):
        files = os.listdir(p)
        tot = 0
        nconf = 'config.json' in files
        for f in files:
            fp = os.path.join(p, f)
            if os.path.isfile(fp):
                try:
                    tot += os.path.getsize(fp)
                except OSError:
                    pass
        o.append('  %-40s files=%2d %.2f GB '
                 'config=%s'
                 % (n, len(files), tot / 1e9,
                    nconf))

# configs of likely third-model candidates
for cand in ('qwen2.5-3b', 'qwen2.5-3b-instruct',
             'qwen2.5-3b-hf', 'glm4-9b-chat-hf',
             'glm-4-9b-chat-hf', 'glm4-9b-hf',
             'qwen2.5-7b', 'qwen2.5-7b-instruct'):
    p = os.path.join(HF, cand, 'config.json')
    if os.path.isfile(p):
        try:
            cfg = json.load(io.open(
                p, encoding='utf-8'))
            o.append('== %s ==' % cand)
            for k in ('model_type', 'architectures',
                      'hidden_size', 'num_hidden_layers',
                      'num_attention_heads',
                      'num_key_value_heads',
                      'intermediate_size',
                      'vocab_size',
                      'torch_dtype'):
                if k in cfg:
                    o.append('  %s = %s'
                             % (k, cfg[k]))
        except Exception as e:
            o.append('== %s: config read fail %s'
                     % (cand, e))

io.open(OUT, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
