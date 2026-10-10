# -*- coding: utf-8 -*-
# p3158_probe.py: safetensors 张量位置 + config 探查（3158 前置）
import os, json, glob
from safetensors import safe_open
ROOT = r'D:\AI2050\Ai2050-OpenOne'
MDIR = {'qwen3-4b': 'qwen3-4b', 'qwen3-14b': 'Qwen3-14B', 'glm4': 'glm4-9b-chat-hf'}
WANT = ('lm_head.weight', 'model.embed_tokens.weight', 'model.norm.weight')
out = []
for m, d in MDIR.items():
    p = os.path.join(ROOT, 'models', 'hf', d)
    lines = ['== %s (%s)' % (m, p)]
    if not os.path.isdir(p):
        lines.append('  DIR MISSING')
        out.extend(lines)
        continue
    cfg = json.load(open(os.path.join(p, 'config.json'), encoding='utf-8'))
    lines.append('  tie=%s hidden=%s vocab=%s layers=%s' % (
        cfg.get('tie_word_embeddings'), cfg.get('hidden_size'),
        cfg.get('vocab_size'), cfg.get('num_hidden_layers')))
    shards = sorted(glob.glob(os.path.join(p, '*.safetensors')))
    lines.append('  shards=%d' % len(shards))
    found = {}
    for sh in shards:
        with safe_open(sh, framework='pt') as f:
            keys = set(f.keys())
            for w in WANT:
                if w in keys and w not in found:
                    found[w] = os.path.basename(sh)
    for w in WANT:
        lines.append('  %s -> %s' % (w, found.get(w, 'ABSENT')))
    out.extend(lines)
open(os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3158_probe.txt'), 'w', encoding='utf-8').write(chr(10).join(out))
print('written')
