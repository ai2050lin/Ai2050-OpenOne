# -*- coding: utf-8 -*-
import os, json
ROOT = r'D:\AI2050\Ai2050-OpenOne'
out = []
for m in ['qwen3-4b', 'qwen3-1.7b', 'qwen2.5-3b-instruct', 'glm4-9b-chat-hf', 'gemma-3-4b-it', 'qwen2-7b']:
    p = os.path.join(ROOT, 'models', 'hf', m, 'config.json')
    if not os.path.exists(p):
        out.append('%-22s MISSING' % m); continue
    J = json.load(open(p, encoding='utf-8'))
    out.append('%-22s tie=%-5s layers=%-4s hidden=%-5s vocab=%-7s arch=%s' % (
        m, J.get('tie_word_embeddings'), J.get('num_hidden_layers'),
        J.get('hidden_size'), J.get('vocab_size'), J.get('architectures')))
open(os.path.join(ROOT, 'gpt5_temp', 'check_tie.txt'), 'w', encoding='utf-8').write('\n'.join(out))
print('\n'.join(out))
