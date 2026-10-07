# -*- coding: utf-8 -*-
"""p3087 env probe 2: GLM4 feasibility + gemma3 config + phase3081 loading."""
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUT = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3087_env2.txt'
o = []

# 1. models root files (the 12 files)
md = os.path.join(ROOT, 'models')
o.append('MODELS ROOT files: %s'
         % sorted(os.listdir(md)))

# 2. search whole workspace (top dirs) for glm
for top in ('models', 'shared', 'scripts'):
    d = os.path.join(ROOT, top)
    if not os.path.isdir(d):
        continue
    hits = []
    for dirpath, dirnames, filenames in \
            os.walk(d):
        for n in dirnames + filenames:
            if 'glm' in n.lower() and \
                    'glm5' not in n.lower():
                hits.append(os.path.join(
                    dirpath[len(ROOT):], n))
        if len(hits) > 30:
            break
    o.append('GLM hits under %s: %d'
             % (top, len(hits)))
    for h in hits[:30]:
        o.append('  %s' % h)

# 3. transformers: does it ship glm4 / chatglm?
import transformers
mdir = os.path.join(
    os.path.dirname(transformers.__file__),
    'models')
mods = sorted(os.listdir(mdir))
glmish = [m for m in mods
          if 'glm' in m or 'chatglm' in m]
o.append('transformers models dir glm-ish: %s'
         % glmish)
o.append('total model modules: %d' % len(mods))
o.append('gemma3 present: %s'
         % ('gemma3' in mods))
o.append('qwen3 present: %s' % ('qwen3' in mods))

# 4. gemma3 text config (for L_INJ planning)
p = os.path.join(
    ROOT, 'models', 'hf', 'gemma-3-4b-it',
    'config.json')
c = json.load(io.open(p, encoding='utf-8'))
tc = c.get('text_config', c)
o.append('GEMMA3 text_config: model_type=%s '
         'layers=%s hidden=%s heads=%s kv=%s '
         'vocab=%s dtype=%s'
         % (tc.get('model_type'),
            tc.get('num_hidden_layers'),
            tc.get('hidden_size'),
            tc.get('num_attention_heads'),
            tc.get('num_key_value_heads'),
            tc.get('vocab_size'),
            c.get('torch_dtype')))

# 5. phase3081 model loading block
scr = os.path.join(
    ROOT, 'tests', 'glm5',
    'phase3081_omega_p78_ds7b_crossmodel.py')
t = io.open(scr, encoding='utf-8').read()
lines = t.split('\n')
for i, ln in enumerate(lines):
    if 'from_pretrained' in ln:
        lo = max(0, i - 6)
        hi = min(len(lines), i + 8)
        o.append('--- 3081 loading block ---')
        for j in range(lo, hi):
            o.append('  L%d: %s'
                     % (j + 1, lines[j][:100]))
        break

# 6. which model path does 3081 use
for i, ln in enumerate(lines):
    if 'models' in ln and ('qwen2-7b' in ln
                           or 'hf' in ln):
        o.append('  path L%d: %s'
                 % (i + 1, ln.strip()[:100]))

io.open(OUT, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
