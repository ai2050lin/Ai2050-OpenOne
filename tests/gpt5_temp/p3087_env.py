# -*- coding: utf-8 -*-
"""p3087 env probe: models dir, GLM4 config, transformers version, DS7B loading strategy."""
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUT = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3087_env.txt'
o = []

# 1. models tree (top levels)
md = os.path.join(ROOT, 'models')
for dirpath, dirnames, filenames in os.walk(md):
    depth = dirpath[len(md):].count(os.sep)
    if depth <= 2:
        o.append('DIR %s (%d files)'
                 % (dirpath[len(md):] or '\\',
                    len(filenames)))
    else:
        dirnames[:] = []
        continue
    if depth == 2:
        for f in filenames[:20]:
            o.append('   %s' % f)

# 2. look for glm4 config.json anywhere in models
for dirpath, dirnames, filenames in os.walk(md):
    for f in filenames:
        if f == 'config.json':
            p = os.path.join(dirpath, f)
            try:
                c = json.load(io.open(
                    p, encoding='utf-8'))
                o.append('CONFIG %s' % p)
                o.append('  architectures=%s'
                         % c.get('architectures'))
                o.append('  model_type=%s '
                         'layers=%s hidden=%s '
                         'heads=%s kv=%s'
                         % (c.get('model_type'),
                            c.get('num_hidden_layers'),
                            c.get('hidden_size'),
                            c.get('num_attention_heads'),
                            c.get('num_key_value_heads')))
                o.append('  torch_dtype=%s '
                         'vocab=%s'
                         % (c.get('torch_dtype'),
                            c.get('vocab_size')))
            except Exception as e:
                o.append('CONFIG %s ERR %s'
                         % (p, e))

# 3. transformers version
try:
    import transformers
    o.append('transformers=%s torch=%s'
             % (transformers.__version__,
                __import__('torch').__version__))
    import transformers.models.auto as auto
    names = [n for n in dir(auto.configuration_auto)
             if 'glm' in n.lower()]
    o.append('glm classes: %s' % names)
except Exception as e:
    o.append('transformers ERR %s' % e)

# 4. how did 3081 load DS7B (grep-like scan of the
#    phase3081 script for device_map / dtype lines)
scr = os.path.join(
    ROOT, 'tests', 'glm5',
    'phase3081_omega_p77_ds7b_crossmodel.py')
if not os.path.exists(scr):
    cands = [f for f in os.listdir(
        os.path.join(ROOT, 'tests', 'glm5'))
        if '3081' in f or 'ds7b' in f]
    o.append('3081 candidates: %s' % cands)
    if cands:
        scr = os.path.join(ROOT, 'tests', 'glm5',
                           cands[0])
o.append('scan %s' % scr)
if os.path.exists(scr):
    t = io.open(scr, encoding='utf-8').read()
    for i, ln in enumerate(t.split('\n'), 1):
        s = ln.strip()
        if any(k in s for k in (
                'from_pretrained', 'device_map',
                'torch_dtype', 'offload',
                'max_memory')):
            o.append('  L%d: %s' % (i, s[:110]))

io.open(OUT, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
