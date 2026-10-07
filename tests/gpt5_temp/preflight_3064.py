# 3064 pre-flight: DS7B size + tokenizer target check (no GPU, no observation)
import os
import io
import json

OUT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\preflight_3064.txt')
lines = []
MD = (r'D:\AI2050\Ai2050-OpenOne\models\hf'
      r'\deepseek-r1-distill-qwen-7b')

tot = 0
for fn in sorted(os.listdir(MD)):
    p = os.path.join(MD, fn)
    if os.path.isfile(p):
        sz = os.path.getsize(p)
        tot += sz
        if fn.endswith('.safetensors'):
            lines.append('%s %.2f GiB'
                         % (fn, sz / 2 ** 30))
lines.append('TOTAL %.2f GiB' % (tot / 2 ** 30))

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MD)
BODIES = (
    'The weather was cold, so',
    'He studied every night because',
    'The experiment failed, therefore',
    'He missed the train, however',
    'The garden grows quickly while',
    'The price was high, yet',
    'She speaks French, although',
    'The road was closed, thus',)
TARGETS = ('so', 'because', 'therefore',
           'however', 'while', 'yet',
           'although', 'thus')
PREFIXES = ('', 'In a formal style,',
            'In Shakespearean style,',
            'Regarding the weather,')
for w in TARGETS:
    wi = tok(' ' + w, add_special_tokens=False)[
        'input_ids']
    lines.append('target %-10s n=%d %s'
                 % (w, len(wi), wi))
lens = []
ok_single = True
for bi in range(8):
    for ci in range(4):
        s = (PREFIXES[ci] + ' ' + BODIES[bi]) \
            if PREFIXES[ci] else BODIES[bi]
        ids = tok(s, add_special_tokens=False)[
            'input_ids']
        lens.append(len(ids))
        t = tok(' ' + TARGETS[bi],
                add_special_tokens=False)[
            'input_ids']
        if len(t) != 1 or ids.count(t[0]) != 1:
            ok_single = False
            lines.append('FAIL pair b=%d c=%d '
                         'target=%s' % (bi, ci, t))
lines.append('n_prompts=%d lens min=%d max=%d'
             % (len(lens), min(lens), max(lens)))
lines.append('ok_single=%s' % ok_single)

with io.open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('WROTE', OUT)
