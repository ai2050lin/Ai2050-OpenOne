# -*- coding: utf-8 -*-
"""phase2920 probe round 2: replacement-word tokenization check."""
from transformers import AutoTokenizer

MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
OUT = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\phase2920_probe2.txt'
tok = AutoTokenizer.from_pretrained(
    MD, local_files_only=True, trust_remote_code=True,
    use_fast=True)
CANDS = {
    'speed_lo_add': ['idle', 'static', 'lagging', 'sleepy',
                     'passive', 'frozen', 'drifting', 'torpid'],
    'moist_lo_add': ['dried', 'barren', 'rainless', 'wilted',
                     'powdery', 'crisp', 'thirsty', 'sunbaked',
                     'wizen'],
    'moist_hi_add': ['rainy', 'marshy', 'misty', 'watery',
                     'dripping', 'sopping'],
    'size_hi_add': ['titanic', 'towering', 'mountainous'],
    'size_lo_add': ['mini', 'weeny', 'midget', 'undersized',
                    'teeny'],
    'speed_hi_add': ['speeding', 'zooming', 'flying', 'racing'],
}
L = ['== round2 tokenization (spaced/bare) ==']
for pool, ws in CANDS.items():
    row = []
    for t in ws:
        n1 = len(tok(' ' + t, add_special_tokens=False)
                 ['input_ids'])
        n2 = len(tok(t, add_special_tokens=False)['input_ids'])
        row.append('%s:%d/%d%s' % (t, n1, n2,
                   '*' if (n1 == 1 or n2 == 1) else ''))
    L.append('  %s: %s' % (pool, ', '.join(row)))
with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(L) + '\n')
print('OK probe2')
