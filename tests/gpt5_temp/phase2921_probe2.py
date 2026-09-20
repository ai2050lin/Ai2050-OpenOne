# -*- coding: utf-8 -*-
"""phase2921 probe round 2: moisture-pool top-up screening."""
from transformers import AutoTokenizer

MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
OUT = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\phase2921_probe2.txt'
tok = AutoTokenizer.from_pretrained(
    MD, local_files_only=True, trust_remote_code=True,
    use_fast=True)
CANDS = {
    'moist_hi_2': ['sweaty', 'lush', 'pulpy', 'teary', 'boggy',
                   'oozy', 'drippy', 'wettish', 'stormy',
                   'splashing', 'sprayed', 'wetting', 'wetter',
                   'wettest', 'damper', 'dampest', 'moister',
                   'drenching', 'dampening', 'watering', 'raining',
                   'splattered', 'dabbled', 'dunked', 'dipped'],
    'moist_lo_2': ['mealy', 'chalky', 'ashy', 'gritty',
                   'scratchy', 'withering', 'baking', 'toasty',
                   'roasted', 'seared', 'dryish', 'singed',
                   'charred', 'dusted', 'sanded', 'crumbled',
                   'granular', 'parch', 'wither', 'sapless'],
}
L = ['== round2 moisture top-up ==']
for pool, ws in CANDS.items():
    ok, bad = [], []
    for t in ws:
        n1 = len(tok(' ' + t, add_special_tokens=False)
                 ['input_ids'])
        n2 = len(tok(t, add_special_tokens=False)['input_ids'])
        (ok if (n1 == 1 or n2 == 1) else bad).append(t)
    L.append('%s ok(%d): %s' % (pool, len(ok), ok))
    L.append('%s rejected(%d): %s' % (pool, len(bad), bad))
with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(L) + '\n')
print('OK probe2 2921')
