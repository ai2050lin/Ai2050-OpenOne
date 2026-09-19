# -*- coding: utf-8 -*-
import json
import hashlib
import os
import sys

import numpy as np

B = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913'
d = B + r'\phase2785\qwen4_flip_deoracle'
r = json.load(open(d + r'\result.json', encoding='utf-8'))
sets = {a: set(int(i) for i, f in r['flip_rows'][a].items() if f)
        for a in ['O', 'D1', 'D2', 'D3']}
print('D1 rows', sorted(sets['D1']))
print('D2==D1', sets['D1'] == sets['D2'], ' D3==D1', sets['D1'] == sets['D3'])
print('O rows', sorted(sets['O']))

sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
import phase2747_rdc_material as mat

material, data = mat.freeze()
crows = [x for x in data['diagnostic']
         if x['kind'] == 'controlled_relation']
zp = np.load(B + r'\phase2774' + r'\qwen4_pull_validation' +
             r'\pull_stats.npz', allow_pickle=False)
w2i = {int(i): k for k, i in enumerate(sorted(
    int(i) for i in np.load(B + r'\phase2761' + r'\qwen4_kc_fault' +
                            r'\fault_scores.npz',
                            allow_pickle=False)['wrong_idx']))}
for i in sorted(sets['D1']):
    k = w2i[i]
    print('D1 flip row', i, 'fam', crows[i]['family'],
          'pull %.4f' % zp['pull'][k])
for f in ['execution.json', 'result.json', 'deoracle_stats.npz']:
    p = os.path.join(d, f)
    print(f, hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16],
          os.path.getsize(p))
ex = json.load(open(d + r'\execution.json', encoding='utf-8'))
print('script', ex['source']['sha256'][:16])
