# -*- coding: utf-8 -*-
import hashlib
import json
import os

B = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913'
d = os.path.join(B, r'phase2787\qwen4_e0_natural_flip')
for f in ['execution.json', 'result.json', 'e0_natural_stats.npz']:
    p = os.path.join(d, f)
    print(f, hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16],
          os.path.getsize(p))
ex = json.load(open(os.path.join(d, 'execution.json'), encoding='utf-8'))
print('script', ex['source']['sha256'][:16])
