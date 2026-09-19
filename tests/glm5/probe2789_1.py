# -*- coding: utf-8 -*-
import hashlib
import json
import os

B = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913'
for tag, rel in [('2788', r'phase2788\qwen4_position_invariance'),
                 ('2789', r'phase2789\qwen4_insitu_stability')]:
    print('PHASE', tag)
    d = os.path.join(B, rel)
    for f in ['execution.json', 'result.json']:
        p = os.path.join(d, f)
        print(' ', f, hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16],
              os.path.getsize(p))
    ex = json.load(open(os.path.join(d, 'execution.json'),
                        encoding='utf-8'))
    print('  script', ex['source']['sha256'][:16])
