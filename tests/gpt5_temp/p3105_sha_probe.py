# -*- coding: utf-8 -*-
"""Verify material.json actual sha8 vs result.json recorded value (3105)."""
import hashlib
import io
import json

OUTD = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3105'
        r'\omega_p103_incontext_truth_consistency')

res = json.load(io.open(OUTD + r'\result.json', encoding='utf-8'))
blob = io.open(OUTD + r'\material.json', 'rb').read()
actual = hashlib.sha256(blob).hexdigest()[:8]
lines = []
lines.append('result.json material_sha8 = %s' % res.get('material_sha8'))
lines.append('material.json actual sha8  = %s' % actual)
lines.append('match = %s' % (res.get('material_sha8') == actual))
lines.append('result.json keys = %s' % sorted(res.keys())[:20])
lines.append('verdict = %s' % res.get('verdict'))
lines.append('n_prompts = %s' % res.get('n_prompts'))
# also check smoke result if exists
import os
smoke = OUTD + r'\smoke'
if os.path.isdir(smoke):
    lines.append('smoke dir files = %s' % os.listdir(smoke))
rep = '\n'.join(lines)
io.open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3105_sha_probe_out.txt',
        'w', encoding='utf-8').write(rep + '\n')
print('probe done')
