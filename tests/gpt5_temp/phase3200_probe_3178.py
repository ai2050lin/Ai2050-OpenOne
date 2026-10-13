# -*- coding: utf-8 -*-
"""Probe sealed p3178 scalars BEFORE 3200 freeze (measured-anchor discipline)."""
import hashlib
import json

P = r'tests/glm5/result/rdc_query_construction_20260913/phase3178/g5b3_ftr14_a2_attribution/result.json'
b = open(P, 'rb').read()
print('file sha8:', hashlib.sha256(b).hexdigest()[:8])
r = json.loads(b.decode('utf-8'))
print('res/seal:', r['res_sha8'], r['seal_sha8'])
print('exch_mean:', repr(r['arm']['exch_mean']))
M = ['qwen3-4b', 'qwen3-14b', 'glm4-9b']
for m in M:
    pm = r['arm']['per_model'][m]
    print(m, 'kout', pm['kout'], 'cos_train_a2 %.4f' % pm['folds']['isa']['cos_train_a2'],
          'cos_held_a2 isa %.4f hasa %.4f' % (pm['folds']['isa']['cos_held_a2'],
                                              pm['folds']['hasa']['cos_held_a2']))
    for rel in ('isa', 'hasa'):
        folds = pm['null'][rel]['folds'] if isinstance(pm['null'][rel], dict) else pm['null'][rel]
        co = sorted(f['cos_out'] for f in folds)
        med = co[len(co) // 2 - 1] / 2 + co[len(co) // 2] / 2 if len(co) % 2 == 0 else co[len(co) // 2]
        # numpy-style median: average of two middle for even n
        n = len(co)
        med = (co[n // 2 - 1] + co[n // 2]) / 2 if n % 2 == 0 else co[n // 2]
        print('  %s: n=%d med_cos_out=%.6f (4dp %.4f) min=%.4f max=%.4f'
              % (rel, n, med, round(med, 4), co[0], co[-1]))
