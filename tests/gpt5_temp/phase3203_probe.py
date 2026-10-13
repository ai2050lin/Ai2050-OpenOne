# -*- coding: utf-8 -*-
"""Phase 3203 probe: sealed scalars availability for GAP-4 residual re-read."""
import io, json, hashlib, os

SD = r'D:/AI2050/Ai2050-OpenOne/tests/glm5/result/rdc_query_construction_20260913'
def sha8(p): return hashlib.sha256(open(p,'rb').read()).hexdigest()[:8]

# candidate phases
for ph, sub in [('3169', None), ('3172', None), ('3175', None),
                ('3188', None), ('3189', None), ('3190', None),
                ('3191', None), ('3192', None)]:
    d = os.path.join(SD, 'phase' + ph)
    if not os.path.isdir(d):
        print(ph, 'MISSING DIR'); continue
    for name in os.listdir(d):
        rj = os.path.join(d, name, 'result.json')
        if os.path.exists(rj):
            r = json.load(io.open(rj, encoding='utf-8'))
            print(ph, name, 'sha8', sha8(rj), '| res', r.get('res_sha8'), 'seal', r.get('seal_sha8'),
                  '| verdict:', str(r.get('verdict'))[:110])
