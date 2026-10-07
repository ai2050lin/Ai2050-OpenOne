# -*- coding: utf-8 -*-
import json, io
RDIR = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913'
D34 = RDIR + r'\phase3134\omega_p132_carrier_matrix_forkcoord_stepscan'
r34 = json.load(io.open(D34 + r'\result.json', encoding='utf-8'))
out = []
out.append('verdict=%s' % r34['verdict'])
pb = r34['part_b']
out.append('part_b keys=%s' % sorted(pb.keys()))
out.append('chg_matrix=%s' % json.dumps(pb.get('chg_matrix'), indent=1))
out.append('dose_grid=%s' % json.dumps(pb.get('dose_grid')))
out.append('doses=%s' % json.dumps(pb.get('doses')))
# grep all keys mentioning dose
for k in sorted(pb.keys()):
    v = pb[k]
    if isinstance(v, (int, float, str, bool)):
        out.append('pb.%s=%r' % (k, v))
io.open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3136_check34.txt','w',encoding='utf-8').write('\n'.join(out))
print('OK')
