# -*- coding: utf-8 -*-
import numpy as np, io
RDIR = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913'
z35 = np.load(RDIR + r'\phase3135\omega_p133_conduction_co36ablation_window\p133_readout.npz', allow_pickle=False)
out = []
for k in ('co36', 'co36_rank', 'top25', 'bot25'):
    if k in z35.files:
        a = z35[k]
        out.append('%s shape=%s dtype=%s' % (k, a.shape, a.dtype))
        out.append('  head=%s' % a[:12].tolist())
        out.append('  tail=%s' % a[-6:].tolist())
cr = z35['co36_rank'].astype(np.int64)
co36 = z35['co36'].astype(np.int64)
out.append('set(rank)==set(co36): %s' % (set(cr.tolist()) == set(co36.tolist())))
out.append('set(rank)==set(range(36)): %s' % (set(cr.tolist()) == set(range(36))))
out.append('len(rank)=%d min=%d max=%d' % (len(cr), cr.min(), cr.max()))
out.append('co36 head=%s' % co36[:12].tolist())
# c1_trials subset names from 3135 result for semantics
import json
r35 = json.load(io.open(RDIR + r'\phase3135\omega_p133_conduction_co36ablation_window\result.json', encoding='utf-8'))
out.append('c1_trials keys=%s' % sorted(r35['part_c']['c1_trials'].keys()))
io.open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3137_rankprobe.txt', 'w', encoding='utf-8').write('\n'.join(out))
print('OK')
