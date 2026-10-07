# -*- coding: utf-8 -*-
"""Diagnose the failed a143 lens anchor from the
run3 npz: inspect the lens cos profile shape."""
import numpy as np

z = np.load(
    r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
    r'\rdc_query_construction_20260913\phase3050'
    r'\omega_p47_kvdeep_dissection_qwen'
    r'\omega_p47_kvdeep_dissection_qwen.npz',
    allow_pickle=True)
cl = z['COS_LENS']  # (36, 24)
ml = z['med_lens']
out = []
out.append('COS_LENS shape=%s' % (cl.shape,))
out.append('COS_LENS[35] per pair: %s'
           % np.array2string(cl[35], precision=4,
                             max_line_width=100))
out.append('med_lens per layer:')
for li in range(36):
    out.append('  L%02d med=%.4f' % (li, ml[li]))
out.append('med_lens unique count=%d'
           % len(np.unique(np.round(ml, 6))))
with open(r'D:\AI2050\Ai2050-OpenOne\gpt5_temp'
          r'\diag3050_lens.txt', 'w',
          encoding='utf-8') as f:
    f.write('\n'.join(out) + '\n')
print('diag written')
