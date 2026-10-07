# -*- coding: utf-8 -*-
"""Extract full ladder curves from 3065 npz."""
import io
import numpy as np

P = (r"D:\AI2050\Ai2050-OpenOne\tests\glm5\result"
     r"\rdc_query_construction_20260913\phase3065"
     r"\omega_p62_v_sign_orchestration"
     r"\omega_p62_v_sign_orchestration.npz")
OUT = (r"D:\AI2050\Ai2050-OpenOne\tests\glm5\result"
       r"\rdc_query_construction_20260913"
       r"\phase3065\omega_p62_v_sign_orchestration"
       r"\ladder_curves.txt")

z = np.load(P)
o = []
for tag in ('qwen3_1p7b', 'qwen3_4b', 'ds7b'):
    mc = z[tag + '_MED_C']
    cv = z[tag + '_MED_CV']
    rho = z[tag + '_MED_RHO']
    o.append('=== %s (NL=%d) ===' % (tag, len(mc)))
    o.append('med_c   : ' + ' '.join(
        '%+.3f' % v for v in mc))
    o.append('med_cV  : ' + ' '.join(
        '%+.3f' % v for v in cv))
    o.append('med_rho : ' + ' '.join(
        '%.3f' % v for v in rho))
    # late layers detail
    o.append('last-6 med_c: ' + ' '.join(
        '%+.4f' % v for v in mc[-6:]))
    o.append('')

# qwen3_4b flip localization: which layer does the
# sign turn negative?
mc4 = z['qwen3_4b_MED_C']
neg_layers = [i for i, v in enumerate(mc4)
              if v < 0]
o.append('qwen3_4b negative layers: %s' % neg_layers)
mc1 = z['qwen3_1p7b_MED_C']
o.append('qwen3_1p7b negative layers: %s'
         % [i for i, v in enumerate(mc1) if v < 0])
mc7 = z['ds7b_MED_C']
o.append('ds7b negative layers: %s'
         % [i for i, v in enumerate(mc7) if v < 0])

with io.open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(o) + '\n')
print('EXTRACT_OK')
