# -*- coding: utf-8 -*-
"""List actual npz keys vs verify need set."""
import io

import numpy as np

NPZ = (r'D:\AI2050\Ai2050-OpenOne\tests'
       r'\glm5\result'
       r'\rdc_query_construction_20260913'
       r'\phase3132'
       r'\omega_p130_forkcausal_'
       r'\single256'.replace(
           r'\single256', r'\single256')
       + '')
NPZ = (r'D:\AI2050\Ai2050-OpenOne\tests'
       r'\glm5\result'
       r'\rdc_query_construction_20260913'
       r'\phase3132'
       r'\omega_p130_forkcausal_single256'
       r'\p130_readout.npz')
OUTF = (r'D:\AI2050\Ai2050-OpenOne'
        r'\tests\gpt5_temp'
        r'\p3132_keys_out.txt')
z = np.load(NPZ, allow_pickle=False)
keys = sorted(z.files)
LAYERS_C = [4, 8, 9, 13, 17, 29, 33, 38]
need = {'co50', 'rho_l17', 'hout_l17',
        'dm31_med17', 'dm31_frac_neg17',
        'tf_idx', 'med_tf_p1', 'med_tf_m1',
        'med_tf_avg', 'sel64', 'sel256'}
for l in LAYERS_C:
    need.add('chg256_%d' % l)
    need.add('same256_%d' % l)
for k in ('17', '38', '33'):
    for m in ('step0', 'allstep'):
        for s in ('p1', 'm1'):
            need.add('same_L%s_%s_%s'
                     % (k, m, s))
for k in ('17', '38'):
    for m in ('step0', 'allstep'):
        for s in ('p1', 'm1'):
            need.add('sameR_L%s_%s_%s'
                     % (k, m, s))
missing = sorted(need - set(keys))
extra = sorted(set(keys) - need)
lines = ['n_keys=%d' % len(keys),
         'missing_from_npz(%d)=%s'
         % (len(missing), missing),
         'extra_in_npz(%d)=%s'
         % (len(extra), extra)]
with io.open(OUTF, 'w',
             encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('done')
