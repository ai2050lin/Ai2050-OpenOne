import json
import os

import numpy as np

RDIR = (r'D:\AI2050\Ai2050-OpenOne'
        r'\tests\glm5\result'
        r'\rdc_query_construction_20260913')
NP76 = os.path.join(
    RDIR, 'phase3076',
    'omega_p73_cross_prompt_family',
    'omega_p73_cross_prompt_family.npz')
z = np.load(NP76)
out = []
for k in sorted(z.files):
    out.append('%s %s %s'
               % (k, z[k].shape, z[k].dtype))
out.append('SMOKE=%s' % bool(z['SMOKE']))
out.append('VERDICT=%s' % z['VERDICT'])
j71 = json.load(open(
    os.path.join(RDIR, 'phase3071',
                 'omega_p68_attn_head_decomp',
                 'result.json'),
    encoding='utf-8'))
out.append('r34_71 len=%s head5=%s'
           % (len(j71['stats']['head']['r34']),
              j71['stats']['head']['r34'][:5]))
p = (r'D:\AI2050\Ai2050-OpenOne'
     r'\tests\gpt5_temp\p3078_probe1.txt')
with open(p, 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
print('OK')
