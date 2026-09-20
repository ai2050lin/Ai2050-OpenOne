# -*- coding: utf-8 -*-
"""list keys of phase2921 npz."""
import numpy as np
import os

OUT = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
       r'\rdc_query_construction_20260913'
       r'\phase2921\attr_vocab_expansion')
z = np.load(os.path.join(OUT, 'attr_vocab_expansion.npz'),
            allow_pickle=True)
L = []
for k in z.files:
    a = z[k]
    L.append('%s shape=%s dtype=%s' % (k, a.shape, a.dtype))
with open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2921_npz_keys.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(L) + '\n')
print('OK keys')
