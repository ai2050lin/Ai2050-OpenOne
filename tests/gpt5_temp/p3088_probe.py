# -*- coding: utf-8 -*-
"""p3088 probe: inspect phase3087 A2 npz keys/values
that become GLM4 unit inputs for the n=12 continuum.
Output -> tests/gpt5_temp/p3088_probe_out.txt (bash shim
loses stdout; file is the only reliable channel)."""
import hashlib
import io
import os

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
P87 = os.path.join(BASE, 'phase3087',
                   'omega_p85_glm4_l37_full_arbitration',
                   'omega_p85_glm4_l37_full_arbitration.npz')
out = []
z = np.load(P87, allow_pickle=False)
keys = sorted(z.files)
out.append('N_KEYS=%d' % len(keys))
out.append('KEYS_BEGIN')
for k in keys:
    out.append('  ' + k)
out.append('KEYS_END')
for k in ('FORWARDS', 'SPEC_CLASS', 'VERDICT',
          'L_INJ', 'L_POST', 'SETUP_OK', 'REPRO_OK',
          'SMOKE', 'ELAPSED'):
    if k in keys:
        out.append('%s = %s' % (k, z[k]))
# any INPUT_SHA-style keys?
sha_keys = [k for k in keys
            if 'SHA' in k.upper() or 'REPRO' in k.upper()]
out.append('SHA_LIKE_KEYS: ' + ' '.join(sha_keys))
for p in ('AB', 'AC', 'BC'):
    t = np.asarray(z['T_' + p], dtype=np.float64)
    u = np.asarray(z['U_' + p], dtype=np.float64)
    out.append('T_%s len=%d med=%+.6f  U_%s len=%d '
               'med=%+.6f  MIG_%s=%+.6f'
               % (p, len(t), float(np.median(t)),
                  p, len(u), float(np.median(u)),
                  p, float(z['MIG_' + p])))
for fk in ('A', 'B', 'C'):
    out.append('E3_TOP3_CS_%s = %.6f'
               % (fk, float(z['E3_TOP3_CS_' + fk])))
with io.open(P87, 'rb') as f:
    out.append('sha8(npz) = %s'
               % hashlib.sha256(f.read()).hexdigest()[:8])
io.open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
        r'\p3088_probe_out.txt', 'w',
        encoding='utf-8').write('\n'.join(out) + '\n')
