# -*- coding: utf-8 -*-
import json, io, hashlib, os
import numpy as np
RDIR = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913'
D38 = RDIR + r'\phase3138\omega_p136_statebank_bicdecomp_probecontrast'
out = []
# full run dir listing
for root, dirs, files in os.walk(D38):
    for f in sorted(files):
        p = os.path.join(root, f)
        out.append('%s (%d bytes)' % (os.path.relpath(p, D38), os.path.getsize(p)))
# result structure
raw = io.open(D38 + r'\result.json', 'rb').read()
out.append('res38 sha8 = %s' % hashlib.sha256(raw).hexdigest()[:8])
r38 = json.loads(raw.decode('utf-8'))
out.append('top keys = %s' % sorted(r38.keys()))
out.append('verdict = %s' % r38['verdict'])
z = np.load(D38 + r'\p136_readout.npz', allow_pickle=False)
out.append('npz keys n=%d: %s' % (len(z.files), sorted(z.files)))
for k in sorted(z.files):
    a = z[k]
    out.append('npz %s shape=%s dtype=%s' % (k, a.shape, a.dtype))
io.open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3139_probe38.txt', 'w', encoding='utf-8').write('\n'.join(out))
print('OK')
