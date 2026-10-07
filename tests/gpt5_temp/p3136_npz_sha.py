# -*- coding: utf-8 -*-
import numpy as np, io, hashlib
RDIR = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913'
z35 = np.load(RDIR + r'\phase3135\omega_p133_conduction_co36ablation_window\p133_readout.npz', allow_pickle=False)
out = []
for k in ('dvec17','dvec29','dvec33','dvec38','dvec_full_17','dvec_full_29','dvec_full_33','dvec_full_38','co36','co36_rank','top25','bot25','only50','ov50'):
    a = z35[k]
    out.append('%s shape=%s dtype=%s sha8=%s' % (k, a.shape, a.dtype, hashlib.sha256(a.tobytes()).hexdigest()[:8]))
z32 = np.load(RDIR + r'\phase3132\omega_p130_forkcausal_single256\p130_readout.npz', allow_pickle=False)
a = z32['co50'].astype(np.int64)
out.append('co50(int64) sha8=%s len=%d' % (hashlib.sha256(a.tobytes()).hexdigest()[:8], len(a)))
u = np.union1d(z35['co36'].astype(np.int64), a)
out.append('union36_50 len=%d sha8=%s' % (len(u), hashlib.sha256(u.tobytes()).hexdigest()[:8]))
dn = z35['dvec_med_norm']
out.append('dvec_med_norm = %r' % [float(v) for v in dn])
io.open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3136_npz_sha.txt','w',encoding='utf-8').write('\n'.join(out))
print('OK')
