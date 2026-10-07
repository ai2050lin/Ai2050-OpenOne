# -*- coding: utf-8 -*-
import json, io, hashlib
import numpy as np
D38 = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
       r'\rdc_query_construction_20260913\phase3138'
       r'\omega_p136_statebank_bicdecomp_probecontrast')
out = []
raw = io.open(D38 + r'\result.json', 'rb').read()
out.append('res38 sha8 = %s' % hashlib.sha256(raw).hexdigest()[:8])
r38 = json.loads(raw.decode('utf-8'))
out.append('verdict = %s' % r38['verdict'])
out.append('runtime=%.1f seal=%s' % (r38['runtime_s'], r38['seal_sha8']))
out.append('bank = %s' % json.dumps(r38['bank'], sort_keys=True)[:400])
pc = r38['part_c']
ns = pc['norm_share']
for l in ('0', '8', '17', '29', '33', '38', '39'):
    out.append('norm_share[%s] = %s' % (l, json.dumps({k: round(v, 4) for k, v in ns[l].items()})))
out.append('ish17 = %s' % json.dumps(pc['ish']['17']))
out.append('ish29 = %s' % json.dumps(pc['ish']['29']))
out.append('retr = %s' % json.dumps({k: round(v, 4) for k, v in pc['retr'].items()}))
out.append('resid_pca1 = %s' % json.dumps({k: round(v, 4) for k, v in pc['resid_pca1_share'].items()}))
out.append('gates = %s' % json.dumps(pc['gates']))
out.append('ish_all_med=%r csh_all_med=%r' % (pc['ish_all_med'], pc['csh_all_med']))
pd_ = r38['part_d']
out.append('auc17 = %s' % json.dumps(pd_['auc']['17']))
out.append('auc_x17 = %s' % json.dumps(pd_['auc_x']['17']))
out.append('auc_x_med=%r gen_ok=%r' % (pd_['auc_x_med'], pd_['auc_gen_ok']))
out.append('part_e = %s' % json.dumps(r38['part_e']))
z = np.load(D38 + r'\p136_readout.npz', allow_pickle=False)
out.append('npz keys n=%d' % len(z.files))
io.open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3138_res.txt', 'w', encoding='utf-8').write('\n'.join(out))
print('OK')
