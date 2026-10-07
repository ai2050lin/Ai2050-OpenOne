# -*- coding: utf-8 -*-
import json, io, hashlib
import numpy as np
D39 = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
       r'\rdc_query_construction_20260913\phase3139'
       r'\omega_p137_idinteract_portconsume_rewrite_xmat')
out = []
raw = io.open(D39 + r'\result.json', 'rb').read()
out.append('res39 sha8 = %s' % hashlib.sha256(raw).hexdigest()[:8])
r39 = json.loads(raw.decode('utf-8'))
out.append('verdict = %s' % r39['verdict'])
out.append('runtime=%.1f seal=%s' % (r39['runtime_s'], r39['seal_sha8']))
pc = r39['part_c']
out.append('ish2_key = %s' % json.dumps({k: pc['ish2'][k] for k in ('17', '29', '33', '38')}))
out.append('ish2_all_med=%r' % pc['ish2_all_med'])
out.append('r_retr_key = %s' % json.dumps({k: pc['r_retr'][k] for k in ('17', '29', '33', '38')}))
out.append('i_templ_key = %s' % json.dumps({k: round(pc['i_templ_share'][k], 4) for k in ('17', '29', '33', '38')}))
out.append('gates = %s' % json.dumps(pc['gates']))
pd_ = r39['part_d']
out.append('port = %s' % json.dumps(pd_['port'], indent=1)[:900])
pe = r39['part_e']
out.append('xphase=%r cinj_quiet=%r iinj_max=%r' % (pe['xphase'], pe['cinj_quiet'], pe['iinj_max']))
out.append('e_trials = %s' % json.dumps({k: v['chg'] for k, v in pe['trials'].items()}))
pf = r39['part_f']
out.append('retr_s = %s' % json.dumps(pf['retr_s']))
out.append('retr_pair = %s' % json.dumps(pf['retr_pair']))
out.append('auc_xmat = %s' % json.dumps(pf['auc_xmat']))
z = np.load(D39 + r'\p137_readout.npz', allow_pickle=False)
out.append('npz keys n=%d' % len(z.files))
io.open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3139_res.txt', 'w', encoding='utf-8').write('\n'.join(out))
print('OK')
