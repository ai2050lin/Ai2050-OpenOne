# -*- coding: utf-8 -*-
import json, io, hashlib, numpy as np
RDIR = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913'
D35 = RDIR + r'\phase3135\omega_p133_conduction_co36ablation_window'
out = []
raw = io.open(D35 + r'\result.json', 'rb').read()
out.append('res35 sha8 = %s' % hashlib.sha256(raw).hexdigest()[:8])
r35 = json.loads(raw.decode('utf-8'))
out.append('verdict = %s' % r35['verdict'])
out.append('runtime = %.1fs, smoke=%s' % (r35['runtime_s'], r35['smoke']))
out.append('seal_sha8 = %s' % r35['seal_sha8'])
pb = r35['part_b']
out.append('xphase_P = %r' % pb['xphase_base_match_P'])
out.append('dvec_med_norm = %s' % pb['dvec_med_norm'])
out.append('conduction = %s' % json.dumps(pb['conduction']))
out.append('b_cond = %s' % pb['b_cond'])
out.append('l17_downstream_cos = %s' % pb['l17_downstream_cos'])
pc = r35['part_c']
out.append('co36_sha8_35 = %s jaccard34=%r delta36=%r' % (pc['co36_sha8'], pc['jaccard34'], pc['delta36']))
out.append('c1_trials = %s' % json.dumps({k: v['chg'] for k, v in pc['c1_trials'].items()}))
out.append('c1_first = %s' % json.dumps({k: v['first'] for k, v in pc['c1_trials'].items()}))
out.append('c_abl = %s' % pc['c_abl'])
out.append('overlap = %s' % json.dumps(pc['overlap']))
out.append('c2_trials = %s' % json.dumps({k: v['chg'] for k, v in pc['c2_trials'].items()}))
out.append('c_ovr = %s' % pc['c_ovr'])
pd = r35['part_d']
out.append('d_trials = %s' % json.dumps({k: {'chg': v['chg'], 'first': v['first']} for k, v in pd['d_trials'].items()}))
out.append('d_win/near/share = %s|%s|%s' % (pd['d_win'], pd['d_near'], pd['d_share']))
out.append('w8 cum = %s' % json.dumps(pd['d_trials']['w8']['cum']))
# npz checks
z35 = np.load(D35 + r'\p133_readout.npz', allow_pickle=False)
out.append('npz keys = %s' % sorted(z35.files))
for k in ('co36','co36_rank','top25','bot25','ov50','only50'):
    a = z35[k]
    out.append('%s sha8=%s len=%d' % (k, hashlib.sha256(a.astype(np.int64).tobytes()).hexdigest()[:8], len(a)))
out.append('dvec17 fp16 sha8 = %s' % hashlib.sha256(z35['dvec17'].tobytes()).hexdigest()[:8])
out.append('cos_29 shape=%s med[30,33,38]=%s' % (z35['cos_29'].shape, [None if np.isnan(float(np.nanmedian(z35['cos_29'][r]))) else round(float(np.nanmedian(z35['cos_29'][r])),4) for r in (30,33,38)]))
out.append('rho_29 med[30,33,38]=%s' % [round(float(np.nanmedian(z35['rho_29'][r])),4) for r in (30,33,38)])
out.append('cos_33 med[34,35,36]=%s' % [round(float(np.nanmedian(z35['cos_33'][r])),4) for r in (34,35,36)])
out.append('cos_38 med[39]=%s' % round(float(np.nanmedian(z35['cos_38'][39])),4))
out.append('fstep_w8 sum>0 = %d' % int((z35['fstep_w8'] >= 0).sum()))
txt = '\n'.join(out)
io.open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3136_anchors.txt','w',encoding='utf-8').write(txt)
print('OK')
