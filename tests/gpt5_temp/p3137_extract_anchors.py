# -*- coding: utf-8 -*-
import json, io, hashlib
import numpy as np
RDIR = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913'
D36 = RDIR + r'\phase3136\omega_p134_conddose_crossmatrix_w8drop'
D34 = RDIR + r'\phase3134\omega_p132_carrier_matrix_forkcoord_stepscan'
out = []
raw = io.open(D36 + r'\result.json', 'rb').read()
out.append('res36 sha8 = %s' % hashlib.sha256(raw).hexdigest()[:8])
r36 = json.loads(raw.decode('utf-8'))
out.append('verdict = %s' % r36['verdict'])
out.append('runtime=%.1f smoke=%s seal=%s' % (r36['runtime_s'], r36['smoke'], r36['seal_sha8']))
out.append('xphase = %r' % r36['part_b']['xphase_base_match_P'])
out.append('dvec_med_norm = %s' % json.dumps(r36['part_b']['dvec_med_norm']))
pd = r36['part_d']
out.append('d_res = %s' % json.dumps({k: v['chg'] for k, v in pd['d_res'].items()}))
out.append('d_first = %s' % json.dumps({k: v['first'] for k, v in pd['d_res'].items()}))
out.append('drop1_contrib = %s' % json.dumps(pd['drop1_contrib']))
out.append('top1_k=%r top1_v=%r sum=%r' % (pd['top1_k'], pd['top1_v'], pd['sum_contrib']))
pc = r36['part_c']
out.append('cmat = %s' % json.dumps({k: v['chg'] for k, v in pc['matrix'].items()}))
out.append('cmat_first = %s' % json.dumps({k: v['first'] for k, v in pc['matrix'].items()}))
out.append('c_diag=%s c_uni=%s' % (pc['c_diag'], pc['c_uni']))
out.append('dose_chg = %s' % json.dumps(r36['part_b']['dose_chg']))
out.append('dose_gates = %s' % json.dumps(r36['part_b']['dose_gates']))
out.append('cosdir_curve = %s' % json.dumps(r36['part_b']['cosdir_curve']))
# npz fstep availability
z36 = np.load(D36 + r'\p134_readout.npz', allow_pickle=False)
out.append('npz36 keys n=%d: %s' % (len(z36.files), sorted(z36.files)[:12]))
for k in ('fstep_w8full', 'fstep_drop0', 'fstep_drop3'):
    if k in z36.files:
        a = z36[k]
        out.append('%s len=%d sum>=0=%d' % (k, len(a), int((a >= 0).sum())))
# 3134 chg_matrix (prompt-only 672 rows)
r34 = json.load(io.open(D34 + r'\result.json', encoding='utf-8'))
out.append('res34 verdict = %s' % r34['verdict'])
pb34 = r34['part_b']
out.append('34 part_b keys = %s' % sorted(pb34.keys()))
if 'chg_matrix' in pb34:
    out.append('34 chg_matrix = %s' % json.dumps(pb34['chg_matrix']))
for k in ('dose_grid', 'doses', 'dose_cond'):
    if k in pb34:
        out.append('34 %s = %s' % (k, json.dumps(pb34[k])))
txt = '\n'.join(out)
io.open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3137_anchors.txt', 'w', encoding='utf-8').write(txt)
print('OK')
