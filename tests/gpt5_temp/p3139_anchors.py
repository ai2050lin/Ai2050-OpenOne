# -*- coding: utf-8 -*-
import json, io, hashlib
import numpy as np
RDIR = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913'
D38 = RDIR + r'\phase3138\omega_p136_statebank_bicdecomp_probecontrast'
D35 = RDIR + r'\phase3135\omega_p133_conduction_co36ablation_window'
out = []
z35 = np.load(D35 + r'\p133_readout.npz', allow_pickle=False)
for k in ('dvec17', 'dvec29', 'dvec33', 'dvec38'):
    a = z35[k]
    out.append('%s shape=%s dtype=%s sha8=%s mednorm=%.6f' % (
        k, a.shape, a.dtype,
        hashlib.sha256(a.tobytes()).hexdigest()[:8],
        float(np.median(np.linalg.norm(a.astype(np.float32), axis=1)))))
out.append('dvec_med_norm_35 = %r' % [float(v) for v in z35['dvec_med_norm']])
z38 = np.load(D38 + r'\p136_readout.npz', allow_pickle=False)
for k in ('retr', 'csh_med', 'ish_med', 'auc_raw_x', 'auc_I_x', 'auc_BI_x',
          'norm_share_B', 'norm_share_I', 'norm_share_C', 'norm_share_R'):
    a = z38[k]
    out.append('%s = %s' % (k, json.dumps([round(float(v), 6) for v in a.tolist()])))
r38 = json.load(io.open(D38 + r'\result.json', encoding='utf-8'))
out.append('res38 verdict = %s' % r38['verdict'])
out.append('res38 auc_x_med = %r auc_gen_ok=%r' % (r38['part_d']['auc_x_med'], r38['part_d']['auc_gen_ok']))
out.append('res38 retr_key = %s' % json.dumps({k: round(v, 6) for k, v in r38['part_c']['retr'].items() if k in ('17', '29', '33', '38')}))
out.append('res38 csh_key = %s' % json.dumps({k: r38['part_c']['csh'][k] for k in ('17', '29', '33', '38')}))
out.append('res38 resid_l29 = %r' % r38['part_c']['resid_pca1_share']['29'])
out.append('res38 bank lens_med = %s' % json.dumps(r38['bank']['lens_med']))
# material.json structure for new-material selection
mat5 = json.load(io.open(RDIR + r'\phase3105\omega_p103_incontext_truth_consistency\material.json', encoding='utf-8'))
out.append('mat keys = %s' % sorted(mat5.keys()))
out.append('n_entities=%d n_preds=%d pair2rel n=%d' % (
    len(mat5['entities']), len(mat5['predicates']), len(mat5['pair2rel'])))
out.append('entities head = %s' % json.dumps(mat5['entities'][:6]))
out.append('predicates = %s' % json.dumps(mat5['predicates']))
out.append('pair2rel head = %s' % json.dumps(dict(list(mat5['pair2rel'].items())[:4])))
out.append('false_rels head = %s' % json.dumps(dict(list(mat5['false_rels'].items())[:3])))
# which (s,o) pairs exist
pairs = sorted(mat5['pair2rel'].keys())
ss = sorted(set(int(p.split('_')[0]) for p in pairs))
oo = sorted(set(int(p.split('_')[1]) for p in pairs))
out.append('pair s range=%d..%d o range=%d..%d n=%d' % (ss[0], ss[-1], oo[0], oo[-1], len(pairs)))
io.open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3139_anchors.txt', 'w', encoding='utf-8').write('\n'.join(out))
print('OK')
