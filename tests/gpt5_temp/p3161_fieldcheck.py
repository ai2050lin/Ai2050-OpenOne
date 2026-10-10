# -*- coding: utf-8 -*-
import json, io, os
P = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913\phase3161\g4p4_head_attribution'
need_m = ['cls', 'ctrl_all3', 'T', 'C4', 'rand_ratio', 'share_nl_none', 'runtime_s',
          'res_sha8', 'seal_sha8', 'det', 'model']
need_det = ['efficacy_maxabs_Lmid1', 'anchor_check']
need_model = ['H', 'HD', 'oproj']
need_s = ['fpmin_q50_none', 'fpmin_sortedR', 'consumption_window', 'gates', 'verdict',
          'res_sha8', 'seal_sha8']
need_g = ['class_agreement', 'ctrl']
out = []
for m in ('qwen3-4b', 'qwen3-14b', 'glm4'):
    r = json.load(io.open(os.path.join(P, m, 'result.json'), encoding='utf-8'))
    miss = [k for k in need_m if k not in r]
    miss += ['det.' + k for k in need_det if k not in r.get('det', {})]
    miss += ['model.' + k for k in need_model if k not in r.get('model', {})]
    out.append('%s missing=%s' % (m, miss))
rs = json.load(io.open(os.path.join(P, 'summary', 'result_summary.json'), encoding='utf-8'))
miss = [k for k in need_s if k not in rs] + ['gates.' + k for k in need_g if k not in rs.get('gates', {})]
out.append('summary missing=%s' % miss)
sm = json.load(io.open(os.path.join(P, 'qwen3-4b', 'smoke', 'result.json'), encoding='utf-8'))
out.append('smoke keys ok=%s verdict=%s' % (all(k in sm for k in ['verdict', 'res_sha8', 'seal_sha8']), sm['verdict'][:40]))
with io.open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3161_fieldcheck.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out) + '\n')
print('done')
