# -*- coding: utf-8 -*-
"""Probe 3105 result.json structure for offline reuse (3106 pre-work)."""
import io
import json

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3105'
     r'\omega_p103_incontext_truth_consistency'
     r'\result.json')
res = json.load(io.open(P, encoding='utf-8'))
lines = []
t1 = res['T1_ctx']
lines.append('T1_ctx keys: %s' % sorted(t1.keys()))
k0 = sorted(t1.keys())[0]
lines.append('T1_ctx[%s] fields: %s' % (k0, sorted(t1[k0].keys())))
for k in sorted(t1.keys()):
    v = t1[k]
    lines.append('  %-16s val=%.4f acc_test=%.4f acc_teE=%s'
                 % (k, v.get('val', float('nan')),
                    v.get('acc_test', float('nan')),
                    ('%.4f' % v['acc_testE'])
                    if v.get('acc_testE') is not None else 'None'))
t2 = res['T2_ctx']
lines.append('T2_ctx keys: %s' % sorted(t2.keys()))
for k in sorted(t2.keys()):
    v = t2[k]
    lines.append('  %-16s val=%.4f auc_test=%.4f auc_teE=%s'
                 % (k, v.get('val', float('nan')),
                    v.get('auc_test', float('nan')),
                    ('%.4f' % v['auc_testE'])
                    if v.get('auc_testE') is not None else 'None'))
lines.append('T1_base: %s' % {k: (v.get('acc_test'), v.get('acc_testE'))
                              for k, v in res['T1_base'].items()})
lines.append('T2_base: %s' % {k: (v.get('auc_test'), v.get('auc_testE'))
                              for k, v in res['T2_base'].items()})
lines.append('gates: %s' % res['gates'])
lines.append('m: %s' % {k: v for k, v in res['m'].items()
                        if not isinstance(v, (list, dict))})
io.open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
        r'\p3106_res_probe_out.txt', 'w',
        encoding='utf-8').write('\n'.join(lines) + '\n')
print('probe done')
