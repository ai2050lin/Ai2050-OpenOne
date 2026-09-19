import json
p = r'D:/AI2050/Ai2050-OpenOne/tests/glm5/result/rdc_query_construction_20260913/phase2853/mlp_transmission/result.json'
v = json.load(open(p, encoding='utf-8'))['verdict']
rep = r'D:/AI2050/Ai2050-OpenOne/tests/glm5/result/_p2853_verdict.txt'
with open(rep, 'w', encoding='utf-8') as f:
    for k in ['n_words', 'T1_verdict', 'g_jac_late_L26_35',
              'g_jac_max_late', 'g_jac_median_abs_late',
              'T2_g_emp_L14_35', 'T2_r_jac_emp',
              'T2_lin_ratio_median', 'max_resid']:
        f.write('%s = %s\n' % (k, json.dumps(v[k])))
    f.write('din_cdir_L26_35 = %s\n' % json.dumps(v['din_cdir_profile'][26:]))
    f.write('dout_cdir_L26_35 = %s\n' % json.dumps(v['dout_cdir_profile'][26:]))
print('done')
