import json
p = r'D:/AI2050/Ai2050-OpenOne/tests/glm5/result/rdc_query_construction_20260913/phase2854/gain_specificity/result.json'
v = json.load(open(p, encoding='utf-8'))['verdict']
rep = r'D:/AI2050/Ai2050-OpenOne/tests/glm5/result/_p2854_verdict.txt'
with open(rep, 'w', encoding='utf-8') as f:
    for k in ['n_words', 'Z1_cdir_specific', 'z_cdir_L26_35',
              'z_min_late', 'z_median_late', 'Z2_operating_point_shift',
              'ratio_clamp_over_base_L26_35', 'g_cdir_base_L26_35',
              'g_cdir_clamp_L26_35', 'g_rand_base_mean_L26_35',
              'D1_r_vs_2853', 'max_resid', 'final_verdict']:
        f.write('%s = %s\n' % (k, json.dumps(v[k])))
print('done')
