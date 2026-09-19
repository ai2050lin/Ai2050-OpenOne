import json
p = r'D:/AI2050/Ai2050-OpenOne/tests/glm5/result/rdc_query_construction_20260913/phase2852/emergence_source/result.json'
v = json.load(open(p, encoding='utf-8'))['verdict']
keys = ['n_words', 'S1_layer_concentration', 'S1_share', 'source_layers',
        'S2_head_localization', 'S2_share', 'top5_head_pairs',
        'n_front_in_top5', 'S3_identity_closed', 'S3_rel_err',
        'profile_delta_L14_35', 'inc_sum_L14_34',
        'deep_neg_total_L30_34', 'deep_attn_L30_34', 'deep_mlp_L30_34',
        'max_resid', 'final_verdict']
rep = r'D:/AI2050/Ai2050-OpenOne/tests/glm5/result/_p2852_verdict.txt'
with open(rep, 'w', encoding='utf-8') as f:
    for k in keys:
        f.write('%s = %s\n' % (k, json.dumps(v[k])))
print('verdict extract done')
