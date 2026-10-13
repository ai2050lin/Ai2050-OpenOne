import json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
D = r'D://AI2050//Ai2050-OpenOne/tests/glm5/result/rdc_query_construction_20260913'
r71 = json.load(open(D + r'/phase3171/g5a8_collapse_mechanism/result.json', encoding='utf-8'))
pm71 = r71['per_model']
print('3171 pm 4b keys:', sorted(pm71['qwen3-4b'].keys()))
print('3171 pm 4b k_main:', json.dumps(pm71['qwen3-4b'].get('k_main'), ensure_ascii=False)[:400])
print('3171 overall:', json.dumps(r71['overall'], ensure_ascii=False)[:400])
r72 = json.load(open(D + r'/phase3172/g5a9_port_calibration/result.json', encoding='utf-8'))
kc8 = r72['per_model']['qwen3-4b']['kcurves']['8']
print('kc8 full keys:', sorted(kc8.keys()))
print('kc8 ratio?', {k: kc8[k] for k in kc8 if 'ratio' in k})
print('kc8 all:', json.dumps(kc8, ensure_ascii=False)[:700])
reg = json.load(open(D + r'/phase3205/g5g3_ftr14_attribution_recheck/atlas_registry_v1_27.json', encoding='utf-8'))
f22 = [f for f in reg['features'] if f['id'] == 'FTR-22'][0]
print('FTR-22 level:', f22['evidence_level'], 'n_anchors:', len(f22['anchors']))
for a in f22['anchors']:
    print('  anchor', a['src'], a['phase'], a['role'])
    print('    asserts:', json.dumps(a['asserts'], ensure_ascii=False)[:1200])
print('FTR-22 values:', json.dumps(f22.get('values'), ensure_ascii=False)[:600])
print('FTR-22 statement:', f22['statement'][:300])
print('FTR-22 scope_limits len:', len(f22['scope_limits']))
print('FTR-22 counter_evidence:', f22['counter_evidence'])
print('FTR-22 model_scope:', f22['model_scope'], 'family:', f22['family'], 'replication:', f22['replication'])
