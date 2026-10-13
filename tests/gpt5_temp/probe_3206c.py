import json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
D = r'D://AI2050//Ai2050-OpenOne/tests/glm5/result/rdc_query_construction_20260913'
r71 = json.load(open(D + r'/phase3171/g5a8_collapse_mechanism/result.json', encoding='utf-8'))
sl = r71['per_model']['qwen3-4b']['slots']
print('slots keys:', sorted(sl.keys()))
print('slots.k_main type:', type(sl.get('k_main')))
print('slots.k_main:', json.dumps(sl.get('k_main'), ensure_ascii=False)[:400])
reg = json.load(open(D + r'/phase3205/g5g3_ftr14_attribution_recheck/atlas_registry_v1_27.json', encoding='utf-8'))
f22 = [f for f in reg['features'] if f['id'] == 'FTR-22'][0]
v = f22['values']
print('values keys:', sorted(v.keys()))
for k in sorted(v): print(' ', k, '=', repr(v[k]))
