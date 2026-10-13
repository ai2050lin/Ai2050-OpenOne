import json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
D = r'D://AI2050//Ai2050-OpenOne/tests/glm5/result/rdc_query_construction_20260913'
r72 = json.load(open(D + r'/phase3172/g5a9_port_calibration/result.json', encoding='utf-8'))
r69 = json.load(open(D + r'/phase3169/g5a6_oov_panel/result.json', encoding='utf-8'))
r71 = json.load(open(D + r'/phase3171/g5a8_collapse_mechanism/result.json', encoding='utf-8'))
print('3172 keys:', sorted(r72.keys()))
print('3172 res/seal:', r72.get('res_sha8'), r72.get('seal_sha8'))
arm72 = r72['arm'] if 'arm' in r72 else r72
print('3172 pooled:', json.dumps(r72.get('pooled') or arm72.get('pooled'), ensure_ascii=False)[:600])
print('3172 overall:', json.dumps(r72.get('overall') or arm72.get('overall'), ensure_ascii=False)[:400])
pm = r72.get('per_model') or arm72.get('per_model')
print('3172 per_model keys:', list(pm.keys()))
print('3172 per_model qwen3-4b keys:', sorted(pm['qwen3-4b'].keys()))
print('3172 kcurves.8 4b:', json.dumps(pm['qwen3-4b'].get('kcurves', {}).get('8'), ensure_ascii=False)[:300])
print('3169 keys:', sorted(r69.keys()), 'res/seal:', r69.get('res_sha8'), r69.get('seal_sha8'))
print('3169 gate:', json.dumps(r69.get('gate') or (r69.get('arm') or {}).get('gate'), ensure_ascii=False)[:400])
print('3171 keys:', sorted(r71.keys()), 'res/seal:', r71.get('res_sha8'), r71.get('seal_sha8'))
sl = r71.get('slots') or (r71.get('arm') or {}).get('slots')
print('3171 slots.k_main:', json.dumps(sl.get('k_main'), ensure_ascii=False)[:400])
print('3171 overall:', json.dumps(r71.get('overall') or (r71.get('arm') or {}).get('overall'), ensure_ascii=False)[:300])
reg = json.load(open(D + r'/phase3205/g5g3_ftr14_attribution_recheck/atlas_registry_v1_27.json', encoding='utf-8'))
f22 = [f for f in reg['features'] if f['id'] == 'FTR-22'][0]
print('FTR-22 level:', f22['evidence_level'], 'n_anchors:', len(f22['anchors']))
for a in f22['anchors']:
    print('  anchor', a['src'], a['phase'], a['role'], json.dumps(a['asserts'], ensure_ascii=False)[:900])
print('FTR-22 values:', json.dumps(f22.get('values'), ensure_ascii=False)[:500])
print('FTR-22 scope_limits tail:', f22['scope_limits'][-200:])
