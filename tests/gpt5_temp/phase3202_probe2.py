# -*- coding: utf-8 -*-
import io, json, hashlib
B181 = r'D:/AI2050/Ai2050-OpenOne/tests/glm5/result/rdc_query_construction_20260913/phase3181/g5b6_ftr04_glm4_attribution'
B176 = r'D:/AI2050/Ai2050-OpenOne/tests/glm5/result/rdc_query_construction_20260913/phase3176/g5b1_elev_upgrade'
B324 = r'D:/AI2050/Ai2050-OpenOne/tests/glm5/result/rdc_query_construction_20260913/phase3201/g5e3_eline_audit'
def sha8(p): return hashlib.sha256(open(p,'rb').read()).hexdigest()[:8]

r81 = json.load(io.open(B181 + r'\result.json', encoding='utf-8'))
print('3181 res_sha8 field', r81.get('res_sha8'), 'seal', r81.get('seal_sha8'), '| file', sha8(B181 + r'\result.json'))
print('3181 design', r81.get('design_sha8'), 'verdict', r81.get('verdict'))
print('3181 geo_sv', json.dumps(r81.get('geo_sv'))[:400])
print('3181 gates', json.dumps(r81.get('gates'))[:400])

r76 = json.load(io.open(B176 + r'\result.json', encoding='utf-8'))
print('3176 file sha8', sha8(B176 + r'\result.json'), 'top keys', list(r76.keys()))
print('3176 res/seal', r76.get('res_sha8'), r76.get('seal_sha8'))

reg = json.load(io.open(B324 + r'\atlas_registry_v1_24.json', encoding='utf-8'))
f04 = [f for f in reg['features'] if f['id'] == 'FTR-04'][0]
print('FTR-04 e_level', f04['evidence_level'])
print('FTR-04 anchors:', json.dumps(f04['anchors'], ensure_ascii=False)[:1500])
print('FTR-04 scope_limits tail:', f04['scope_limits'][-300:])
print('FTR-04 ce n', len(f04['counter_evidence']))
