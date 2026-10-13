# -*- coding: utf-8 -*-
"""Phase 3202 probe: unit-level field availability in sealed 3181/3176."""
import io, json, hashlib

B181 = r'D:/AI2050/Ai2050-OpenOne/tests/glm5/result/rdc_query_construction_20260913/phase3181/g5b6_ftr04_glm4_attribution'
B176 = r'D:/AI2050/Ai2050-OpenOne/tests/glm5/result/rdc_query_construction_20260913/phase3176/g5b1_elev_upgrade'

def sha8(p): return hashlib.sha256(open(p,'rb').read()).hexdigest()[:8]

r = json.load(io.open(B181 + r'\result.json', encoding='utf-8'))
f = json.load(io.open(B181 + r'\final_result.json', encoding='utf-8'))
print('3181 result sha8', sha8(B181 + r'\result.json'), 'final', sha8(B181 + r'\final_result.json'))
print('3181 top keys:', list(r.keys()))
print('3181 final keys:', list(f.keys()))
arm = r.get('arm') or f.get('arm')
print('arm type', type(arm).__name__)
print(json.dumps(arm, ensure_ascii=False)[:3000])
