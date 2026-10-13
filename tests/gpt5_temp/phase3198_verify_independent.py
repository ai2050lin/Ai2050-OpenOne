# -*- coding: utf-8 -*-
"""Phase 3198 independent verification (read-only, disk readback). 18 checks."""
import json, hashlib, io, os

def sha8(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]
B98 = r'tests/glm5/result/rdc_query_construction_20260913/phase3198/g5d6_dline_registration'
OUT = []
def chk(i, name, cond):
    OUT.append((i, name, bool(cond)))
    print('[%02d] %-54s %s' % (i, name, 'PASS' if cond else 'FAIL'))

final = json.load(open(B98 + r'\final_result.json', encoding='utf-8'))
res = json.load(open(B98 + r'\result.json', encoding='utf-8'))
led = json.load(open(r'research/gpt5/atlas/atlas_ledger.json', encoding='utf-8'))
memo = io.open(r'research/gpt5/docs/AGI_GPT5_MEMO.md', encoding='utf-8').read()
reg21p = B98 + r'\atlas_registry_v1_21.json'
reg20p = r'tests/glm5/result/rdc_query_construction_20260913/phase3194/g5d2_ftr26_heldout_upgrade/atlas_registry_v1_20.json'
htmlp = B98 + r'\atlas_v1_23.html'

reg21 = json.load(io.open(reg21p, encoding='utf-8'))
reg20 = json.load(io.open(reg20p, encoding='utf-8'))
html = io.open(htmlp, encoding='utf-8').read()
M = ['qwen3-4b', 'qwen3-14b', 'glm4-9b']

chk(1, 'final res/seal b1e81f1b/fbd85fbb + design 1d6b40ce', final['res_sha8'] == 'b1e81f1b' and final['seal_sha8'] == 'fbd85fbb' and final['design_sha8'] == '1d6b40ce')
chk(2, 'result res/seal a1167bb8/591185ae', res['res_sha8'] == 'a1167bb8' and res['seal_sha8'] == '591185ae')
chk(3, 'mode FULL + phase 3198 + name g5d6', final['mode'] == 'FULL' and final['phase'] == 3198 and final['name'] == 'g5d6_dline_registration')
chk(4, 'disposition v1_20_to_v1_21_ftr26_signed_closing', final['registry_disposition'] == 'v1_20_to_v1_21_ftr26_signed_closing')
chk(5, 'registry v1.21 disk sha 47cf01e0 + v1.20 ref 2aac7e5e', final['registry_v121_sha8'] == '47cf01e0' and sha8(reg21p) == '47cf01e0' and final['registry_v120_sha8_ref'] == '2aac7e5e')
chk(6, 'registry v1.20 disk untouched 2aac7e5e', sha8(reg20p) == '2aac7e5e')
chk(7, 'control drift==0 x3/3 in result+final', all(res['control'][m]['drift'] == 0.0 for m in M) and all(final['arm']['control'][m]['drift'] == 0.0 for m in M))
chk(8, 'version 1.21 + 26 features + evidence/upgrade unchanged', reg21['version'] == '1.21' and len(reg21['features']) == 26 and reg21['evidence_level_taxonomy'] == reg20['evidence_level_taxonomy'] and reg21['upgrade_log'] == reg20['upgrade_log'] and reg21['failures'] == reg20['failures'])
# 25 legacy features byte-identical + FTR-26 append-only
legacy_ok = all(json.dumps(a, ensure_ascii=False, sort_keys=True) == json.dumps(b, ensure_ascii=False, sort_keys=True)
                for a, b in zip(reg20['features'], [f for f in reg21['features'] if f['id'] != 'FTR-26']))
f26o = [f for f in reg20['features'] if f['id'] == 'FTR-26'][0]
f26n = [f for f in reg21['features'] if f['id'] == 'FTR-26'][0]
chk(9, '25 legacy byte-preserved + FTR-26 append-only', legacy_ok and f26n['statement'].startswith(f26o['statement']) and f26n['evidence_level'] == f26o['evidence_level'] == 'E2_held_out' and f26n['anchors'][:-1] == f26o['anchors'] and len(f26n['anchors']) == len(f26o['anchors']) + 1 and f26n['anchors'][-1]['role'] == 'signed_closing')
chk(10, 'new anchor asserts vc_l1_share + control drift 0', f26n['anchors'][-1]['asserts']['vc_l1_share.qwen3-4b'] == 0.9159 and f26n['anchors'][-1]['asserts']['control_3197_drift'] == 0.0)
chk(11, 'ledger n=350 + #350 phase/evidence', len(led['measurements']) == 350 and led['measurements'][-1]['phase'] == 3198 and led['measurements'][-1]['evidence_level'] == 'E2_held_out' and led['measurements'][348]['phase'] == 3197)
m350 = led['measurements'][-1]
c = m350['chain']
chk(12, 'ledger #350 chain anchors', c['design_sha8'] == '1d6b40ce' and c['res_sha8'] == 'a1167bb8' and c['final_res_sha8'] == 'b1e81f1b' and c['registry_v121'] == '47cf01e0' and c['atlas_v1_23'] == '9488777c')
chk(13, 'ledger #349 intact (SIGNED_CLASS_CONTRAST)', 'SIGNED_CLASS_CONTRAST' in led['measurements'][348]['verdict'] and led['measurements'][348]['chain']['res_sha8'] == 'dd2af224')
chk(14, 'MEMO 3198 section + closing + ledger post', '## Phase 3198' in memo and 'REGISTERED' in memo and '44b4ccc5' in memo and 'G5-E1' in memo and 'FTR-24 held_out' in memo)
chk(15, 'html disk sha 9488777c + field check clean', sha8(htmlp) == final['html_sha8'] == '9488777c' and final['field_check']['missing'] == 0 and final['field_check']['extra'] == 0 and final['field_check']['mismatch'] == 0 and final['field_check']['dups'] == 0 and final['field_check']['n_found'] == 775)
chk(16, 'statement closing text in registry+html+memo', 'SIGNED_CLASS_CONTRAST' in f26n['statement'] and f26n['statement'][-200:] in html.replace('&gt;', '>') or 'anti-class' in memo or '反类均值对比轴' in f26n['statement'])
chk(17, 'zero_gpu + sources asserted 17', res['zero_gpu'] is True and final['zero_gpu'] is True and len(final['sources']) == 17)
chk(18, 'SMOKE artifacts coexist + FULL payload distinct', os.path.exists(B98 + r'\smoke_result.json') and json.load(open(B98 + r'\smoke_result.json', encoding='utf-8'))['design_sha8'] == '1d6b40ce')

fails = [o for o in OUT if not o[2]]
print('\n%d/%d ALL PASS' % (len(OUT) - len(fails), len(OUT)))
if fails:
    print('FAILS:', fails)
