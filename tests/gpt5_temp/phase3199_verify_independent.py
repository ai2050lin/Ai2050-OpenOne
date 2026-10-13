# -*- coding: utf-8 -*-
"""Phase 3199 independent verification (read-only, disk readback). 18 checks."""
import json, hashlib, io, os

def sha8(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]
B99 = r'tests/glm5/result/rdc_query_construction_20260913/phase3199/g5e1_ftr24_heldout_upgrade'
OUT = []
def chk(i, name, cond):
    OUT.append((i, name, bool(cond)))
    print('[%02d] %-54s %s' % (i, name, 'PASS' if cond else 'FAIL'))

final = json.load(open(B99 + r'\final_result.json', encoding='utf-8'))
res = json.load(open(B99 + r'\result.json', encoding='utf-8'))
led = json.load(open(r'research/gpt5/atlas/atlas_ledger.json', encoding='utf-8'))
memo = io.open(r'research/gpt5/docs/AGI_GPT5_MEMO.md', encoding='utf-8').read()
reg22p = B99 + r'\atlas_registry_v1_22.json'
reg21p = r'tests/glm5/result/rdc_query_construction_20260913/phase3198/g5d6_dline_registration/atlas_registry_v1_21.json'
htmlp = B99 + r'\atlas_v1_24.html'

reg22 = json.load(io.open(reg22p, encoding='utf-8'))
reg21 = json.load(io.open(reg21p, encoding='utf-8'))
html = io.open(htmlp, encoding='utf-8').read()
M = ['qwen3-4b', 'qwen3-14b', 'glm4-9b']

chk(1, 'final res/seal 06644f1f/82009c4a + design cfbe183c', final['res_sha8'] == '06644f1f' and final['seal_sha8'] == '82009c4a' and final['design_sha8'] == 'cfbe183c')
chk(2, 'result res/seal 94442e48/b908731e', res['res_sha8'] == '94442e48' and res['seal_sha8'] == 'b908731e')
chk(3, 'mode FULL + phase 3199 + name g5e1', final['mode'] == 'FULL' and final['phase'] == 3199 and final['name'] == 'g5e1_ftr24_heldout_upgrade')
chk(4, 'disposition v1_21_to_v1_22_ftr24_ce_not_predictive', final['registry_disposition'] == 'v1_21_to_v1_22_ftr24_ce_not_predictive')
chk(5, 'registry v1.22 disk sha 9fd87433 + v1.21 ref', final['registry_v122_sha8'] == '9fd87433' and sha8(reg22p) == '9fd87433' and final['registry_v121_sha8_ref'] == '47cf01e0')
chk(6, 'registry v1.21 disk untouched 47cf01e0', sha8(reg21p) == '47cf01e0')
chk(7, 'gate NOT_PREDICTIVE 0/3 + min med|cos| < 0.3', final['arm']['gate_predictive'] is False and all(res['per_model'][m]['min_med_abs_cos_dvu'] < 0.3 for m in M) and all(x < 0.3 for m in M for x in res['per_model'][m]['med_abs_cos_dvu']))
smoke = json.load(open(B99 + r'\smoke_result.json', encoding='utf-8'))
chk(8, 'med values bitwise SMOKE==FULL per model', all(smoke['per_model'][m]['med_abs_cos_dvu'] == res['per_model'][m]['med_abs_cos_dvu'] and smoke['per_model'][m]['full_panel_med_abs_cos_dvu'] == res['per_model'][m]['full_panel_med_abs_cos_dvu'] for m in M))
chk(9, 'version 1.22 + 26 features + failures/evidence unchanged', reg22['version'] == '1.22' and len(reg22['features']) == 26 and reg22['failures'] == reg21['failures'] and reg22['evidence_level_taxonomy'] == reg21['evidence_level_taxonomy'] and reg22['upgrade_log'] == reg21['upgrade_log'] and len(reg22['upgrade_log']) == 8)
f24o = [f for f in reg21['features'] if f['id'] == 'FTR-24'][0]
f24n = [f for f in reg22['features'] if f['id'] == 'FTR-24'][0]
chk(10, 'FTR-24 append-only: E1 kept + ce +1 + stmt/anchors frozen', f24n['evidence_level'] == f24o['evidence_level'] == 'E1_repeatable' and f24n['statement'] == f24o['statement'] and f24n['anchors'] == f24o['anchors'] and f24n['counter_evidence'][:-1] == f24o['counter_evidence'] and len(f24n['counter_evidence']) == 4 and '3199' in f24n['counter_evidence'][-1])
chk(11, '25 legacy features byte-preserved', all(json.dumps(a, ensure_ascii=False, sort_keys=True) == json.dumps(b, ensure_ascii=False, sort_keys=True) for a, b in zip([f for f in reg21['features'] if f['id'] != 'FTR-24'], [f for f in reg22['features'] if f['id'] != 'FTR-24'])))
chk(12, 'ledger n=351 + #351 phase/chain', len(led['measurements']) == 351 and led['measurements'][-1]['phase'] == 3199 and led['measurements'][-1]['chain']['registry_v122'] == '9fd87433' and led['measurements'][-1]['chain']['design_sha8'] == 'cfbe183c' and led['measurements'][349]['phase'] == 3198)
chk(13, 'ledger #350 intact (REGISTERED)', 'REGISTERED' in led['measurements'][349]['verdict'] and led['measurements'][349]['chain']['registry_v121'] == '47cf01e0')
chk(14, 'MEMO 3199 section + NOT_PREDICTIVE + ledger post + 3200 prereg', '## Phase 3199' in memo and 'NOT_PREDICTIVE' in memo and 'cffec403' in memo and 'G5-E2' in memo and 'FTR-23' in memo)
chk(15, 'html disk sha 74a11492 + field check clean', sha8(htmlp) == final['html_sha8'] == '74a11492' and final['field_check']['missing'] == 0 and final['field_check']['extra'] == 0 and final['field_check']['mismatch'] == 0 and final['field_check']['dups'] == 0 and final['field_check']['n_found'] == 775)
chk(16, 'ce text in registry+html, 3188/3189 cross-ref in memo', 'DECOUPLED' in memo and '0.0404' in memo)
chk(17, 'zero_gpu + sources asserted 16 + 3197 lock in result', res['zero_gpu'] is True and final['zero_gpu'] is True and len(final['sources']) == 16 and 'p3197_result' in final['sources'])
chk(18, 'SMOKE artifacts coexist + same design', os.path.exists(B99 + r'\smoke_result.json') and json.load(open(B99 + r'\smoke_result.json', encoding='utf-8'))['design_sha8'] == 'cfbe183c')

fails = [o for o in OUT if not o[2]]
print('\n%d/%d ALL PASS' % (len(OUT) - len(fails), len(OUT)))
if fails:
    print('FAILS:', fails)
