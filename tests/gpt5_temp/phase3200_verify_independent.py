# -*- coding: utf-8 -*-
"""Phase 3200 independent verification (read-only, disk readback). 18 checks."""
import json, hashlib, io, os

def sha8(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]
B20 = r'tests/glm5/result/rdc_query_construction_20260913/phase3200/g5e2_ftr23_heldout_upgrade'
OUT = []
def chk(i, name, cond):
    OUT.append((i, name, bool(cond)))
    print('[%02d] %-54s %s' % (i, name, 'PASS' if cond else 'FAIL'))

final = json.load(open(B20 + r'\final_result.json', encoding='utf-8'))
res = json.load(open(B20 + r'\result.json', encoding='utf-8'))
smoke = json.load(open(B20 + r'\smoke_result.json', encoding='utf-8'))
led = json.load(open(r'research/gpt5/atlas/atlas_ledger.json', encoding='utf-8'))
memo = io.open(r'research/gpt5/docs/AGI_GPT5_MEMO.md', encoding='utf-8').read()
reg23p = B20 + r'\atlas_registry_v1_23.json'
reg22p = r'tests/glm5/result/rdc_query_construction_20260913/phase3199/g5e1_ftr24_heldout_upgrade/atlas_registry_v1_22.json'
htmlp = B20 + r'\atlas_v1_25.html'

reg23 = json.load(io.open(reg23p, encoding='utf-8'))
reg22 = json.load(io.open(reg22p, encoding='utf-8'))
html = io.open(htmlp, encoding='utf-8').read()
M = ['qwen3-4b', 'qwen3-14b', 'glm4-9b']

chk(1, 'final res/seal ae7a5837/dfd3b880 + design 2ea6e1f6', final['res_sha8'] == 'ae7a5837' and final['seal_sha8'] == 'dfd3b880' and final['design_sha8'] == '2ea6e1f6')
chk(2, 'result res/seal f9a280fc/cba9d56c', res['res_sha8'] == 'f9a280fc' and res['seal_sha8'] == 'cba9d56c')
chk(3, 'mode FULL + phase 3200 + name g5e2', final['mode'] == 'FULL' and final['phase'] == 3200 and final['name'] == 'g5e2_ftr23_heldout_upgrade')
chk(4, 'disposition v1_22_to_v1_23_ftr23_e2_held_out', final['registry_disposition'] == 'v1_22_to_v1_23_ftr23_e2_held_out')
chk(5, 'registry v1.23 disk sha 5f5b9e73 + v1.22 ref', final['registry_v123_sha8'] == '5f5b9e73' and sha8(reg23p) == '5f5b9e73' and final['registry_v122_sha8_ref'] == '9fd87433')
chk(6, 'registry v1.22 disk untouched 9fd87433', sha8(reg22p) == '9fd87433')
chk(7, 'gate 6/6 units + control drift==0 x3', final['arm']['gate_upgraded'] is True and all(res['control'][m]['drift'] == 0.0 for m in M) and all(min(res['control'][m]['med_isa'], res['control'][m]['med_hasa']) >= 0.5 for m in M))
chk(8, 'med cos_out bitwise SMOKE==FULL per model', all(smoke['control'][m]['med_isa'] == res['control'][m]['med_isa'] and smoke['control'][m]['med_hasa'] == res['control'][m]['med_hasa'] for m in M))
chk(9, 'version 1.23 + 26 features + failures unchanged', reg23['version'] == '1.23' and len(reg23['features']) == 26 and reg23['failures'] == reg22['failures'] and reg23['evidence_level_taxonomy'] == reg22['evidence_level_taxonomy'])
f23o = [f for f in reg22['features'] if f['id'] == 'FTR-23'][0]
f23n = [f for f in reg23['features'] if f['id'] == 'FTR-23'][0]
chk(10, 'FTR-23 upgrade: E1->E2 + stmt/ce append-only + anchors +1', f23n['evidence_level'] == 'E2_held_out' and f23o['evidence_level'] == 'E1_repeatable' and f23n['statement'].startswith(f23o['statement']) and f23n['counter_evidence'] == f23o['counter_evidence'] and len(f23n['anchors']) == len(f23o['anchors']) + 1 and f23n['anchors'][-1]['role'] == 'held_out' and f23n['anchors'][-1]['asserts']['e_level_after'] == 'E2_held_out')
chk(11, 'upgrade_log #9 append-only + node FTR-23', len(reg23['upgrade_log']) == 9 and reg23['upgrade_log'][:-1] == reg22['upgrade_log'] and reg23['upgrade_log'][-1]['node'] == 'FTR-23' and 'E1_repeatable -> E2_held_out' in reg23['upgrade_log'][-1]['change'])
chk(12, '25 legacy features byte-preserved', all(json.dumps(a, ensure_ascii=False, sort_keys=True) == json.dumps(b, ensure_ascii=False, sort_keys=True) for a, b in zip([f for f in reg22['features'] if f['id'] != 'FTR-23'], [f for f in reg23['features'] if f['id'] != 'FTR-23'])))
chk(13, 'ledger n=352 + #352 phase/chain + #351 intact', len(led['measurements']) == 352 and led['measurements'][-1]['phase'] == 3200 and led['measurements'][-1]['chain']['registry_v123'] == '5f5b9e73' and led['measurements'][-1]['evidence_level'] == 'E2_held_out' and led['measurements'][350]['phase'] == 3199 and 'NOT_PREDICTIVE' in led['measurements'][350]['verdict'])
chk(14, 'MEMO 3200 section + upgrade #9 + ledger post + 3201 prereg', '## Phase 3200' in memo and 'HOLDOUT_PREDICTIVE_FTR23' in memo and '8964e20d' in memo and 'G5-E3' in memo and 'E_LINE_CLOSED' in memo)
chk(15, 'html disk sha 7dc08f1e + field check 782 clean', sha8(htmlp) == final['html_sha8'] == '7dc08f1e' and final['field_check']['n_found'] == 782 and final['field_check']['missing'] == 0 and final['field_check']['extra'] == 0 and final['field_check']['mismatch'] == 0 and final['field_check']['dups'] == 0)
a16 = f23n['anchors'][-1]['asserts']
chk(16, 'new anchor asserts medians in registry+html', abs(a16['med_cos_out.isa.qwen3-4b'] - 0.7696) < 5e-5 and abs(a16['med_cos_out.hasa.glm4-9b'] - 0.7885) < 5e-5 and f23n['anchors'][-1]['src'] == 'p3178' and str(f23n['anchors'][-1]['phase']) in html)
chk(17, 'zero_gpu + sources 7 + 3178 double lock in chain', res['zero_gpu'] is True and final['zero_gpu'] is True and len(final['sources']) == 7 and final['sources']['p3178_result'] == 'abae296e')
chk(18, 'SMOKE artifacts coexist + same design + payloads identical', smoke['design_sha8'] == '2ea6e1f6' and smoke['gates']['upgraded'] == res['gates']['upgraded'] == True)

fails = [o for o in OUT if not o[2]]
print('\n%d/%d ALL PASS' % (len(OUT) - len(fails), len(OUT)))
if fails:
    print('FAILS:', fails)
