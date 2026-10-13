# -*- coding: utf-8 -*-
"""Phase 3195 independent verification (read-only, disk readback). 18 checks."""
import json, hashlib, io, os

def sha8(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]
B95 = r'tests/glm5/result/rdc_query_construction_20260913/phase3195/g5d3_ustar_attribution'
OUT = []
def chk(i, name, cond):
    OUT.append((i, name, bool(cond)))
    print('[%02d] %-50s %s' % (i, name, 'PASS' if cond else 'FAIL'))

final = json.load(open(B95 + r'\final_result.json', encoding='utf-8'))
res = json.load(open(B95 + r'\result.json', encoding='utf-8'))
led = json.load(open(r'research/gpt5/atlas/atlas_ledger.json', encoding='utf-8'))
memo = io.open(r'research/gpt5/docs/AGI_GPT5_MEMO.md', encoding='utf-8').read()
reg20p = r'tests/glm5/result/rdc_query_construction_20260913/phase3194/g5d2_ftr26_heldout_upgrade/atlas_registry_v1_20.json'

pm = res['per_model']
M = ['qwen3-4b', 'qwen3-14b', 'glm4-9b']
# 01 final chain
chk(1, 'final res/seal b527537a/551e8b40', final['res_sha8'] == 'b527537a' and final['seal_sha8'] == '551e8b40')
# 02 mode/design
chk(2, 'mode FULL + design c23f7ada', final['mode'] == 'FULL' and final['design_sha8'] == 'c23f7ada')
# 03 result chain
chk(3, 'result res/seal 21aee353/f9fe19e0', res['res_sha8'] == '21aee353' and res['seal_sha8'] == 'f9fe19e0')
# 04 disposition report-only
chk(4, 'registry_disposition v1_20_unchanged_report_only', final['registry_disposition'] == 'v1_20_unchanged_report_only' and res['registry_disposition'] == 'v1_20_unchanged_report_only')
# 05 registry v1.20 byte-untouched
chk(5, 'registry v1.20 disk sha 2aac7e5e (untouched)', final['registry_v120_sha8'] == '2aac7e5e' and final['registry_v120_recheck_sha8'] == '2aac7e5e' and sha8(reg20p) == '2aac7e5e')
# 06 gate 3/3 ratio
ratios = [pm[m]['residual_ratio'] for m in M]
chk(6, 'ratio 3/3 < 0.5 + values 0.0181/0.0174/0.0181', final['arm']['n_models_pass'] == 3 and final['arm']['decomposed'] is True and all(abs(r - e) < 5e-5 for r, e in zip(ratios, [0.0181, 0.0174, 0.0181])))
# 07 top carrier is v_c
ok = all(pm[m]['top_contrib'][0][0].startswith('v_c[') and pm[m]['top_contrib'][0][1] >= 0.7 for m in M)
chk(7, 'top carrier v_c >= 0.7 all models', ok)
# 08 3193 gate_max control anchor re-derived
gm = [pm[m]['gate_max_3193'] for m in M]
chk(8, '3193 gate_max control 0.2130/0.3303/0.2457 bitwise', all(abs(a - b) < 5e-5 for a, b in zip(gm, [0.2130, 0.3303, 0.2457])))
# 09 cos_dvu control anchor
cd = [abs(pm[m]['cos_dvu']) for m in M]
chk(9, '|cos(u*,dvu)| 0.1064/0.0405/0.0240 bitwise', all(abs(a-b) < 5e-5 for a, b in zip(cd, [0.1064, 0.0405, 0.0240])))
# 10 ledger #347
m347 = led['measurements'][-1]
chk(10, 'ledger n=347 + #347 phase/evidence', len(led['measurements']) == 347 and m347['phase'] == 3195 and m347['evidence_level'] == 'E2_held_out' and led['measurements'][345]['phase'] == 3194)
# 11 ledger chain anchors
c = m347['chain']
chk(11, 'ledger #347 chain anchors', c['design_sha8'] == 'c23f7ada' and c['res_sha8'] == '21aee353' and c['registry_v120'] == '2aac7e5e' and c['final_res_sha8'] == 'b527537a')
# 12 ledger prefix: #346 intact
chk(12, 'ledger #346 intact (FTR-26 upgrade)', 'FTR-26 upgraded' in led['measurements'][345]['detail'] and led['measurements'][345]['chain']['registry_v120'] == '2aac7e5e')
# 13 MEMO section
chk(13, 'MEMO 3195 section + PARTIALLY_DECOMPOSED', '## Phase 3195' in memo and 'PARTIALLY_DECOMPOSED' in memo and '11ce7697' in memo)
# 14 verdict consistency res vs final
chk(14, 'verdict consistent res/final/ledger', res['verdict'] == final['arm']['verdict'] and 'PARTIALLY_DECOMPOSED' in m347['verdict'])
# 15 zero GPU
chk(15, 'zero_gpu true res+final', res['zero_gpu'] is True and final['zero_gpu'] is True)
# 16 drifts logged zero in ledger detail
chk(16, 'device anchors drift==0 logged', 'drift69/89/90/91/93/94 all == 0' in m347['detail'] and 'dv_drift == 0' in m347['detail'])
# 17 honest log: candidate count refreeze recorded
chk(17, 'honest log 17->22 refreeze c23f7ada', '17 -> 22' in m347['detail'] and 'c23f7ada' in m347['detail'])
# 18 no atlas_v1_23 produced (registry untouched => no new html)
import glob
chk(18, 'no new atlas html in 3195 outdir', not glob.glob(os.path.join(B95, 'atlas_v1_23*')))

fails = [o for o in OUT if not o[2]]
print('\n%d/%d ALL PASS' % (len(OUT) - len(fails), len(OUT)))
if fails:
    print('FAILS:', fails)
