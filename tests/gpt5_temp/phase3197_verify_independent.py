# -*- coding: utf-8 -*-
"""Phase 3197 independent verification (read-only, disk readback). 18 checks."""
import json, hashlib, io, os, glob

def sha8(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]
B97 = r'tests/glm5/result/rdc_query_construction_20260913/phase3197/g5d5_signed_ls_shares'
OUT = []
def chk(i, name, cond):
    OUT.append((i, name, bool(cond)))
    print('[%02d] %-54s %s' % (i, name, 'PASS' if cond else 'FAIL'))

final = json.load(open(B97 + r'\final_result.json', encoding='utf-8'))
res = json.load(open(B97 + r'\result.json', encoding='utf-8'))
led = json.load(open(r'research/gpt5/atlas/atlas_ledger.json', encoding='utf-8'))
memo = io.open(r'research/gpt5/docs/AGI_GPT5_MEMO.md', encoding='utf-8').read()
reg20p = r'tests/glm5/result/rdc_query_construction_20260913/phase3194/g5d2_ftr26_heldout_upgrade/atlas_registry_v1_20.json'

pm = res['per_model']
M = ['qwen3-4b', 'qwen3-14b', 'glm4-9b']
chk(1, 'final res/seal f8a39985/2d361f0d + design c3a64857', final['res_sha8'] == 'f8a39985' and final['seal_sha8'] == '2d361f0d' and final['design_sha8'] == 'c3a64857')
chk(2, 'result res/seal dd2af224/4311d796', res['res_sha8'] == 'dd2af224' and res['seal_sha8'] == '4311d796')
chk(3, 'mode FULL + phase 3197 + name g5d5', final['mode'] == 'FULL' and final['phase'] == 3197 and final['name'] == 'g5d5_signed_ls_shares')
chk(4, 'disposition v1_20_unchanged_report_only', final['registry_disposition'] == 'v1_20_unchanged_report_only' and res['registry_disposition'] == 'v1_20_unchanged_report_only')
chk(5, 'registry v1.20 disk untouched 2aac7e5e', final['registry_v120_recheck_sha8'] == '2aac7e5e' and sha8(reg20p) == '2aac7e5e')
chk(6, 'gate SIGNED_CLASS_CONTRAST 3/3', final['arm']['n_models_pass'] == 3 and final['arm']['contrast'] is True)
chk(7, 'ls_resid 0.0181/0.0174/0.0181 + ratio bitwise', all(abs(pm[m]['ls_resid'] - e) < 5e-5 for m, e in zip(M, [0.0181, 0.0174, 0.0181])) and all(abs(pm[m]['residual_ratio'] - e) < 1e-12 for m, e in zip(M, [0.01806072230473137, 0.017358575654316224, 0.01814552246882951])))
chk(8, 'vc_l1_share 0.9159/0.8905/0.9244', all(abs(pm[m]['vc_l1_share'] - e) < 5e-5 for m, e in zip(M, [0.9159, 0.8905, 0.9244])))
# all four v_c weights negative, magnitude in [-0.40, -0.25]
ok = all(all(-0.40 < pm[m]['signed_weights']['v_c[%d]' % c] < -0.25 for c in range(6, 10)) for m in M)
chk(9, 'v_c x4 weights all negative in [-0.40,-0.25] x3', ok)
chk(10, 'next-largest |w| <= 0.05 (v_c dominance)', all(sorted(abs(v) for v in pm[m]['signed_weights'].values())[-5] < 0.05 for m in M))
chk(11, 'ledger n=349 + #349 phase/evidence', len(led['measurements']) == 349 and led['measurements'][-1]['phase'] == 3197 and led['measurements'][-1]['evidence_level'] == 'E2_held_out' and led['measurements'][347]['phase'] == 3196)
m349 = led['measurements'][-1]
c = m349['chain']
chk(12, 'ledger #349 chain anchors', c['design_sha8'] == 'c3a64857' and c['res_sha8'] == 'dd2af224' and c['final_res_sha8'] == 'f8a39985' and c['registry_v120'] == '2aac7e5e')
chk(13, 'ledger #348 intact (MIXED + negative result)', 'MIXED' in led['measurements'][347]['verdict'] and led['measurements'][347]['chain']['res_sha8'] == 'e69ab1e7')
chk(14, 'MEMO 3197 section + anti-class-mean', '## Phase 3197' in memo and 'anti-class-mean' in memo and '694bfcbf' in memo)
chk(15, 'verdict consistent res/final/ledger', res['verdict'] == final['arm']['verdict'] and 'SIGNED_CLASS_CONTRAST' in m349['verdict'])
chk(16, 'zero_gpu + drifts logged', res['zero_gpu'] is True and final['zero_gpu'] is True and 'drift69/89/90/91/93/94 all == 0' in m349['detail'])
chk(17, 'honest log sha measured probe', 'f3f86f47' in m349['detail'] and 'f086e6a3' in m349['detail'])
chk(18, 'no new atlas html in 3197 outdir', not glob.glob(os.path.join(B97, 'atlas_v1_23*')))

fails = [o for o in OUT if not o[2]]
print('\n%d/%d ALL PASS' % (len(OUT) - len(fails), len(OUT)))
if fails:
    print('FAILS:', fails)
