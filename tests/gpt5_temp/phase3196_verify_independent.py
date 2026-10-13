# -*- coding: utf-8 -*-
"""Phase 3196 independent verification (read-only, disk readback). 18 checks."""
import json, hashlib, io, os, glob

def sha8(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]
B96 = r'tests/glm5/result/rdc_query_construction_20260913/phase3196/g5d4_ustar_nnls_shares'
OUT = []
def chk(i, name, cond):
    OUT.append((i, name, bool(cond)))
    print('[%02d] %-52s %s' % (i, name, 'PASS' if cond else 'FAIL'))

final = json.load(open(B96 + r'\final_result.json', encoding='utf-8'))
res = json.load(open(B96 + r'\result.json', encoding='utf-8'))
led = json.load(open(r'research/gpt5/atlas/atlas_ledger.json', encoding='utf-8'))
memo = io.open(r'research/gpt5/docs/AGI_GPT5_MEMO.md', encoding='utf-8').read()
reg20p = r'tests/glm5/result/rdc_query_construction_20260913/phase3194/g5d2_ftr26_heldout_upgrade/atlas_registry_v1_20.json'

pm = res['per_model']
M = ['qwen3-4b', 'qwen3-14b', 'glm4-9b']
chk(1, 'final res/seal 7313de4d/f57cc8b7 + design 6723c371', final['res_sha8'] == '7313de4d' and final['seal_sha8'] == 'f57cc8b7' and final['design_sha8'] == '6723c371')
chk(2, 'result res/seal e69ab1e7/c0cdd97f', res['res_sha8'] == 'e69ab1e7' and res['seal_sha8'] == 'c0cdd97f')
chk(3, 'mode FULL + phase 3196 + name g5d4', final['mode'] == 'FULL' and final['phase'] == 3196 and final['name'] == 'g5d4_ustar_nnls_shares')
chk(4, 'disposition v1_20_unchanged_report_only', final['registry_disposition'] == 'v1_20_unchanged_report_only' and res['registry_disposition'] == 'v1_20_unchanged_report_only')
chk(5, 'registry v1.20 disk untouched 2aac7e5e', final['registry_v120_recheck_sha8'] == '2aac7e5e' and sha8(reg20p) == '2aac7e5e')
# gate 0/3 MIXED
chk(6, 'gate MIXED 0/3 (dominated False)', final['arm']['n_models_pass'] == 0 and final['arm']['dominated'] is False)
# vc_share zero
chk(7, 'vc_share 0.0000 x3', all(pm[m]['vc_share'] == 0.0 for m in M))
chk(8, 'known_share 0.1999/0.3998/0.1139', all(abs(pm[m]['known_share'] - e) < 5e-5 for m, e in zip(M, [0.1999, 0.3998, 0.1139])))
# key negative result: nnls_resid >> ratio
chk(9, 'nnls_resid 0.8184/0.8125/0.8236 + ratio 0.018 bitwise', all(abs(pm[m]['nnls_resid'] - e) < 5e-5 for m, e in zip(M, [0.8184, 0.8125, 0.8236])) and all(abs(pm[m]['residual_ratio'] - e) < 1e-12 for m, e in zip(M, [0.01806072230473137, 0.017358575654316224, 0.01814552246882951])))
# top share is kout_pc (4b/glm4) or u_seen (14b) - never v_c; share >= 0.2
chk(10, 'top NNLS share kout_pc/u_seen (never v_c), >= 0.2', all((pm[m]['top_contrib'][0][0].startswith('kout_pc') or pm[m]['top_contrib'][0][0] == 'u_seen') and pm[m]['top_contrib'][0][1] >= 0.2 for m in M))
# ledger
m348 = led['measurements'][-1]
chk(11, 'ledger n=348 + #348 phase 3196', len(led['measurements']) == 348 and m348['phase'] == 3196 and led['measurements'][346]['phase'] == 3195)
c = m348['chain']
chk(12, 'ledger #348 chain anchors', c['design_sha8'] == '6723c371' and c['res_sha8'] == 'e69ab1e7' and c['final_res_sha8'] == '7313de4d' and c['registry_v120'] == '2aac7e5e')
chk(13, 'ledger #347 intact', 'PARTIALLY_DECOMPOSED' in led['measurements'][346]['verdict'] and led['measurements'][346]['chain']['res_sha8'] == '21aee353')
chk(14, 'MEMO 3196 section + key result', '## Phase 3196' in memo and '86f3a444' in memo and 'MIXED' in memo)
chk(15, 'verdict consistent res/final/ledger', res['verdict'] == final['arm']['verdict'] and 'MIXED' in m348['verdict'])
chk(16, 'zero_gpu + drifts logged', res['zero_gpu'] is True and final['zero_gpu'] is True and 'drift69/89/90/91/93/94 all == 0' in m348['detail'])
chk(17, 'honest log leftover return + signed interp', 'decomposed -> dominated' in m348['detail'] and 'SIGNED contrast mixture' in m348['detail'])
chk(18, 'no new atlas html in 3196 outdir', not glob.glob(os.path.join(B96, 'atlas_v1_23*')))

fails = [o for o in OUT if not o[2]]
print('\n%d/%d ALL PASS' % (len(OUT) - len(fails), len(OUT)))
if fails:
    print('FAILS:', fails)
