# -*- coding: utf-8 -*-
# Phase 3174 verify: independent re-implementation audit.
# Re-derives seal bytes, re-runs shadow comparison from the on-disk shadow dir,
# re-navigates a sample of anchors, re-computes FTR-22 values, re-reads all
# five writes. No imports from the main script.
import hashlib
import io
import json
import os
import re

ROOT = r'D:\AI2050\Ai2050-OpenOne'
SRC = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
PDIR = os.path.join(SRC, 'phase3174', 'g5a11_audit')
SHADOW = os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3174_shadow')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
DAILY = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-09.md')
WMEM = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
OUT = os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3174_verify_out.txt')
L = []
N_OK = 0


def chk(name, ok, note=''):
    global N_OK
    if ok:
        N_OK += 1
    L.append(('OK   ' if ok else 'FAIL ') + name + ('  | ' + str(note) if note else ''))


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def jload(p):
    return json.load(io.open(p, encoding='utf-8'))


R = jload(os.path.join(PDIR, 'result.json'))
SM = jload(os.path.join(PDIR, 'smoke_result.json'))

# 1. seal byte-level reconstruction
core = {k: v for k, v in R.items() if k not in ('res_sha8', 'seal_sha8')}
raw1 = json.dumps(core, ensure_ascii=False, indent=1, sort_keys=True).encode('utf-8')
res8 = hashlib.sha256(raw1).hexdigest()[:8]
mid = json.dumps(dict(core, res_sha8=res8), ensure_ascii=False, indent=1, sort_keys=True).encode('utf-8')
seal8 = hashlib.sha256(mid).hexdigest()[:8]
chk('1.1 res content sha reconstruct', res8 == R['res_sha8'], res8)
chk('1.2 seal sha reconstruct', seal8 == R['seal_sha8'], seal8)
sc = {k: v for k, v in SM.items() if k not in ('res_sha8', 'seal_sha8')}
sr8 = hashlib.sha256(json.dumps(sc, ensure_ascii=False, indent=1, sort_keys=True).encode('utf-8')).hexdigest()[:8]
chk('1.3 smoke res reconstruct', sr8 == SM['res_sha8'], sr8)

# 2. shadow comparison re-run from disk
reg_a = open(os.path.join(SRC, 'phase3173', 'g5a10_atlas_v13', 'atlas_registry_v1_2.json'), 'rb').read()
reg_b = open(os.path.join(SHADOW, 'atlas_registry_v1_2.json'), 'rb').read()
chk('2.1 shadow registry byte-identical', reg_a == reg_b)
gl_a = open(os.path.join(SRC, 'phase3173', 'g5a10_atlas_v13', 'gap_ledger_v1_4.json'), 'rb').read()
gl_b = open(os.path.join(SHADOW, 'gap_ledger_v1_4.json'), 'rb').read()
chk('2.2 shadow gap byte-identical', gl_a == gl_b)
ha = io.open(os.path.join(SRC, 'phase3173', 'g5a10_atlas_v13', 'atlas_v1_3.html'), encoding='utf-8').read().splitlines()
hb = io.open(os.path.join(SHADOW, 'atlas_v1_3.html'), encoding='utf-8').read().splitlines()
chk('2.3 html line counts equal', len(ha) == len(hb), str(len(ha)))
diff = [i for i in range(len(ha)) if ha[i] != hb[i]]
TS = re.compile(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}')
chk('2.4 html diff lines all timestamp',
    all(TS.search(ha[i]) and TS.search(hb[i]) for i in diff) and len(diff) == R['shadow']['html_diff_lines'],
    'diff=%d' % len(diff))
ka = re.findall(r'data-k="([^"]+)"', '\n'.join(ha))
kb = re.findall(r'data-k="([^"]+)"', '\n'.join(hb))
chk('2.5 data-k 637 keys identical', ka == kb and len(ka) == 637, str(len(ka)))

# 3. anchor re-navigation sample (independent resolver)
def nav(d, key):
    cur = d
    for pt in key.split('.'):
        if isinstance(cur, dict):
            if pt in cur:
                cur = cur[pt]
                continue
            if pt.startswith('node:') and isinstance(cur.get('nodes'), list):
                nid = pt.split(':', 1)[1]
                hit = [it for it in cur['nodes'] if isinstance(it, dict) and it.get('id') == nid]
                if hit:
                    cur = hit[0]
                    continue
            return None
        if isinstance(cur, list) and pt.isdigit():
            cur = cur[int(pt)]
            continue
        return None
    return cur


A = jload(os.path.join(SRC, 'phase3154', 'g1p4_mfd_multifactor_disentangle', 'summary', 'result_summary.json'))
chk('3.1 FTR-01 p3154 res content sha', A['res_sha8'] == '2402f401', A.get('res_sha8'))
chk('3.2 FTR-01 shares_mean_kout.C', abs(nav(A, 'shares_mean_kout.C') - 0.5762712045154778) < 1e-9)
AUD = jload(os.path.join(SRC, 'phase3162', 'g5a1_atlas_foundation', 'result_audit.json'))
chk('3.3 node:N01 status', nav(AUD, 'node:N01.status') == 'disk_verified', nav(AUD, 'node:N01.status'))
P = jload(os.path.join(SRC, 'phase3156', 'g3p1_position_shift_family', 'qwen3-4b', 'result.json'))
chk('3.4 p3156 ctx_effect.zh_k1[7]', abs(nav(P, 'ctx_effect.zh_k1.7') - 103.15874481201172) < 1e-6)
Q = jload(os.path.join(ROOT, 'tests', 'deepseek', 'result', 'q06_result.json'))
chk('3.5 q06 wilson[1]', abs(nav(Q, 'C_steer_main.wilson.1') - 0.010113689495831947) < 1e-12)
AG = jload(os.path.join(ROOT, 'research', 'deepseek', 'atlas', 'a_gate_closure_v1.json'))
chk('3.6 agate gate_closed', nav(AG, 'gate_closed') is True)
R69 = jload(os.path.join(SRC, 'phase3169', 'g5a6_oov_panel', 'result.json'))
R71 = jload(os.path.join(SRC, 'phase3171', 'g5a8_collapse_mechanism', 'result.json'))
R72 = jload(os.path.join(SRC, 'phase3172', 'g5a9_port_calibration', 'result.json'))
chk('3.7 3169 content sha', R69['res_sha8'] == '49430a39' and R69['seal_sha8'] == '018c6024')
chk('3.8 3171 content sha', R71['res_sha8'] == '6a29201c' and R71['seal_sha8'] == 'cc7ccedd')
chk('3.9 3172 content sha', R72['res_sha8'] == '463f42c8' and R72['seal_sha8'] == 'cdb85525')

# 4. FTR-22 independent recompute
rk = {k: R72['pooled'][k]['ratio'] for k in ('0', '1', '2', '4', '8')}
chk('4.1 k0 bitwise 3169', rk['0'] == R69['gate']['ratio_B'] == 2.5388114997805062)
chk('4.2 k8 value', rk['8'] == 1.800176217907229)
port = (rk['0'] - rk['8']) / rk['0']
chk('4.3 port_frac', abs(port - 0.2909) < 0.001, '%.6f' % port)
MS = ['qwen3-4b', 'qwen3-14b', 'glm4-9b']
r8m = {m: R72['per_model'][m]['kcurves']['8']['ratio'] for m in MS}
chk('4.4 k8 band x3', all(1.5 <= r8m[m] < 2.0 for m in MS))
V = R['e1_assessment']
chk('4.5 e1 assessment ids', [a['id'] for a in V] == ['FTR-04', 'FTR-06', 'FTR-08', 'FTR-14', 'FTR-15'])
zs = [a['id'] for a in V if a['zero_gpu_candidate']]
chk('4.6 zero-gpu candidates', zs == ['FTR-04', 'FTR-06', 'FTR-08'], ','.join(zs))
rs4 = R71['per_model']['qwen3-4b']['slots']['k_main']['ratio_S']
rsg = R71['per_model']['glm4-9b']['slots']['k_main']['ratio_S']
chk('4.7 p3171 mixed regime', rs4 < 0.8 and rsg > 0.8 and R71['overall']['main_cls'] == 'mixed_across_models')
chk('4.8 prereg sha on disk', R['prereg_sha8'] == sha8_file(os.path.join(PDIR, 'prereg_3175_structural_residual_draft.json')), R['prereg_sha8'])
PR = jload(os.path.join(PDIR, 'prereg_3175_structural_residual_draft.json'))
chk('4.9 prereg schema', PR['prereg_id'] == 'G5-A12' and PR['target_phase'] == 3175 and
    PR['status'] == 'draft_pending_freeze' and 'primary_gate' in PR and 'immutable_predicates' in PR)
chk('4.10 prereg k0 predicate', '2.5388114997805062' in json.dumps(PR['immutable_predicates']))
chk('4.11 ranking structure', [x['rank'] for x in R['ranking']] == [1, 2, 3] and
    R['ranking'][0]['item'] == 'structural_residual_localization' and
    R['ranking'][1]['gpu'] is False and R['ranking'][2]['gpu'] is False)
chk('4.12 taxonomy notes', R['taxonomy']['tag_notes'] == ['FTR-05', 'FTR-09', 'FTR-12', 'FTR-13', 'FTR-18', 'FTR-19'])
chk('4.13 premise correction', 'E2_predictive' in R['taxonomy']['premise_correction']['registry_v12_reality'])

# 5. five-write readback
led = jload(LEDGER)
ms2 = led['measurements']
e74 = [m for m in ms2 if m.get('phase') == 3174]
chk('5.1 ledger n>=326 with 3174', len(ms2) >= 326 and len(e74) == 1, 'n=%d' % len(ms2))
chk('5.2 ledger chain field', isinstance(led.get('ledger_sha256_8'), str) and len(led['ledger_sha256_8']) == 8)
memo2 = open(MEMO, 'rb').read().decode('utf-8')
chk('5.3 MEMO Phase 3174', '## Phase 3174' in memo2)
chk('5.4 MEMO res+seal', R['res_sha8'] in memo2 and R['seal_sha8'] in memo2)
chk('5.5 MEMO prereg 3175', '预注册 3175' in memo2 and R['prereg_sha8'] in memo2)
d2 = io.open(DAILY, encoding='utf-8').read()
chk('5.6 daily 3174', '3174 图谱完整性审计' in d2)
w2 = open(WMEM, 'rb').read().decode('utf-8')
chk('5.7 MEMORY 3174', '3174 图谱 v1.3 完整性审计' in w2)
chk('5.8 snapshots exist', os.path.exists(MEMO + '.snap3174') and os.path.exists(WMEM + '.snap3174'))

# 6. immutability: sealed 3173 artifacts untouched
chk('6.1 sealed 3173 result untouched',
    sha8_file(os.path.join(SRC, 'phase3173', 'g5a10_atlas_v13', 'result.json')) == 'f98ae4c4')
chk('6.2 sealed 3173 html untouched',
    sha8_file(os.path.join(SRC, 'phase3173', 'g5a10_atlas_v13', 'atlas_v1_3.html')) == 'ff875ca3')
chk('6.3 sealed 3173 registry untouched',
    sha8_file(os.path.join(SRC, 'phase3173', 'g5a10_atlas_v13', 'atlas_registry_v1_2.json')) == '0e5abcaf')

L.append('TOTAL PASS=%d FAIL=%d' % (N_OK, len(L) - N_OK))
with io.open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(L) + '\n')
print('verify done: PASS=%d FAIL=%d' % (N_OK, len(L) - N_OK - 1))
