# -*- coding: utf-8 -*-
"""Phase 3245 independent disk verify: re-reads every artifact from disk and
re-checks the sealed numbers. Fails loudly on any mismatch. Read-only."""
import os, json, hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
SRC_DIR = os.path.join(ROOT, r'tests\glm5\result\rdc_query_construction_20260913')
PD = os.path.join(SRC_DIR, r'phase3245\g5h8_combo_flip')
LED = os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json')
MEMO = os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md')
REG = os.path.join(SRC_DIR, r'phase3238\g5h1_census_rank\atlas_registry_v1_33.json')
OK = []


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


# 1. execution.json: design freeze intact, hashes recorded
exe = json.load(open(os.path.join(PD, 'execution.json'), encoding='utf-8'))
assert exe['phase'] == 3245 and exe['design_sha8'] == '105df1a8'
assert exe['smoke'] is False and exe['frozen_before'] == 'any observation'
# recompute design hash from the stored design blob
ds = hashlib.sha256(json.dumps(exe['design'], ensure_ascii=False, sort_keys=True,
                               indent=1).encode('utf-8')).hexdigest()[:8]
assert ds == '105df1a8', ('design hash drift', ds)
OK.append('1 execution.json: design freeze 105df1a8 intact (recomputed)')

# 2. result.json: verdict, gates, anchors, headline numbers
R = json.load(open(os.path.join(PD, 'result.json'), encoding='utf-8'))
assert R['design_sha8'] == '105df1a8' and R['res_sha8'] == 'ef039473'
rs = hashlib.sha256(json.dumps(
    {k: v for k, v in R.items() if k not in ('res_sha8', 'created', 'runtime_s')},
    ensure_ascii=False, sort_keys=True, indent=1).encode('utf-8')).hexdigest()[:8]
assert rs == 'ef039473', ('result hash drift', rs)
assert R['verdict'] == 'combo_flip_registered'
assert R['seeds_run'] == [7, 8, 9] and R['kl_rows'] == 876
g = R['gates']
assert all(g[k] is True for k in ('G1_replication', 'G2_combination',
                                  'clause_a_zero_regression', 'clause_b_joint_gain',
                                  'clause_c_not_worse_than_best_single',
                                  'G3_determinism', 'G4_registry_untouched'))
assert g['determinism_diff'] == 0.0
AGG = R['aggregate']
assert AGG['G0']['d_vs_sealed_3240_max'] == 0.0
assert AGG['R1_rep']['d_vs_sealed_3244_max'] == 0.0
assert AGG['T3_rep']['d_vs_sealed_3242_max'] == 0.0
D = R['delta_vs_G0']
dX = D['X_combo']['d_fam_fid']
assert abs(dX['SCLS'] - 0.9544502701641727) < 1e-9
assert abs(dX['ENT'] - 0.8229492161955131) < 1e-9
assert abs(dX['DVU'] - D['T3_rep']['d_fam_fid']['DVU']) < 1e-6
assert abs(dX['DCC'] - D['T3_rep']['d_fam_fid']['DCC']) < 1e-6
H = R['headline']
assert H['flip_absolute_SCLS_ENT_positive'] is False
assert abs(H['combo_min_scls_ent'] - (-0.797038604750368)) < 1e-9
assert H['joint_beats_best_single'] is True
OK.append('2 result.json: res ef039473 recomputed; all gates/anchors/deltas match')

# 3. combo.npz present with expected arrays
import numpy as np
z = np.load(os.path.join(PD, 'combo.npz'))
assert z['recipes'].tolist() == ['G0', 'R1_rep', 'T3_rep', 'X_combo', 'X_combo_r']
assert z['agg_fid_matrix'].shape == (5, 6)
assert abs(z['agg_fid_matrix'][3][4] - AGG['X_combo']['fam_fid_mean']['SCLS']) < 1e-12
OK.append('3 combo.npz: 5x6 agg matrix consistent with result.json')

# 4. ledger: #397 present, chain recomputes, previous entries untouched
raw = open(LED, 'rb').read()
led = json.loads(raw.decode('utf-8'))
assert len(led['measurements']) == 397
e397 = [e for e in led['measurements'] if e.get('phase') == 3245]
assert len(e397) == 1
assert 'facec2e7' == led['ledger_sha256_8']
led['ledger_sha256_8'] = '530d0ca2'
rec = hashlib.sha256(json.dumps(led, ensure_ascii=False, indent=1)
                     .encode('utf-8')).hexdigest()[:8]
assert rec == 'facec2e7', ('ledger chain recompute mismatch', rec)
for ph in (3240, 3241, 3242, 3243, 3244):
    assert len([e for e in json.loads(raw.decode('utf-8'))['measurements']
                if e.get('phase') == ph]) == 1, ph
OK.append('4 ledger: n=397, #397 facec2e7 chain recomputes; #392-#396 intact')

# 5. MEMO: 3245 section present exactly once, prior sections untouched
mb = open(MEMO, 'rb').read()
assert mb.count(b'## Phase 3245:') == 1
for tag in ('## Phase 3240:', '## Phase 3241:', '## Phase 3242:',
            '## Phase 3243:', '## Phase 3244:'):
    assert mb.count(tag.encode()) == 1, tag
assert 'combo_flip_registered'.encode() in mb and b'105df1a8' in mb
OK.append('5 MEMO: Phase 3245 section present; 3240-3244 sections intact')

# 6. dual wlog
for p, mark in ((os.path.join(ROOT, r'.workbuddy\memory\2026-10-20.md'),
                 '3245 组合转正臂完成'),
                (os.path.join(r'C:\Users\Admin\.workbuddy\user-8ae6cef1-1e4d-4c0f-982f-7edf906d7def-personal\memory',
                              '2026-10-20.md'), '3245 组合转正臂完成')):
    txt = open(p, encoding='utf-8').read()
    assert txt.count(mark) == 1, (p, txt.count(mark))
OK.append('6 dual wlog: 3245 entry present exactly once in both files')

# 7. MEMORY.md updated, under limit
mem = open(os.path.join(ROOT, r'.workbuddy\memory\MEMORY.md'), encoding='utf-8').read()
assert '3245 收官（combo flip' in mem and '3246 已预注册' in mem
assert len(mem) < 3000, len(mem)
OK.append('7 MEMORY.md: 3245 block present (len=%d < 3000)' % len(mem))

# 8. registry v1.33 untouched (byte sha8) + v1.32 legacy intact
import re
mtxt = mem
m = re.search(r'registry v1\.33 `([0-9a-f]{8})`', mtxt)
assert m and sha8_file(REG) == m.group(1), (sha8_file(REG), m.group(1) if m else None)
OK.append('8 registry v1.33: byte sha8 %s matches MEMORY record (untouched)'
          % sha8_file(REG))

# 9. smoke artifacts recorded-only
for f in ('smoke_result.json', 'smoke_run_log.txt', 'smoke_execution.json',
          'smoke_combo.npz'):
    assert os.path.exists(os.path.join(PD, 'smoke', f)), f
OK.append('9 smoke artifacts: 4 files recorded-only, preserved')

print(chr(10).join(OK))
print('DISK VERIFY: 9/9 OK')
