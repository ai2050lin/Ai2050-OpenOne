# -*- coding: utf-8 -*-
"""Phase 3247 independent disk verify: re-reads every artifact from disk and
re-checks the sealed numbers. Fails loudly on any mismatch. Read-only."""
import os, json, hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
SRC_DIR = os.path.join(ROOT, r'tests\glm5\result\rdc_query_construction_20260913')
PD = os.path.join(SRC_DIR, r'phase3247\g5h10_oov_info_menu')
LED = os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json')
MEMO = os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md')
REG = os.path.join(SRC_DIR, r'phase3238\g5h1_census_rank\atlas_registry_v1_33.json')
OK = []


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


# 1. execution.json: design freeze intact, hash recomputed
exe = json.load(open(os.path.join(PD, 'execution.json'), encoding='utf-8'))
assert exe['phase'] == 3247 and exe['design_sha8'] == 'd0def478'
assert exe['smoke'] is False and exe['frozen_before'] == 'any observation'
ds = hashlib.sha256(json.dumps(exe['design'], ensure_ascii=False, sort_keys=True,
                               indent=1).encode('utf-8')).hexdigest()[:8]
assert ds == 'd0def478', ('design hash drift', ds)
assert 'prereg_correction' not in exe['design'] or not exe['design'].get('prereg_correction')
OK.append('1 execution.json: design freeze d0def478 intact (recomputed); no SMOKE correction (one-pass)')

# 2. result.json
R = json.load(open(os.path.join(PD, 'result.json'), encoding='utf-8'))
assert R['design_sha8'] == 'd0def478' and R['res_sha8'] == 'c5510e5f'
rs = hashlib.sha256(json.dumps(
    {k: v for k, v in R.items() if k not in ('res_sha8', 'created', 'runtime_s')},
    ensure_ascii=False, sort_keys=True, indent=1).encode('utf-8')).hexdigest()[:8]
assert rs == 'c5510e5f', ('result hash drift', rs)
assert R['verdict'] == 'info_menu_registered'
assert R['seeds_run'] == [7, 8, 9] and R['kl_rows'] == 876
g = R['gates']
assert all(g[k] is True for k in ('G1_replication', 'G2_characterization',
                                  'clause_a_transfer_realism',
                                  'clause_b_closure_recorded',
                                  'G3_determinism', 'G4_registry_untouched'))
assert g['determinism_diff'] == 0.0
AGG = R['aggregate']
assert AGG['G0']['d_vs_sealed_3240_max'] == 0.0
assert AGG['T3_rep']['d_vs_sealed_3242_max'] == 0.0
assert AGG['XC_b4']['d_vs_sealed_3246_max'] == 0.0
assert AGG['T1_rep']['d_vs_sealed_3242_max'] == 0.0
H = R['headline']
assert H['best_menu_item'] == 'XC_poly_b4'
assert abs(H['closure_table']['XC_poly_b4'] - 0.142) < 0.001
real = H['transfer_realism']
for fam in ('rbf', 'poly', 'joint'):
    assert real[fam]['margin'] > 0.0
assert AGG['XC_poly_b4']['fam_fid_mean']['SCLS'] > AGG['XC_b4']['fam_fid_mean']['SCLS']
assert AGG['XC_poly_b4']['fam_fid_mean']['ENT'] > AGG['XC_b4']['fam_fid_mean']['ENT']
assert abs(AGG['XC_poly_b4']['fam_fid_mean']['SCLS'] - (-0.651)) < 0.001
assert abs(AGG['ORC_ls_cls']['fam_fid_mean']['SCLS'] - 0.083) < 0.01
assert AGG['K_rbf_T3_perm']['r2_oov_mean'] < AGG['K_rbf_T3']['r2_oov_mean'] - 0.05
OK.append('2 result.json: res c5510e5f recomputed; gates/anchors/closure-table match')

# 3. menu.npz consistent
import numpy as np
z = np.load(os.path.join(PD, 'menu.npz'))
assert z['agg_fid_matrix'].shape == (13, 6)
i_xp = z['recipes'].tolist().index('XC_poly_b4')
i_scls = z['fams'].tolist().index('SCLS')
assert abs(z['agg_fid_matrix'][i_xp][i_scls]
           - AGG['XC_poly_b4']['fam_fid_mean']['SCLS']) < 1e-12
OK.append('3 menu.npz: 13x6 agg matrix consistent with result.json')

# 4. ledger #399 + chain + previous entries
raw = open(LED, 'rb').read()
led = json.loads(raw.decode('utf-8'))
assert len(led['measurements']) == 399
assert led['ledger_sha256_8'] == '94fc1ffe'
led['ledger_sha256_8'] = 'a451686b'
rec = hashlib.sha256(json.dumps(led, ensure_ascii=False, indent=1)
                     .encode('utf-8')).hexdigest()[:8]
assert rec == '94fc1ffe', ('ledger chain recompute mismatch', rec)
for ph in (3240, 3241, 3242, 3243, 3244, 3245, 3246):
    assert len([e for e in json.loads(raw.decode('utf-8'))['measurements']
                if e.get('phase') == ph]) == 1, ph
OK.append('4 ledger: n=399, #399 94fc1ffe chain recomputes; #392-#398 intact')

# 5. MEMO
mb = open(MEMO, 'rb').read()
assert mb.count(b'## Phase 3247:') == 1
for tag in ('## Phase 3240:', '## Phase 3241:', '## Phase 3242:',
            '## Phase 3243:', '## Phase 3244:', '## Phase 3245:',
            '## Phase 3246:'):
    assert mb.count(tag.encode()) == 1, tag
assert 'XC_poly_b4'.encode() in mb and b'd0def478' in mb
OK.append('5 MEMO: Phase 3247 section present; 3240-3246 sections intact')

# 6. dual wlog
for p in (os.path.join(ROOT, r'.workbuddy\memory\2026-10-20.md'),
          os.path.join(r'C:\Users\Admin\.workbuddy\user-8ae6cef1-1e4d-4c0f-982f-7edf906d7def-personal\memory',
                       '2026-10-20.md')):
    txt = open(p, encoding='utf-8').read()
    assert txt.count('3247 OOV 信息合法来源菜单刻画完成') == 1, (p, txt.count('3247 OOV 信息合法来源菜单刻画完成'))
OK.append('6 dual wlog: 3247 entry present exactly once in both files')

# 7. MEMORY.md
mem = open(os.path.join(ROOT, r'.workbuddy\memory\MEMORY.md'), encoding='utf-8').read()
assert '3247 收官（info menu' in mem and '3248 已预注册' in mem
assert len(mem) < 3000, len(mem)
OK.append('7 MEMORY.md: 3247 block present (len=%d < 3000)' % len(mem))

# 8. registry untouched
import re
m = re.search(r'registry v1\.33 `([0-9a-f]{8})`', mem)
assert m and sha8_file(REG) == m.group(1), (sha8_file(REG), m.group(1) if m else None)
OK.append('8 registry v1.33: byte sha8 %s matches MEMORY record (untouched)'
          % sha8_file(REG))

# 9. smoke artifacts recorded-only (preserved)
for f in ('smoke_result.json', 'smoke_run_log.txt', 'smoke_execution.json',
          'smoke_menu.npz'):
    assert os.path.exists(os.path.join(PD, 'smoke', f)), f
OK.append('9 smoke artifacts: 4 files recorded-only, preserved')

print(chr(10).join(OK))
print('DISK VERIFY: 9/9 OK')
