# -*- coding: utf-8 -*-
"""Phase 3246 independent disk verify: re-reads every artifact from disk and
re-checks the sealed numbers. Fails loudly on any mismatch. Read-only."""
import os, json, hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
SRC_DIR = os.path.join(ROOT, r'tests\glm5\result\rdc_query_construction_20260913')
PD = os.path.join(SRC_DIR, r'phase3246\g5h9_regime_boundary')
LED = os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json')
MEMO = os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md')
REG = os.path.join(SRC_DIR, r'phase3238\g5h1_census_rank\atlas_registry_v1_33.json')
OK = []


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


# 1. execution.json: design freeze intact, hash recomputed
exe = json.load(open(os.path.join(PD, 'execution.json'), encoding='utf-8'))
assert exe['phase'] == 3246 and exe['design_sha8'] == '88ab83b9'
assert exe['smoke'] is False and exe['frozen_before'] == 'any observation'
ds = hashlib.sha256(json.dumps(exe['design'], ensure_ascii=False, sort_keys=True,
                               indent=1).encode('utf-8')).hexdigest()[:8]
assert ds == '88ab83b9', ('design hash drift', ds)
assert 'prereg_correction' in exe['design'] and 'ORACLE REDESIGN' in exe['design']['prereg_correction']
OK.append('1 execution.json: design freeze 88ab83b9 intact (recomputed); 2 SMOKE corrections documented')

# 2. result.json
R = json.load(open(os.path.join(PD, 'result.json'), encoding='utf-8'))
assert R['design_sha8'] == '88ab83b9' and R['res_sha8'] == 'df7ec079'
rs = hashlib.sha256(json.dumps(
    {k: v for k, v in R.items() if k not in ('res_sha8', 'created', 'runtime_s')},
    ensure_ascii=False, sort_keys=True, indent=1).encode('utf-8')).hexdigest()[:8]
assert rs == 'df7ec079', ('result hash drift', rs)
assert R['verdict'] == 'regime_boundary_registered'
assert R['seeds_run'] == [7, 8, 9] and R['kl_rows'] == 876
g = R['gates']
assert all(g[k] is True for k in ('G1_replication', 'G2_characterization',
                                  'clause_a_span_monotone',
                                  'clause_c_oracle_recorded',
                                  'G3_determinism', 'G4_registry_untouched'))
assert g['determinism_diff'] == 0.0
AGG = R['aggregate']
assert AGG['G0']['d_vs_sealed_3240_max'] == 0.0
assert AGG['T3_rep']['d_vs_sealed_3242_max'] == 0.0
assert AGG['XC_b1']['d_vs_sealed_3245_max'] == 0.0
assert AGG['T1_rep']['d_vs_sealed_3242_max'] == 0.0
H = R['headline']
assert H['flip_location'] == 'INFORMATION_GAP'
assert abs(H['orc_ls_full_min_scls_ent'] - 0.329) < 0.001
assert H['orc_ls_full_flips'] is True
assert AGG['ORC_ls_cls']['fam_fid_mean']['SCLS'] > 0
assert AGG['ORC_full']['fam_fid_mean']['SCLS'] < 0  # pull ceiling
seq = [H['span_ladder_SCLS'][k] for k in ('G0', 'T1_rep', 'T3_rep', 'SJE', 'S55')]
assert all(seq[i + 1] - seq[i] >= -0.05 for i in range(4)) and seq[-1] > seq[0]
assert abs(H['beta_range_minSCLSENT'] - 0.0861) < 0.001
assert AGG['XC_b4']['r2_oov_mean'] > 2 * AGG['XC_b1']['r2_oov_mean']
OK.append('2 result.json: res df7ec079 recomputed; gates/anchors/FLIP-location match')

# 3. boundary.npz consistent
import numpy as np
z = np.load(os.path.join(PD, 'boundary.npz'))
assert z['agg_fid_matrix'].shape == (13, 6)
i_orc = z['recipes'].tolist().index('ORC_ls_full')
i_scls = z['fams'].tolist().index('SCLS')
assert abs(z['agg_fid_matrix'][i_orc][i_scls]
           - AGG['ORC_ls_full']['fam_fid_mean']['SCLS']) < 1e-12
OK.append('3 boundary.npz: 13x6 agg matrix consistent with result.json')

# 4. ledger #398 + chain + previous entries
raw = open(LED, 'rb').read()
led = json.loads(raw.decode('utf-8'))
assert len(led['measurements']) == 398
assert led['ledger_sha256_8'] == 'a451686b'
led['ledger_sha256_8'] = 'facec2e7'
rec = hashlib.sha256(json.dumps(led, ensure_ascii=False, indent=1)
                     .encode('utf-8')).hexdigest()[:8]
assert rec == 'a451686b', ('ledger chain recompute mismatch', rec)
for ph in (3240, 3241, 3242, 3243, 3244, 3245):
    assert len([e for e in json.loads(raw.decode('utf-8'))['measurements']
                if e.get('phase') == ph]) == 1, ph
OK.append('4 ledger: n=398, #398 a451686b chain recomputes; #392-#397 intact')

# 5. MEMO
mb = open(MEMO, 'rb').read()
assert mb.count(b'## Phase 3246:') == 1
for tag in ('## Phase 3240:', '## Phase 3241:', '## Phase 3242:',
            '## Phase 3243:', '## Phase 3244:', '## Phase 3245:'):
    assert mb.count(tag.encode()) == 1, tag
assert 'INFORMATION_GAP'.encode() in mb and b'88ab83b9' in mb
OK.append('5 MEMO: Phase 3246 section present; 3240-3245 sections intact')

# 6. dual wlog
for p in (os.path.join(ROOT, r'.workbuddy\memory\2026-10-20.md'),
          os.path.join(r'C:\Users\Admin\.workbuddy\user-8ae6cef1-1e4d-4c0f-982f-7edf906d7def-personal\memory',
                       '2026-10-20.md')):
    txt = open(p, encoding='utf-8').read()
    assert txt.count('3246 声明体制边界刻画完成') == 1, (p, txt.count('3246 声明体制边界刻画完成'))
OK.append('6 dual wlog: 3246 entry present exactly once in both files')

# 7. MEMORY.md
mem = open(os.path.join(ROOT, r'.workbuddy\memory\MEMORY.md'), encoding='utf-8').read()
assert '3246 收官（regime boundary' in mem and '3247 已预注册' in mem
assert len(mem) < 3000, len(mem)
OK.append('7 MEMORY.md: 3246 block present (len=%d < 3000)' % len(mem))

# 8. registry untouched
import re
m = re.search(r'registry v1\.33 `([0-9a-f]{8})`', mem)
assert m and sha8_file(REG) == m.group(1), (sha8_file(REG), m.group(1) if m else None)
OK.append('8 registry v1.33: byte sha8 %s matches MEMORY record (untouched)'
          % sha8_file(REG))

# 9. smoke artifacts recorded-only (3 runs preserved)
for f in ('smoke_result.json', 'smoke_run_log.txt', 'smoke_execution.json',
          'smoke_boundary.npz'):
    assert os.path.exists(os.path.join(PD, 'smoke', f)), f
OK.append('9 smoke artifacts: 4 files recorded-only, preserved')

print(chr(10).join(OK))
print('DISK VERIFY: 9/9 OK')
