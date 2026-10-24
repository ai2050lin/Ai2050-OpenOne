# -*- coding: utf-8 -*-
"""Phase 3249 independent disk verify: re-reads every artifact from disk and
re-checks the sealed numbers. Fails loudly on any mismatch. Read-only."""
import os, json, hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
SRC_DIR = os.path.join(ROOT, r'tests\glm5\result\rdc_query_construction_20260913')
PD = os.path.join(SRC_DIR, r'phase3249\g5h12_deg_extension')
LED = os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json')
MEMO = os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md')
REG = os.path.join(SRC_DIR, r'phase3238\g5h1_census_rank\atlas_registry_v1_33.json')
OK = []


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


# 1. execution.json: design freeze intact, hash recomputed
exe = json.load(open(os.path.join(PD, 'execution.json'), encoding='utf-8'))
assert exe['phase'] == 3249 and exe['design_sha8'] == '6eb9bed8'
assert exe['smoke'] is False and exe['frozen_before'] == 'any observation'
ds = hashlib.sha256(json.dumps(exe['design'], ensure_ascii=False, sort_keys=True,
                               indent=1).encode('utf-8')).hexdigest()[:8]
assert ds == '6eb9bed8', ('design hash drift', ds)
OK.append('1 execution.json: design freeze 6eb9bed8 intact (recomputed); no SMOKE correction')

# 2. result.json
R = json.load(open(os.path.join(PD, 'result.json'), encoding='utf-8'))
assert R['design_sha8'] == '6eb9bed8' and R['res_sha8'] == '8104fd1b'
rs = hashlib.sha256(json.dumps(
    {k: v for k, v in R.items() if k not in ('res_sha8', 'created', 'runtime_s')},
    ensure_ascii=False, sort_keys=True, indent=1).encode('utf-8')).hexdigest()[:8]
assert rs == '8104fd1b', ('result hash drift', rs)
assert R['verdict'] == 'deg_extension_registered'
assert R['seeds_run'] == [7, 8, 9] and R['kl_rows'] == 876
g = R['gates']
assert all(g[k] is True for k in ('G1_replication', 'G2_characterization',
                                  'clause_a_transfer_realism',
                                  'clause_b_probe_closure_recorded',
                                  'G3_determinism', 'G4_registry_untouched'))
assert g['determinism_diff'] == 0.0
AGG = R['aggregate']
assert AGG['G0']['d_vs_sealed_3240_max'] == 0.0
assert AGG['T3_rep']['d_vs_sealed_3242_max'] == 0.0
assert AGG['XC_b4']['d_vs_sealed_3246_max'] == 0.0
assert AGG['XC_p2_b4']['d_vs_sealed_3247_max'] == 0.0
assert AGG['XC_p3_b4']['d_vs_sealed_3248_max'] == 0.0
assert AGG['K_p3_T3']['d_vs_sealed_3248_max'] == 0.0
H = R['headline']
assert H['saturated'] is False and H['best_new_item'] == 'XC_p5_b4'
assert abs(H['best_new_closure'] - 0.102) < 0.001
assert H['numeric_probe_ok'] is True
dl = H['extension_axes']['degree_ladder_SCLS']
assert dl['XC_p2_b4'] < dl['XC_p3_b4'] < dl['XC_p4_b4'] < dl['XC_p5_b4']
assert abs(AGG['XC_p5_b4']['fam_fid_mean']['SCLS'] - (-0.467)) < 0.001
assert abs(AGG['K_p3_T3']['fam_fid_mean']['SCLS'] - (-0.529)) < 0.001
assert abs(AGG['ORC_ls_cls']['fam_fid_mean']['SCLS'] - 0.083) < 0.01
OK.append('2 result.json: res 8104fd1b recomputed; gates/anchors/closure match')

# 3. degext.npz consistent
import numpy as np
z = np.load(os.path.join(PD, 'degext.npz'))
assert z['agg_fid_matrix'].shape == (11, 6)
i_p5 = z['recipes'].tolist().index('XC_p5_b4')
i_scls = z['fams'].tolist().index('SCLS')
assert abs(z['agg_fid_matrix'][i_p5][i_scls]
           - AGG['XC_p5_b4']['fam_fid_mean']['SCLS']) < 1e-12
OK.append('3 degext.npz: 11x6 agg matrix consistent with result.json')

# 4. ledger #401 + chain + previous entries
raw = open(LED, 'rb').read()
led = json.loads(raw.decode('utf-8'))
assert len(led['measurements']) == 401
assert led['ledger_sha256_8'] == '87e2762c'
led['ledger_sha256_8'] = '652aa8a2'
rec = hashlib.sha256(json.dumps(led, ensure_ascii=False, indent=1)
                     .encode('utf-8')).hexdigest()[:8]
assert rec == '87e2762c', ('ledger chain recompute mismatch', rec)
for ph in (3240, 3241, 3242, 3243, 3244, 3245, 3246, 3247, 3248):
    assert len([e for e in json.loads(raw.decode('utf-8'))['measurements']
                if e.get('phase') == ph]) == 1, ph
OK.append('4 ledger: n=401, #401 87e2762c chain recomputes; #392-#400 intact')

# 5. MEMO
mb = open(MEMO, 'rb').read()
assert mb.count(b'## Phase 3249:') == 1
for tag in ('## Phase 3240:', '## Phase 3241:', '## Phase 3242:',
            '## Phase 3243:', '## Phase 3244:', '## Phase 3245:',
            '## Phase 3246:', '## Phase 3247:', '## Phase 3248:'):
    assert mb.count(tag.encode()) == 1, tag
assert 'XC_p5_b4'.encode() in mb and b'6eb9bed8' in mb
OK.append('5 MEMO: Phase 3249 section present; 3240-3248 sections intact')

# 6. dual wlog
for p in (os.path.join(ROOT, r'.workbuddy\memory\2026-10-20.md'),
          os.path.join(r'C:\Users\Admin\.workbuddy\user-8ae6cef1-1e4d-4c0f-982f-7edf906d7def-personal\memory',
                       '2026-10-20.md')):
    txt = open(p, encoding='utf-8').read()
    assert txt.count('3249 degree 阶梯延长完成') == 1, (p, txt.count('3249 degree 阶梯延长完成'))
OK.append('6 dual wlog: 3249 entry present exactly once in both files')

# 7. MEMORY.md
mem = open(os.path.join(ROOT, r'.workbuddy\memory\MEMORY.md'), encoding='utf-8').read()
assert '3249 收官（deg extension' in mem and '3250 已预注册' in mem
assert len(mem) < 3000, len(mem)
OK.append('7 MEMORY.md: 3249 block present (len=%d < 3000)' % len(mem))

# 8. registry untouched
import re
m = re.search(r'registry v1\.33 `([0-9a-f]{8})`', mem)
assert m and sha8_file(REG) == m.group(1), (sha8_file(REG), m.group(1) if m else None)
OK.append('8 registry v1.33: byte sha8 %s matches MEMORY record (untouched)'
          % sha8_file(REG))

# 9. smoke artifacts recorded-only (preserved)
for f in ('smoke_result.json', 'smoke_run_log.txt', 'smoke_execution.json',
          'smoke_degext.npz'):
    assert os.path.exists(os.path.join(PD, 'smoke', f)), f
OK.append('9 smoke artifacts: 4 files recorded-only, preserved')

print(chr(10).join(OK))
print('DISK VERIFY: 9/9 OK')
