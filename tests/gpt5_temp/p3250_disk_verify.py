# -*- coding: utf-8 -*-
"""Phase 3250 independent disk verify (dual arm): re-reads every artifact from
disk and re-checks the sealed numbers. Fails loudly on any mismatch. Read-only."""
import os, json, hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
SRC_DIR = os.path.join(ROOT, r'tests\glm5\result\rdc_query_construction_20260913')
PD = os.path.join(SRC_DIR, r'phase3250\g5h13_kernel_boundary')
LED = os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json')
MEMO = os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md')
REG = os.path.join(SRC_DIR, r'phase3238\g5h1_census_rank\atlas_registry_v1_33.json')
OK = []


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


# 1. execution.json: design freeze intact, hash recomputed
exe = json.load(open(os.path.join(PD, 'execution.json'), encoding='utf-8'))
assert exe['phase'] == 3250 and exe['design_sha8'] == 'c3aa103e'
assert exe['smoke'] is False and exe['frozen_before'] == 'any observation'
ds = hashlib.sha256(json.dumps(exe['design'], ensure_ascii=False, sort_keys=True,
                               indent=1).encode('utf-8')).hexdigest()[:8]
assert ds == 'c3aa103e', ('design hash drift', ds)
OK.append('1 execution.json: design freeze c3aa103e intact (recomputed); no SMOKE correction')

# 2. result.json
R = json.load(open(os.path.join(PD, 'result.json'), encoding='utf-8'))
assert R['design_sha8'] == 'c3aa103e' and R['res_sha8'] == '9d0f1825'
rs = hashlib.sha256(json.dumps(
    {k: v for k, v in R.items() if k not in ('res_sha8', 'created', 'runtime_s')},
    ensure_ascii=False, sort_keys=True, indent=1).encode('utf-8')).hexdigest()[:8]
assert rs == '9d0f1825', ('result hash drift', rs)
assert R['verdict'] == 'kernel_boundary_registered'
assert R['seeds_run'] == [7, 8, 9] and R['kl_rows'] == 876
g = R['gates']
assert all(g[k] is True for k in ('G1_replication', 'G2_characterization',
                                  'clause_a_transfer_realism',
                                  'clause_b_probe_closure_recorded',
                                  'G3_determinism', 'G4_registry_untouched'))
assert g['determinism_diff'] == 0.0
AGG = R['aggregate']
for key, ph in (('G0', 3240), ('T3_rep', 3242), ('XC_b4', 3246),
                ('XC_p2_b4', 3247), ('XC_p3_b4', 3248), ('K_p3_T3', 3248),
                ('XC_p5_b4', 3249), ('XC_p3_b6', 3249)):
    assert AGG[key]['d_vs_sealed_%d_max' % ph] == 0.0, (key, ph)
H = R['headline']
assert H['saturated'] is False and H['best_new_item'] == 'XC_p5_b8'
assert abs(H['best_new_closure'] - 0.032) < 0.001
assert H['numeric_probe_ok'] is True
ct = H['cond_trend_max']
assert max(ct.values()) < 1e6
assert abs(AGG['XC_p5_b8']['fam_fid_mean']['SCLS'] - (-0.449)) < 0.001
assert abs(AGG['ORC_ls_cls']['fam_fid_mean']['SCLS'] - 0.083) < 0.01
assert abs(AGG['XC_hyb_b4']['fam_fid_mean']['SCLS']
           - AGG['XC_p3_b4']['fam_fid_mean']['SCLS']) < 1e-3
OK.append('2 result.json: res 9d0f1825 recomputed; 8 anchors 0.0; boundary/closure match')

# 3. kbound.npz consistent
import numpy as np
z = np.load(os.path.join(PD, 'kbound.npz'))
assert z['agg_fid_matrix'].shape == (15, 6)
i_p5b8 = z['recipes'].tolist().index('XC_p5_b8')
i_scls = z['fams'].tolist().index('SCLS')
assert abs(z['agg_fid_matrix'][i_p5b8][i_scls]
           - AGG['XC_p5_b8']['fam_fid_mean']['SCLS']) < 1e-12
OK.append('3 kbound.npz: 15x6 agg matrix consistent with result.json')

# 4. ledger #402 + chain + previous entries
raw = open(LED, 'rb').read()
led = json.loads(raw.decode('utf-8'))
assert len(led['measurements']) == 402
assert led['ledger_sha256_8'] == 'a79a1e4f'
led['ledger_sha256_8'] = '87e2762c'
rec = hashlib.sha256(json.dumps(led, ensure_ascii=False, indent=1)
                     .encode('utf-8')).hexdigest()[:8]
assert rec == 'a79a1e4f', ('ledger chain recompute mismatch', rec)
for ph in (3240, 3241, 3242, 3243, 3244, 3245, 3246, 3247, 3248, 3249):
    assert len([e for e in json.loads(raw.decode('utf-8'))['measurements']
                if e.get('phase') == ph]) == 1, ph
OK.append('4 ledger: n=402, #402 a79a1e4f chain recomputes; #392-#401 intact')

# 5. MEMO
mb = open(MEMO, 'rb').read()
assert mb.count(b'## Phase 3250:') == 1
for tag in ('## Phase 3240:', '## Phase 3241:', '## Phase 3242:',
            '## Phase 3243:', '## Phase 3244:', '## Phase 3245:',
            '## Phase 3246:', '## Phase 3247:', '## Phase 3248:',
            '## Phase 3249:'):
    assert mb.count(tag.encode()) == 1, tag
assert 'XC_p5_b8'.encode() in mb and b'c3aa103e' in mb
OK.append('5 MEMO: Phase 3250 section present; 3240-3249 sections intact')

# 6. dual wlog
for p in (os.path.join(ROOT, r'.workbuddy\memory\2026-10-20.md'),
          os.path.join(r'C:\Users\Admin\.workbuddy\user-8ae6cef1-1e4d-4c0f-982f-7edf906d7def-personal\memory',
                       '2026-10-20.md')):
    txt = open(p, encoding='utf-8').read()
    assert txt.count('3250 核函数族数值边界') == 1, (p, txt.count('3250 核函数族数值边界'))
OK.append('6 dual wlog: 3250 entry present exactly once in both files')

# 7. MEMORY.md
mem = open(os.path.join(ROOT, r'.workbuddy\memory\MEMORY.md'), encoding='utf-8').read()
assert '3250 收官（kernel boundary' in mem and '3251 已预注册' in mem
assert len(mem) < 3000, len(mem)
OK.append('7 MEMORY.md: 3250 block present (len=%d < 3000)' % len(mem))

# 8. registry untouched (byte sha8 vs the sealed v1.33 value a30215fd;
#    the MEMORY hash annotation was compressed away in the 3250 closeout)
sha_reg = sha8_file(REG)
assert sha_reg == 'a30215fd', sha_reg
OK.append('8 registry v1.33: byte sha8 %s == sealed a30215fd (untouched)' % sha_reg)

# 9. smoke artifacts recorded-only (preserved)
for f in ('smoke_result.json', 'smoke_run_log.txt', 'smoke_execution.json',
          'smoke_kbound.npz'):
    assert os.path.exists(os.path.join(PD, 'smoke', f)), f
OK.append('9 smoke artifacts: 4 files recorded-only, preserved')

# 10. engineering arm: endpoint re-check (direct coroutine, read-only)
import subprocess, sys
py = r'C:\Users\Admin\.workbuddy\binaries\python\versions\3.11.9\python.exe'
out = subprocess.run([py, os.path.join(ROOT, r'tests\gpt5_temp\p3250_endpoint_check.py')],
                     capture_output=True, text=True, cwd=ROOT)
assert 'ENDPOINT CHECK OK' in out.stdout, out.stdout + out.stderr
fj = open(os.path.join(ROOT, 'frontend', 'src', 'components', 'app', 'rdc_fusion',
                       'FidelityMode.jsx'), encoding='utf-8').read()
assert 'GapLineChart' in fj and 'XC_p5_b8' in fj
OK.append('10 engineering arm: endpoint check re-runs OK; FidelityMode v2 markers present')

print(chr(10).join(OK))
print('DISK VERIFY: 10/10 OK')
