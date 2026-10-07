# -*- coding: utf-8 -*-
"""Phase 3087 independent verify: seal hashes,
npz scalars/anchors, A1 verdict re-derivation,
spectrum class re-derivation, G_DS, execution,
ledger (n/hash), MEMO/audit/wlog/MEMORY,
artifact inventory.  Exit nonzero on any
failure.  Output -> file."""
import hashlib
import io
import json
import os

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913'
     r'\phase3087'
     r'\omega_p85_glm4_l37_full_arbitration')
R_A1 = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3087'
        r'\omega_p84_glm4_layer_scan')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
AUDIT = ROOT + (r'\research\gpt5\docs'
                r'\hdmcc_knowledge_map_review_'
                r'20260921.md')
WLOG = ROOT + r'\.workbuddy\memory\2026-09-22.md'
MEMW = ROOT + r'\.workbuddy\memory\MEMORY.md'
OUTF = (ROOT + r'\tests\gpt5_temp'
        r'\phase3087_verify_out.txt')

n_pass = 0
n_fail = 0
lines = []

def chk(name, cond):
    global n_pass, n_fail
    if cond:
        n_pass += 1
        lines.append('PASS %s' % name)
    else:
        n_fail += 1
        lines.append('FAIL %s' % name)

def sha8(path):
    with io.open(path, 'rb') as f:
        return hashlib.sha256(
            f.read()).hexdigest()[:8]

# ---- 1. seal hashes ----
seal = json.load(io.open(
    R + r'\seal.json', encoding='utf-8'))
chk('seal_npz8', seal['npz_sha256_8']
    == sha8(R + r'\omega_p85_glm4_l37_full_'
            r'arbitration.npz'))
chk('seal_result8', seal['result_sha256_8']
    == sha8(R + r'\result.json'))
chk('seal_exec8', seal['exec_sha256_8']
    == sha8(R + r'\execution.json'))
chk('seal_script8', seal['script_sha256_8']
    == sha8(ROOT + r'\tests\glm5'
            r'\phase3087_omega_p85_glm4_l37_'
            r'full_arbitration.py'))
chk('seal_verdict',
    seal['verdict'] == 'fourth_mixed_absent')
chk('seal_setup_ok',
    seal['setup_ok'] is True)
chk('a1_npz8_seal', '912ec7df'
    == sha8(R_A1 + r'\omega_p84_glm4_layer_'
            r'scan.npz'))

# ---- 2. npz scalars ----
z = np.load(R + r'\omega_p85_glm4_l37_full_'
            r'arbitration.npz',
            allow_pickle=False)
chk('smoke_false', bool(z['SMOKE']) is False)
chk('forwards_20922',
    int(z['FORWARDS']) == 20922)
chk('linj_37', int(z['L_INJ']) == 37)
chk('lpost_38', int(z['L_POST']) == 38)
chk('setup_ok', bool(z['SETUP_OK']))
chk('verdict', str(z['VERDICT'])
    == 'fourth_mixed_absent')
chk('tied_false', bool(z['TIED']) is False)

# ---- 3. b anchors bit-0 ----
for b in ('B0', 'B1', 'B4', 'B6', 'B7A',
          'B8'):
    for fk in 'ABC':
        chk('%s_DIFF_%s_zero'
            % (b.lower(), fk),
            float(z[b + '_DIFF_' + fk]) == 0.0)
        chk('%s_OK_%s' % (b.lower(), fk),
            bool(z[b + '_OK_' + fk]))
for fk in 'ABC':
    chk('b3_ok_' + fk, bool(z['B3_OK_' + fk]))

# ---- 4. repro anchors ----
chk('repro_ok', bool(z['REPRO_OK']))
for fk in 'ABC':
    chk('repro_%s_parts' % fk,
        bool(z['REPRO_NNEG_OK_' + fk])
        and bool(z['REPRO_TOP8_OK_' + fk])
        and float(z['REPRO_MEDC_DIFF_' + fk])
        <= 1e-9
        and float(z['REPRO_CS1H_DIFF_' + fk])
        <= 1e-9
        and float(z['REPRO_RALL_DIFF_' + fk])
        <= 1e-9)

# ---- 5. spectrum class re-derivation ----
top3 = {fk: float(z['E3_TOP3_CS_' + fk])
        for fk in 'ABC'}
mn = min(top3.values())
mx = max(top3.values())
derived = ('trunk' if mn >= 0.9
           else ('dispersed' if mx <= 0.5
                 else 'mixed'))
chk('spec_class_rederived',
    str(z['SPEC_CLASS']) == derived
    == 'mixed')
chk('top3_values',
    abs(top3['A'] - 0.6992) < 5e-4
    and abs(top3['B'] - 0.7858) < 5e-4
    and abs(top3['C'] - 0.7442) < 5e-4)

# ---- 6. gate stats ----
chk('gds_count_0', int(z['GDS_COUNT']) == 0)
chk('gds_min_sp',
    abs(float(z['GDS_MIN_SP'])
        - (-0.2443)) < 5e-4)
chk('stouffer',
    abs(float(z['STOUFFER_Z']) - 1.564) < 5e-3)
for k in ('AB', 'AC', 'BC'):
    chk('t_len_' + k, len(z['T_' + k]) == 24)
    chk('u_len_' + k, len(z['U_' + k]) == 24)

# ---- 7. A1 re-derivation ----
za1 = np.load(R_A1 + r'\omega_p84_glm4_layer_'
              r'scan.npz', allow_pickle=False)
chk('a1_verdict',
    str(za1['VERDICT']) == 'layer_rescue')
best = max((31, 34, 37, 38),
           key=lambda L: min(
               int(za1['N_NEG_L%d_%s' % (L, fk)])
               for fk in 'ABC'))
chk('a1_best_l37', best == 37)
for fk in 'ABC':
    d = abs(float(z['MED_C_' + fk])
            - float(za1['MED_C_L37_' + fk]))
    chk('a2_vs_a1_medc_' + fk, d <= 1e-9)
    chk('a2_vs_a1_nneg_' + fk,
        int(z['N_NEG_' + fk])
        == int(za1['N_NEG_L37_' + fk]))

# ---- 8. execution.json ----
exe = json.load(io.open(
    R + r'\execution.json', encoding='utf-8'))
chk('exec_phase', exe['phase'] == 3087)
chk('exec_name',
    exe['name']
    == 'omega_p85_glm4_l37_full_arbitration')
chk('exec_smoke', exe['smoke'] is False)
chk('exec_created',
    exe['created'].startswith('2026-09-22'))
chk('exec_prereg',
    isinstance(exe.get('prereg'), dict)
    and 'mode' in exe['prereg']
    and 'verdict' in exe['prereg'])

# ---- 9. ledger ----
led = json.load(io.open(
    LEDGER, encoding='utf-8'))
chk('ledger_n226',
    len(led['measurements']) == 226)
m87 = [m for m in led['measurements']
       if isinstance(m, dict)
       and m.get('phase') == 3087]
chk('meas3087_present', len(m87) == 1)
chk('meas3087_verdict',
    m87[0]['verdict']
    == 'fourth_mixed_absent')
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'
       ][0]
chk('l14_n194',
    len(l14['connects']) == 194)
chk('l14_last_3087',
    isinstance(l14['connects'][-1], dict)
    and l14['connects'][-1].get('phase')
    == 3087)
blob = json.dumps(
    {k: v for k, v in led.items()
     if k != 'ledger_sha256_8'},
    sort_keys=True, ensure_ascii=False)
chk('ledger_sha',
    hashlib.sha256(
        blob.encode('utf-8')).hexdigest()[:8]
    == led['ledger_sha256_8'])

# ---- 10. docs ----
memo = io.open(MEMO, encoding='utf-8').read()
chk('memo_phase3087',
    '## Phase 3087:' in memo)
chk('memo_after_3086',
    memo.find('## Phase 3086:')
    < memo.find('## Phase 3087:'))
chk('memo_key_values',
    'fourth_mixed_absent' in memo
    and '0.6992' in memo
    and '20922' in memo)
aud = io.open(AUDIT, encoding='utf-8').read()
chk('audit_49',
    u'## 四十九、3087' in aud)
wl = io.open(WLOG, encoding='utf-8').read()
chk('wlog_3087', 'Phase 3087' in wl)
memw = io.open(MEMW, encoding='utf-8').read()
chk('memory_max3087', 'max=3087' in memw)

# ---- 11. artifact inventory ----
for fn in ('result.json', 'seal.json',
           'execution.json', 'run_log.txt',
           'omega_p85_glm4_l37_full_'
           'arbitration.npz',
           'closeout_log.txt'):
    chk('file_' + fn[:18],
        os.path.isfile(R + '\\' + fn))
for fn in ('result.json', 'seal.json',
           'execution.json', 'run_log.txt',
           'omega_p84_glm4_layer_scan.npz'):
    chk('a1file_' + fn[:18],
        os.path.isfile(R_A1 + '\\' + fn))
chk('no_misplaced_a2_in_3085',
    not os.path.isdir(
        ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3085'
        r'\omega_p85_glm4_l37_full_'
        r'arbitration'))

lines.append('SUMMARY pass=%d fail=%d'
             % (n_pass, n_fail))
with io.open(OUTF, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('VERIFY_DONE pass=%d fail=%d'
      % (n_pass, n_fail))
