# -*- coding: utf-8 -*-
"""Phase 3116 disk verification: every closeout write
re-checked on the real disk."""
import hashlib
import io
import json
import os

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3116'
        r'\omega_p114_full_mlp_sweep_decouple')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
WLOG_D = ROOT + r'\.workbuddy\memory\2026-09-23.md'
WLOG_C = (r'C:\Users\Admin\WorkBuddy'
          r'\2026-09-17-01-30-05'
          r'\.workbuddy\memory\2026-09-23.md')
MEMO_W = ROOT + r'\.workbuddy\memory\MEMORY.md'
SCR_MAIN = ROOT + (r'\tests\glm5'
                   r'\phase3116_omega_p114_full_mlp_'
                   r'sweep_decouple.py')
SCR_CLOSE = ROOT + (r'\tests\gpt5_temp'
                    r'\phase3116_closeout.py')

ok = []
fail = []


def chk(name, cond):
    (ok if cond else fail).append(name)


def has_any(t, *alts):
    return any(a in t for a in alts)


# 1. result.json
r = json.load(io.open(OUTD + r'\result.json',
                      encoding='utf-8'))
chk('r01 verdict', r['verdict'] ==
    'interaction_dominant|'
    'no_behavioral_decoupling')
chk('r02 n_records=2016', r['n_records'] == 2016)
chk('r03 smoke=False', r['smoke'] is False)
chk('r04 cov 2.278',
    abs(r['coverage']['cov'] - 2.277798) < 1e-4)
chk('r05 d_all -0.9463',
    abs(r['coverage']['d_all']
        - (-0.94633)) < 1e-4)
chk('r06 sum_single -3.1019',
    abs(r['coverage']['sum_single']
        - (-3.101879)) < 1e-4)
chk('r07 top3 order',
    [t['layer'] for t in r['top3_causal']]
    == [26, 33, 31])
chk('r08 top1 L26 -0.4636',
    abs(r['top3_causal'][0]['dmp_rel']
        - (-0.46360)) < 1e-4)
chk('r09 L35 +0.376',
    abs(r['sweep']['abl_mlp_L35']['dmp_rel']
        - 0.3764) < 1e-3)
chk('r10 L24 +0.346',
    abs(r['sweep']['abl_mlp_L24']['dmp_rel']
        - 0.3460) < 1e-3)
chk('r11 segments',
    abs(r['segments']['early_L12_19']
        - (-1.3089)) < 1e-3
    and abs(r['segments']['write_L20_28']
            - (-1.5028)) < 1e-3
    and abs(r['segments']['late_L29_35']
            - (-0.2902)) < 1e-3)
chk('r12 overlap bit-exact',
    all(r['overlap_check'][k]['abs_diff'] == 0.0
        for k in ('L20', 'L24', 'L28', 'L32')))
chk('r13 agree equal 0.0283',
    abs(r['decouple']['agree_clean']
        - 0.028274) < 1e-4
    and r['decouple']['agree_clean']
    == r['decouple']['agree_abl'])
chk('r14 yes_rate bit-identical',
    r['decouple']['yes_rate_clean']
    == r['decouple']['yes_rate_abl']
    and abs(r['decouple']['yes_rate_clean']
            - 0.588542) < 1e-5)
chk('r15 diverge median 0',
    r['decouple']['diverge_median_pos'] == 0.0)
chk('r16 selfcheck<0.01',
    max(r['selfcheck_rel'].values()) < 0.01)

# 2. design_seal.json
s = json.load(io.open(OUTD + r'\design_seal.json',
                      encoding='utf-8'))
chk('s01 sweep layers 24',
    s['sweep_layers'] == list(range(12, 36)))
chk('s02 coverage gate', '0.20'
    in s['gates']['coverage'])
chk('s03 decouple gate', '0.05'
    in s['gates']['decouple']
    and '0.02' in s['gates']['decouple'])
chk('s04 yes family frozen',
    s['yes_family'] == ['yes', 'Yes', ' yes',
                        ' Yes', 'YES'])

# 3. run_log.txt
rl = io.open(OUTD + r'\run_log.txt',
             encoding='utf-8').read()
chk('l01 VERDICT line',
    'VERDICT: interaction_dominant|'
    'no_behavioral_decoupling' in rl)
chk('l02 overlap lines', rl.count(
    'diff 0.00e+00') == 4)
chk('l03 COVERAGE line',
    'COVERAGE: sum_single=-3.1019 d_all=-0.9463 '
    'cov=2.278' in rl)
chk('l04 DECOUPLE line',
    'agree_clean=0.0283 agree_abl=0.0283 '
    'diff=+0.0000' in rl)
chk('l05 all selfchecks',
    rl.count('selfcheck') == 5)

# 4. npz artifact
npz = OUTD + r'\sweep_readout.npz'
chk('n01 npz exists>100KB',
    os.path.exists(npz)
    and os.path.getsize(npz) > 1e5)
z = np.load(npz)
chk('n02 cond keys', 'baseline__m' in z
    and 'abl_all_mlp__m' in z
    and 'abl_mlp_L26__m' in z)

# 5. closeout_log.txt
cl = io.open(OUTD + r'\closeout_log.txt',
             encoding='utf-8').read()
chk('c01 five lines', len(cl.strip()
    .splitlines()) == 5)
chk('c02 memory <3000',
    'memory updated 2993 chars' in cl)

# 6. ledger
led = json.load(io.open(LEDGER, encoding='utf-8'))
m16 = [m for m in led['measurements']
       if m.get('phase') == 3116]
chk('g01 meas3116 exists', len(m16) == 1)
chk('g02 meas verdict', m16[0]['verdict'] ==
    'interaction_dominant|'
    'no_behavioral_decoupling')
chk('g03 L14 connects',
    'meas3116_omega_p114_full_mlp_sweep_decouple'
    in [l for l in led['linkage']
        if l.get('link_id')
        == 'L14_readout_spectrum_cross_model']
    [0]['connects'])
led2 = dict(led)
sha = led2.pop('ledger_sha256_8')
chk('g04 sha self-consistent',
    hashlib.sha256(json.dumps(
        led2, sort_keys=True,
        ensure_ascii=False)
        .encode('utf-8')).hexdigest()[:8] == sha)
chk('g05 n_measurements=253',
    len(led['measurements']) == 253)

# 7. MEMO
memo = io.open(MEMO, encoding='utf-8').read()
tail = memo[-6500:]
chk('m01 header', '## Phase 3116:' in memo)
chk('m02 title', '全层 MLP 扫描+行为验证' in tail)
chk('m03 key numbers', all(
    has_any(tail, v, v.replace('-', '\u2212'))
    for v in ('-0.464', '-0.448', '-0.416',
              '-0.946', '+0.376', '+0.346',
              '2.28', '5.4%', '97.2%',
              '0.588542', '0.0283', '-3.102')))
chk('m04 balanced state', '平衡态' in tail)
chk('m05 withdrawn', '撤回' in tail
    and '分布熵调节' in tail)
chk('m06 prereg 3117',
    '成对消融' in tail
    and ('温度 0.7' in tail or '采样' in tail))
chk('m07 append-only (no 3117 yet)',
    '## Phase 3117:' not in memo)

# 8. wlogs
wd = io.open(WLOG_D, encoding='utf-8').read()
wc = io.open(WLOG_C, encoding='utf-8').read()
chk('w01 wlog D', 'Phase 3116 Omega-P114' in wd)
chk('w02 wlog C', 'Phase 3116 Omega-P114' in wc)

# 9. MEMORY.md
mem = io.open(MEMO_W, encoding='utf-8').read()
chk('e01 len<3000', len(mem) < 3000)
chk('e02 max=3116', 'max=3116' in mem)
chk('e03 chain 3116',
    '## 机制链状态（3116）' in mem)
chk('e04 next 3117',
    '下一 3117' in mem and '成对消融' in mem)
chk('e05 entropy correction',
    '分布熵调节' in mem)

# 10. scripts exist
chk('p01 main script',
    os.path.exists(SCR_MAIN)
    and os.path.getsize(SCR_MAIN) > 15000)
chk('p02 closeout script',
    os.path.exists(SCR_CLOSE))

rep = ['VERIFY %d/%d PASS'
       % (len(ok), len(ok) + len(fail))]
rep += ['PASS ' + n for n in ok]
rep += ['FAIL ' + n for n in fail]
with io.open(OUTD + r'\disk_verify_out.txt', 'w',
             encoding='utf-8') as f:
    f.write('\n'.join(rep) + '\n')
print('\n'.join(rep[:3]))
