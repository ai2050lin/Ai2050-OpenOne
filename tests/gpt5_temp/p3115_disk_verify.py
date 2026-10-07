# -*- coding: utf-8 -*-
"""Phase 3115 disk verification: every closeout write
re-checked on the real disk."""
import hashlib
import io
import json
import os

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3115'
        r'\omega_p113_joint_mlp_erase_purpose')
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
                   r'\phase3115_omega_p113_joint_mlp_'
                   r'erase_purpose.py')
SCR_CLOSE = ROOT + (r'\tests\gpt5_temp'
                    r'\phase3115_closeout.py')

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
    'write_joint_partial|additive|'
    'erase_serves_generation')
chk('r02 n_records=2016', r['n_records'] == 2016)
chk('r03 smoke=False', r['smoke'] is False)
chk('r04 L20 -0.3576',
    abs(r['dmp_rel']['L20_mlp']
        - (-0.357599)) < 1e-4)
chk('r05 joint28 -0.1879',
    abs(r['dmp_rel']['joint20_28']
        - (-0.187935)) < 1e-4)
chk('r06 joint32 -0.1492',
    abs(r['dmp_rel']['joint20_32']
        - (-0.149178)) < 1e-4)
chk('r07 single_sum -0.1760',
    abs(r['dmp_rel']['single_sum_L20_L24_L28']
        - (-0.176007)) < 1e-4)
chk('r08 additive |diff|<0.05',
    abs(r['dmp_rel']['joint_minus_single_sum'])
    < 0.05)
chk('r09 js delta +0.0535',
    abs(r['erase_purpose']['js_delta_mean']
        - 0.053497) < 1e-4)
chk('r10 sign rate 0.976',
    abs(r['erase_purpose']['js_delta_sign_rate']
        - 0.97619) < 1e-3)
chk('r11 jac down',
    r['erase_purpose']['jac_delta_mean'] < 0)
chk('r12 m_consistency=0',
    r['m_consistency_max_abs_vs_3113'] == 0.0)
chk('r13 selfcheck<0.01',
    max(r['selfcheck_rel'].values()) < 0.01)
chk('r14 frozen singles',
    abs(r['dmp_rel']['frozen_L24_mlp_3114']
        - 0.34603) < 1e-4
    and abs(r['dmp_rel']['frozen_L28_mlp_3114']
            + 0.164439) < 1e-4)

# 2. design_seal.json
s = json.load(io.open(OUTD + r'\design_seal.json',
                      encoding='utf-8'))
chk('s01 ablations 5',
    s['ablations'] == ['baseline', 'abl_L20_mlp',
                       'abl_L32_mlp',
                       'abl_L20_28_mlp_joint',
                       'abl_L20_32_mlp_joint'])
chk('s02 erase gate', 'sign_rate >= 0.60'
    in s['gates']['erase'])
chk('s03 js_def', 'natural log' in s['js_def'])
chk('s04 frozen singles seal',
    abs(s['frozen_single_dmp_rel_3114']['L24_mlp']
        - 0.3460303786765555) < 1e-9)

# 3. run_log.txt
rl = io.open(OUTD + r'\run_log.txt',
             encoding='utf-8').read()
chk('l01 baseline mpair 8.2458',
    'mpair=8.2458' in rl)
chk('l02 VERDICT line',
    'VERDICT: write_joint_partial|additive|'
    'erase_serves_generation' in rl)
chk('l03 js lines', 'js mean=0.159273' in rl
    and 'js mean=0.212770' in rl)
chk('l04 m_consistency 0',
    'm_consistency=0.000e+00' in rl)
chk('l05 2016 rebuilt',
    'records rebuilt: 2016 (672 pairs)' in rl)

# 4. npz artifact
npz = OUTD + r'\ablation_readout.npz'
chk('n01 npz exists>100KB',
    os.path.exists(npz)
    and os.path.getsize(npz) > 1e5)
z = np.load(npz)
chk('n02 js keys', 'baseline__js' in z
    and 'abl_L32_mlp__js' in z)
chk('n03 cond keys', 'baseline__m' in z
    and 'abl_L20_28_mlp_joint__m' in z)

# 5. closeout_log.txt
cl = io.open(OUTD + r'\closeout_log.txt',
             encoding='utf-8').read()
chk('c01 five lines', len(cl.strip()
    .splitlines()) == 5)
chk('c02 ledger sha', 'sha=f488aa7d' in cl)

# 6. ledger
led = json.load(io.open(LEDGER, encoding='utf-8'))
m15 = [m for m in led['measurements']
       if m.get('phase') == 3115]
chk('g01 meas3115 exists', len(m15) == 1)
chk('g02 meas verdict', m15[0]['verdict'] ==
    'write_joint_partial|additive|'
    'erase_serves_generation')
chk('g03 L14 connects',
    'meas3115_omega_p113_joint_mlp_erase_purpose'
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
chk('g05 n_measurements=252',
    len(led['measurements']) == 252)

# 7. MEMO
memo = io.open(MEMO, encoding='utf-8').read()
tail = memo[-6000:]
chk('m01 header', '## Phase 3115:' in memo)
chk('m02 title', '联合消融+擦除目的性' in tail)
chk('m03 key numbers', all(
    has_any(tail, v, v.replace('-', '\u2212'))
    for v in ('-0.3576', '-0.1879', '-0.1492',
              '+0.1030', '8.2458', '5.2971',
              '6.6961', '7.0157', '9.0950',
              '0.0535', '0.976')))
chk('m04 decoupling', '信念-生成解耦' in tail)
chk('m05 distributed', '深度分布式' in tail)
chk('m06 prereg 3116',
    'L12–L36' in tail or 'L12-L36' in tail)
chk('m07 append-only (no 3116 yet)',
    '## Phase 3116:' not in memo)

# 8. wlogs
wd = io.open(WLOG_D, encoding='utf-8').read()
wc = io.open(WLOG_C, encoding='utf-8').read()
chk('w01 wlog D', 'Phase 3115 Omega-P113' in wd)
chk('w02 wlog C', 'Phase 3115 Omega-P113' in wc)

# 9. MEMORY.md
mem = io.open(MEMO_W, encoding='utf-8').read()
chk('e01 len<3000', len(mem) < 3000)
chk('e02 max=3115', 'max=3115' in mem)
chk('e03 chain 3115',
    '## 机制链状态（3115）' in mem)
chk('e04 next 3116',
    '下一 3116' in mem and 'L12–L36' in mem)
chk('e05 3114 compressed',
    'head=相关' in mem)

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
