# -*- coding: utf-8 -*-
"""Phase 3117 independent disk verification (~50 checks).
All assertions recomputed from disk bytes; writes a
report file (bash shim loses stdout)."""
import hashlib
import io
import json
import os

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3117'
        r'\omega_p115_pair_cancellation_sampling')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
WLOG_D = ROOT + r'\.workbuddy\memory'
WLOG_C = (r'C:\Users\Admin\WorkBuddy'
          r'\2026-09-17-01-30-05'
          r'\.workbuddy\memory')
MEMO_W = WLOG_D + r'\MEMORY.md'
VF = ROOT + (r'\tests\gpt5_temp'
             r'\p3117_verify_stdout.txt')
res = json.load(io.open(OUTD + r'\result.json',
                        encoding='utf-8'))
seal = json.load(io.open(OUTD + r'\design_seal.json',
                         encoding='utf-8'))
rlog = io.open(OUTD + r'\run_log.txt',
               encoding='utf-8').read()
clog = io.open(OUTD + r'\closeout_log.txt',
               encoding='utf-8').read()
ledger = json.load(io.open(LEDGER,
                           encoding='utf-8'))
memo = io.open(MEMO, encoding='utf-8').read()
memw = io.open(MEMO_W, encoding='utf-8').read()

R = []


def has(t, s):
    return s in t


def has_any(t, a, b):
    return (a in t) or (b in t)


def chk(cid, cond):
    R.append((cid, bool(cond)))


v = res['verdict']
# result.json
chk('r01 verdict', v == 'cancellation_pairwise_'
    'buffered|sampled_consequence_confirmed')
chk('r02 n_records', res['n_records'] == 2016)
chk('r03 smoke=False', res['smoke'] is False)
chk('r04 n_samp=300 K=5 t=0.7',
    res['n_samp_pairs'] == 300
    and res['k_reps'] == 5
    and abs(res['temp'] - 0.7) < 1e-9)
cs = res['cancel_summary']
chk('r05 canc_add_err_rel_mean',
    abs(cs['cancel_add_err_rel_mean']
        - 1.6267068748718683) < 1e-9)
chk('r06 canc gate', cs['gate']
    == 'cancellation_pairwise_buffered')
chk('r07 same_neg 0.8053',
    abs(cs['same_neg_add_frac_mean']
        - 0.8053048843012826) < 1e-9)
chk('r08 same_pos 1.1487',
    abs(cs['same_pos_add_frac_mean']
        - 1.1487073654751507) < 1e-9)
chk('r09 cann_eff_mean 0.6575',
    abs(cs['cancel_cann_eff_mean']
        - 0.6574552401059749) < 1e-9)
p = res['pairs']['pair_L26_L24']
chk('r10 L26L24 d=-0.4622',
    abs(p['d_pair']
        - (-0.4622163357799242)) < 1e-9
    and p['class'] == 'cancel')
p2 = res['pairs']['pair_L24_L35']
chk('r11 L24L35 superadd 1.149',
    abs(p2['add_frac']
        - 1.1487073654751507) < 1e-9
    and p2['class'] == 'same_pos')
chk('r12 overlap L24/L35 = 0.0',
    res['overlap_check']['L24']['abs_diff'] == 0.0
    and res['overlap_check']['L35']['abs_diff']
    == 0.0)
chk('r13 js overlap 0.0',
    res['js_overlap_check']['base_diff'] == 0.0
    and res['js_overlap_check']['abl_diff'] == 0.0)
js = res['js_stats']
chk('r14 js delta 0.0535 sign 0.976',
    abs(js['delta_mean']
        - 0.05349716588260983) < 1e-9
    and abs(js['delta_sign_rate']
            - 0.9761904761904762) < 1e-9)
gc = res['greedy_control']
chk('r15 greed full_agree',
    abs(gc['full_agree_P'] - 0.67) < 1e-9
    and abs(gc['full_agree_A1']
            - 0.6033333333333334) < 1e-9)
chk('r16 greed yes-swap 0',
    gc['div_yes_swap_rate_P'] == 0.0
    and gc['div_yes_swap_rate_A1'] == 0.0
    and gc['n_diverged'] == {'P': 1, 'A1': 25})
gf = res['greedy_first_all']
chk('r17 first_all P 0.9985 A1 0.7976',
    abs(gf['P'] - 0.9985119047619048) < 1e-9
    and abs(gf['A1'] - 0.7976190476190477) < 1e-9)
sm = res['sampled']
chk('r18 seq_agree 0.4443',
    abs(sm['seq_agree']
        - 0.44433333333333336) < 1e-9
    and sm['n_seq'] == 3000
    and sm['gate']
    == 'sampled_consequence_confirmed')
chk('r19 yes_rate drop',
    abs(sm['yes_rate_clean']
        - 0.5856666666666667) < 1e-9
    and abs(sm['yes_rate_abl'] - 0.546) < 1e-9)
chk('r20 selfcheck<0.01',
    max(res['selfcheck_rel'].values()) < 0.01
    and len(res['selfcheck_rel']) == 6)
# seal
chk('s01 seal phase 3117',
    seal['phase'] == 3117 and seal['smoke'] is False)
chk('s02 seal 12 pairs',
    len(seal['pairs']) == 12
    and seal['pair_class_rule'].startswith('cancel'))
chk('s03 seal seed rule',
    'no cond term' in seal['sampling']['seed_rule']
    and seal['sampling']['temp'] == 0.7)
# run_log
chk('l01 VERDICT line',
    has(rlog, 'VERDICT: '
        'cancellation_pairwise_buffered|'
        'sampled_consequence_confirmed'))
chk('l02 overlap 0.00e+00 x4',
    rlog.count('diff 0.00e+00') == 4)
chk('l03 CANC line', has(rlog, 'CANC: mean '
    'add_err_rel=1.627 over 7 cancel pairs -> '
    'cancellation_pairwise_buffered'))
chk('l04 SAMPLED line', has(rlog,
    'SAMPLED: seq_agree=0.4443 (1333/3000)'))
chk('l05 GREED line', has(rlog, 'GREED CONTROL: '
    'full_agree P=0.6700 A1=0.6033'))
chk('l06 selfchecks x6',
    rlog.count('rel L2 = ') == 6)
# npz
f_npz = OUTD + r'\pair_readout.npz'
chk('n01 npz exists>100KB',
    os.path.getsize(f_npz) > 100 * 1024)
z = np.load(f_npz, allow_pickle=False)
keys = set(z.files)
need = {'js_base', 'js_abl32', 'samp_pks',
        'greed_P_clean', 'greed_A1_abl',
        'samp_P_clean', 'samp_A1_abl',
        'pair_L26_L24__m', 'baseline__m'}
chk('n02 npz keys', need.issubset(keys))
chk('n03 js arrays 672',
    z['js_base'].shape == (672,)
    and z['js_abl32'].shape == (672,))
chk('n04 samp shape 300x5x12',
    z['samp_P_clean'].shape == (300, 5, 12)
    and z['samp_A1_abl'].shape == (300, 5, 12))
# closeout_log
chk('c01 five lines', len(clog.strip()
    .split('\n')) == 5)
chk('c02 ledger n=254', has(clog, 'n=254 l14=222'))
chk('c03 memory 2619', has(clog, '2619 chars'))
# ledger
m17 = [m for m in ledger['measurements']
       if m.get('phase') == 3117]
chk('g01 meas3117 exists', len(m17) == 1)
chk('g02 meas verdict',
    m17 and m17[0]['verdict'] == v)
l14 = [l for l in ledger['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
chk('g03 L14 connects 222',
    len(l14['connects']) == 222
    and l14['connects'][-1].startswith(
        'meas3117_omega_p115'))
saved = ledger['ledger_sha256_8']
ledger.pop('ledger_sha256_8', None)
blob = json.dumps(ledger, sort_keys=True,
                  ensure_ascii=False)
chk('g04 sha self-consistent',
    hashlib.sha256(blob.encode('utf-8'))
    .hexdigest()[:8] == saved)
chk('g05 n_measurements=254',
    len(ledger['measurements']) == 254)
# MEMO tail
tail = memo[-6000:]
chk('m01 header 3117',
    has(tail, '## Phase 3117:'))
chk('m02 title keys', has_any(tail,
    'add_err_rel=1.627', 'add_err_rel=1.627')
    and has(tail, '20.2%') and has(tail, '0.4443'))
chk('m03 pair numbers', has_any(tail,
    '\u22120.462', '-0.462')
    and has_any(tail, '\u22120.118', '-0.118')
    and has_any(tail, '+0.830', '+0.830'))
chk('m04 first_all', has(tail, '0.9985')
    and has(tail, '0.7976'))
chk('m05 yes rates', has(tail, '58.57%')
    and has(tail, '54.60%'))
chk('m06 prereg 3118', has(tail, '3118')
    and has(tail, 'margin'))
chk('m07 append-only', not has(memo,
    '## Phase 3118:'))
# wlogs
wd = io.open(WLOG_D + r'\2026-09-23.md',
             encoding='utf-8').read()
wc = io.open(WLOG_C + r'\2026-09-23.md',
             encoding='utf-8').read()
chk('w01 wlog D', has(wd, 'Phase 3117 Omega-P115'))
chk('w02 wlog C', has(wc, 'Phase 3117 Omega-P115'))
# MEMORY
chk('e01 len<3000', len(memw) < 3000)
chk('e02 max=3117', has(memw, 'max=3117'))
chk('e03 chain 3117',
    has(memw, '\uff083117\uff09')
    and has(memw, '1.627'))
chk('e04 next 3118', has(memw, '3118')
    and has(memw, 'margin'))
chk('e05 regime refinement',
    has(memw, 'regime'))
# scripts
chk('p01 main script', os.path.getsize(
    ROOT + r'\tests\glm5'
    r'\phase3117_omega_p115_pair_cancellation_'
    'sampling.py') > 20000)
chk('p02 closeout script', os.path.getsize(
    ROOT + r'\tests\gpt5_temp'
    r'\phase3117_closeout.py') > 5000)

npass = sum(1 for (_, ok) in R if ok)
lines = ['VERIFY %d/%d PASS' % (npass, len(R))]
for (cid, ok) in R:
    lines.append(('PASS ' if ok else 'FAIL ')
                 + cid)
out = '\n'.join(lines) + '\n'
io.open(VF, 'w', encoding='utf-8').write(out)
io.open(OUTD + r'\disk_verify_out.txt', 'w',
        encoding='utf-8').write(out)
print('VERIFY %d/%d' % (npass, len(R)))
