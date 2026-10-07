# -*- coding: utf-8 -*-
"""Phase 3118 independent disk verification (~67 checks).
All assertions recomputed from disk bytes where possible;
writes a report file (bash shim loses stdout)."""
import hashlib
import io
import json
import os

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3118'
        r'\omega_p116_autoregressive_margin_trajectory')
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
             r'\p3118_verify_stdout.txt')
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
# ---- result.json values ----
chk('r01 verdict', v == 'belief_decays_in_generation'
    '|temporal_compensation'
    '|top_ablation_changes_behavior')
chk('r02 n_records 2016 n_pairs 672',
    res['n_records'] == 2016
    and res['n_pairs'] == 672)
chk('r03 smoke=False', res['smoke'] is False)
chk('r04 top3 [26,33,31]',
    res['top3'] == [26, 33, 31])
auc = res['auc_curve']
chk('r05 auc0 0.98090942',
    len(auc) == 13
    and abs(auc[0] - 0.9809094210600907) < 1e-12)
chk('r06 auc_last 0.67182185',
    abs(auc[12] - 0.6718218537414966) < 1e-12)
tk = res['track']
chk('r07 decay 0.30908757 gate',
    abs(tk['decay'] - 0.3090875673185941) < 1e-12
    and tk['gate']
    == 'belief_decays_in_generation')
chk('r08 curve min 0.51779071 max 0.98090942',
    abs(min(auc) - 0.5177907100340136) < 1e-12
    and abs(max(auc) - 0.9809094210600907) < 1e-12)
chk('r09 curve steps t2 0.7969 t6 0.9720',
    abs(auc[2] - 0.7968617134353742) < 1e-12
    and abs(auc[6] - 0.9720339958900227) < 1e-12)
ts = res['temporal_state']
chk('r10 temporal mean 0.52351781 gate',
    abs(ts['ratio_mean'] - 0.5235178059211297)
    < 1e-12
    and ts['gate'] == 'temporal_compensation')
chk('r11 temporal L26 ratio 0.44719893',
    abs(ts['by_layer']['L26']['ratio']
        - 0.44719892847360415) < 1e-12)
chk('r12 temporal L33 0.53066562 L31 0.59268887',
    abs(ts['by_layer']['L33']['ratio']
        - 0.5306656233054948) < 1e-12
    and abs(ts['by_layer']['L31']['ratio']
            - 0.59268886598429) < 1e-12)
cl = res['closed_loop']
chk('r13 closed L26 ratio 1.07774586',
    abs(cl['L26']['ratio'] - 1.0777458644936844)
    < 1e-12)
chk('r14 closed L33 1.22062155 L31 1.72770623',
    abs(cl['L33']['ratio'] - 1.2206215491289525)
    < 1e-12
    and abs(cl['L31']['ratio']
            - 1.7277062340789917) < 1e-12)
bh = res['behavior']
chk('r15 yes clean 0.58854167 abl26 0.66443452',
    abs(bh['yes_rate']['clean']
        - 0.5885416666666666) < 1e-12
    and abs(bh['yes_rate']['abl_L26']
            - 0.6644345238095238) < 1e-12)
chk('r16 beh max_diff 0.07589286 gate',
    abs(bh['max_diff'] - 0.0758928571428572) < 1e-12
    and bh['gate']
    == 'top_ablation_changes_behavior')
chk('r17 agree clean 0.02827381 L26 0.07142857',
    abs(bh['agree_P_A1']['clean']
        - 0.028273809523809524) < 1e-12
    and abs(bh['agree_P_A1']['abl_L26']
            - 0.07142857142857142) < 1e-12)
sm = res['sampled']
chk('r18 sampled seq_agree 0.33666667 n=1200',
    abs(sm['seq_agree'] - 0.33666666666666667)
    < 1e-12 and sm['n_seq'] == 1200)
chk('r19 sampled yes 0.59416667/0.66416667'
    ' ratio 0.48095252',
    abs(sm['yes_rate_clean']
        - 0.5941666666666666) < 1e-12
    and abs(sm['yes_rate_abl_L26']
            - 0.6641666666666667) < 1e-12
    and abs(sm['state_ratio']
            - 0.4809525247666324) < 1e-12)
gr = res['greedy_recheck']
chk('r20 recheck clean 0.99361359 own>cross',
    abs(gr['cleanseq_clean']
        - 0.9936135912698412) < 1e-12
    and gr['L26seq_L26'] > gr['L26seq_clean']
    and gr['L33seq_L33'] > gr['L33seq_clean']
    and gr['L31seq_L31'] > gr['L31seq_clean'])
# ---- seal ----
chk('s01 seal phase 3118 smoke False',
    seal['phase'] == 3118
    and seal['smoke'] is False)
chk('s02 seal top3 n_new replay',
    seal['top3'] == [26, 33, 31]
    and seal['n_new'] == 12
    and len(seal['replay_matrix']) == 1)
chk('s03 seal seed rule temp',
    'no cond term' in seal['sampled']['seed_rule']
    and seal['sampled']['temp'] == 0.7
    and seal['sampled']['K'] == 2)
# ---- run_log ----
chk('l01 VERDICT line',
    has(rlog, 'VERDICT: '
        'belief_decays_in_generation|'
        'temporal_compensation|'
        'top_ablation_changes_behavior'))
chk('l02 TRACK decay line',
    has(rlog, 'TRACK: AUC(0)=0.9809 '
        'AUC(12)=0.6718 decay=0.3091 -> '
        'belief_decays_in_generation'))
chk('l03 TRACK curve line', has(rlog,
    'TRACK curve: 0.981 0.559 0.797 0.534 '
    '0.760 0.879 0.972 0.853 0.518 0.645 '
    '0.658 0.712 0.672'))
chk('l04 TEMPORAL mean line', has(rlog,
    'TEMPORAL: mean ratio=0.524 over TOP3'
    ' -> temporal_compensation'))
chk('l05 BEHAV line', has(rlog,
    'BEHAV: yes_rate clean=0.5885 L26=0.6644'
    ' L33=0.5982 L31=0.5558')
    and has(rlog, 'max_diff=0.0759 -> '
        'top_ablation_changes_behavior'))
chk('l06 CLOSED+SAMPLED lines',
    has(rlog, 'CLOSED L31: head=1.2662'
        ' tail=2.1877 ratio=1.728')
    and has(rlog, 'SAMPLED: seq_agree=0.3367'
        ' yes clean=0.5942 abl26=0.6642'))
# ---- npz ----
f_npz = OUTD + r'\traj_readout.npz'
chk('n01 npz exists>500KB',
    os.path.getsize(f_npz) > 500 * 1024)
z = np.load(f_npz, allow_pickle=False)
keys = set(z.files)
need = {'m_base_check', 'pk', 'cond', 'truth',
        'auc_curve', 'gt_cleanseq_clean__P',
        'gt_cleanseq_clean__A1',
        'gt_cleanseq_L26__P',
        'gt_L31seq_clean__A1',
        'gen_clean__P', 'gen_abl_L26__A1',
        'st_samp_cleanseq_clean',
        'st_samp_cleanseq_L26',
        'st_samp_L26seq_clean', 'samp_pks'}
chk('n02 npz keys', need.issubset(keys))
chk('n03 npz auc_curve == result.json',
    z['auc_curve'].shape == (13,)
    and np.allclose(z['auc_curve'],
                    np.array(auc), atol=1e-12))
chk('n04 gt/gen/st shapes',
    z['gt_cleanseq_clean__P'].shape == (672, 13)
    and z['gt_cleanseq_clean__A1'].shape
    == (672, 13)
    and z['gen_clean__P'].shape == (672, 12)
    and z['gen_abl_L26__A1'].shape == (672, 12)
    and z['st_samp_cleanseq_L26'].shape
    == (1200, 13))
chk('n05 m_base 2016 rows; samp_pks 300',
    z['m_base_check'].shape[0] == 2016
    and len(z['samp_pks']) == 300)
chk('n06 pk/cond/truth 2016',
    len(z['pk']) == 2016
    and len(z['cond']) == 2016
    and len(z['truth']) == 2016)


def auc_mw(pos_vals, neg_vals):
    """Mann-Whitney AUC with average ranks
    for ties (no scipy) - copied from main
    script for independent recomputation."""
    x = np.concatenate([pos_vals, neg_vals])
    n1 = len(pos_vals)
    n2 = len(neg_vals)
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x), dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    r1 = ranks[:n1].sum()
    return float((r1 - n1 * (n1 + 1) / 2.0)
                 / (n1 * n2))


# ---- independent recomputation from npz ----
P0 = z['gt_cleanseq_clean__P'].astype(np.float64)
A0 = z['gt_cleanseq_clean__A1'].astype(np.float64)
auc2 = [auc_mw(P0[:, t], A0[:, t])
        for t in range(13)]
chk('x01 recompute auc0', abs(auc2[0]
    - 0.9809094210600907) < 1e-9)
chk('x02 recompute full curve+decay',
    np.allclose(np.array(auc2), np.array(auc),
                atol=1e-9)
    and abs(auc2[0] - auc2[12]
            - 0.3090875673185941) < 1e-9)
dmP = (z['gt_cleanseq_L26__P']
       - z['gt_cleanseq_clean__P'])
dmA = (z['gt_cleanseq_L26__A1']
       - z['gt_cleanseq_clean__A1'])
dm = np.concatenate([dmP, dmA],
                    axis=0).astype(np.float64)
head = np.abs(dm[:, 0:3]).mean()
tail = np.abs(dm[:, 9:13]).mean()
chk('x03 recompute temporal L26 ratio',
    abs(tail / max(head, 1e-9)
        - 0.44719892847360415) < 1e-9)
dP = (z['gt_L31seq_L31__P']
      - z['gt_cleanseq_clean__P'])
dA = (z['gt_L31seq_L31__A1']
      - z['gt_cleanseq_clean__A1'])
dm2 = np.concatenate([dP, dA],
                     axis=0).astype(np.float64)
h2 = np.abs(dm2[:, 0:3]).mean()
t2 = np.abs(dm2[:, 9:13]).mean()
chk('x04 recompute closed L31 ratio',
    abs(t2 / max(h2, 1e-9)
        - 1.7277062340789917) < 1e-9)
dms = (z['st_samp_cleanseq_L26']
       - z['st_samp_cleanseq_clean']) \
    .astype(np.float64)
hs = np.abs(dms[:, 0:3]).mean()
tsl = np.abs(dms[:, 9:13]).mean()
chk('x05 recompute sampled state_ratio',
    abs(tsl / max(hs, 1e-9)
        - 0.4809525247666324) < 1e-9)
YF = {7414, 9454, 9693, 9834, 14004}


def yes_rate(arr):
    return float(np.isin(arr[:, :8], list(YF))
                 .any(axis=1).sum()) / len(arr)


yr_c = (yes_rate(z['gen_clean__P'])
        + yes_rate(z['gen_clean__A1'])) / 2.0
yr_a = (yes_rate(z['gen_abl_L26__P'])
        + yes_rate(z['gen_abl_L26__A1'])) / 2.0
chk('x06 recompute yes rates',
    abs(yr_c - 0.5885416666666666) < 1e-12
    and abs(yr_a - 0.6644345238095238) < 1e-12)
gpc = z['gen_clean__P'][:, :8]
gac = z['gen_clean__A1'][:, :8]
ag = int((gpc == gac).all(axis=1).sum())
chk('x07 recompute agree clean 19/672',
    abs(ag / 672.0 - 0.028273809523809524)
    < 1e-12)
# ---- closeout_log ----
chk('c01 five lines', len(clog.strip()
    .split('\n')) == 5)
chk('c02 ledger n=255 l14=223',
    has(clog, 'n=255 l14=223'))
chk('c03 memory 2641', has(clog, '2641 chars'))
# ---- ledger ----
m18 = [m for m in ledger['measurements']
       if m.get('phase') == 3118]
chk('g01 meas3118 exists unique',
    len(m18) == 1)
chk('g02 meas verdict', m18
    and m18[0]['verdict'] == v)
l14 = [l for l in ledger['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
chk('g03 L14 connects 223',
    len(l14['connects']) == 223
    and l14['connects'][-1].startswith(
        'meas3118_omega_p116'))
saved = ledger['ledger_sha256_8']
ledger.pop('ledger_sha256_8', None)
blob = json.dumps(ledger, sort_keys=True,
                  ensure_ascii=False)
chk('g04 sha 0541d3be self-consistent',
    hashlib.sha256(blob.encode('utf-8'))
    .hexdigest()[:8] == saved
    and saved == '0541d3be')
chk('g05 n_measurements=255',
    len(ledger['measurements']) == 255)
# ---- MEMO tail ----
tail = memo[-6000:]
chk('m01 header 3118', has(tail, '## Phase 3118:'))
chk('m02 title numbers 0.981 0.672 +7.6pp',
    has(tail, '0.981') and has(tail, '0.672')
    and has(tail, '+7.6pp'))
chk('m03 ratio numbers', has(tail,
    '0.447/0.531/0.593')
    and has(tail, '1.078/1.221/1.728'))
chk('m04 yes rates 58.85 66.44',
    has(tail, '58.85%') and has(tail, '66.44%'))
chk('m05 bit-level clean 0.588542',
    has(tail, '0.588542'))
chk('m06 L32 zero-change contrast',
    has(tail, 'L32')
    and has_any(tail, '\u22123.3pp', '-3.3pp'))
chk('m07 prereg 3119 tokens',
    has(tail, '3119')
    and has(tail, '振荡 token 归因')
    and has(tail, 'margin(t)'))
chk('m08 timestamp 14:49',
    has(tail, '2026-09-23 14:49'))
chk('m09 append-only', not has(memo,
    '## Phase 3119:'))
# ---- wlogs ----
wd = io.open(WLOG_D + r'\2026-09-23.md',
             encoding='utf-8').read()
wc = io.open(WLOG_C + r'\2026-09-23.md',
             encoding='utf-8').read()
chk('w01 wlog D', has(wd, 'Phase 3118'))
chk('w02 wlog C', has(wc, 'Phase 3118'))
# ---- MEMORY ----
chk('e01 len<3000', len(memw) < 3000)
chk('e02 max=3118', has(memw, 'max=3118'))
chk('e03 chain 3118 numbers',
    has(memw, '\uff083118\uff09')
    and has(memw, '0.981')
    and has(memw, '0.672'))
chk('e04 compensation vs snowball',
    has(memw, '时间补偿')
    and has(memw, '1.08'))
chk('e05 next 3119', has(memw, '3119')
    and has(memw, '写入地图'))
# ---- scripts ----
chk('p01 main script', os.path.getsize(
    ROOT + r'\tests\glm5\phase3118_omega_p116_'
    'autoregressive_margin_trajectory.py')
    > 20000)
chk('p02 closeout script', os.path.getsize(
    ROOT + r'\tests\gpt5_temp'
    r'\phase3118_closeout.py') > 5000)

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
