# -*- coding: utf-8 -*-
"""Phase 3119 independent disk verification (~70 checks).
All assertions recomputed from disk bytes where
possible; writes a report file (bash shim loses
stdout)."""
import hashlib
import io
import json
import os

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3119'
        r'\omega_p117_oscillation_attribution_'
        'writemap_linearity')
D18 = (ROOT + r'\tests\glm5\result'
       r'\rdc_query_construction_20260913'
       r'\phase3118'
       r'\omega_p116_autoregressive_margin_'
       'trajectory')
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
             r'\p3119_verify_stdout.txt')
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


V = res['verdict']
# ---- result.json values ----
chk('r01 verdict', V == 'attribution_selection_'
    'dominant|compensation_layer_specific|'
    'rewrite_nonlinear')
chk('r02 n_records 2016 pairs 672',
    res['n_records'] == 2016
    and res['n_pairs'] == 672
    and res['n_pairs_partb'] == 672)
chk('r03 smoke=False', res['smoke'] is False)
chk('r04 layers_scan 22 L14..L35',
    len(res['layers_scan']) == 22
    and sorted(res['layers_scan'])
    == list(range(14, 36)))
pa = res['part_a']
chk('r05 partA verdict', pa['verdict']
    == 'attribution_selection_dominant')
chk('r06 step t1 yesP 0.9970 noA1 0.8185',
    abs(pa['step_table'][0]['yes_frac_P']
        - 0.9970238095238095) < 1e-12
    and abs(pa['step_table'][0]['no_frac_A1']
            - 0.8184523809523809) < 1e-12)
chk('r07 steps t2..t12 all content tokens',
    all(pa['step_table'][i]['yes_frac_P'] == 0.0
        and pa['step_table'][i]['yes_frac_A1']
        == 0.0
        and pa['step_table'][i]['no_frac_P'] == 0.0
        and pa['step_table'][i]['no_frac_A1']
        == 0.0
        for i in range(1, 12)))
chk('r08 A1 no dm +8.7657 n=550',
    abs(pa['cls_stats']['A1']['no']['mean_dm']
        - 8.765673626986416) < 1e-12
    and pa['cls_stats']['A1']['no']['n'] == 550)
chk('r09 A1 yes dm -1.6445; P yes -0.4724',
    abs(pa['cls_stats']['A1']['yes']['mean_dm']
        - (-1.6445414438720578)) < 1e-12
    and abs(pa['cls_stats']['P']['yes']['mean_dm']
            - (-0.4724376698928093)) < 1e-12)
chk('r10 sel AUC mean 0.8366 present',
    abs(pa['sel_auc']['mean']
        - 0.8365936518210295) < 1e-12
    and pa['sel_auc']['verdict']
    == 'self_selection_present')
chk('r11 decile yes P 3/3 A1 2/2; no A1 0/6',
    pa['decile_yes']['P']['pass_rate'] == 1.0
    and pa['decile_yes']['P']['n_valid'] == 3
    and pa['decile_yes']['A1']['pass_rate'] == 1.0
    and pa['decile_no']['A1']['n_valid'] == 6
    and pa['decile_no']['A1']['n_pass'] == 0)
chk('r12 yes_semantic False no_eval False',
    pa['yes_semantic_confirmed'] is False
    and pa['no_evaluable'] is False)
chk('r13 combo yes,no -9.1998 n=548',
    abs(pa['combo_table']['1_2']['mean_dgap']
        - (-9.199765738779611)) < 1e-12
    and pa['combo_table']['1_2']['n'] == 548)
chk('r14 combo yes,yes +1.006 n=121',
    abs(pa['combo_table']['1_1']['mean_dgap']
        - 1.0059828566125602) < 1e-12
    and pa['combo_table']['1_1']['n'] == 121)
chk('r15 combo other,other +0.1227 n=7392',
    abs(pa['combo_table']['0_0']['mean_dgap']
        - 0.12273390999152547) < 1e-12
    and pa['combo_table']['0_0']['n'] == 7392)
chk('r16 cross combos n=1/0 gap rejected',
    pa['combo_table']['1_0']['n'] == 1
    and pa['combo_table']['0_1']['n'] == 0
    and pa['gap_attribution_confirmed'] is False)
chk('r17 curve recompute exact',
    pa['selfcheck_curve_rel_max'] == 0.0
    and len(pa['auc_curve_recomputed']) == 13)
pb = res['part_b']
chk('r18 partB verdict', pb['verdict']
    == 'compensation_layer_specific')
chk('r19 integrity overlap 6x 0.0',
    pb['integrity_confirmed'] is True
    and all(pb['overlap'][k][d] == 0.0
            for k in ('L26', 'L31', 'L33')
            for d in ('dP', 'dA')))
wm = pb['write_map']
chk('r20 TOP3 ratios bit vs 3118',
    abs(wm['L26']['ratio']
        - 0.44719892847360415) < 1e-12
    and abs(wm['L33']['ratio']
            - 0.5306656233054948) < 1e-12
    and abs(wm['L31']['ratio']
            - 0.59268886598429) < 1e-12)
chk('r21 amplifiers L30 1.3774 L32 1.1480',
    abs(wm['L30']['ratio']
        - 1.3773594868400552) < 1e-12
    and abs(wm['L32']['ratio']
            - 1.1479748121428726) < 1e-12)
chk('r22 amplifier late peaks 9/5',
    wm['L30']['peak_t'] == 9
    and wm['L32']['peak_t'] == 5)
chk('r23 shrinkers list 10 layers',
    pb['shrinkers'] == [26, 14, 16, 18, 19, 22,
                        23, 24, 31, 33])
chk('r24 L35 0.9194 rebuild_absent',
    abs(wm['L35']['ratio']
        - 0.9193717473737328) < 1e-12
    and pb['l35_verdict'] == 'rebuild_layer_absent')
chk('r25 L24 strongest shrinker 0.3637',
    abs(wm['L24']['ratio']
        - 0.3637059816752286) < 1e-12
    and wm['L24']['ratio']
    == min(wm[k]['ratio'] for k in wm))
pc = res['part_c']
chk('r26 partC verdicts', pc['verdict_lin']
    == 'rewrite_nonlinear'
    and pc['verdict_tok']
    == 'token_injection_untracked'
    and pc['verdict_int']
    == 'multiplicative_component_present')
chk('r27 R2 persist/base/full/inter',
    abs(pc['r2_persist']
        - (-0.2453164373288632)) < 1e-12
    and abs(pc['r2_base']
            - 0.18926222482334454) < 1e-12
    and abs(pc['r2_full']
            - 0.18936137607996972) < 1e-12
    and abs(pc['r2_inter']
            - 0.2918723204575383) < 1e-12)
chk('r28 d_tok 1e-4 d_int 0.1025 spearman 0.455',
    abs(pc['d_r2_token']
        - 9.915125662518509e-05) < 1e-12
    and abs(pc['d_r2_int']
            - 0.10251094437756858) < 1e-12
    and abs(pc['spearman_pred_true']
            - 0.45522346949311665) < 1e-12)
chk('r29 e_yes class means sign pattern',
    pc['e_yes_class_means']['yes'] > 0.9
    and pc['e_yes_class_means']['no'] < -0.5
    and abs(pc['e_yes_class_means']['other']
            - (-0.052956800907850266)) < 1e-12)
chk('r30 n_train=n_test=8064',
    pc['n_train'] == 8064
    and pc['n_test'] == 8064)
# ---- seal ----
chk('s01 seal phase 3119 smoke False',
    seal['phase'] == 3119
    and seal['smoke'] is False)
chk('s02 seal layers L14..L35 top3',
    seal['layers_scan'] == list(range(14, 36))
    and seal['top3'] == [26, 33, 31])
chk('s03 seal part_c spec split',
    'pair parity' in seal['part_c_spec']['split']
    and seal['part_c_spec']['n_samples'] == 16128)
chk('s04 seal gates frozen', 'ratio<0.7'
    in seal['gates']['B_map']
    and '1.1' in seal['gates']['B_map']
    and '0.10' in seal['gates']['C_tok']
    and 'bit-exact' in seal['gates']['B_ovl'])
# ---- run_log ----
chk('l01 VERDICT line',
    has(rlog, 'VERDICT: attribution_selection_'
        'dominant|compensation_layer_specific|'
        'rewrite_nonlinear'))
chk('l02 overlap lines 3x 0.00e+00 x2',
    rlog.count('OVERLAP L26: dP=0.00e+00 '
               'dA=0.00e+00') == 1
    and rlog.count('OVERLAP L31: dP=0.00e+00 '
                   'dA=0.00e+00') == 1
    and rlog.count('OVERLAP L33: dP=0.00e+00 '
                   'dA=0.00e+00') == 1)
chk('l03 replay 22 layers done',
    rlog.count('replay L') == 22)
chk('l04 WMAP CLASS line', has(rlog,
    'WMAP CLASS: shrinkers(<0.7)=[26, 14, 16, 18,'
    ' 19, 22, 23, 24, 31, 33] amplifiers(>1.1)='
    '[30, 32]'))
chk('l05 PART C line', has(rlog,
    'PART C: R2 persist=-0.2453 base=0.1893 '
    'full=0.1894 inter=0.2919'))
chk('l06 L35 line', has(rlog,
    'L35: ratio=0.919 -> rebuild_layer_absent'))
# ---- npz + independent recomputation ----
f_npz = OUTD + r'\wmap_readout.npz'
chk('n01 npz exists>1MB',
    os.path.getsize(f_npz) > 1024 * 1024)
z = np.load(f_npz, allow_pickle=False)
z18 = np.load(D18 + r'\traj_readout.npz',
              allow_pickle=False)
chk('n02 npz shapes',
    z['wmP'].shape == (22, 672, 13)
    and z['wmA'].shape == (22, 672, 13)
    and z['wmap_per_step'].shape == (22, 13))
lay = [int(x) for x in z['layers']]
chk('n03 layers order L26 first',
    lay == [26] + [L for L in range(14, 36)
                   if L != 26])
chk('n04 recompute L26 ratio exact',
    abs(float(np.abs(np.concatenate(
        [z['wmP'][0] - z['cleanP_ref'],
         z['wmA'][0] - z['cleanA_ref']],
        axis=0).astype(np.float64))
        [:, 9:13].mean())
        / max(float(np.abs(np.concatenate(
            [z['wmP'][0] - z['cleanP_ref'],
             z['wmA'][0] - z['cleanA_ref']],
            axis=0).astype(np.float64))
            [:, 0:3].mean()), 1e-9)
        - 0.44719892847360415) < 1e-9)
i30 = lay.index(30)
chk('n05 recompute L30 ratio exact',
    abs(float(np.abs(np.concatenate(
        [z['wmP'][i30] - z['cleanP_ref'],
         z['wmA'][i30] - z['cleanA_ref']],
        axis=0).astype(np.float64))
        [:, 9:13].mean())
        / max(float(np.abs(np.concatenate(
            [z['wmP'][i30] - z['cleanP_ref'],
             z['wmA'][i30] - z['cleanA_ref']],
            axis=0).astype(np.float64))
            [:, 0:3].mean()), 1e-9)
        - 1.3773594868400552) < 1e-9)
chk('n06 wm bit-exact vs 3118 for TOP3',
    all(float(np.abs(
        z['wmP'][lay.index(X)]
        - z18['gt_cleanseq_%s__P'
             % ('L%d' % X)]).max()) == 0.0
        for X in (26, 31, 33)))
# ---- closeout_log ----
chk('c01 five lines', len(clog.strip()
    .split('\n')) == 5)
chk('c02 ledger n=256 l14=224',
    has(clog, 'n=256 l14=224'))
chk('c03 memo +3232', has(clog, '3232 chars'))
chk('c04 memory 2798', has(clog, '2798 chars'))
# ---- ledger ----
m19 = [m for m in ledger['measurements']
       if m.get('phase') == 3119]
chk('g01 meas3119 exists unique',
    len(m19) == 1)
chk('g02 meas verdict', m19
    and m19[0]['verdict'] == V)
l14 = [l for l in ledger['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
chk('g03 L14 connects 224',
    len(l14['connects']) == 224
    and l14['connects'][-1].startswith(
        'meas3119_omega_p117'))
saved = ledger['ledger_sha256_8']
ledger.pop('ledger_sha256_8', None)
blob = json.dumps(ledger, sort_keys=True,
                  ensure_ascii=False)
chk('g04 sha c309103c self-consistent',
    hashlib.sha256(blob.encode('utf-8'))
    .hexdigest()[:8] == saved
    and saved == 'c309103c')
chk('g05 n_measurements=256',
    len(ledger['measurements']) == 256)
# ---- MEMO tail ----
tail = memo[-6500:]
chk('m01 header 3119', has(tail, '## Phase 3119:'))
chk('m02 title keys', has(tail, 't=1')
    and has(tail, '内容 token 步驱动')
    and has(tail, 'L30/L32')
    and has(tail, '0.19'))
chk('m03 key numbers', has_any(tail,
    '−9.20', '-9.20')
    and has(tail, '+8.77')
    and has(tail, '1.377')
    and has(tail, '1.148'))
chk('m04 ratio numbers', has(tail, '0.364')
    and has(tail, '0.447/0.531/0.593')
    and has(tail, '0.919'))
chk('m05 partC numbers', has(tail, '+0.0001')
    and has(tail, '+10.3pp'))
chk('m06 hypothesis refuted stated',
    has(tail, '方向否定')
    or has(tail, '方向被否定'))
chk('m07 prereg 3120', has(tail, '3120')
    and has(tail, '事实重述')
    and has(tail, '回归均值'))
chk('m08 timestamp', has_any(tail,
    '2026-09-23 15:', '2026-09-23 16:'))
chk('m09 append-only', not has(memo,
    '## Phase 3120:'))
# ---- wlogs ----
wd = io.open(WLOG_D + r'\2026-09-23.md',
             encoding='utf-8').read()
wc = io.open(WLOG_C + r'\2026-09-23.md',
             encoding='utf-8').read()
chk('w01 wlog D', has(wd, 'Phase 3119 Omega-P117'))
chk('w02 wlog C', has(wc, 'Phase 3119 Omega-P117'))
# ---- MEMORY ----
chk('e01 len<3000', len(memw) < 3000)
chk('e02 max=3119', has(memw, 'max=3119'))
chk('e03 chain 3119',
    has(memw, '\uff083119\uff09')
    and has(memw, '层特异')
    and has(memw, '去信念化'))
chk('e04 3118 compressed retained',
    has(memw, '3118\uff08T4\uff09')
    and has(memw, '0.981'))
chk('e05 next 3120', has(memw, '3120')
    and has(memw, '回归均值'))
# ---- scripts ----
chk('p01 main script', os.path.getsize(
    ROOT + r'\tests\glm5\phase3119_omega_p117_'
    'oscillation_attribution_writemap_'
    'linearity.py') > 20000)
chk('p02 closeout script', os.path.getsize(
    ROOT + r'\tests\gpt5_temp'
    r'\phase3119_closeout.py') > 5000)

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
