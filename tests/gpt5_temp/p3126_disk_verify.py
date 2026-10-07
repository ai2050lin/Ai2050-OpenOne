# -*- coding: utf-8 -*-
"""Phase 3126 disk verify (independent):
A0 result -> A1 seal -> A2 npz shapes ->
A3 frozen inputs (24/25) -> B1 Part-A refit
+ D6 decomp + AR(2) + G6 recompute ->
B2 permutation bit-recompute (seed 3126,
200+200 full-pipeline) -> B3 Part-A gates ->
B4 Part-B recompute (readout AUC / E
curves via span_idx / lstar / sign /
vs-3124 corr) -> B5 wspec write-chain ->
C Part-C regen stats recompute -> C2 full
verdict re-assembly -> D ledger -> E MEMO
-> F wlogs -> G MEMORY. All recomputes
deterministic (no_mc): rng only the
frozen seed-3126 permutation."""
import hashlib
import io
import json
import os
import re

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913')
D18 = RDIR + r'\phase3118' \
      r'\omega_p116_autoregressive_margin_' \
      'trajectory'
D20 = RDIR + r'\phase3120' \
      r'\omega_p118_content_attr_amplifier_' \
      'behavior_opshape'
D24 = RDIR + r'\phase3124' \
      r'\omega_p122_resid_cm_kalman_l35rel_' \
      'glm4x'
D25 = RDIR + r'\phase3125' \
      r'\omega_p123_third_comp_qwen_' \
      'inputstream'
OUTD = RDIR + r'\phase3126' \
       r'\omega_p124_glm4_anchoredlast_' \
       'regen_writechain'
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
WLOG_D = ROOT + r'\.workbuddy\memory'
WLOG_C = (r'C:\Users\Admin\WorkBuddy'
          r'\2026-09-17-01-30-05'
          r'\.workbuddy\memory')
MEMO_W = WLOG_D + r'\MEMORY.md'
VF = (ROOT + r'\tests\gpt5_temp'
      r'\p3126_verify_out.txt')
fails = []
o = []


def sec(name, conds):
    bad = [i for i, c in enumerate(conds)
           if not c]
    o.append('%s: %d/%d pass%s'
             % (name, len(conds) - len(bad),
                len(conds),
                (' FAIL at ' + str(bad))
                if bad else ''))
    if bad:
        fails.append((name, bad))


NP = 672
N_NEW = 12
NLG = 40
TRAIL_D = 6
N_PERM = 200
PERM_SEED = 3126
SCOND = ('s0', 's1', 's2', 's3')
DIRS = ('P', 'A1')

# ---------- A0. result ----------
rf = io.open(OUTD + r'\result.json',
             encoding='utf-8')
res = json.load(rf)
rf.close()
V25 = ('common_mode_partial|trail_present|'
       'ar_absent|trajectory_transient|'
       'third_partial|path_valid|'
       'replay_bit_exact|readout_ok|spans_ok|'
       'qis_syntax_L23|qis_syntax_L22|'
       'qis_content_L28|qis_content_L32|'
       'qis_syn_final_negative|'
       'qis_syn_final_negative|'
       'qis_cont_final_negative|'
       'qis_cont_final_negative')
pa = res['part_a']
pb = res['part_b']
pc = res['part_c']
pd = res['part_d']
verdict = res['verdict']
sec('A0_result', [
    res['phase'] == 3126,
    res['smoke'] is False,
    res['n_pairs'] == NP,
    res['np_b'] == NP,
    res['np_reg'] == 96,
    res['runtime_s'] > 0,
    len(verdict.split('|')) == 19,
    'long_range' in verdict,
    'trail_' in verdict,
    'path_valid' in verdict,
    'readout' in verdict,
    'spans' in verdict,
    'behavior_shift' in verdict,
    'writechain' in verdict,
    isinstance(pa, dict),
    isinstance(pb, dict) and
    isinstance(pc, dict) and
    isinstance(pd, dict)])

# ---------- A1. seal ----------
sf = io.open(OUTD + r'\design_seal.json',
             encoding='utf-8')
seal = json.load(sf)
sf.close()
gpb = pb['gen_probe_bos']
sec('A1_seal', [
    seal['phase'] == 3126,
    seal['smoke'] is False,
    seal['np_a'] == NP,
    seal['np_b'] == NP,
    seal['np_reg'] == 96,
    seal['name'] == res['name'],
    '3126' in seal['part_a']['no_mc'],
    'LAST' in seal['part_b']['span_fix'],
    gpb['prefix_ids'] == [151331, 151333],
    gpb['prefix_decoded'] == '[gMASK]<sop>',
    gpb['gen_enc_len'] - gpb['plain_len'] == 2,
    isinstance(seal['part_b']['gates'],
               dict)])

# ---------- A2. npz shapes ----------
zn = np.load(OUTD + r'\p124_readout.npz',
             allow_pickle=False)
ck = []
for dc in DIRS:
    ck.append(zn['G6_%s' % dc].shape == (10, 6))
    ck.append(zn['wspec_%s' % dc].shape
              == (NLG, 13))
    ck.append(zn['wspec_%s' % dc].dtype
              == np.float32)
    ck.append(zn['span_idx_%s' % dc].shape
              == (NP, 2))
    ck.append(zn['span_idx_%s' % dc].dtype
              == np.int16)
    ck.append(zn['gen_base_%s' % dc].shape
              == (NP, N_NEW))
    ck.append(zn['gen_base_%s' % dc].dtype
              == np.int32)
    for c in SCOND:
        ck.append(zn['mlg_%s_%s' % (c, dc)]
                  .shape == (NP, NLG + 1, 13))
        ck.append(zn['mlg_%s_%s' % (c, dc)]
                  .dtype == np.float32)
        ck.append(zn['regen_%s_%s' % (c, dc)]
                  .shape == (96, N_NEW))
    for nm in ('syn', 'cont'):
        ck.append(zn['E_%s_%s' % (nm, dc)]
                  .shape == (NLG + 1,))
allfin = all(
    bool(np.isfinite(zn['mlg_%s_%s'
                         % (c, dc)]).all())
    for dc in DIRS for c in SCOND)
allfin = allfin and all(
    bool(np.isfinite(zn['wspec_%s' % dc])
         .all()) for dc in DIRS)
ck.append(allfin)
sec('A2_npz_shape', ck)

# ---------- A3. frozen inputs ----------
r24 = json.load(io.open(
    D24 + r'\result.json', encoding='utf-8'))
r25 = json.load(io.open(
    D25 + r'\result.json', encoding='utf-8'))
z18 = np.load(D18 + r'\traj_readout.npz',
              allow_pickle=False)
z20 = np.load(D20 + r'\p118_readout.npz',
              allow_pickle=False)
z24 = np.load(D24 + r'\p122_readout.npz',
              allow_pickle=False)
sec('A3_frozen_inputs', [
    r24['smoke'] is False,
    abs(r24['part_a']['decomp']['P']
        ['cm_share']
        - 0.19001305759179912) < 1e-12,
    r25['smoke'] is False,
    r25['verdict'] == V25,
    abs(r25['part_a']['decomp3']['P']
        ['trail_share']
        - 0.08599867672568678) < 1e-12,
    abs(r25['part_a']['decomp3']['A1']
        ['trail_share']
        - 0.05022617239377846) < 1e-12,
    len(z18['auc_curve']) == N_NEW + 1,
    z20['annP_c'].shape == (NP, N_NEW),
    z20['annA_c'].shape == (NP, N_NEW),
    'gt_cleanseq_clean__P' in z18,
    'gt_cleanseq_clean__A1' in z18,
    'E_syn_P' in z24])


# ---------- B1. Part-A pipeline ----
def content_table(m, cls):
    dm = m[:, 1:] - m[:, :-1]
    tab = np.zeros((10, N_NEW),
                   dtype=np.float64)
    for k in range(N_NEW):
        col = cls[:, k]
        dv = dm[:, k]
        allm = dv.mean()
        for cc in range(10):
            sel = (col == cc)
            if sel.any():
                tab[cc, k] = float(
                    dv[sel].mean() - allm)
    return tab


def refit(m, cls, tab):
    dm = m[:, 1:] - m[:, :-1]
    dev = tab[cls, np.arange(N_NEW)]
    dm_res = dm - dev
    X = np.stack([m[:, :-1].ravel(),
                  np.ones(m[:, :-1].size)], 1)
    y = dm_res.ravel()
    bb, *_ = np.linalg.lstsq(X, y,
                             rcond=None)
    S_dc = float(bb[0])
    MS_dc = float(-bb[1] / bb[0])
    resid = (y - X @ bb).reshape(NP, N_NEW)
    return S_dc, MS_dc, resid


def trailX(cls, D):
    NIK = cls.size
    X = np.zeros((NIK, 10 * D))
    for d in range(1, D + 1):
        rows_k = np.arange(d, N_NEW)
        idx_rows = (np.repeat(
            np.arange(NP) * N_NEW,
            len(rows_k))
            + np.tile(rows_k, NP))
        cls_d = cls[:, :N_NEW - d].ravel()
        X[idx_rows, cls_d * D + (d - 1)] = 1.0
    return X


def seq_shares(m, MS, resid, cls, D):
    ss_tot = float((resid * resid).sum())
    NIK = resid.size
    Z1 = np.zeros((NIK, N_NEW))
    Z1[np.arange(NIK),
       np.arange(NIK) % N_NEW] = 1.0
    b1, *_ = np.linalg.lstsq(
        Z1, resid.ravel(), rcond=None)
    r1 = (resid.ravel() - Z1 @ b1).reshape(
        NP, N_NEW)
    ss_step = ss_tot - float((r1 * r1).sum())
    x = (m[:, :-1] - MS).ravel()
    Z2 = np.stack([x, np.ones(NIK)], 1)
    b2, *_ = np.linalg.lstsq(
        Z2, r1.ravel(), rcond=None)
    r2 = (r1.ravel() - Z2 @ b2).reshape(
        NP, N_NEW)
    ss_m = float((r1 * r1).sum()) \
        - float((r2 * r2).sum())
    X_tr = trailX(cls, D)
    btr, *_ = np.linalg.lstsq(
        X_tr, r2.ravel(), rcond=None)
    r3 = (r2.ravel() - X_tr @ btr).reshape(
        NP, N_NEW)
    ss_trail = float((r2 * r2).sum()) \
        - float((r3 * r3).sum())
    return (ss_step, ss_m, ss_trail, ss_tot,
            r3, X_tr, btr)


MSEQ = {'P': z18['gt_cleanseq_clean__P'][:NP]
        .astype(np.float64),
        'A1': z18['gt_cleanseq_clean__A1'][:NP]
        .astype(np.float64)}
ANN = {'P': z20['annP_c'][:NP],
       'A1': z20['annA_c'][:NP]}
fit = {}
c1 = []
for dc in DIRS:
    tab = content_table(MSEQ[dc], ANN[dc])
    S_dc, MS_dc, resid = refit(MSEQ[dc],
                               ANN[dc], tab)
    fit[dc] = {'S': S_dc, 'MS': MS_dc,
               'resid': resid, 'col': ANN[dc]}
    c1.append(abs(S_dc - pa['refit'][dc]['S'])
              < 1e-12)
    c1.append(abs(MS_dc - pa['refit'][dc]
                  ['MS']) < 1e-12)
# D6 decomposition recompute
dec6r = {}
ar6r = {}
G6r = {}
r4r = {}
dec3r = {}
for dc in DIRS:
    m = MSEQ[dc]
    R = fit[dc]['resid']
    cls = fit[dc]['col']
    (ss_step, ss_m, ss_tr3, ss_tot,
     _, _, _) = seq_shares(m, fit[dc]['MS'],
                           R, cls, 3)
    dec3r[dc] = ss_tr3 / ss_tot
    (ss_step, ss_m, ss_tr6, ss_tot,
     r3, X_tr, btr) = seq_shares(
        m, fit[dc]['MS'], R, cls, TRAIL_D)
    y_ar = r3[:, 2:].ravel()
    X_ar = np.stack(
        [r3[:, 1:N_NEW - 1].ravel(),
         r3[:, 0:N_NEW - 2].ravel(),
         np.ones(y_ar.size)], 1)
    bar, *_ = np.linalg.lstsq(X_ar, y_ar,
                              rcond=None)
    phi1, phi2, c_ar = (float(bar[0]),
                        float(bar[1]),
                        float(bar[2]))
    r4 = r3.copy()
    r4[:, 2:] = r3[:, 2:] - (
        phi1 * r3[:, 1:N_NEW - 1]
        + phi2 * r3[:, 0:N_NEW - 2] + c_ar)
    ss_ar = float(((r3[:, 2:] - r4[:, 2:])
                   ** 2).sum())
    ss_rem = float((r4 * r4).sum())
    leak = (1.0 - ss_step / ss_tot
            - ss_m / ss_tot
            - ss_tr6 / ss_tot
            - ss_ar / ss_tot
            - ss_rem / ss_tot)
    dec6r[dc] = {'cm_share': ss_step / ss_tot,
                 'm_share': ss_m / ss_tot,
                 'trail6_share':
                     ss_tr6 / ss_tot,
                 'ar6_share': ss_ar / ss_tot,
                 'rem6_share':
                     ss_rem / ss_tot,
                 'leak': leak,
                 'ss_tot': ss_tot}
    ar6r[dc] = {'phi1': phi1, 'phi2': phi2,
                'c': c_ar}
    G6r[dc] = btr.reshape(10, TRAIL_D)
    r4r[dc] = r4
for dc in DIRS:
    d6 = dec6r[dc]
    d6r = pa['decomp6'][dc]
    for k in ('cm_share', 'm_share',
              'trail6_share', 'ar6_share',
              'rem6_share', 'ss_tot'):
        c1.append(abs(d6[k] - d6r[k]) < 1e-9)
    c1.append(abs(d6['leak']) < 1e-9)
    c1.append(abs(d6['leak']
                  - d6r['leak']) < 1e-9)
    for k in ('phi1', 'phi2', 'c'):
        c1.append(abs(ar6r[dc][k]
                      - pa['ar6_params'][dc][k])
                  < 1e-9)
    g_npz = zn['G6_%s' % dc]
    c1.append(float(np.abs(G6r[dc] - g_npz)
                    .max()) < 1e-9)
    g_res = np.array(pa['trail6_G'][dc])
    c1.append(float(np.abs(G6r[dc] - g_res)
                    .max()) < 1e-12)
    c1.append(abs(dec3r[dc]
                  - pa['decomp3_repl'][dc])
              < 1e-12)
sec('B1_partA_refit_decomp', c1)

# ---------- B2. permutation ------
perm_stat_r = {}
rgA = np.random.default_rng(PERM_SEED)
for dc in DIRS:
    m = MSEQ[dc]
    R = fit[dc]['resid']
    cls = fit[dc]['col']
    obs = dec6r[dc]['trail6_share']
    nulls = np.zeros(N_PERM)
    for rep in range(N_PERM):
        pm = np.argsort(
            rgA.random((NP, N_NEW)), axis=1)
        cls_p = np.take_along_axis(
            cls, pm, axis=1)
        tab_p = content_table(m, cls_p)
        S_p, MS_p, resid_p = refit(m, cls_p,
                                   tab_p)
        (_, _, ss_tr, ss_tot_p,
         _, _, _) = seq_shares(
            m, MS_p, resid_p, cls_p,
            TRAIL_D)
        nulls[rep] = ss_tr / ss_tot_p
    mu = float(nulls.mean())
    sd = float(nulls.std())
    zsc = (obs - mu) / sd if sd > 0 else 0.0
    p_hat = float((nulls >= obs).mean())
    perm_stat_r[dc] = {
        'obs': obs, 'null_mean': mu,
        'null_std': sd, 'z': zsc,
        'p_hat': p_hat}
c2 = []
for dc in DIRS:
    pr = pa['perm'][dc]
    qr = perm_stat_r[dc]
    for k in ('obs', 'null_mean',
              'null_std', 'z', 'p_hat'):
        c2.append(abs(qr[k] - pr[k]) < 1e-9)
    c2.append(pr['reps'] == N_PERM)
    c2.append(pr['seed'] == PERM_SEED)
sec('B2_partA_perm', c2)

# ---------- B3. Part-A gates -----
t3_min = min(r25['part_a']['decomp3']['P']
             ['trail_share'],
             r25['part_a']['decomp3']['A1']
             ['trail_share'])
t6_min = min(dec6r['P']['trail6_share'],
             dec6r['A1']['trail6_share'])
a_long_r = ('long_range_trail_present'
            if t6_min - t3_min >= 0.02
            else 'long_range_trail_absent')
z_min = min(perm_stat_r['P']['z'],
            perm_stat_r['A1']['z'])
a_sig_r = ('trail_significant'
           if z_min >= 4.0
           else 'trail_notsig')
gp_ = G6r['P'].ravel()
ga_ = G6r['A1'].ravel()
g_corr_r = float(np.corrcoef(gp_, ga_)[0, 1])
a_share_r = ('trail_shared_directions'
             if g_corr_r >= 0.5
             else 'trail_dirspecific')
sec_r = {}
for dc in DIRS:
    m = MSEQ[dc]
    r4 = r4r[dc]
    cls = fit[dc]['col']
    ss_tot = dec6r[dc]['ss_tot']
    ss_r4 = float((r4 * r4).sum())
    rows = r4[:, 1:].ravel()
    N2 = rows.size
    Xj = np.zeros((N2, 100))
    ck_ = cls[:, 1:].ravel()
    ckm1 = cls[:, :-1].ravel()
    Xj[np.arange(N2), ck_ * 10 + ckm1] = 1.0
    bj, *_ = np.linalg.lstsq(Xj, rows,
                             rcond=None)
    rj = rows - Xj @ bj
    gain_j = (float((r4[:, 1:] ** 2).sum())
              - float((rj * rj).sum())) / ss_tot
    x = (m[:, :-1] - fit[dc]['MS']).ravel()
    Xq = np.stack([x, x * x,
                   np.ones(x.size)], 1)
    bq, *_ = np.linalg.lstsq(Xq, r4.ravel(),
                             rcond=None)
    rq = r4.ravel() - Xq @ bq
    gain_q = (ss_r4 - float((rq * rq)
                            .sum())) / ss_tot
    sec_r[dc] = {'joint_gain': gain_j,
                 'mquad_gain': gain_q}
sec_min_r = min(
    max(sec_r['P']['joint_gain'],
        sec_r['P']['mquad_gain']),
    max(sec_r['A1']['joint_gain'],
        sec_r['A1']['mquad_gain']))
a_second_r = ('second_order_candidate_'
              'present' if sec_min_r >= 0.05
              else 'second_order_absent')
c3 = []
for dc in DIRS:
    for k in ('joint_gain', 'mquad_gain'):
        c3.append(abs(sec_r[dc][k]
                      - pa['second'][dc][k])
                  < 1e-9)
lg = pa['long_gate']
c3.append(abs(lg['trail3_min'] - t3_min)
          < 1e-12)
c3.append(abs(lg['trail6_min'] - t6_min)
          < 1e-9)
c3.append(abs(lg['delta']
              - (t6_min - t3_min)) < 1e-9)
c3.append(lg['verdict'] == a_long_r)
c3.append(pa['g_struct']['corr'] == verdict
          or abs(pa['g_struct']['corr']
                 - g_corr_r) < 1e-9)
c3.append(pa['g_struct']['verdict']
          == a_share_r)
for dc in DIRS:
    le_r = pa['g_struct']['lag_energy'][dc]
    le_c = [float((G6r[dc][:, d] ** 2).sum())
            for d in range(TRAIL_D)]
    c3.append(all(abs(a - b) < 1e-9
                  for a, b in zip(le_c, le_r)))
c3.append(pa['second_verdict']
          == a_second_r)
c3.append(a_long_r in verdict)
c3.append(a_sig_r in verdict)
c3.append(a_share_r in verdict)
c3.append(a_second_r in verdict)
sec('B3_partA_gates', c3)


# ---------- B4. Part-B ----------
def auc_mw(pos_vals, neg_vals):
    x = np.concatenate([pos_vals, neg_vals])
    n1 = len(pos_vals)
    n2 = len(neg_vals)
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x),
                     dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) \
                and sx[j + 1] == sx[i]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    r1 = ranks[:n1].sum()
    return float((r1 - n1 * (n1 + 1) / 2.0)
                 / (n1 * n2))


auc_r = auc_mw(
    zn['mlg_s0_P'][:, NLG, -1]
    .astype(np.float64),
    zn['mlg_s0_A1'][:, NLG, -1]
    .astype(np.float64))
E_r = {}
n_valid = {}
c4 = []
for dc in DIRS:
    sidx = zn['span_idx_%s' % dc]
    valid = sidx[:, 0] >= 0
    n_valid[dc] = int(valid.sum())
    s0v = zn['mlg_s0_%s' % dc][valid] \
        .astype(np.float64)
    d1 = zn['mlg_s1_%s' % dc][valid] \
        .astype(np.float64) - s0v
    d2 = zn['mlg_s2_%s' % dc][valid] \
        .astype(np.float64) - s0v
    d3 = zn['mlg_s3_%s' % dc][valid] \
        .astype(np.float64) - s0v
    E_r[dc] = {
        'syn': (d1.mean(2) - d2.mean(2))
        .mean(0),
        'cont': (d1.mean(2) - d3.mean(2))
        .mean(0)}
    for nm in ('syn', 'cont'):
        e_npz = zn['E_%s_%s' % (nm, dc)] \
            .astype(np.float64)
        e_rec = E_r[dc][nm]
        c4.append(float(np.abs(e_rec
                               - e_npz).max())
                  < 1e-4)
        c4.append(float(np.abs(
            e_rec - np.array(
                pb['curves']['E_%s_%s'
                             % (nm, dc)]))
            .max()) < 1e-4)
        c4.append(abs(float(e_rec[NLG])
                      - float(e_npz[NLG]))
                  < 1e-4)
c4.append(n_valid['P']
          == pb['n_span']['P'])
c4.append(n_valid['A1']
          == pb['n_span']['A1'])
c4.append(abs(auc_r - pb['readout_auc'])
          < 1e-4)
c4.append(pb['readout_auc'] > 0.6)
# lstar + sign recompute
lstar_r = {}
sign_r = {}
for dc in DIRS:
    lstar_r[dc] = {}
    sign_r[dc] = {}
    for nm, cvv in (('syn',
                     E_r[dc]['syn']),
                    ('cont',
                     E_r[dc]['cont'])):
        eff = abs(float(cvv[NLG]))
        lr = None
        la = None
        for L in range(20, NLG + 1):
            if lr is None and eff > 0 \
                    and cvv[L] <= -0.3 * eff:
                lr = L
            th = 0.05 if dc == 'P' else 0.025
            if la is None \
                    and cvv[L] <= -th:
                la = L
        lstar_r[dc][nm] = {'rel': lr,
                           'abs': la}
        sign_r[dc][nm] = (
            'positive' if cvv[NLG] > 0
            else 'negative')
        c4.append(lstar_r[dc][nm]['rel']
                  == pb['lstar'][dc][nm]['rel'])
        c4.append(lstar_r[dc][nm]['abs']
                  == pb['lstar'][dc][nm]['abs'])
        c4.append(sign_r[dc][nm]
                  == pb['sign'][dc][nm])
# vs-3124 corr recompute
corr_r = {}
for dc in DIRS:
    for nm in ('syn', 'cont'):
        c_new = E_r[dc][nm]
        c_old = z24['E_%s_%s' % (nm, dc)] \
            .astype(np.float64)
        corr_r['%s_%s' % (nm, dc)] = float(
            np.corrcoef(c_new, c_old)[0, 1])
a1_min_r = min(corr_r['syn_A1'],
               corr_r['cont_A1'])
b_base_r = ('baseline_replicated'
            if a1_min_r >= 0.9
            else 'baseline_diverged')
for k in ('syn_P', 'syn_A1', 'cont_P',
          'cont_A1'):
    c4.append(abs(corr_r[k]
                  - pb['vs_3124']['corr'][k])
              < 1e-4)
c4.append(abs(a1_min_r
              - pb['vs_3124']['a1_min_corr'])
          < 1e-4)
c4.append(pb['vs_3124']['verdict']
          == b_base_r)
c4.append(b_base_r in verdict)
c4.append(pb['path']['verdict']
          == 'path_valid')
c4.append(pb['path']['r'] > 0.9999)
c4.append(pb['repro']['verdict']
          in ('replay_bit_exact',
              'replay_ok'))
c4.append(pb['spans_verdict']
          in ('spans_ok', 'spans_sparse'))
sg24_r = {}
for dc in DIRS:
    for nm in ('syn', 'cont'):
        c_old = z24['E_%s_%s'
                    % (nm, dc)] \
            .astype(np.float64)
        sg24_r['%s_%s' % (nm, dc)] = (
            'positive' if c_old[NLG] > 0
            else 'negative')
c4.append(pb['vs_3124']['sign_3124']
          == sg24_r)
sec('B4_partB_readout', c4)

# ---------- B5. wspec ----------
c5 = []
wd_r = {}
for dc in DIRS:
    s0m = zn['mlg_s0_%s' % dc] \
        .astype(np.float64)
    D_l = s0m[:, 1:, :] - s0m[:, :-1, :]
    ws = D_l.mean(0)
    w_npz = zn['wspec_%s' % dc] \
        .astype(np.float64)
    c5.append(float(np.abs(ws - w_npz).max())
              < 1e-4)
    W = ws.mean(1)
    W0 = ws[:, 0]
    aw = np.abs(W)
    top3 = np.argsort(aw)[::-1][:3]
    c3v = float(aw[top3].sum()
                / max(aw.sum(), 1e-12))
    pos_b = [int(L) for L in range(NLG)
             if W[L] >= 0.3]
    neg_b = [int(L) for L in range(NLG)
             if W[L] <= -0.3]
    peak = int(np.argmax(aw))
    wd_r[dc] = {'c3': c3v, 'top3': top3,
                'peak': peak}
    lay = pd['layers'][dc]
    c5.append(float(np.abs(
        W - np.array(lay['W_mean'])).max())
        < 1e-4)
    c5.append(float(np.abs(
        W0 - np.array(lay['W_ans0'])).max())
        < 1e-4)
    c5.append([int(v) for v in top3]
              == lay['top3_layers'])
    c5.append(abs(c3v - lay['c3']) < 1e-4)
    c5.append(pos_b
              == lay['pos_band_ge_0.3'])
    c5.append(neg_b
              == lay['neg_band_le_-0.3'])
    c5.append(peak == lay['peak_layer'])
c3_min_r = min(wd_r['P']['c3'],
               wd_r['A1']['c3'])
d_chain_r = ('writechain_located'
             if c3_min_r >= 0.4
             else 'writechain_diffuse')
c5.append(abs(c3_min_r - pd['c3_min'])
          < 1e-4)
c5.append(pd['verdict'] == d_chain_r)
c5.append(d_chain_r in verdict)
sec('B5_wspec', c5)

# ---------- C. Part-C recompute ---
c6 = []
for dc in DIRS:
    gb = zn['gen_base_%s' % dc]
    st = pc['stats'][dc]
    for c in SCOND:
        rg = zn['regen_%s_%s' % (c, dc)]
        ag_l = []
        fd_l = []
        for j in range(96):
            base12 = gb[j]
            g12 = rg[j]
            ag_l.append(float(np.mean(
                [a == b for a, b
                 in zip(g12, base12)])))
            fd = N_NEW
            for k in range(N_NEW):
                if int(g12[k]) \
                        != int(base12[k]):
                    fd = k
                    break
            fd_l.append(float(fd))
        c6.append(abs(float(np.mean(ag_l))
                      - st['agree_mean'][c])
                  < 1e-9)
        c6.append(abs(float(np.mean(fd_l))
                      - st['first_div_mean'][c])
                  < 1e-9)
        fr = st['flip_rate'][c]
        na = st['n_ans'][c]
        c6.append(fr is None or 0.0 <= fr <= 1.0)
        if na > 0 and fr is not None:
            c6.append(abs(fr * na
                          - round(fr * na))
                      < 1e-6)
        else:
            c6.append(fr is None)
    c6.append(st['n_span_sub'] <= 96)
sm_r = min(
    pc['stats'][dc]['agree_mean']['s0']
    - float(np.mean([
        pc['stats'][dc]['agree_mean'][c]
        for c in ('s1', 's2', 's3')]))
    for dc in DIRS)
c_shift_r = ('behavior_shift_present'
             if sm_r >= 0.10
             else 'behavior_shift_absent')
c6.append(abs(sm_r - pc['shift_min'])
          < 1e-12)
c6.append(pc['verdict'] == c_shift_r)
c6.append(c_shift_r in verdict)
sec('C_partC_regen', c6)

# ---------- C2. verdict ------
vtoks = verdict.split('|')
sec('C2_verdict_full', [
    len(vtoks) == 19,
    vtoks[0] == a_long_r,
    vtoks[1] == a_sig_r,
    vtoks[2] == a_share_r,
    vtoks[3] == a_second_r,
    vtoks[4] == 'path_valid',
    vtoks[7] in ('spans_ok',
                 'spans_sparse'),
    vtoks[8] == b_base_r,
    vtoks[17] == c_shift_r,
    vtoks[18] == d_chain_r,
    vtoks[9].startswith('qis_'),
    vtoks[13].startswith('qis_')])

# ---------- D. ledger ----------
led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
m6 = [m for m in led['measurements']
      if m.get('phase') == 3126]
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][
    0]
stored_sha8 = led.get('ledger_sha256_8')
led.pop('ledger_sha256_8', None)
blob = json.dumps(led, sort_keys=True,
                  ensure_ascii=False)
sha8 = hashlib.sha256(
    blob.encode('utf-8')).hexdigest()[:8]
sec('D_ledger', [
    len(led['measurements']) == 263,
    len(m6) == 1,
    m6 and m6[0]['meas_id']
    == 'meas3126_omega_p124_glm4_'
    'anchoredlast_regen_writechain',
    m6 and m6[0]['verdict'] == verdict,
    m6 and m6[0]['claim'].find(
        'Omega-P124') >= 0,
    len(l14['connects']) == 231,
    stored_sha8 is not None,
    sha8 == stored_sha8])

# ---------- E. MEMO ----------
memo = io.open(MEMO, encoding='utf-8').read()
cnt = memo.count('## Phase 3126:')
tail = memo[memo.index('## Phase 3126:'):] \
    if cnt else ''
mline = [l for l in tail.splitlines()
         if l.startswith('## Phase 3126:')][0] \
    if cnt else ''
has_ts = bool(re.search(
    r'\[2026-\d{2}-\d{2} \d{2}:\d{2}\]',
    mline)) if cnt else False
sec('E_memo', [
    cnt == 1,
    has_ts,
    sum(s in tail
        for s in verdict.split('|')) >= 4,
    u'\u03a9-P124' in tail,
    'anchored-last' in tail,
    'gMASK' in tail,
    '3127' in tail,
    'GLM4' in tail or 'glm4' in tail,
    V25.split('|')[1] in tail,
    'trail6' in tail or 'TRAIL_D=6' in tail,
    'baseline_replicated' in tail
    or 'baseline_diverged' in tail,
    'behavior_shift' in tail])

# ---------- F. wlogs ----------
cf = []
wlog_sha_ok = []
for wdir in (WLOG_D, WLOG_C):
    t = ''
    for dstr in ('2026-09-24.md',
                 '2026-09-25.md'):
        pth = os.path.join(wdir, dstr)
        if os.path.exists(pth):
            t = io.open(pth,
                        encoding='utf-8').read()
            break
    cf += ['Phase 3126' in t,
           'closeout' in t.lower()]
    wlog_sha_ok.append(sha8 in t)
sec('F_wlogs', cf + wlog_sha_ok)

# ---------- G. MEMORY ----------
mem = io.open(MEMO_W, encoding='utf-8').read()
sec('G_memory', [
    len(mem) < 3000,
    len(mem) >= 2000,
    mem.count(u'\u673a\u5236\u94fe\u72b6'
              u'\u6001\uff083126\uff09') == 1,
    mem.count(u'- 3126\uff08T4\uff09') == 1,
    mem.count('max=3126') == 1,
    '3126' in mem])

# ---------- report ----------
o.append('TOTAL FAILS: %d' % len(fails))
o.append('sha8=%s' % sha8)
io.open(VF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('verify done, fails=%d' % len(fails))
