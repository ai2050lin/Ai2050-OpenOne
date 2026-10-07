# -*- coding: utf-8 -*-
"""Phase 3124 disk verify (independent):
A load+result asserts -> B deterministic
recompute from frozen npz (refit/decomp/det/
kalman/own/l35/glm4 curves+lstar+readout)
-> C seal -> D ledger -> E MEMO -> F wlogs
-> G MEMORY. No Monte-Carlo in this phase
(no_mc): all recomputes are deterministic."""
import hashlib
import io
import json
import math
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
D22 = RDIR + r'\phase3122' \
      r'\omega_p120_write_content_readout_' \
      'sentence_causal_dist_recon'
D23 = RDIR + r'\phase3123' \
      r'\omega_p121_dirfit_anchor_l35loc_' \
      'syntax_trace'
OUTD = RDIR + r'\phase3124' \
       r'\omega_p122_resid_cm_kalman_l35rel_' \
       'glm4x'
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
      r'\p3124_verify_out.txt')
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


V = ('common_mode_partial|det_core_failed|'
     'r2_weak|lag1_negative|anchor_static|'
     'own_anchor_insufficient|'
     'semantic_component_present|path_valid|'
     'replay_bit_exact|readout_ok|spans_ok|'
     'glm_syntax_L20|glm_syntax_L20|'
     'glm_content_L20|glm_content_L20')
NP = 672
N_NEW = 12

# ---------- A. load ----------
res = json.load(io.open(
    os.path.join(OUTD, 'result.json'),
    encoding='utf-8'))
seal = json.load(io.open(
    os.path.join(OUTD, 'design_seal.json'),
    encoding='utf-8'))
r23 = json.load(io.open(
    os.path.join(D23, 'result.json'),
    encoding='utf-8'))
z18 = np.load(os.path.join(
    D18, 'traj_readout.npz'),
    allow_pickle=False)
z20 = np.load(os.path.join(
    D20, 'p118_readout.npz'),
    allow_pickle=False)
z22 = np.load(os.path.join(
    D22, 'p120_readout.npz'),
    allow_pickle=False)
z23 = np.load(os.path.join(
    D23, 'p121_readout.npz'),
    allow_pickle=False)
z24 = np.load(os.path.join(
    OUTD, 'p122_readout.npz'),
    allow_pickle=False)
sec('A0_load', [res['verdict'] == V,
                res['smoke'] is False,
                res['n_pairs'] == NP,
                res['np_d'] == 672,
                abs(res['runtime_s']
                    - 13475.6) < 0.05,
                seal['phase'] == 3124,
                seal['smoke'] is False,
                'no_mc' in seal['part_d']])

pa = res['part_a']
chk = [abs(pa['refit']['P']['S']
           - (-0.601611613690753)) < 1e-12,
       abs(pa['refit']['P']['MS']
           - (-4.955594255793292)) < 1e-12,
       abs(pa['refit']['A1']['S']
           - (-0.5540798866844862)) < 1e-12,
       abs(pa['refit']['A1']['MS']
           - (-7.154421990932929)) < 1e-12]
dcP = pa['decomp']['P']
dcA = pa['decomp']['A1']
chk += [abs(dcP['cm_share']
            - 0.19001305759179912) < 1e-12,
        abs(dcP['m_share']
            - 0.0033137063011249046) < 1e-12,
        abs(dcP['rem_share']
            - 0.806673236107076) < 1e-12,
        abs(dcP['ss_tot']
            - 69737.75834882268) < 1e-9,
        dcP['leak'] == 0.0,
        abs(dcA['cm_share']
            - 0.35574441732458434) < 1e-12,
        abs(dcA['m_share']
            - 3.0830390620402785e-06) < 1e-12,
        abs(dcA['rem_share']
            - 0.6442524996363537) < 1e-12,
        abs(dcA['ss_tot']
            - 103458.4879387006) < 1e-9,
        abs(pa['cm_min']
            - 0.19001305759179912) < 1e-12,
        pa['cm_verdict']
        == 'common_mode_partial']
ds = pa['det_sim']
chk += [abs(ds['r_det']
            - (-0.12297439326221062)) < 1e-12,
        ds['verdict'] == 'det_core_failed',
        abs(ds['r2_median']['P']
            - 0.32964950758136624) < 1e-12,
        abs(ds['r2_median']['A1']
            - (-0.015582100480399985)) < 1e-12,
        ds['r2_verdict'] == 'r2_weak',
        abs(ds['auc_det'][0]
            - 0.9809094210600907) < 1e-12,
        abs(ds['auc_det'][4]
            - 0.739937641723356) < 1e-12,
        abs(ds['auc_det'][12]
            - 0.9714405293367347) < 1e-12]
an = pa['analytic']
chk += [abs(an['sig_P']
            - 3.206171299571775) < 1e-12,
        abs(an['sig_A1']
            - 4.001745467245389) < 1e-12,
        abs(an['gap']
            - 2.198827735139629) < 1e-12,
        abs(an['phi']
            - 0.6659700011599246) < 1e-12,
        abs(an['phi_3123_plateau']
            - 0.6635588242718963) < 1e-12]
pb = res['part_b']
chk += [abs(pb['lag1']['P']
            - (-0.1674141723780209)) < 1e-12,
        abs(pb['lag1']['A1']
            - (-0.006200686258676985)) < 1e-12,
        pb['lag1_verdict'] == 'lag1_negative',
        abs(pb['kalman']['P']['q_hat']
            - 0.00014353249592527267) < 1e-12,
        abs(pb['kalman']['P']['ratio']
            - 0.0001) < 1e-12,
        abs(pb['kalman']['P']['sig_a2']
            - 1.4353249592527266) < 1e-12,
        abs(pb['kalman']['A1']['q_hat']
            - 0.0001481438608183718) < 1e-12,
        abs(pb['kalman']['A1']['ratio']
            - 0.0001) < 1e-12,
        abs(pb['kalman']['A1']['sig_a2']
            - 1.4814386081837179) < 1e-12,
        pb['kalman_verdict'] == 'anchor_static',
        abs(pb['own_sim']['r_own']
            - 0.0062127328674929) < 1e-12,
        pb['own_sim']['verdict']
        == 'own_anchor_insufficient',
        abs(pb['own_sim']['r2_median']['P']
            - 0.4035387815911334) < 1e-12,
        abs(pb['own_sim']['r2_median']['A1']
            - 0.016867642371469482) < 1e-12,
        abs(pb['own_sim']['auc_own'][12]
            - 0.92605362457483) < 1e-12]
pc = res['part_c']
l35P = pc['l35rel']['P']
l35A = pc['l35rel']['A1']
chk += [abs(l35P['W_ans']
            - (-11.373681399084273)) < 1e-9,
        abs(l35P['W_oth']
            - (-11.479271332374513)) < 1e-9,
        abs(l35P['norm_share']
            - 0.5009523266638253) < 1e-12,
        abs(l35P['t_norm']
            - 2.6518068179125405) < 1e-12,
        abs(l35P['t_sem']
            - (-2.641724475918722)) < 1e-12,
        abs(l35P['leak']
            - (-0.11567227528405821)) < 1e-12,
        abs(l35A['W_ans']
            - (-8.029358918468157)) < 1e-9,
        abs(l35A['W_oth']
            - (-11.716023877190498)) < 1e-9,
        abs(l35A['norm_share']
            - 0.4717725310807821) < 1e-12,
        abs(l35A['t_norm']
            - (-1.7147589040668996)) < 1e-12,
        abs(l35A['t_sem']
            - (-1.919956538433672)) < 1e-12,
        abs(l35A['leak']
            - (-0.051949516221769354)) < 1e-12,
        pc['verdict']
        == 'semantic_component_present']
pd_ = res['part_d']
chk += [pd_['interference']
        == 'input_prompt_span_equal_len',
        abs(pd_['path']['r']
            - 0.9999857905405052) < 1e-12,
        pd_['path']['verdict'] == 'path_valid',
        pd_['repro']['max_diff'] == 0.0,
        pd_['repro']['verdict']
        == 'replay_bit_exact',
        abs(pd_['readout_auc']
            - 0.8313137755102041) < 1e-12,
        pd_['readout_verdict'] == 'readout_ok',
        pd_['n_span'] == {'P': 672, 'A1': 672},
        pd_['spans_verdict'] == 'spans_ok']
lsp = pd_['lstar']
chk += [lsp['P']['syn']
        == {'rel': 20, 'abs': 20},
        lsp['P']['cont']
        == {'rel': 20, 'abs': 20},
        lsp['A1']['syn']
        == {'rel': 20, 'abs': 20},
        lsp['A1']['cont']
        == {'rel': 20, 'abs': 20}]
cv = pd_['curves']
chk += [cv['E_syn_P'][0] == 0.0,
        abs(cv['E_syn_P'][9]
            - (-0.34420040916078365)) < 1e-12,
        abs(cv['E_syn_P'][20]
            - (-0.12357999268414346)) < 1e-12,
        abs(cv['E_syn_P'][40]
            - 0.0593891150570533) < 1e-12,
        abs(cv['E_syn_A1'][9]
            - (-0.31807351703226805)) < 1e-12,
        abs(cv['E_syn_A1'][40]
            - (-0.1140573490683959)) < 1e-12,
        abs(cv['E_cont_P'][9]
            - (-0.2808397461834192)) < 1e-12,
        abs(cv['E_cont_P'][20]
            - (-0.3004961357086951)) < 1e-12,
        abs(cv['E_cont_P'][26]
            - 0.7840392479669615) < 1e-12,
        abs(cv['E_cont_P'][40]
            - 0.3408039071573277) < 1e-12,
        abs(cv['E_cont_A1'][20]
            - (-0.2806829876836132)) < 1e-12,
        abs(cv['E_cont_A1'][26]
            - 0.8675801791301875) < 1e-12,
        abs(cv['E_cont_A1'][40]
            - 0.08595561432300791) < 1e-12]
ft = pd_['first_tokens']
chk += [ft['P'][0] == [9450, 670, 'Yes'],
        ft['A1'][0] == [2753, 555, 'No']]
sec('A_result', chk)

# ---------- B1. refit ----------
mP18 = z18['gt_cleanseq_clean__P']
mA18 = z18['gt_cleanseq_clean__A1']
MSEQ = {'P': mP18[:NP].astype(np.float64),
        'A1': mA18[:NP].astype(np.float64)}
ANN = {'P': z20['annP_c'][:NP],
       'A1': z20['annA_c'][:NP]}
auc18 = z18['auc_curve']


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


content = {}
for dc in ('P', 'A1'):
    dm = MSEQ[dc][:, 1:] - MSEQ[dc][:, :-1]
    tab = np.zeros((10, N_NEW),
                   dtype=np.float64)
    for k in range(N_NEW):
        col = ANN[dc][:, k].astype(np.int64)
        dv = dm[:, k]
        allm = dv.mean()
        for cc in range(10):
            sel = (col == cc)
            if sel.any():
                tab[cc, k] = float(
                    dv[sel].mean() - allm)
    content[dc] = tab

fit = {}
for dc in ('P', 'A1'):
    m = MSEQ[dc]
    dm = m[:, 1:] - m[:, :-1]
    col = ANN[dc].astype(np.int64)
    dev = content[dc][col, np.arange(N_NEW)]
    dm_res = dm - dev
    X = np.stack([m[:, :-1].ravel(),
                  np.ones(m[:, :-1].size)],
                 1)
    y = dm_res.ravel()
    bb, *_ = np.linalg.lstsq(X, y,
                             rcond=None)
    S_dc = float(bb[0])
    MS_dc = float(-bb[1] / bb[0])
    resid = (y - X @ bb).reshape(NP, N_NEW)
    a_full = m[:, :-1] - dm_res / S_dc
    a_i = a_full.mean(1)
    fit[dc] = {'S': S_dc, 'MS': MS_dc,
               'resid': resid,
               'a_full': a_full, 'a': a_i,
               'col': col, 'dm_res': dm_res}
r23a = r23['part_a']['dirfit']
c1 = [abs(fit['P']['S'] - r23a['P']['S'])
      < 1e-9,
      abs(fit['P']['MS'] - r23a['P']['MS'])
      < 1e-9,
      abs(fit['A1']['S'] - r23a['A1']['S'])
      < 1e-9,
      abs(fit['A1']['MS'] - r23a['A1']['MS'])
      < 1e-9,
      abs(fit['P']['S'] - pa['refit']['P']['S'])
      < 1e-12,
      abs(fit['A1']['S']
          - pa['refit']['A1']['S']) < 1e-12,
      np.max(np.abs(
          fit['P']['a']
          - z23['anchors_P'].astype(
              np.float64))) < 1e-5,
      np.max(np.abs(
          fit['A1']['a']
          - z23['anchors_A1'].astype(
              np.float64))) < 1e-5,
      np.max(np.abs(
          fit['P']['a']
          - z24['anchors_P'].astype(
              np.float64))) < 1e-5,
      np.max(np.abs(
          fit['A1']['a']
          - z24['anchors_A1'].astype(
              np.float64))) < 1e-5]
sec('B1_refit_anchors', c1)

# ---------- B2. decomp + det + analytic ----------
c2 = []
for dc in ('P', 'A1'):
    R = fit[dc]['resid'].ravel()
    ss_tot = float((R * R).sum())
    NIK = R.size
    Z1 = np.zeros((NIK, N_NEW))
    Z1[np.arange(NIK),
       np.arange(NIK) % N_NEW] = 1.0
    b1, *_ = np.linalg.lstsq(Z1, R,
                             rcond=None)
    r1 = R - Z1 @ b1
    ss_step = ss_tot - float((r1 * r1)
                             .sum())
    x = (MSEQ[dc][:, :-1]
         - fit[dc]['MS']).ravel()
    Z2 = np.stack([x, np.ones(NIK)], 1)
    b2, *_ = np.linalg.lstsq(Z2, r1,
                             rcond=None)
    r2 = r1 - Z2 @ b2
    ss_m = float((r1 * r1).sum()) \
        - float((r2 * r2).sum())
    ss_rem = float((r2 * r2).sum())
    dec = pa['decomp'][dc]
    c2 += [abs(ss_step / ss_tot
               - dec['cm_share']) < 1e-12,
           abs(ss_m / ss_tot
               - dec['m_share']) < 1e-12,
           abs(ss_rem / ss_tot
               - dec['rem_share']) < 1e-12,
           abs(ss_tot - dec['ss_tot'])
           < 1e-9]
auc_det = np.zeros(N_NEW + 1)
det_arr = {}
r2_med = {}
for dc in ('P', 'A1'):
    m = MSEQ[dc]
    col = fit[dc]['col']
    S_dc = fit[dc]['S']
    MS_dc = fit[dc]['MS']
    mh = m[:, 0].copy()
    snap = [mh.copy()]
    for k in range(N_NEW):
        mh = mh + S_dc * (mh - MS_dc) \
            + content[dc][col[:, k], k]
        snap.append(mh.copy())
    SArr = np.stack(snap, 1)
    det_arr[dc] = SArr
    denom = ((m - m.mean(1, keepdims=True))
             ** 2).sum(1)
    r2i = 1.0 - ((SArr - m) ** 2).sum(1) \
        / np.maximum(denom, 1e-12)
    r2_med[dc] = float(np.median(r2i))
for t in range(N_NEW + 1):
    auc_det[t] = auc_mw(det_arr['P'][:, t],
                        det_arr['A1'][:, t])
r_det = float(np.corrcoef(auc_det, auc18)
              [0, 1])
md_det = float(np.max(np.abs(
    auc_det - z24['auc_det'])))
c2 += [md_det < 1e-10,
       abs(r_det - ds['r_det']) < 1e-12,
       abs(r2_med['P']
           - ds['r2_median']['P']) < 1e-12,
       abs(r2_med['A1']
           - ds['r2_median']['A1']) < 1e-12]
sigP = (float(np.std(fit['P']['resid']))
        / math.sqrt(1.0 - (1.0
                           + fit['P']['S'])
                    ** 2))
sigA1 = (float(np.std(fit['A1']['resid']))
         / math.sqrt(1.0 - (1.0
                            + fit['A1']['S'])
                     ** 2))
gap = (float(fit['P']['a'].mean())
       - float(fit['A1']['a'].mean()))
phi = 0.5 * (1.0 + math.erf(
    (gap / math.sqrt(sigP ** 2
                     + sigA1 ** 2))
    / math.sqrt(2.0)))
c2 += [abs(sigP - an['sig_P']) < 1e-12,
       abs(sigA1 - an['sig_A1']) < 1e-12,
       abs(gap - an['gap']) < 1e-12,
       abs(phi - an['phi']) < 1e-12,
       abs(phi - float(z23['auc_sim_dir'][12]))
       < 0.03]
sec('B2_decomp_det_analytic', c2)

# ---------- B3. kalman + own ----------
c3 = []
own_s_store = {}
own_r2_store = {}
for dc in ('P', 'A1'):
    af = fit[dc]['a_full']
    afc = af - af.mean(1, keepdims=True)
    num = (afc[:, :-1]
           * afc[:, 1:]).sum(1)
    den = np.sqrt(
        (afc[:, :-1] ** 2).sum(1)
        * (afc[:, 1:] ** 2).sum(1))
    lag1_dc = float(
        (num / np.maximum(den, 1e-12))
        .mean())
    c3.append(abs(lag1_dc
                  - pb['lag1'][dc]) < 1e-12)
    m = MSEQ[dc]
    S_dc = fit[dc]['S']
    MS_dc = fit[dc]['MS']
    dm_res = fit[dc]['dm_res']
    sig_e2 = float(
        np.std(fit[dc]['resid'])) ** 2
    sig_a2 = float(fit[dc]['a'].var())
    z = dm_res - S_dc * m[:, :-1]
    q_grid = sig_a2 * np.logspace(-4, 1, 22)
    lls = np.zeros(len(q_grid))
    for qi, q in enumerate(q_grid):
        a = np.full(NP, MS_dc)
        P = np.full(NP, sig_a2)
        ll = 0.0
        for t in range(N_NEW):
            Pm = P + q
            v = S_dc * S_dc * Pm + sig_e2
            innov = z[:, t] + S_dc * a
            ll += -0.5 * float(
                (np.log(2.0 * math.pi * v)
                 + innov * innov / v).sum())
            K = -S_dc * Pm / v
            a = a + K * innov
            P = (1.0 - S_dc * S_dc * Pm / v) \
                * Pm
        lls[qi] = ll
    q_hat = float(q_grid[
        int(np.argmax(lls))])
    kl = pb['kalman'][dc]
    c3 += [abs(q_hat - kl['q_hat']) < 1e-12,
           abs(q_hat / sig_a2
               - kl['ratio']) < 1e-12,
           abs(sig_a2 - kl['sig_a2'])
           < 1e-12]
    # RTS smoother at q_hat
    a = np.full(NP, MS_dc)
    P = np.full(NP, sig_a2)
    af_st = np.zeros((NP, N_NEW))
    Pf_st = np.zeros((NP, N_NEW))
    for t in range(N_NEW):
        Pm = P + q_hat
        v = S_dc * S_dc * Pm + sig_e2
        innov = z[:, t] + S_dc * a
        K = -S_dc * Pm / v
        a = a + K * innov
        P = (1.0 - S_dc * S_dc * Pm / v) \
            * Pm
        af_st[:, t] = a
        Pf_st[:, t] = P
    a_s = np.zeros((NP, N_NEW))
    a_s[:, N_NEW - 1] = af_st[:, N_NEW - 1]
    for t in range(N_NEW - 2, -1, -1):
        Pp_next = Pf_st[:, t] + q_hat
        C = Pf_st[:, t] \
            / np.maximum(Pp_next, 1e-12)
        a_s[:, t] = af_st[:, t] + C * (
            a_s[:, t + 1] - af_st[:, t])
    md_rts = float(np.max(np.abs(
        a_s - z24['rts_%s' % dc].astype(
            np.float64))))
    c3.append(md_rts < 1e-4)
    # own deterministic sim
    col = fit[dc]['col']
    mh = m[:, 0].copy()
    snap = [mh.copy()]
    for k in range(N_NEW):
        mh = mh + S_dc * (mh - a_s[:, k]) \
            + content[dc][col[:, k], k]
        snap.append(mh.copy())
    SArr = np.stack(snap, 1)
    denom = ((m - m.mean(1, keepdims=True))
             ** 2).sum(1)
    r2i = 1.0 - ((SArr - m) ** 2).sum(1) \
        / np.maximum(denom, 1e-12)
    own_s_store[dc] = SArr
    own_r2_store[dc] = float(
        np.median(r2i))
auc_own = np.zeros(N_NEW + 1)
for t in range(N_NEW + 1):
    auc_own[t] = auc_mw(
        own_s_store['P'][:, t],
        own_s_store['A1'][:, t])
r_own = float(np.corrcoef(auc_own, auc18)
              [0, 1])
md_own = float(np.max(np.abs(
    auc_own - z24['auc_own'])))
c3 += [md_own < 1e-10,
       abs(r_own - pb['own_sim']['r_own'])
       < 1e-12,
       abs(own_r2_store['P']
           - pb['own_sim']['r2_median']['P'])
       < 1e-12,
       abs(own_r2_store['A1']
           - pb['own_sim']['r2_median']['A1'])
       < 1e-12]
sec('B3_kalman_rts_own', c3)

# ---------- B4. L35 decomposition ----------
c4 = []
for dc in ('P', 'A1'):
    w = z22['wrec_pd_%s' % dc].astype(
        np.float64)[35][:, :N_NEW]
    p = z22['wrec_pn_%s' % dc].astype(
        np.float64)[35][:, :N_NEW]
    ok = np.abs(p) > 1e-9
    wn = np.where(ok,
                  w / np.where(ok, p, 1.0),
                  np.nan)
    ann = ANN[dc]
    am = (ann == 9) & ok
    om = (~am) & ok
    W_ans = float(np.nanmean(w[am]))
    W_oth = float(np.nanmean(w[om]))
    Pn_ans = float(np.nanmean(p[am]))
    Pn_oth = float(np.nanmean(p[om]))
    Wn_ans = float(np.nanmean(wn[am]))
    Wn_oth = float(np.nanmean(wn[om]))
    t_norm = Pn_oth * (Wn_oth - Wn_ans)
    t_sem = Wn_ans * (Pn_oth - Pn_ans)
    leak = (W_oth - W_ans) - t_norm - t_sem
    denom2 = abs(t_norm) + abs(t_sem)
    nsh = (abs(t_norm) / denom2
           if denom2 > 0 else None)
    l35 = pc['l35rel'][dc]
    c4 += [abs(W_ans - l35['W_ans']) < 1e-9,
           abs(W_oth - l35['W_oth']) < 1e-9,
           abs(Pn_ans - l35['Pn_ans'])
           < 1e-9,
           abs(Pn_oth - l35['Pn_oth'])
           < 1e-9,
           abs(Wn_ans - l35['Wn_ans'])
           < 1e-9,
           abs(Wn_oth - l35['Wn_oth'])
           < 1e-9,
           abs(t_norm - l35['t_norm'])
           < 1e-9,
           abs(t_sem - l35['t_sem']) < 1e-9,
           abs(leak - l35['leak']) < 1e-9,
           abs(nsh - l35['norm_share'])
           < 1e-12]
sec('B4_l35_decomp', c4)

# ---------- B5. GLM4 npz cross-check ----------
c5 = []
for dc in ('P', 'A1'):
    for nm in ('syn', 'cont'):
        e_npz = z24['E_%s_%s' % (nm, dc)] \
            .astype(np.float64)
        e_res = np.array(
            cv['E_%s_%s' % (nm, dc)])
        c5.append(np.max(np.abs(
            e_npz - e_res)) < 1e-4)
auc_fin = auc_mw(
    z24['mlg_s0_P'][:, 40, -1].astype(
        np.float64),
    z24['mlg_s0_A1'][:, 40, -1].astype(
        np.float64))
c5.append(abs(auc_fin
              - pd_['readout_auc']) < 1e-4)
# lstar recompute from npz E curves
for dc in ('P', 'A1'):
    for nm in ('syn', 'cont'):
        cvv = z24['E_%s_%s' % (nm, dc)] \
            .astype(np.float64)
        eff = abs(float(cvv[40]))
        lr = None
        la = None
        for L in range(20, 41):
            if lr is None and eff > 0 \
                    and cvv[L] <= -0.3 * eff:
                lr = L
            th = 0.05 if dc == 'P' else 0.025
            if la is None \
                    and cvv[L] <= -th:
                la = L
        c5 += [lr == 20, la == 20]
md_ad = float(np.max(np.abs(
    z24['auc_det']
    - np.array(ds['auc_det']))))
md_ao = float(np.max(np.abs(
    z24['auc_own']
    - np.array(pb['own_sim']['auc_own']))))
c5 += [md_ad < 1e-12, md_ao < 1e-12,
       z24['mlg_s0_P'].shape == (672, 41, 13),
       z24['mlg_s3_A1'].shape == (672, 41, 13)]
sec('B5_glm4_npz', c5)

# ---------- C. seal ----------
c6 = [seal['part_a']['cm_gate'].find('0.2') >= 0,
      seal['part_a']['det_sim'].find(
          'no-noise') >= 0,
      seal['part_b']['kalman'].find(
          'logspace(-4,1,22)') >= 0,
      seal['part_c']['l35rel'].find(
          'norm_share_A1') >= 0,
      seal['part_d']['gates']['D_path']
      .find('0.9999') >= 0,
      seal['part_d']['gates']['D_lstar_rel']
      .find('-0.3') >= 0,
      seal['created'].startswith('2026-')]
sec('C_seal', c6)

# ---------- D. ledger ----------
led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
m4 = [m for m in led['measurements']
      if m.get('phase') == 3124]
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][
    0]
led.pop('ledger_sha256_8', None)
blob = json.dumps(led, sort_keys=True,
                  ensure_ascii=False)
sha8 = hashlib.sha256(
    blob.encode('utf-8')).hexdigest()[:8]
sec('D_ledger',
    [len(led['measurements']) == 261,
     len(m4) == 1,
     len(l14['connects']) == 229,
     m4 and m4[0]['meas_id']
     == 'meas3124_omega_p122_resid_cm_kalman_'
     'l35rel_glm4x',
     m4 and m4[0]['verdict'] == V,
     m4 and m4[0]['claim'].find(
         'Omega-P122 (3124') >= 0,
     sha8 == 'a7c7d98f'])

# ---------- E. MEMO ----------
memo = io.open(MEMO, encoding='utf-8').read()
cnt = memo.count('## Phase 3124:')
tail = memo[memo.index('## Phase 3124:'):] \
    if cnt else ''
mline = [l for l in tail.splitlines()
         if l.startswith('## Phase 3124:')][0] \
    if cnt else ''
has_ts = bool(re.search(
    r'\[2026-\d{2}-\d{2} \d{2}:\d{2}\]',
    mline)) if cnt else False
f1 = u'\u6b8b\u5dee\u4e09\u6210\u5206\u5b9a' \
     u'\u6027'
f2 = u'\u7b2c\u4e09\u7cfb\u7edf\u6210\u5206'
f3 = 'semantic_component_present'
f4 = u'\u5199\u5165\u94fe\u4e0a\u6e38'
f5 = 'norm_share'
f6 = 'common_mode_partial'
sec('E_memo',
    [cnt == 1,
     has_ts,
     f1 in tail, f2 in tail, f3 in tail,
     f4 in tail, f5 in tail, f6 in tail,
     tail.rstrip().endswith(
         u'`tests/gpt5_temp/p3124_span_probe'
         u'.py`\u3002'),
     '3125' in tail])

# ---------- F. wlogs ----------
cf = []
for wdir in (WLOG_D, WLOG_C):
    t = io.open(wdir + '\\' + '2026-09-24.md',
                encoding='utf-8').read()
    cf += ['Phase 3124 Omega-P122' in t,
           'Phase 3124 closeout' in t,
           'a7c7d98f' in t]
sec('F_wlogs', cf)

# ---------- G. MEMORY ----------
mem = io.open(MEMO_W,
              encoding='utf-8').read()
sec('G_memory',
    [len(mem) == 2968,
     mem.count(u'\u673a\u5236\u94fe\u72b6'
               u'\u6001\uff083124\uff09') == 1,
     mem.count(u'- 3124\uff08T4\uff09') == 1,
     mem.count('max=3124') == 1,
     mem.count(u'- 3118\u20133120\uff1a') == 1,
     mem.count(u'- 3119\uff1a') == 0,
     mem.count(u'- 3113\uff1a') == 0,
     mem.count(u'- 3123\uff08T4\uff09') == 1])

# ---------- report ----------
o.append('TOTAL FAILS: %d' % len(fails))
io.open(VF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('verify done, fails=%d' % len(fails))
