# -*- coding: utf-8 -*-
"""Phase 3125 disk verify (independent):
A load+result asserts -> B1 refit recompute
from frozen npz -> B2 decomp3/trail kernel/
AR/r4 bit-compare vs p123_readout.npz +
sim3 recompute -> B3 permutation traj stats
recompute -> B4 part-B npz recompute (E
curves/readout/lstar/span_idx) -> C seal
-> D ledger -> E MEMO -> F wlogs -> G
MEMORY. All recomputes deterministic
(no_mc): rng only the frozen seed-3125
permutation."""
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
D24 = RDIR + r'\phase3124' \
      r'\omega_p122_resid_cm_kalman_l35rel_' \
      'glm4x'
OUTD = RDIR + r'\phase3125' \
       r'\omega_p123_third_comp_qwen_inputstream'
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
      r'\p3125_verify_out.txt')
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


V = ('common_mode_partial|trail_present|'
     'ar_absent|trajectory_transient|'
     'third_partial|path_valid|'
     'replay_bit_exact|readout_ok|spans_ok|'
     'qis_syntax_L23|qis_syntax_L22|'
     'qis_content_L28|qis_content_L32|'
     'qis_syn_final_negative|'
     'qis_syn_final_negative|'
     'qis_cont_final_negative|'
     'qis_cont_final_negative')
NP = 672
N_NEW = 12
NL = 36
TRAIL_D = 3
N_PERM = 200
PERM_SEED = 3125

# ---------- A. load ----------
res = json.load(io.open(
    os.path.join(OUTD, 'result.json'),
    encoding='utf-8'))
seal = json.load(io.open(
    os.path.join(OUTD, 'design_seal.json'),
    encoding='utf-8'))
z18 = np.load(os.path.join(
    D18, 'traj_readout.npz'),
    allow_pickle=False)
z20 = np.load(os.path.join(
    D20, 'p118_readout.npz'),
    allow_pickle=False)
z24 = np.load(os.path.join(
    D24, 'p122_readout.npz'),
    allow_pickle=False)
z25 = np.load(os.path.join(
    OUTD, 'p123_readout.npz'),
    allow_pickle=False)
sec('A0_load', [res['verdict'] == V,
                res['smoke'] is False,
                res['n_pairs'] == NP,
                res['np_b'] == NP,
                abs(res['runtime_s']
                    - 245.9) < 0.05,
                res['phase'] == 3125,
                res['name'] ==
                'omega_p123_third_comp_'
                'qwen_inputstream',
                seal['phase'] == 3125,
                seal['smoke'] is False,
                seal['np_a'] == NP,
                seal['np_b'] == NP,
                'no_mc' in seal['part_a']])

pa = res['part_a']
chk = [abs(pa['refit']['P']['S']
           - (-0.601611613690753)) < 1e-12,
       abs(pa['refit']['P']['MS']
           - (-4.955594255793292)) < 1e-12,
       abs(pa['refit']['A1']['S']
           - (-0.5540798866844862)) < 1e-12,
       abs(pa['refit']['A1']['MS']
           - (-7.154421990932929)) < 1e-12]
d3P = pa['decomp3']['P']
d3A = pa['decomp3']['A1']
chk += [abs(d3P['ss_tot']
            - 69737.75834882268) < 1e-9,
        abs(d3P['cm_share']
            - 0.19001305759179912) < 1e-12,
        abs(d3P['m_share']
            - 0.0033137063011249046) < 1e-12,
        abs(d3P['trail_share']
            - 0.08599867672568678) < 1e-12,
        abs(d3P['ar_share']
            - 0.005767845108351762) < 1e-12,
        abs(d3P['rem_share']
            - 0.7149067142730373) < 1e-12,
        abs(d3P['leak']) < 1e-12,
        abs(d3A['ss_tot']
            - 103458.4879387006) < 1e-9,
        abs(d3A['cm_share']
            - 0.35574441732458434) < 1e-12,
        abs(d3A['m_share']
            - 3.0830390620402785e-06) < 1e-12,
        abs(d3A['trail_share']
            - 0.05022617239377846) < 1e-12,
        abs(d3A['ar_share']
            - 0.007261861528795945) < 1e-12,
        abs(d3A['rem_share']
            - 0.5867644657137793) < 1e-12,
        abs(d3A['leak']) < 1e-12,
        pa['cm_verdict']
        == 'common_mode_partial',
        pa['trail_verdict'] == 'trail_present',
        pa['ar_verdict'] == 'ar_absent',
        pa['traj_verdict']
        == 'trajectory_transient']
GP = pa['trail_G']['P']
GA = pa['trail_G']['A1']
chk += [abs(GP[1][0]
            - (-1.6769770017180752)) < 1e-12,
        abs(GP[0][1]
            - (-1.7570000874491265)) < 1e-12,
        abs(GP[4][2]
            - (-2.032119834120507)) < 1e-12,
        abs(GP[9][1]
            - 1.6959194067754657) < 1e-12,
        abs(GA[1][0]
            - (-0.4542977728738911)) < 1e-12,
        abs(GA[0][1]
            - (-1.728273987975595)) < 1e-12,
        abs(GA[4][2]
            - (-2.0623726211758497)) < 1e-12,
        abs(GA[9][1]
            - 0.44877744803404757) < 1e-12,
        all(GP[5]) is False,
        all(GA[3]) is False]
arP = pa['ar_params']['P']
arA = pa['ar_params']['A1']
chk += [abs(arP['phi1']
            - (-0.09355245050878877)) < 1e-12,
        abs(arP['phi2']
            - (-0.03499462595239264)) < 1e-12,
        abs(arP['c']
            - (-0.0072109727195960125))
        < 1e-12,
        abs(arA['phi1']
            - (-0.10965806973954667)) < 1e-12,
        abs(arA['phi2']
            - (-0.0773445705673305)) < 1e-12,
        abs(arA['c']
            - (-0.009942803709820168))
        < 1e-12]
tsP = pa['traj_stat']['P']
tsA = pa['traj_stat']['A1']
chk += [abs(tsP['rho_within']
            - (-0.000633567318687721)) < 1e-12,
        abs(tsP['rho_perm_mean']
            - (-0.001760361987296985))
        < 1e-12,
        abs(tsP['rho_perm_std']
            - 0.014041536747155588) < 1e-12,
        tsP['perm_reps'] == 200,
        abs(tsA['rho_within']
            - (-0.018863878190179637)) < 1e-12,
        abs(tsA['rho_perm_mean']
            - 0.00116289141241039) < 1e-12,
        abs(tsA['rho_perm_std']
            - 0.012302487752804975) < 1e-12,
        tsA['perm_reps'] == 200]
s3 = pa['sim3']
chk += [abs(s3['r_sim3']
            - 0.3768119309633145) < 1e-12,
        s3['verdict'] == 'third_partial',
        abs(s3['r_det_3124_ref']
            - (-0.12297439326221062)) < 1e-12,
        abs(s3['auc_sim3'][0]
            - 0.9809094210600907) < 1e-12,
        abs(s3['auc_sim3'][4]
            - 0.739937641723356) < 1e-12,
        abs(s3['auc_sim3'][12]
            - 0.7930706136621315) < 1e-12]
pb = res['part_b']
chk += [pb['interference']
        == 'input_prompt_span_equal_len_'
        'anchored_last',
        pb['ids'] == {'yes': 9834, 'no': 902,
                      'dot': 13, 'n_layers': 36},
        abs(pb['path']['r']
            - 0.99997947625086) < 1e-12,
        pb['path']['verdict'] == 'path_valid',
        pb['repro']['max_diff'] == 0.0,
        pb['repro']['verdict']
        == 'replay_bit_exact',
        abs(pb['readout_auc']
            - 0.7269898844954649) < 1e-12,
        pb['readout_verdict'] == 'readout_ok',
        pb['n_span'] == {'P': 672, 'A1': 672},
        pb['spans_verdict'] == 'spans_ok']
lsp = pb['lstar']
chk += [lsp['P']['syn']
        == {'rel': 23, 'abs': 22},
        lsp['P']['cont']
        == {'rel': 28, 'abs': 27},
        lsp['A1']['syn']
        == {'rel': 22, 'abs': 20},
        lsp['A1']['cont']
        == {'rel': 32, 'abs': 30}]
sg = pb['sign']
chk += [sg['P']['syn'] == 'negative',
        sg['P']['cont'] == 'negative',
        sg['A1']['syn'] == 'negative',
        sg['A1']['cont'] == 'negative']
cv = pb['curves']
chk += [len(cv['E_syn_P']) == 37,
        len(cv['E_cont_A1']) == 37,
        cv['E_syn_P'][0] == 0.0,
        abs(cv['E_syn_P'][23]
            - (-0.2861765851113166)) < 1e-12,
        abs(cv['E_syn_P'][36]
            - (-0.8362901988763356)) < 1e-12,
        cv['E_syn_A1'][0] == 0.0,
        abs(cv['E_syn_A1'][22]
            - (-0.1131437063138701)) < 1e-12,
        abs(cv['E_syn_A1'][36]
            - (-0.26489097411120044)) < 1e-12,
        cv['E_cont_P'][0] == 0.0,
        abs(cv['E_cont_P'][28]
            - (-0.40839819107443537)) < 1e-12,
        abs(cv['E_cont_P'][36]
            - (-0.8324938236459432)) < 1e-12,
        cv['E_cont_A1'][0] == 0.0,
        abs(cv['E_cont_A1'][32]
            - (-0.17526632380729779)) < 1e-12,
        abs(cv['E_cont_A1'][36]
            - (-0.1783839988389185)) < 1e-12]
sec('A_result', chk)

# ---------- B1. refit recompute ----------
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
               'resid': resid, 'a': a_i,
               'col': col}
c1 = [abs(fit['P']['S'] - (-0.601611613690753))
      < 1e-12,
      abs(fit['P']['MS']
          - (-4.955594255793292)) < 1e-12,
      abs(fit['A1']['S']
          - (-0.5540798866844862)) < 1e-12,
      abs(fit['A1']['MS']
          - (-7.154421990932929)) < 1e-12,
      abs(fit['P']['S']
          - pa['refit']['P']['S']) < 1e-12,
      abs(fit['A1']['S']
          - pa['refit']['A1']['S']) < 1e-12,
      np.max(np.abs(
          fit['P']['a']
          - z24['anchors_P'].astype(
              np.float64))) < 1e-5,
      np.max(np.abs(
          fit['A1']['a']
          - z24['anchors_A1'].astype(
              np.float64))) < 1e-5]
sec('B1_refit_anchors', c1)

# ---------- B2. decomp3 + trail + AR + r4 + sim3 ----------
c2 = []
r4_store = {}
for dc in ('P', 'A1'):
    R = fit[dc]['resid']
    ss_tot = float((R * R).sum())
    NIK = R.size
    Z1 = np.zeros((NIK, N_NEW))
    Z1[np.arange(NIK),
       np.arange(NIK) % N_NEW] = 1.0
    b1, *_ = np.linalg.lstsq(Z1, R.ravel(),
                             rcond=None)
    r1 = (R.ravel() - Z1 @ b1).reshape(
        NP, N_NEW)
    ss_step = ss_tot - float((r1 * r1)
                             .sum())
    x = (MSEQ[dc][:, :-1]
         - fit[dc]['MS']).ravel()
    Z2 = np.stack([x, np.ones(NIK)], 1)
    b2, *_ = np.linalg.lstsq(Z2, r1.ravel(),
                             rcond=None)
    r2 = (r1.ravel() - Z2 @ b2).reshape(
        NP, N_NEW)
    ss_m = float((r1 * r1).sum()) \
        - float((r2 * r2).sum())
    X_tr = np.zeros((NIK, 10 * TRAIL_D))
    col = fit[dc]['col']
    for d in range(1, TRAIL_D + 1):
        rows_k = np.arange(d, N_NEW)
        idx_rows = (np.repeat(
            np.arange(NP) * N_NEW,
            len(rows_k))
            + np.tile(rows_k, NP))
        cls_d = col[:, :N_NEW - d].ravel()
        X_tr[idx_rows,
             cls_d * TRAIL_D + (d - 1)] \
            = 1.0
    btr, *_ = np.linalg.lstsq(
        X_tr, r2.ravel(), rcond=None)
    r3 = (r2.ravel() - X_tr @ btr).reshape(
        NP, N_NEW)
    ss_trail = float((r2 * r2).sum()) \
        - float((r3 * r3).sum())
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
            - ss_trail / ss_tot
            - ss_ar / ss_tot
            - ss_rem / ss_tot)
    dec = pa['decomp3'][dc]
    arp = pa['ar_params'][dc]
    c2 += [abs(ss_tot - dec['ss_tot'])
           < 1e-9,
           abs(ss_step / ss_tot
               - dec['cm_share']) < 1e-12,
           abs(ss_m / ss_tot
               - dec['m_share']) < 1e-12,
           abs(ss_trail / ss_tot
               - dec['trail_share']) < 1e-12,
           abs(ss_ar / ss_tot
               - dec['ar_share']) < 1e-12,
           abs(ss_rem / ss_tot
               - dec['rem_share']) < 1e-12,
           abs(leak - dec['leak']) < 1e-15,
           abs(phi1 - arp['phi1']) < 1e-12,
           abs(phi2 - arp['phi2']) < 1e-12,
           abs(c_ar - arp['c']) < 1e-12]
    md_r4 = float(np.max(np.abs(
        r4 - z25['r4_%s' % dc])))
    c2.append(md_r4 < 1e-9)
    r4_store[dc] = r4
# G table bit-compare vs result (exact json roundtrip)
Grec = {}
for dc in ('P', 'A1'):
    G = np.array(pa['trail_G'][dc],
                 dtype=np.float64)
    Grec[dc] = G
    rows_k = np.arange(1, N_NEW)
    idx_rows = (np.repeat(
        np.arange(NP) * N_NEW, len(rows_k))
        + np.tile(rows_k, NP))
    cls1 = fit[dc]['col'][:, :N_NEW - 1] \
        .ravel()
    X1 = np.zeros((R.size, 10 * TRAIL_D))
    X1[idx_rows, cls1 * TRAIL_D] = 1.0
# sim3 deterministic re-simulation
auc_sim3 = np.zeros(N_NEW + 1)
sim_arr = {}
for dc in ('P', 'A1'):
    m = MSEQ[dc]
    col = fit[dc]['col']
    S_dc = fit[dc]['S']
    MS_dc = fit[dc]['MS']
    phi1 = pa['ar_params'][dc]['phi1']
    phi2 = pa['ar_params'][dc]['phi2']
    c_ar = pa['ar_params'][dc]['c']
    mh = m[:, 0].copy()
    snap = [mh.copy()]
    e1 = np.zeros(NP)
    e2 = np.zeros(NP)
    for k in range(N_NEW):
        t_tr = np.zeros(NP)
        if k >= 1:
            t_tr += Grec[dc][col[:, k - 1], 0]
        if k >= 2:
            t_tr += Grec[dc][col[:, k - 2], 1]
        if k >= 3:
            t_tr += Grec[dc][col[:, k - 3], 2]
        e_k = phi1 * e1 + phi2 * e2 + c_ar
        mh = mh + S_dc * (mh - MS_dc) \
            + content[dc][col[:, k], k] \
            + t_tr + e_k
        snap.append(mh.copy())
        e2 = e1
        e1 = e_k
    sim_arr[dc] = np.stack(snap, 1)
for t in range(N_NEW + 1):
    auc_sim3[t] = auc_mw(
        sim_arr['P'][:, t],
        sim_arr['A1'][:, t])
r3v = float(np.corrcoef(auc_sim3, auc18)
            [0, 1])
md_as = float(np.max(np.abs(
    auc_sim3 - z25['auc_sim3'])))
c2 += [md_as < 1e-9,
       np.max(np.abs(
           auc_sim3
           - np.array(s3['auc_sim3'])))
       < 1e-12,
       abs(r3v - s3['r_sim3']) < 1e-12,
       ('third_sufficient' if r3v >= 0.5
        else ('third_partial'
              if r3v >= 0.3
              else 'third_insufficient'))
       == s3['verdict']]
sec('B2_decomp3_trail_ar_sim3', c2)

# ---------- B3. permutation traj stats ----------
c3 = []
for dc in ('P', 'A1'):
    r4 = r4_store[dc]
    xx = r4[:, :-1].ravel()
    yy = r4[:, 1:].ravel()
    rho_w = float(np.corrcoef(xx, yy)[0, 1])
    rg = np.random.default_rng(PERM_SEED)
    rho_p = np.zeros(N_PERM)
    for rep in range(N_PERM):
        pm = rg.permutation(NP)
        rho_p[rep] = float(np.corrcoef(
            r4[pm, :-1].ravel(), yy)[0, 1])
    st = pa['traj_stat'][dc]
    c3 += [abs(rho_w - st['rho_within'])
           < 1e-12,
           abs(float(rho_p.mean())
               - st['rho_perm_mean']) < 1e-12,
           abs(float(rho_p.std())
               - st['rho_perm_std']) < 1e-12,
           (rho_w >= 0.1 and rho_w
            > float(rho_p.mean())
            + 4.0 * float(rho_p.std()))
           == (st['rho_within'] >= 0.1
               and st['rho_within']
               > st['rho_perm_mean']
               + 4.0 * st['rho_perm_std'])]
cm_min = min(d3P['cm_share'],
             d3A['cm_share'])
c3.append(('common_mode_dominant'
           if cm_min >= 0.2
           else ('common_mode_partial'
                 if cm_min >= 0.05
                 else 'iid_like'))
          == pa['cm_verdict'])
tr_min = min(d3P['trail_share'],
             d3A['trail_share'])
c3.append(('trail_present' if tr_min >= 0.05
           else 'trail_absent')
          == pa['trail_verdict'])
ar_min = min(d3P['ar_share'],
             d3A['ar_share'])
c3.append(('ar_present' if ar_min >= 0.05
           else 'ar_absent')
          == pa['ar_verdict'])
c3.append(pa['traj_verdict']
          == 'trajectory_transient')
sec('B3_perm_traj_gates', c3)

# ---------- B4. part-B npz recompute ----------
c4 = []
for dc in ('P', 'A1'):
    si = z25['span_idx_%s' % dc]
    w = si[:, 1] - si[:, 0] + 1
    c4 += [si.shape == (NP, 2),
           int(si.min()) >= 0,
           int(w.max()) <= N_NEW]
    for nm in ('syn', 'cont'):
        s0 = z25['mlg_s0_%s' % dc].astype(
            np.float64)
        s1 = z25['mlg_s1_%s' % dc].astype(
            np.float64)
        s2 = z25['mlg_s2_%s' % dc].astype(
            np.float64)
        s3a = z25['mlg_s3_%s' % dc].astype(
            np.float64)
        d1 = (s1 - s0).mean(2)
        d2 = (s2 - s0).mean(2)
        d3b = (s3a - s0).mean(2)
        e_syn = (d1 - d2).mean(0)
        e_con = (d1 - d3b).mean(0)
        e_np = ('E_syn_%s' % dc if nm
                == 'syn'
                else 'E_cont_%s' % dc)
        ev = (e_syn if nm == 'syn'
              else e_con)
        c4.append(np.max(np.abs(
            ev - z25[e_np].astype(
                np.float64))) < 1e-4)
auc_fin = auc_mw(
    z25['mlg_s0_P'][:, NL, -1].astype(
        np.float64),
    z25['mlg_s0_A1'][:, NL, -1].astype(
        np.float64))
c4 += [abs(auc_fin - pb['readout_auc'])
       < 1e-4,
       z25['mlg_s0_P'].shape == (672, 37, 13),
       z25['mlg_s3_A1'].shape == (672, 37, 13)]
# lstar recompute from result curves (exact)
for dc in ('P', 'A1'):
    for nm in ('syn', 'cont'):
        cvv = np.array(cv['E_%s_%s'
                          % (nm, dc)])
        eff = abs(float(cvv[NL]))
        lr = None
        la = None
        for L in range(20, NL + 1):
            if lr is None and eff > 0 \
                    and cvv[L] <= -0.3 * eff:
                lr = L
            th = 0.05 if dc == 'P' \
                else 0.025
            if la is None \
                    and cvv[L] <= -th:
                la = L
        c4 += [lr == lsp[dc][nm]['rel'],
               la == lsp[dc][nm]['abs']]
md_ad = float(np.max(np.abs(
    z25['auc_sim3']
    - np.array(s3['auc_sim3']))))
c4.append(md_ad < 1e-12)
sec('B4_partb_npz', c4)

# ---------- C. seal ----------
c6 = [seal['part_a']['trail_gate']
      .find('0.05') >= 0,
      seal['part_a']['ar_gate']
      .find('0.05') >= 0,
      seal['part_a']['traj_gate']
      .find('default_rng(3125)') >= 0,
      seal['part_a']['traj_gate']
      .find('perm_mean + 4*perm_std') >= 0,
      seal['part_a']['sim3']
      .find('third_partial') >= 0,
      seal['part_a']['decomp3_seq']
      .find('TRAIL_D') >= 0
      or seal['part_a']['decomp3_seq']
      .find('d=1..3') >= 0,
      seal['part_b']['hs_trap']
      .find('final-normed') >= 0,
      seal['part_b']['interference']
      .find('anchored_last') >= 0,
      seal['part_b']['span_fix']
      .find('LAST') >= 0,
      seal['part_b']['gates']['B_path']
      .find('0.9999') >= 0,
      seal['part_b']['gates']['B_lstar']
      .find('-0.3') >= 0,
      seal['created'].startswith('2026-')]
sec('C_seal', c6)

# ---------- D. ledger ----------
led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
m5 = [m for m in led['measurements']
      if m.get('phase') == 3125]
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
    [len(led['measurements']) == 262,
     len(m5) == 1,
     m5 and m5[0]['meas_id']
     == 'meas3125_omega_p123_third_comp_'
     'qwen_inputstream',
     m5 and m5[0]['verdict'] == V,
     m5 and m5[0]['claim'].find(
         'Omega-P123') >= 0,
     len(l14['connects']) == 230,
     sha8 == 'ddf922b0'])

# ---------- E. MEMO ----------
memo = io.open(MEMO, encoding='utf-8').read()
cnt = memo.count('## Phase 3125:')
tail = memo[memo.index('## Phase 3125:'):] \
    if cnt else ''
mline = [l for l in tail.splitlines()
         if l.startswith('## Phase 3125:')][0] \
    if cnt else ''
has_ts = bool(re.search(
    r'\[2026-\d{2}-\d{2} \d{2}:\d{2}\]',
    mline)) if cnt else False
f1 = u'\u7b2c\u4e09\u6210\u5206=\u5185\u5bb9' \
     u'\u5c3e\u8ff9'
f2 = u'\u6a21\u578b\u7279\u5f02'
f3 = u'\u672b\u9879\u5df2\u542b final norm'
sec('E_memo',
    [cnt == 1,
     has_ts,
     f1 in tail,
     '0.08599867672568678' in tail,
     '0.05022617239377846' in tail,
     'third_partial' in tail,
     '0.3768119309633145' in tail,
     '0.7269898844954649' in tail,
     f2 in tail,
     'anchored-last' in tail,
     f3 in tail,
     tail.rstrip().endswith(
         u'`tests/gpt5_temp/p3125_path_probe'
         u'.py`\u3002'),
     '3126' in tail])

# ---------- F. wlogs ----------
cf = []
for wdir in (WLOG_D, WLOG_C):
    t = io.open(wdir + '\\' + '2026-09-24.md',
                encoding='utf-8').read()
    cf += ['Phase 3125 Omega-P123' in t,
           'Phase 3125 closeout' in t,
           'ddf922b0' in t]
sec('F_wlogs', cf)

# ---------- G. MEMORY ----------
mem = io.open(MEMO_W,
              encoding='utf-8').read()
sec('G_memory',
    [len(mem) == 2968,
     mem.count(u'\u673a\u5236\u94fe\u72b6'
               u'\u6001\uff083125\uff09') == 1,
     mem.count(u'- 3125\uff08T4\uff09') == 1,
     mem.count(u'- 3124\uff08T4\uff09') == 1,
     mem.count('max=3125') == 1,
     mem.count(u'- 3121\uff1a') == 0,
     mem.count('3122/3123 L36') == 1,
     mem.count('span=Facts') == 1])

# ---------- report ----------
o.append('TOTAL FAILS: %d' % len(fails))
io.open(VF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('verify done, fails=%d' % len(fails))
