# -*- coding: utf-8 -*-
"""Phase 3121 independent disk verify.
Recomputes Part A/B/C metrics from npz raw bytes
(aligned dtype paths), re-derives ledger sha8,
checks MEMO/wlog/MEMORY.md on disk."""
import hashlib
import io
import json
import os
import re
import sys

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUTD = (RDIR + r'\phase3121'
        r'\omega_p119_repl_causality_erase_'
        'polarity_recon')
D18 = (RDIR + r'\phase3118'
       r'\omega_p116_autoregressive_margin_'
       'trajectory')
D20 = (RDIR + r'\phase3120'
       r'\omega_p118_content_attr_amplifier_'
       'behavior_opshape')
D05 = (RDIR + r'\phase3105'
       r'\omega_p103_incontext_truth_'
       'consistency')
MDIR = os.path.join(ROOT, 'models', 'hf',
                    'qwen3-4b')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
WLOG_D = ROOT + r'\.workbuddy\memory'
WLOG_C = (r'C:\Users\Admin\WorkBuddy'
          r'\2026-09-17-01-30-05\.workbuddy'
          r'\memory')
MEMO_W = WLOG_D + r'\MEMORY.md'
o = []
n_fail = [0]


def ck(name, cond, detail=''):
    if cond:
        o.append('PASS %s' % name)
    else:
        n_fail[0] += 1
        o.append('FAIL %s %s' % (name, detail))


def close(a, b, tol=1e-12):
    return abs(float(a) - float(b)) <= tol


V = ('content_nonspecific|'
     'syntax_polarity_absent|'
     'erase_joint_behavioral|'
     'readout_robust|'
     'reconstruction_failed|'
     'curve_shape_weak|coverage_ok|'
     'position_independent_confirmed')
N_NEW = 12
NP_ = 672

# ================= A. result.json =================
res = json.load(io.open(OUTD + r'\result.json',
                        encoding='utf-8'))
ck('R.verdict', res['verdict'] == V)
ck('R.smoke', res['smoke'] is False)
ck('R.n_pairs', res['n_pairs'] == 672)
ck('R.np_a', res['np_a'] == 672)
ck('R.np_g', res['np_g'] == 672)
ck('R.runtime', close(res['runtime_s'], 669.4,
                      0.05), res['runtime_s'])
pa = res['part_a']
ck('R.pa.verdict', pa['verdict'] ==
   'content_nonspecific|syntax_polarity_absent')
ck('R.pa.repro_v', pa['repro']['verdict'] ==
   'replay_bit_exact')
ck('R.pa.repro_d', pa['repro']['max_diff'] == 0.0)
ck('R.pa.nspanP', pa['n_span_P'] == 305)
ck('R.pa.nspanA1', pa['n_span_A1'] == 320)
ck('R.pa.gates', pa['gates'] == {
    'P': 'content_nonspecific',
    'A1': 'content_nonspecific',
    'syntax_P': 'syntax_polarity_absent',
    'syntax_A1': 'syntax_polarity_absent'})
eP = pa['effects']['P']
ck('R.eP.n', eP['n_span'] == 305
   and eP['n_len2'] == 305)
ck('R.eP.D0', eP['D_mean']['c0'] == 0.0)
ck('R.eP.D1', close(eP['D_mean']['c1'],
                    -2.0944437736859087, 1e-9))
ck('R.eP.D2', close(eP['D_mean']['c2'],
                    0.4315817940430563, 1e-9))
ck('R.eP.D3', close(eP['D_mean']['c3'],
                    1.1288977380170198, 1e-9))
ck('R.eP.E10', close(eP['E_10_mean'],
                     -2.526025567728965, 1e-9))
ck('R.eP.E31', close(eP['E_31_mean'],
                     3.2233415117029285, 1e-9))
ck('R.eP.s10', close(eP['E_10_sd'],
                     1.42277025689462, 1e-9))
ck('R.eP.s31', close(eP['E_31_sd'],
                     0.8628208373949953, 1e-9))
eA = pa['effects']['A1']
ck('R.eA.n', eA['n_span'] == 320
   and eA['n_len2'] == 320)
ck('R.eA.D1', close(eA['D_mean']['c1'],
                    0.5678164122719318, 1e-9))
ck('R.eA.D2', close(eA['D_mean']['c2'],
                    1.844489301871508, 1e-9))
ck('R.eA.D3', close(eA['D_mean']['c3'],
                    2.423060708269477, 1e-9))
ck('R.eA.E10', close(eA['E_10_mean'],
                     -1.2766728895995765, 1e-9))
ck('R.eA.E31', close(eA['E_31_mean'],
                     1.855244295997545, 1e-9))
pb = res['part_b']
ck('R.pb.verdict', pb['verdict'] ==
   'erase_joint_behavioral|readout_robust')
ck('R.pb.yclean', close(pb['yes_rate_clean'],
                        0.5885416666666666))
ck('R.pb.yjoint', close(pb['yes_rate_joint'],
                        0.5275297619047619))
ck('R.pb.delta', close(pb['yes_delta'],
                       -0.06101190476190477))
ck('R.pb.jv', pb['joint_verdict'] ==
   'erase_joint_behavioral')
ck('R.pb.famr', close(pb['fam_r'],
                      0.8499039671426715, 1e-9))
ck('R.pb.fv', pb['fam_verdict'] ==
   'readout_robust')
ft = pb['first_token']
ck('R.ft.clean', close(ft['clean']['first_yes'],
                       0.5885416666666666))
ck('R.ft.L30', close(ft['abl_L30']['first_yes'],
                     0.9955357142857143))
ck('R.ft.L32', close(ft['abl_L32']['first_yes'],
                     0.9955357142857143))
ck('R.ft.joint', close(ft['abl_joint']['first_yes'],
                       0.5275297619047619))
pc = res['part_c']
ck('R.pc.verdict', pc['verdict'] ==
   'reconstruction_failed|curve_shape_weak|'
   'coverage_ok')
ck('R.pc.r2two', close(pc['r2_two'],
                       0.12843996911885736))
ck('R.pc.r2lin', close(pc['r2_lin'],
                       0.04183500685653896))
ck('R.pc.r2per', close(pc['r2_persistence'],
                       -0.9618412331055202))
ck('R.pc.dr2', close(pc['d_r2'],
                     0.0866049622623184))
ck('R.pc.fitv', pc['fit_verdict'] ==
   'reconstruction_failed')
ck('R.pc.aucr', close(pc['auc_r'],
                      -0.07890096967521298))
ck('R.pc.aucrl', close(pc['auc_r_lin'],
                       2.276099300428211e-16,
                       1e-18))
ck('R.pc.curvev', pc['curve_verdict'] ==
   'curve_shape_weak')
ck('R.pc.sigma', close(pc['sigma'],
                       3.3923936726908717, 1e-9))
ck('R.pc.cov', close(pc['coverage'],
                     0.9671347966269841, 1e-9))
ck('R.pc.mcv', pc['mc_verdict'] == 'coverage_ok')
ck('R.pc.slope', close(pc['slope_3120'],
                       -0.6801486439276976))
ck('R.pc.mstar', close(pc['mstar_3120'],
                       -6.052991245322225))
pd_ = res['part_d']
ck('R.pd.verdict_kept', pd_['verdict'] ==
   'position_independent_confirmed')
ck('R.pd.nzP', pd_['syntax_n_zero_P'] == 6)
ck('R.pd.nzA', pd_['syntax_n_zero_A1'] == 5)

# ================= B. npz recompute =================
z119 = np.load(OUTD + r'\p119_readout.npz',
               allow_pickle=True)
z18 = np.load(os.path.join(D18,
                           'traj_readout.npz'),
              allow_pickle=False)
z20 = np.load(os.path.join(D20,
                           'p118_readout.npz'),
              allow_pickle=False)
annP = z20['annP_c']
annA = z20['annA_c']
ck('B.ann.shape', annP.shape == (672, N_NEW)
   and annP.dtype == np.int8)

# --- B1. replay bit-exact (float32 path) ---
mP18 = z18['gt_cleanseq_clean__P']
mA18 = z18['gt_cleanseq_clean__A1']
dP = float(np.abs(z119['pa_c0_dn__P']
                  - mP18[:NP_]).max())
dA = float(np.abs(z119['pa_c0_dn__A1']
                  - mA18[:NP_]).max())
ck('B1.reproP', dP == 0.0, dP)
ck('B1.reproA1', dA == 0.0, dA)

# --- B2. independent span retrieval ---
def find_span(ann_row):
    best = None
    k1 = None
    for k in range(1, N_NEW):
        isf = int(ann_row[k]) in (3, 4)
        if isf and k1 is None:
            k1 = k
        if (not isf or k == N_NEW - 1) \
                and k1 is not None:
            k2 = (k if isf and k == N_NEW - 1
                  else k - 1)
            if best is None or \
                    (k2 - k1) > (best[1]
                                 - best[0]):
                best = (k1, k2)
            k1 = None
    return best

hist = {'P': [0] * 12, 'A1': [0] * 12}
spans = {'P': [], 'A1': []}
for dc, ann in (('P', annP), ('A1', annA)):
    for j in range(NP_):
        b = find_span(ann[j])
        if b is None:
            hist[dc][0] += 1
        else:
            spans[dc].append((j, b[0], b[1],
                              b[1] - b[0] + 1))
            hist[dc][b[1] - b[0] + 1] += 1
ck('B2.histP', hist['P'] == [367, 0, 0, 0, 0, 0,
                             0, 63, 146, 82, 14,
                             0], str(hist['P']))
ck('B2.histA1', hist['A1'] == [352, 0, 0, 0, 0,
                               0, 0, 87, 148, 73,
                               12, 0],
   str(hist['A1']))
si_len = z119['span_info_len']
si_dir = z119['span_info_dir']
ck('B2.spaninfo_lenP',
   int((si_len[:NP_] == 0).sum()) == 367
   and int((si_len[:NP_] == 7).sum()) == 63
   and int((si_len[:NP_] == 10).sum()) == 14)
ck('B2.spaninfo_lenA1',
   int((si_len[NP_:] == 0).sum()) == 352
   and int((si_len[NP_:] == 7).sum()) == 87
   and int((si_len[NP_:] == 10).sum()) == 12)
ck('B2.spaninfo_len_total',
   len(si_len) == 2 * NP_)

# --- B3. Part A effects recompute ---
CONDS = ['c0', 'c1', 'c2', 'c3']


def eff_recompute(dc):
    ann = annP if dc == 'P' else annA
    base = z119['pa_c0_dn__%s' % dc]
    Ds = {c: [] for c in CONDS}
    lens = []
    for (j, k1, k2, ln) in spans[dc]:
        lens.append(ln)
        for c in CONDS[1:]:
            arr = z119['pa_%s_dn__%s' % (c, dc)]
            diff = (arr[j].astype(np.float64)
                    - base[j].astype(np.float64))
            Ds[c].append(float(
                diff[k1 + 1:].mean()))
    Dm = {c: float(np.mean(Ds[c]))
          for c in CONDS[1:]}
    e10 = [a - b for (a, b, ln) in
           zip(Ds['c1'], Ds['c2'], lens)
           if ln >= 2]
    e31 = [a - b for (a, b) in
           zip(Ds['c3'], Ds['c1'])]
    return {'n_span': len(spans[dc]),
            'n_len2': len(e10),
            'D_mean': Dm,
            'E_10_mean': float(np.mean(e10)),
            'E_31_mean': float(np.mean(e31)),
            'E_10_sd': float(np.std(e10)),
            'E_31_sd': float(np.std(e31))}


rP = eff_recompute('P')
rA = eff_recompute('A1')
for dc, rr, ee in (('P', rP, eP),
                   ('A1', rA, eA)):
    ck('B3.%s.nspan' % dc,
       rr['n_span'] == ee['n_span']
       and rr['n_len2'] == ee['n_len2'])
    for c in CONDS[1:]:
        ck('B3.%s.D%s' % (dc, c),
           close(rr['D_mean'][c],
                 ee['D_mean'][c], 1e-9),
           '%r vs %r' % (rr['D_mean'][c],
                         ee['D_mean'][c]))
    ck('B3.%s.E10' % dc,
       close(rr['E_10_mean'], ee['E_10_mean'],
             1e-9))
    ck('B3.%s.E31' % dc,
       close(rr['E_31_mean'], ee['E_31_mean'],
             1e-9))
    ck('B3.%s.s10' % dc,
       close(rr['E_10_sd'], ee['E_10_sd'],
             1e-9))
    ck('B3.%s.s31' % dc,
       close(rr['E_31_sd'], ee['E_31_sd'],
             1e-9))

# --- B4. family readout (tokenizer) ---
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MDIR)
yfam = []
for s in ['yes', 'Yes', ' yes', ' Yes',
          'YES']:
    yfam += list(tok(s, add_special_tokens=False)
                 ['input_ids'])
yfam = sorted(set(yfam))
nfam = []
for s in ['no', 'No', ' no', ' No', 'NO']:
    nfam += list(tok(s, add_special_tokens=False)
                 ['input_ids'])
nfam = sorted(set(nfam))
yf = set(yfam)
nfs = set(nfam)
inc_dn = []
inc_fm = []
for dc in ('P', 'A1'):
    for (kdn, kfm) in (('pa_c0_dn__%s' % dc,
                        'pa_c0_fam__%s' % dc),
                       ('mj_dn_%s' % dc,
                        'mj_fam_%s' % dc)):
        a = z119[kdn].astype(np.float64)
        b = z119[kfm].astype(np.float64)
        inc_dn.append(a[:, 1:] - a[:, :-1])
        inc_fm.append(b[:, 1:] - b[:, :-1])
xd = np.concatenate([x.ravel() for x in inc_dn])
yd = np.concatenate([x.ravel() for x in inc_fm])
r_fam = float(np.corrcoef(xd, yd)[0, 1])
ck('B4.famr', close(r_fam, 0.8499039671426715,
                    1e-9), r_fam)

# --- B5. Part B yes rates + fork ---
gP18 = z18['gen_clean__P']
gA18 = z18['gen_clean__A1']


def any_yes(seq2d, nmatch=8):
    cnt = 0
    for row in seq2d:
        if any(int(t) in yf
               for t in row[:nmatch]):
            cnt += 1
    return cnt / float(seq2d.shape[0])


clean_arr = np.vstack([gP18[:NP_],
                       gA18[:NP_]])
yr_clean = any_yes(clean_arr)
ck('B5.yclean', close(yr_clean,
                      0.5885416666666666))
ck('B5.yclean_int', round(yr_clean * 1344) == 791,
   yr_clean * 1344)
gj = np.vstack([z119['gj_P'], z119['gj_A1']])
yr_joint = any_yes(gj)
ck('B5.yjoint', close(yr_joint,
                      0.5275297619047619))
ck('B5.yjoint_int', round(yr_joint * 1344) == 709,
   yr_joint * 1344)
z20g30 = np.vstack([z20['gen_abl_L30__P'],
                    z20['gen_abl_L30__A1']])
z20g32 = np.vstack([z20['gen_abl_L32__P'],
                    z20['gen_abl_L32__A1']])
forks = (('clean', clean_arr),
         ('abl_L30', z20g30[:NP_]),
         ('abl_L32', z20g32[:NP_]),
         ('abl_joint', gj))
for (nm, arr) in forks:
    fy = float(np.mean([int(t) in yf
                        for t in arr[:, 0]]))
    fn = float(np.mean([int(t) in nfs
                        for t in arr[:, 0]]))
    y8 = any_yes(arr)
    ft2 = ft[nm]
    ck('B5.%s.fy' % nm, close(fy,
                              ft2['first_yes']))
    ck('B5.%s.fn' % nm, close(fn,
                              ft2['first_no']))
    ck('B5.%s.y8' % nm, close(y8,
                              ft2['yes_rate8']))
ck('B5.bug_L30_int',
   round(ft['abl_L30']['first_yes'] * 672)
   == 669,
   ft['abl_L30']['first_yes'] * 672)
ck('B5.bug_L32_int',
   round(ft['abl_L32']['first_yes'] * 672)
   == 669,
   ft['abl_L32']['first_yes'] * 672)

# --- B6. Part C recompute ---
mP = mP18.astype(np.float64)
mA = mA18.astype(np.float64)
MSEQ = {'P': mP, 'A1': mA}
ANN = {'P': annP, 'A1': annA}
S = float(res['part_c']['slope_3120'])
MS = float(res['part_c']['mstar_3120'])
res20 = json.load(io.open(
    os.path.join(D20, 'result.json'),
    encoding='utf-8'))
S20 = float(res20['part_c']['primary']
            ['slope_lin'])
M20 = float(res20['part_c']
            ['fixed_point_raw_lin'])
ck('B6.S_cross', close(S, S20)
   and close(S, -0.6801486439276976))
ck('B6.MS_cross', close(MS, M20)
   and close(MS, -6.052991245322225))
content = {}
for dc in ('P', 'A1'):
    dm = (MSEQ[dc][:, 1:] - MSEQ[dc][:, :-1])
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
ck('B6.contentP', np.allclose(
    content['P'], z119['content_P'],
    rtol=0, atol=0))
ck('B6.contentA1', np.allclose(
    content['A1'], z119['content_A1'],
    rtol=0, atol=0))


def recon(use_content):
    out = {}
    for dc in ('P', 'A1'):
        seq = MSEQ[dc]
        mh = np.zeros((NP_, N_NEW + 1),
                      dtype=np.float64)
        mh[:, 0] = seq[:, 0]
        for k in range(N_NEW):
            step = S * (mh[:, k] - MS)
            if use_content:
                col = ANN[dc][:, k] \
                    .astype(np.int64)
                step = step + \
                    content[dc][col, k]
            mh[:, k + 1] = mh[:, k] + step
        out[dc] = mh
    return out


rec_two = recon(True)
rec_lin = recon(False)
rec_per = {d: np.repeat(MSEQ[d][:, :1],
                        N_NEW + 1, axis=1)
           for d in ('P', 'A1')}
ck('B6.rec_two', np.allclose(
    rec_two['P'], z119['rec_two_P'],
    rtol=0, atol=0)
    and np.allclose(rec_two['A1'],
                    z119['rec_two_A1'],
                    rtol=0, atol=0))


def r2_all(rec):
    num = 0.0
    gm = 0.0
    den = 0.0
    n = 0
    for dc in ('P', 'A1'):
        X = MSEQ[dc][:, 1:].ravel()
        Y = rec[dc][:, 1:].ravel()
        gm += X.sum()
        n += len(X)
        num += ((X - Y) ** 2).sum()
    gm /= n
    for dc in ('P', 'A1'):
        X = MSEQ[dc][:, 1:].ravel()
        den += ((X - gm) ** 2).sum()
    return float(1.0 - num / den)


r2_two = r2_all(rec_two)
r2_lin = r2_all(rec_lin)
r2_per = r2_all(rec_per)
ck('B6.r2two', close(r2_two,
                     pc['r2_two'], 1e-12),
   r2_two)
ck('B6.r2lin', close(r2_lin, pc['r2_lin'],
                     1e-12), r2_lin)
ck('B6.r2per', close(r2_per,
                     pc['r2_persistence'],
                     1e-12), r2_per)


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


auc_two = np.array([
    auc_mw(rec_two['P'][:, t],
           rec_two['A1'][:, t])
    for t in range(N_NEW + 1)])
ck('B6.auc_two', np.allclose(
    auc_two, z119['auc_two'],
    rtol=0, atol=1e-12))
auc18 = z18['auc_curve']
ck('B6.auc18_len', len(auc18) == N_NEW + 1)
ck('B6.auc18_0', close(float(auc18[0]),
                       0.9809094210600907))
auc_r = float(np.corrcoef(auc_two,
                          auc18)[0, 1])
auc_lin = np.array([
    auc_mw(rec_lin['P'][:, t],
           rec_lin['A1'][:, t])
    for t in range(N_NEW + 1)])
auc_r_lin = float(np.corrcoef(auc_lin,
                              auc18)[0, 1])
ck('B6.aucr', close(auc_r, pc['auc_r'],
                    1e-12), auc_r)
ck('B6.aucrl', close(auc_r_lin,
                     pc['auc_r_lin'], 1e-16),
   auc_r_lin)
resid = []
for dc in ('P', 'A1'):
    dm = (MSEQ[dc][:, 1:] - MSEQ[dc][:, :-1])
    lin = S * (MSEQ[dc][:, :-1] - MS)
    col = ANN[dc].astype(np.int64)
    dev = content[dc][col, np.arange(N_NEW)]
    resid.append((dm - lin - dev).ravel())
sigma = float(np.std(np.concatenate(resid)))
ck('B6.sigma', close(sigma, pc['sigma'], 1e-9),
   sigma)
rng_mc = np.random.default_rng(3121)
covered = 0
tot = 0
for rep in range(20):
    for dc in ('P', 'A1'):
        seq = MSEQ[dc]
        mh = seq[:, 0].copy()
        for k in range(N_NEW):
            ck_col = ANN[dc][:, k] \
                .astype(np.int64)
            step = S * (mh - MS) \
                + content[dc][ck_col, k] \
                + rng_mc.normal(0.0, sigma,
                                len(mh))
            mh = mh + step
            lo_b = mh - 1.96 * sigma \
                * np.sqrt(k + 1)
            hi_b = mh + 1.96 * sigma \
                * np.sqrt(k + 1)
            covered += int(
                ((seq[:, k + 1] >= lo_b)
                 & (seq[:, k + 1] <= hi_b))
                .sum())
            tot += len(mh)
cov = covered / float(tot)
ck('B6.coverage', close(cov, pc['coverage'],
                        1e-9), cov)
ck('B6.coverage_int', tot == 20 * 2 * 672 * 12,
   tot)

# ================= C. ledger =================
led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
ms = led['measurements']
ck('C.n', len(ms) == 258, len(ms))
m31 = [m for m in ms
       if m.get('meas_id') ==
       'meas3121_omega_p119_repl_causality_'
       'erase_polarity_recon']
ck('C.meas31_exists', len(m31) == 1)
if m31:
    m31 = m31[0]
    ck('C.meas31.phase', m31['phase'] == 3121)
    ck('C.meas31.verdict', m31['verdict'] == V)
    ck('C.meas31.claim_len',
       len(m31['claim']) > 3000,
       len(m31['claim']))
    ck('C.meas31.anchors',
       'seed 3121' in m31['anchors']
       and 'D-POS' in m31['anchors'])
    ck('C.meas31.artifacts',
       sorted(m31['artifacts'].keys())
       == ['readout', 'result', 'seal'])
    ck('C.meas31.note_bugs',
       'boolean' in m31['note'])
m30 = [m for m in ms
       if m.get('meas_id', '').startswith(
           'meas3120_')]
ck('C.meas30_kept', len(m30) == 1)
l14 = [l for l in led['linkage']
       if l.get('link_id') ==
       'L14_readout_spectrum_cross_model'][0]
ck('C.l14_n', len(l14['connects']) == 226,
   len(l14['connects']))
ck('C.l14_last',
   l14['connects'][-1] ==
   'meas3121_omega_p119_repl_causality_'
   'erase_polarity_recon')
stored_sha = led.get('ledger_sha256_8', '')
led2 = json.loads(json.dumps(
    led, ensure_ascii=False))
led2.pop('ledger_sha256_8', None)
blob = json.dumps(led2, sort_keys=True,
                  ensure_ascii=False)
sha_re = hashlib.sha256(
    blob.encode('utf-8')).hexdigest()[:8]
ck('C.sha8', sha_re == stored_sha,
   '%s vs %s' % (sha_re, stored_sha))

# ================= D. MEMO =================
memo = io.open(MEMO, encoding='utf-8').read()
i21 = memo.find('## Phase 3121:')
i20 = memo.find('## Phase 3120:')
ck('D.m21_exists', i21 > 0)
ck('D.m20_before', i20 > 0 and i21 > i20)
head = memo[i21:i21 + 800]
ck('D.m21_bracket',
   re.search(r'## Phase 3121: .* \[\d{4}-\d{2}'
             r'-\d{2} \d{2}:\d{2}\]',
             head) is not None, head[:120])
tail = memo[i21:]
ck('D.m21_no_placeholder',
   '[[NOW]]' not in tail)
for s in ['−2.094', '+0.4316', '+1.1289',
          '−2.526', '+3.223', '0.5275', '0.8499',
          '0.128', '0.9671', '0.0655', '0.3318',
          '+4.846', '+7.951',
          'position_confounded', '范式失效',
          '断言牵引', '分布现象']:
    ck('D.m21.key[%s]' % s, s in tail)
for s in ['### 1. 三大发现', '### 2. 关键数值',
          '### 3. 硬伤', '### 4. 机制拼图更新',
          '### 5. 3122 预注册']:
    ck('D.m21.sec[%s]' % s, s in tail)
ck('D.m21.tail_artifact',
   tail.rstrip().endswith(chr(96) + 'p3121_probe3.py' + chr(96) + chr(12290)))
ck('D.m21_bugs_recorded',
   'vstack[:NP_G]' in tail
   and 'dm[sel]' in tail)

# ================= E. workspace logs =====
for wdir in (WLOG_D, WLOG_C):
    wl = wdir + '\\' + '2026-09-23.md'
    tag = 'D' if wdir == WLOG_D else 'C'
    try:
        t = io.open(wl,
                    encoding='utf-8').read()
    except IOError:
        t = ''
    ck('E.%s.exp' % tag,
       'Phase 3121 Omega-P119' in t)
    ck('E.%s.clo' % tag,
       'Phase 3121 closeout finished' in t)
    ck('E.%s.sha' % tag,
       'sha=c5eae421' in t)

# ================= F. MEMORY.md ==========
mem = io.open(MEMO_W, encoding='utf-8').read()
ck('F.header', '## 机制链状态（3121）' in mem)
ck('F.line3121',
   '- 3121（T4）' in mem
   and '范式失效' in mem
   and '0.066' in mem)
ck('F.line3120_slim',
   '- 3120（T4）：重述步双向' in mem)
ck('F.max', 'max=3121' in mem)
ck('F.next', '下一 3122' in mem
   and '句级连贯替换' in mem)
ck('F.no_old_header',
   '## 机制链状态（3120）' not in mem)
ck('F.len', len(mem) < 3000, len(mem))

# ================= report ================
out = '\n'.join(o)
io.open(ROOT + r'\tests\gpt5_temp'
        r'\p3121_verify_stdout.txt', 'w',
        encoding='utf-8').write(
    out + '\nTOTAL=%d FAIL=%d\n'
    % (len(o), n_fail[0]))
try:
    io.open(OUTD + r'\disk_verify_out.txt',
            'w', encoding='utf-8').write(
        out + '\nTOTAL=%d FAIL=%d\n'
        % (len(o), n_fail[0]))
except Exception:
    pass
print('verify done TOTAL=%d FAIL=%d'
      % (len(o), n_fail[0]))
sys.exit(1 if n_fail[0] else 0)
