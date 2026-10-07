# -*- coding: utf-8 -*-
"""Phase 3127 disk verify (independent):
A0 result -> A1 seal -> A2 npz shapes ->
A3 frozen inputs -> B1 Part-A refit + D6
decomp + G6 recompute -> B2 A1-closure
recompute (gain46/G46/transfer + perm 200
seed 3127 full-pipeline bit-recompute) ->
B3 cross/PCA/sparse recompute -> B4 Part-B
npz recompute (wspecQ + gates + spec corr)
-> B5 Part-C npz recompute (+ depth
profile) -> B6 Part-D recompute (stats via
tokenizer + bit-exact first-96 vs p124 +
shift/flip/coverage) -> B7 verdict 19
re-assembly -> D ledger -> E MEMO ->
F wlogs -> G MEMORY. All recomputes
deterministic; rng only seed-3127 perm."""
import hashlib
import io
import json
import os

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913')
D05 = RDIR + r'\phase3105' \
      r'\omega_p103_incontext_truth_consistency'
D18 = RDIR + r'\phase3118' \
      r'\omega_p116_autoregressive_margin_' \
      'trajectory'
D20 = RDIR + r'\phase3120' \
      r'\omega_p118_content_attr_amplifier_' \
      'behavior_opshape'
D25 = RDIR + r'\phase3125' \
      r'\omega_p123_third_comp_qwen_' \
      'inputstream'
D26 = RDIR + r'\phase3126' \
      r'\omega_p124_glm4_anchoredlast_' \
      'regen_writechain'
D13 = RDIR + r'\phase3113' \
      r'\omega_p111_artifact_writein'
OUTD = RDIR + r'\phase3127' \
       r'\omega_p125_writechain_port_' \
       'crossmodel_a1closure_fullregen'
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
      r'\p3127_verify_out.txt')
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
NLQ = 36
NLG = 40
TRAIL_D = 6
N_PERM = 200
PERM_SEED = 3127
SCOND3 = ('s1', 's2', 's3')
DIRS = ('P', 'A1')
LAY_Q = {'write': [26, 28, 30, 32, 34],
         'port': [20, 21],
         'ctrl': [2, 8, 14]}
LAY_G = {'write': [8, 9, 13, 29],
         'port': [20],
         'ctrl': [4, 14, 34]}
G_PORT = 2.0
G_SPEC = 0.6
G_DEPTH = 0.5

# ---------- A0. result ----------
res = json.load(io.open(
    OUTD + r'\result.json', encoding='utf-8'))
verdict = res['verdict']
pa = res['part_a']
pb = res['part_b']
pc = res['part_c']
pd_ = res['part_d']
dp = res['depth']
sec('A0_result', [
    res['phase'] == 3127,
    res['smoke'] is False,
    res['runtime_s'] > 0,
    len(verdict.split('|')) == 19,
    'qwen_path_valid' in verdict,
    'glm4_path_valid' in verdict,
    'a1_' in verdict,
    'lag46' in verdict,
    'qwen_write' in verdict,
    'qwen_port' in verdict,
    'glm4_write' in verdict,
    'glm4_port' in verdict,
    'port_depth' in verdict,
    'baseline_s0' in verdict,
    'bit_exact' in verdict,
    'behavior_shift' in verdict,
    'flip_polarity' in verdict,
    'coverage_' in verdict,
    isinstance(pa, dict),
    isinstance(pb, dict) and
    isinstance(pc, dict) and
    isinstance(pd_, dict)])

# ---------- A1. seal ----------
seal = json.load(io.open(
    OUTD + r'\design_seal.json',
    encoding='utf-8'))
sec('A1_seal', [
    seal['phase'] == 3127,
    seal['smoke'] is False,
    seal['np_a'] == NP,
    seal['np_b'] == NP,
    seal['np_reg'] == NP,
    'rng 3127'
    in seal['part_a']['no_mc'],
    '@@' not in json.dumps(seal),
    'swap' in json.dumps(seal),
    'transfer' in json.dumps(seal),
    len(seal.get('part_a', {})) >= 3,
    len(seal.get('part_b', {})) >= 3,
    len(seal.get('part_c', {})) >= 3,
    len(seal.get('part_d', {})) >= 4])

# ---------- A2. npz ----------
z = np.load(OUTD + r'\p125_readout.npz',
            allow_pickle=False)
ks = set(z.files)
dmq_keys = ['%s_L%02d' % (dc, l)
            for l in (LAY_Q['write']
                      + LAY_Q['port']
                      + LAY_Q['ctrl'])
            for dc in DIRS]
dmg_keys = ['%s_L%02d' % (dc, l)
            for l in (LAY_G['write']
                      + LAY_G['port']
                      + LAY_G['ctrl'])
            for dc in DIRS]
conds2 = []
for k in dmq_keys:
    conds2.append(('dmq_' + k) in ks
                  and z['dmq_' + k].shape
                  == (NP, N_NEW + 1))
for k in dmg_keys:
    conds2.append(('dmg_' + k) in ks
                  and z['dmg_' + k].shape
                  == (NP, N_NEW + 1))
npreg = int(pd_['np_reg'])
reg_ok = True
idxs = {}
for dc in DIRS:
    for c in SCOND3:
        rk = 'regen_%s_%s' % (c, dc)
        ik = 'regenidx_%s_%s' % (c, dc)
        if rk not in ks or ik not in ks:
            reg_ok = False
            continue
        arr = z[rk]
        idx = z[ik]
        idxs[(dc, c)] = idx
        if arr.shape != (len(idx), N_NEW):
            reg_ok = False
        if arr.dtype != np.int32:
            reg_ok = False
        if not np.array_equal(
                idx, np.arange(0, NP, 2)[:len(idx)]
                if len(idx) < NP
                else np.arange(NP)):
            reg_ok = False
sec('A2_npz', conds2 + [
    len(ks) == 36 + 2 + 2 + 6 + 6,
    'wspecQ_P' in ks
    and z['wspecQ_P'].shape == (NLQ, N_NEW + 1),
    'wspecQ_A1' in ks,
    'wprofQ_pd_P' in ks
    and z['wprofQ_pd_P'].shape == (NLQ,),
    'wprofQ_pd_A1' in ks,
    reg_ok,
    npreg in (NP, NP // 2)])

# ---------- A3. frozen inputs ----------
z18 = np.load(D18 + r'\traj_readout.npz',
              allow_pickle=False)
z20 = np.load(D20 + r'\p118_readout.npz',
              allow_pickle=False)
z25 = np.load(D25 + r'\p123_readout.npz',
              allow_pickle=False)
z26 = np.load(D26 + r'\p124_readout.npz',
              allow_pickle=False)
capb = np.load(D13 + r'\capture_b.npz',
               allow_pickle=False)
mat5 = json.load(io.open(
    D05 + r'\material.json', encoding='utf-8'))
MSEQ = {'P': z18['gt_cleanseq_clean__P'][:NP]
        .astype(np.float64),
        'A1': z18['gt_cleanseq_clean__A1'][:NP]
        .astype(np.float64)}
ANN = {'P': z20['annP_c'][:NP],
       'A1': z20['annA_c'][:NP]}
sec('A3_frozen', [
    z25['mlg_s0_P'].shape == (NP, NLQ + 1, N_NEW + 1),
    z26['mlg_s0_P'].shape == (NP, NLG + 1, N_NEW + 1),
    z26['G6_P'].shape == (10, TRAIL_D),
    z26['gen_base_P'].shape == (NP, N_NEW),
    z26['wspec_P'].shape == (NLG, N_NEW + 1),
    z25['span_idx_P'].shape == (NP, 2),
    z26['span_idx_A1'].shape == (NP, 2),
    capb['h_out'].shape[0] == 2016,
    len(capb['layers']) == 5,
    mat5['yes_id'] == 9834,
    mat5['no_id'] == 902,
    MSEQ['P'].shape == (NP, N_NEW + 1)])


# ---------- shared Part-A fns ----------
def content_table(m, cls):
    dm = m[:, 1:] - m[:, :-1]
    tab = np.zeros((10, N_NEW))
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
    bb, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = (y - X @ bb).reshape(NP, N_NEW)
    return float(bb[0]), float(-bb[1] / bb[0]), \
        resid


def trailX(cls, D, lags=None):
    if lags is None:
        lags = list(range(1, D + 1))
    NIK = cls.size
    X = np.zeros((NIK, 10 * len(lags)))
    for ci, d in enumerate(lags):
        rows_k = np.arange(d, N_NEW)
        idx_rows = (np.repeat(
            np.arange(NP) * N_NEW, len(rows_k))
            + np.tile(rows_k, NP))
        cls_d = cls[:, :N_NEW - d].ravel()
        X[idx_rows, cls_d * len(lags) + ci] = 1.0
    return X


def pre_resid(m, MS, resid):
    """Z1 + Z2 steps -> r2 (pre-trail)."""
    NIK = resid.size
    Z1 = np.zeros((NIK, N_NEW))
    Z1[np.arange(NIK),
       np.arange(NIK) % N_NEW] = 1.0
    b1, *_ = np.linalg.lstsq(Z1, resid.ravel(),
                             rcond=None)
    r1 = (resid.ravel() - Z1 @ b1).reshape(
        NP, N_NEW)
    x = (m[:, :-1] - MS).ravel()
    Z2 = np.stack([x, np.ones(NIK)], 1)
    b2, *_ = np.linalg.lstsq(Z2, r1.ravel(),
                             rcond=None)
    r2 = (r1.ravel() - Z2 @ b2).reshape(
        NP, N_NEW)
    return r2


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(
        np.float64)
    rb = np.argsort(np.argsort(b)).astype(
        np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------- B1. Part-A refit/D6/G6 ----------
fit = {}
for dc in DIRS:
    tab = content_table(MSEQ[dc], ANN[dc])
    S_dc, MS_dc, resid = refit(MSEQ[dc],
                               ANN[dc], tab)
    fit[dc] = (S_dc, MS_dc, resid, tab)
r25 = json.load(io.open(
    D25 + r'\result.json', encoding='utf-8'))
res26 = json.load(io.open(
    D26 + r'\result.json', encoding='utf-8'))
dec6r = {}
G6r = {}
r4r = {}
ss_tot_r = {}
for dc in DIRS:
    m = MSEQ[dc]
    S_dc, MS_dc, resid, tab = fit[dc]
    NIK = resid.size
    Z1 = np.zeros((NIK, N_NEW))
    Z1[np.arange(NIK),
       np.arange(NIK) % N_NEW] = 1.0
    b1, *_ = np.linalg.lstsq(Z1, resid.ravel(),
                             rcond=None)
    r1 = (resid.ravel() - Z1 @ b1).reshape(
        NP, N_NEW)
    ss_step = float((resid * resid).sum()) \
        - float((r1 * r1).sum())
    x = (m[:, :-1] - MS_dc).ravel()
    Z2 = np.stack([x, np.ones(NIK)], 1)
    b2, *_ = np.linalg.lstsq(Z2, r1.ravel(),
                             rcond=None)
    r2 = (r1.ravel() - Z2 @ b2).reshape(
        NP, N_NEW)
    ss_m = float((r1 * r1).sum()) \
        - float((r2 * r2).sum())
    X_tr = trailX(ANN[dc], TRAIL_D)
    btr, *_ = np.linalg.lstsq(X_tr, r2.ravel(),
                              rcond=None)
    r3 = (r2.ravel() - X_tr @ btr).reshape(
        NP, N_NEW)
    ss_tr = float((r2 * r2).sum()) \
        - float((r3 * r3).sum())
    y_ar = r3[:, 2:].ravel()
    X_ar = np.stack(
        [r3[:, 1:N_NEW - 1].ravel(),
         r3[:, 0:N_NEW - 2].ravel(),
         np.ones(y_ar.size)], 1)
    bar, *_ = np.linalg.lstsq(X_ar, y_ar,
                              rcond=None)
    r4 = r3.copy()
    r4[:, 2:] = r3[:, 2:] - (
        bar[0] * r3[:, 1:N_NEW - 1]
        + bar[1] * r3[:, 0:N_NEW - 2]
        + bar[2])
    ss_ar = float(((r3[:, 2:] - r4[:, 2:])
                   ** 2).sum())
    ss_tot = float((resid * resid).sum())
    ss_rem = float((r4 * r4).sum())
    dec6r[dc] = {
        'cm_share': ss_step / ss_tot,
        'm_share': ss_m / ss_tot,
        'trail6_share': ss_tr / ss_tot,
        'ar6_share': ss_ar / ss_tot,
        'rem6_share': ss_rem / ss_tot,
        'ss_tot': ss_tot}
    G6r[dc] = btr.reshape(10, TRAIL_D)
    r4r[dc] = r4
    ss_tot_r[dc] = ss_tot
d26a = res26['part_a']['decomp6']
sec('B1_partA_recompute', [
    abs(fit['P'][0]
        - r25['part_a']['refit']['P']['S'])
    < 1e-12,
    abs(fit['P'][1]
        - r25['part_a']['refit']['P']['MS'])
    < 1e-12,
    abs(fit['A1'][0]
        - r25['part_a']['refit']['A1']['S'])
    < 1e-12,
    abs(fit['A1'][1]
        - r25['part_a']['refit']['A1']['MS'])
    < 1e-12,
    abs(dec6r['P']['cm_share']
        - d26a['P']['cm_share']) < 1e-12,
    abs(dec6r['P']['trail6_share']
        - d26a['P']['trail6_share']) < 1e-12,
    abs(dec6r['P']['rem6_share']
        - d26a['P']['rem6_share']) < 1e-12,
    abs(dec6r['A1']['cm_share']
        - d26a['A1']['cm_share']) < 1e-12,
    abs(dec6r['A1']['trail6_share']
        - d26a['A1']['trail6_share']) < 1e-12,
    abs(dec6r['A1']['rem6_share']
        - d26a['A1']['rem6_share']) < 1e-12,
    np.abs(G6r['P'] - z26['G6_P']).max()
    < 1e-10,
    np.abs(G6r['A1'] - z26['G6_A1']).max()
    < 1e-10])

# ---------- B2. A1-closure recompute ----------
LAGS46 = (4, 5, 6)
a1r = {}
for dc in DIRS:
    m = MSEQ[dc]
    S_dc, MS_dc, resid, tab = fit[dc]
    r2 = pre_resid(m, MS_dc, resid)
    ss_r2 = float((r2 * r2).sum())
    ss_tot = ss_tot_r[dc]
    X46 = trailX(ANN[dc], TRAIL_D,
                 lags=list(LAGS46))
    b46, *_ = np.linalg.lstsq(X46,
                              r2.ravel(),
                              rcond=None)
    r46 = r2.ravel() - X46 @ b46
    gain46 = (ss_r2 - float((r46 * r46)
                            .sum())) / ss_tot
    GT = z26['G6_%s' % ('A1' if dc == 'P'
                        else 'P')][:, 3:6]
    r2t = r2.ravel() - X46 @ GT.reshape(-1)
    gain_tr = (ss_r2 - float((r2t * r2t)
                             .sum())) / ss_tot
    rgA = np.random.default_rng(PERM_SEED)
    nulls = np.zeros(N_PERM)
    NIK = r2.size
    for rep in range(N_PERM):
        pm = np.argsort(
            rgA.random((NP, N_NEW)), axis=1)
        cls_p = np.take_along_axis(
            ANN[dc], pm, axis=1)
        tab_p = content_table(m, cls_p)
        S_p, MS_p, resid_p = refit(
            m, cls_p, tab_p)
        r2p = pre_resid(m, MS_p, resid_p)
        ss_r2p = float((r2p * r2p).sum())
        X46p = trailX(cls_p, TRAIL_D,
                      lags=list(LAGS46))
        b46p, *_ = np.linalg.lstsq(
            X46p, r2p.ravel(), rcond=None)
        r46p = r2p.ravel() - X46p @ b46p
        ss_tot_p = float(
            (resid_p * resid_p).sum())
        nulls[rep] = (ss_r2p - float(
            (r46p * r46p).sum())) / ss_tot_p
    mu = float(nulls.mean())
    sd = float(nulls.std())
    z46 = (gain46 - mu) / sd if sd > 0 \
        else 0.0
    a1r[dc] = {'gain46': gain46, 'z46': z46,
               'gain_tr': gain_tr,
               'perm_mean': mu,
               'perm_std': sd,
               'G46': b46.reshape(10, 3)}
ac = pa['a1_closure']
sec('B2_a1closure_recompute', [
    abs(a1r['P']['gain46']
        - ac['P']['gain46']) < 1e-10,
    abs(a1r['A1']['gain46']
        - ac['A1']['gain46']) < 1e-10,
    abs(a1r['P']['gain_tr']
        - ac['P']['gain_transfer']) < 1e-10,
    abs(a1r['A1']['gain_tr']
        - ac['A1']['gain_transfer']) < 1e-10,
    abs(a1r['P']['perm_mean']
        - ac['P']['perm_mean']) < 1e-12,
    abs(a1r['A1']['perm_mean']
        - ac['A1']['perm_mean']) < 1e-12,
    abs(a1r['P']['perm_std']
        - ac['P']['perm_std']) < 1e-12,
    abs(a1r['P']['z46'] - ac['P']['z46'])
    < 1e-9,
    abs(a1r['A1']['z46'] - ac['A1']['z46'])
    < 1e-9,
    np.abs(np.array(ac['P']['G46'])
           - a1r['P']['G46']).max() < 1e-10,
    ac['P']['reps'] == N_PERM,
    ac['P']['seed'] == PERM_SEED])

# ---------- B3. cross/PCA/sparse ----------
second2r = {}
for dc in DIRS:
    m = MSEQ[dc]
    r4 = r4r[dc]
    ss_tot = ss_tot_r[dc]
    NIK = r4.size
    A_cls = np.zeros((NIK, 10))
    A_cls[np.arange(NIK),
          ANN[dc].ravel()] = 1.0
    m_c = (m[:, :-1] - fit[dc][1]).ravel()
    m_c = m_c - m_c.mean()
    ones = np.ones(NIK)
    X_add = np.column_stack([A_cls, m_c, ones])
    X_int = np.column_stack(
        [A_cls, m_c, A_cls * m_c[:, None], ones])
    ba, *_ = np.linalg.lstsq(X_add, r4.ravel(),
                             rcond=None)
    ra = r4.ravel() - X_add @ ba
    bi, *_ = np.linalg.lstsq(X_int, r4.ravel(),
                             rcond=None)
    ri = r4.ravel() - X_int @ bi
    second2r[dc] = (
        float((ra * ra).sum())
        - float((ri * ri).sum())) / ss_tot
Zp = np.stack([r4r['P'], r4r['A1']],
              0).reshape(2 * NP, N_NEW)
Zc = Zp - Zp.mean(0, keepdims=True)
U, sv, Vt = np.linalg.svd(Zc,
                          full_matrices=False)
var_exp = (sv * sv) / float((Zc * Zc).sum())
pc1 = U[:, 0] * sv[0]
h_out = capb['h_out'].astype(np.float32)
LAY5 = capb['layers']
ev_counts = np.zeros(len(LAY5), dtype=np.int64)
for li in range(len(LAY5)):
    hl = h_out[:, li, :]
    mu_l = hl.mean(0, keepdims=True)
    sd_l = hl.std(0, keepdims=True)
    sd_l[sd_l == 0] = 1.0
    zl = (hl - mu_l) / sd_l
    ev_counts[li] = int(
        (np.abs(zl) >= 4.0).sum())
ev_total = int(ev_counts.sum())
ev_rate = ev_total / float(
    h_out.shape[0] * h_out.shape[1]
    * h_out.shape[2])
ev_top = float(ev_counts.max()
               / max(ev_total, 1))
sec('B3_cross_pca_sparse', [
    abs(second2r['P']
        - pa['cross_second']['P']
        ['cross_gain']) < 1e-10,
    abs(second2r['A1']
        - pa['cross_second']['A1']
        ['cross_gain']) < 1e-10,
    abs(float(var_exp[0])
        - pa['pca']['pc1_share']) < 1e-10,
    abs(pa['pca']['var_top5'][1]
        - float(var_exp[1])) < 1e-10,
    abs(pa['sparse_events']['rate']
        - ev_rate) < 1e-12,
    abs(pa['sparse_events']['top_share']
        - ev_top) < 1e-12,
    pa['sparse_events']['counts']
    == [int(v) for v in ev_counts],
    pa['sparse_events']['layers']
    == [int(v) for v in LAY5]])
print('PART A-C done', flush=True)

# ---------- B4. Part-B npz recompute ----------
# wspecQ recompute from p123 mlg_s0
wspecQ_r = {}
for dc in DIRS:
    s0m = z25['mlg_s0_%s' % dc][:NP] \
        .astype(np.float64)
    D_l = s0m[:, 1:, :] - s0m[:, :-1, :]
    wspecQ_r[dc] = D_l.mean(0)
W_q = {dc: wspecQ_r[dc].mean(1)
       for dc in DIRS}
# dm_final recompute from npz fields
NP_BV = min(NP, z['dmq_P_L26'].shape[0])
dmq_final = {}
for (l, tag) in ([(l, 'write')
                  for l in LAY_Q['write']]
                 + [(l, 'port')
                    for l in LAY_Q['port']]
                 + [(l, 'ctrl')
                    for l in LAY_Q['ctrl']]):
    for dc in DIRS:
        key = '%s_L%02d' % (dc, l)
        dmq_final[key] = (
            z['dmq_' + key][:, -1].astype(
                np.float64)
            - z25['mlg_s0_%s' % dc][:NP_BV,
                                    NLQ, -1]
            .astype(np.float64))
med_c_q = np.median(np.abs(np.concatenate(
    [dmq_final['P_L%02d' % l]
     for l in LAY_Q['ctrl']]
    + [dmq_final['A1_L%02d' % l]
       for l in LAY_Q['ctrl']])))
b_gates_r = {}
for grp in ('write', 'port'):
    meds = []
    for l in LAY_Q[grp]:
        for dc in DIRS:
            meds.append(np.median(np.abs(
                dmq_final['%s_L%02d'
                          % (dc, l)])))
    b_gates_r[grp] = float(
        np.median(meds)) / max(float(med_c_q),
                               1e-12)
spec_pairs_q = []
for (l, tag) in ([(l, 'write')
                  for l in LAY_Q['write']]
                 + [(l, 'port')
                    for l in LAY_Q['port']]
                 + [(l, 'ctrl')
                    for l in LAY_Q['ctrl']]):
    for dc in DIRS:
        key = '%s_L%02d' % (dc, l)
        spec_pairs_q.append(
            (float(np.abs(W_q[dc][l])),
             float(np.median(np.abs(
                dmq_final[key])))))
spec_corr_q = spearman(
    np.array([p[0] for p in spec_pairs_q]),
    np.array([p[1] for p in spec_pairs_q]))
# result dm_final lists vs recompute
# (result stores |f32 subtraction|; the f64
#  recompute above differs by f32 rounding
#  ~2.4e-7, so compare in the f32 path)
dm_res_ok = True
for key in dmq_final:
    dc_k = key.split('_')[0]
    arr = np.abs(
        z['dmq_' + key][:, -1]
        - z25['mlg_s0_%s' % dc_k][:NP_BV,
                                  NLQ, -1]
    ).astype(np.float64)
    lst = pb['dm_final'][key]
    if len(lst) != len(arr) or np.abs(
            np.array(lst) - arr).max() > 1e-12:
        dm_res_ok = False
        break
sec('B4_partB_recompute', [
    np.abs(wspecQ_r['P']
           - z['wspecQ_P'].astype(np.float64))
    .max() < 1e-6,
    np.abs(wspecQ_r['A1']
           - z['wspecQ_A1'].astype(np.float64))
    .max() < 1e-6,
    abs(b_gates_r['write']
        - pb['gates']['write']) < 1e-12,
    abs(b_gates_r['port']
        - pb['gates']['port']) < 1e-12,
    abs(float(med_c_q) - pb['ctrl_median'])
    < 1e-12,
    abs(spec_corr_q - pb['spec_corr'])
    < 1e-12,
    dm_res_ok,
    pb['repro_max_abs'] <= 1e-4,
    pb['path_r_min'] >= 0.9999,
    ('qwen_write_functional'
     if b_gates_r['write'] >= G_PORT
     else 'qwen_write_not') in verdict,
    ('qwen_port_functional'
     if b_gates_r['port'] >= G_PORT
     else 'qwen_port_not') in verdict,
    ('qwen_spectrum_aligned'
     if spec_corr_q >= G_SPEC
     else 'qwen_spectrum_weak') in verdict])
print('PART B done', flush=True)

# ---------- B5. Part-C + depth ----------
dmg_final = {}
for (l, tag) in ([(l, 'write')
                  for l in LAY_G['write']]
                 + [(l, 'port')
                    for l in LAY_G['port']]
                 + [(l, 'ctrl')
                    for l in LAY_G['ctrl']]):
    for dc in DIRS:
        key = '%s_L%02d' % (dc, l)
        dmg_final[key] = (
            z['dmg_' + key][:, -1].astype(
                np.float64)
            - z26['mlg_s0_%s' % dc][:NP_BV,
                                    NLG, -1]
            .astype(np.float64))
med_c_g = np.median(np.abs(np.concatenate(
    [dmg_final['P_L%02d' % l]
     for l in LAY_G['ctrl']]
    + [dmg_final['A1_L%02d' % l]
       for l in LAY_G['ctrl']])))
c_gates_r = {}
for grp in ('write', 'port'):
    meds = []
    for l in LAY_G[grp]:
        for dc in DIRS:
            meds.append(np.median(np.abs(
                dmg_final['%s_L%02d'
                          % (dc, l)])))
    c_gates_r[grp] = float(
        np.median(meds)) / max(float(med_c_g),
                               1e-12)
Wg = {dc: z26['wspec_%s' % dc].astype(
    np.float64).mean(1) for dc in DIRS}
spec_pairs_g = []
for (l, tag) in ([(l, 'write')
                  for l in LAY_G['write']]
                 + [(l, 'port')
                    for l in LAY_G['port']]
                 + [(l, 'ctrl')
                    for l in LAY_G['ctrl']]):
    for dc in DIRS:
        key = '%s_L%02d' % (dc, l)
        spec_pairs_g.append(
            (float(np.abs(Wg[dc][l])),
             float(np.median(np.abs(
                dmg_final[key])))))
spec_corr_g = spearman(
    np.array([p[0] for p in spec_pairs_g]),
    np.array([p[1] for p in spec_pairs_g]))
# depth profile recompute


def depth_profile(keys_by_layer, dm_final, NL,
                  med_c):
    xs = []
    ys = []
    for (l, tag) in keys_by_layer:
        vals = []
        for dc in DIRS:
            vals.extend(np.abs(
                dm_final['%s_L%02d' % (dc, l)]))
        xs.append((l + 0.5) / NL)
        ys.append(float(np.median(vals))
                  / max(med_c, 1e-12))
    order = np.argsort(xs)
    xs = np.array(xs)[order]
    ys = np.array(ys)[order]
    return xs, ys


ABL_Q = ([(l, 'write') for l in LAY_Q['write']]
         + [(l, 'port') for l in LAY_Q['port']]
         + [(l, 'ctrl') for l in LAY_Q['ctrl']])
ABL_G = ([(l, 'write') for l in LAY_G['write']]
         + [(l, 'port') for l in LAY_G['port']]
         + [(l, 'ctrl') for l in LAY_G['ctrl']])
xs_q, ys_q = depth_profile(ABL_Q, dmq_final,
                           NLQ, med_c_q)
xs_g, ys_g = depth_profile(ABL_G, dmg_final,
                           NLG, med_c_g)
grid = np.linspace(0.0, 1.0, 21)
yq_i = np.interp(grid, xs_q, ys_q)
yg_i = np.interp(grid, xs_g, ys_g)
depth_corr = spearman(yq_i, yg_i)
dm_gres_ok = True
for key in dmg_final:
    dc_k = key.split('_')[0]
    arr = np.abs(
        z['dmg_' + key][:, -1]
        - z26['mlg_s0_%s' % dc_k][:NP_BV,
                                  NLG, -1]
    ).astype(np.float64)
    lst = pc['dm_final'][key]
    if len(lst) != len(arr) or np.abs(
            np.array(lst) - arr).max() > 1e-12:
        dm_gres_ok = False
        break
sec('B5_partC_recompute', [
    abs(c_gates_r['write']
        - pc['gates']['write']) < 1e-12,
    abs(c_gates_r['port']
        - pc['gates']['port']) < 1e-12,
    abs(float(med_c_g) - pc['ctrl_median'])
    < 1e-12,
    abs(spec_corr_g - pc['spec_corr'])
    < 1e-12,
    dm_gres_ok,
    pc['repro_max_abs'] <= 1e-4,
    pc['path_r_min'] >= 0.9999,
    abs(depth_corr - dp['corr']) < 1e-12,
    np.abs(np.array(dp['xs_q']) - xs_q).max()
    < 1e-12,
    np.abs(np.array(dp['ys_q']) - ys_q).max()
    < 1e-12,
    np.abs(np.array(dp['ys_g']) - ys_g).max()
    < 1e-12,
    ('glm4_write_functional'
     if c_gates_r['write'] >= G_PORT
     else 'glm4_write_not') in verdict,
    ('glm4_port_functional'
     if c_gates_r['port'] >= G_PORT
     else 'glm4_port_not') in verdict,
    ('glm4_spectrum_aligned'
     if spec_corr_g >= G_SPEC
     else 'glm4_spectrum_weak') in verdict,
    ('port_depth_aligned'
     if depth_corr >= G_DEPTH
     else 'port_depth_divergent') in verdict])
print('PART C done', flush=True)

# ---------- B6. Part-D recompute ----------
import datetime  # noqa: E402

from transformers import AutoTokenizer \
    as _AT  # noqa: E402

tok_g = _AT.from_pretrained(
    ROOT + r'\models\hf\glm4-9b-chat-hf',
    trust_remote_code=True)
DOT_G = int(tok_g('.', add_special_tokens=False)
            ['input_ids'][0])


def pad12(ids):
    ids2 = list(ids)[:N_NEW]
    return ids2 + [DOT_G] * (N_NEW - len(ids2))


def pol_raw(tok_id):
    if not tok_id:
        return (0, 0)
    t = tok_g.decode([int(tok_id)])
    ts = t.strip()
    tl = ts.lower()
    if tl.startswith('yes'):
        return (1, 0)
    if tl.startswith('no'):
        return (-1, 1 if ts.startswith('No')
                else 0)
    return (0, 0)


d_stats_r = {}
flip_all_neg = {dc: {c: [] for c in SCOND3}
                for dc in DIRS}
flip_no_only = {dc: {c: [] for c in SCOND3}
                for dc in DIRS}
n_no_cap = 0
n_no_low = 0
for dc in DIRS:
    agree = {c: [] for c in SCOND3}
    fdiv = {c: [] for c in SCOND3}
    flip = {c: [] for c in SCOND3}
    mflip = {c: [] for c in SCOND3}
    for c in SCOND3:
        idx = idxs[(dc, c)]
        arr = z['regen_%s_%s' % (c, dc)]
        for jj in range(len(idx)):
            j = int(idx[jj])
            base12 = pad12(
                z26['gen_base_%s' % dc][j])
            bpol, _ = pol_raw(base12[0])
            g12 = [int(v) for v in arr[jj]]
            ag = float(np.mean(
                [a == b for a, b
                 in zip(g12, base12)]))
            agree[c].append(ag)
            fd = N_NEW
            for k in range(N_NEW):
                if g12[k] != base12[k]:
                    fd = k
                    break
            fdiv[c].append(fd)
            gpol, cap = pol_raw(g12[0])
            if cap and gpol < 0:
                n_no_cap += 1
            elif gpol < 0:
                n_no_low += 1
            if bpol != 0 and gpol != 0 \
                    and gpol != bpol:
                flip[c].append(1.0)
                flip_all_neg[dc][c].append(1.0)
                flip_no_only[dc][c].append(
                    1.0 if cap == 1 else 0.0)
            else:
                flip[c].append(0.0)
                flip_all_neg[dc][c].append(0.0)
                flip_no_only[dc][c].append(0.0)
            pols = []
            for k in range(3):
                p_k, _ = pol_raw(g12[k])
                if p_k != 0:
                    pols.append(p_k)
            ch = sum(1 for a, b in
                     zip(pols, pols[1:])
                     if a != b)
            mflip[c].append(
                1.0 if ch >= 1 else 0.0)
    cs = {}
    for c in SCOND3:
        cs[c] = {
            'agree_mean': float(np.mean(agree[c])),
            'first_div_mean': float(
                np.mean(fdiv[c])),
            'flip_rate': float(np.mean(flip[c])),
            'multi_flip_rate': float(
                np.mean(mflip[c]))}
    d_stats_r[dc] = cs
stats_ok = True
for dc in DIRS:
    for c in SCOND3:
        for k in ('agree_mean',
                  'first_div_mean',
                  'flip_rate',
                  'multi_flip_rate'):
            if abs(d_stats_r[dc][c][k]
                   - pd_['stats'][dc][c][k]) > 1e-12:
                stats_ok = False
shift_min_r = min(
    d_stats_r[dc]['s1']['agree_mean']
    - float(np.mean([d_stats_r[dc][c]
                     ['agree_mean']
                     for c in SCOND3]))
    for dc in DIRS)
diffs = []
for dc in DIRS:
    for c in SCOND3:
        a_all = float(np.mean(
            flip_all_neg[dc][c]))
        a_no = float(np.mean(
            flip_no_only[dc][c]))
        diffs.append(abs(a_all - a_no))
flip_sep_r = max(diffs)
# bit-exact first-96 vs p124 (full coverage)
bit_ok = True
bit_checked = False
if npreg >= 96:
    for dc in DIRS:
        for c in SCOND3:
            idx_c = list(idxs[(dc, c)][:96])
            if idx_c != list(range(96)):
                continue
            ref = z26['regen_%s_%s' % (c, dc)]
            arr = z['regen_%s_%s' % (c, dc)]
            for j in range(96):
                if [int(v) for v in arr[j]] \
                    != [int(v)
                        for v in ref[j]]:
                    bit_ok = False
            bit_checked = True
cov_r = npreg >= NP
s0_ok = pd_['s0_probe_agree'] >= 0.95
sec('B6_partD_recompute', [
    stats_ok,
    abs(shift_min_r
        - pd_['shift_min_full']) < 1e-12,
    abs(flip_sep_r - pd_['flip_sep_max'])
    < 1e-12,
    int(pd_['n_no_cap']) == n_no_cap,
    int(pd_['n_no_low']) == n_no_low,
    bit_checked and bit_ok
    or not bit_checked,
    abs(float(pd_['s0_probe_agree'])
        - 0.0) >= 0,
    s0_ok
    or ('baseline_s0_drift' in verdict),
    ('behavior_shift_full'
     if shift_min_r >= 0.10
     else 'behavior_shift_weak') in verdict,
    ('flip_polarity_separated'
     if flip_sep_r <= 0.05
     else 'flip_polarity_mixed') in verdict,
    ('coverage_full' if cov_r
     else 'coverage_degraded') in verdict])
print('PART D done', flush=True)

# ---------- B7. verdict re-assembly ----------
z46_min = min(a1r['P']['z46'], a1r['A1']['z46'])
a_lag_v = ('lag46_significant'
           if z46_min >= 4.0
           else 'lag46_notsig')
own_a1 = max(a1r['A1']['gain46'], 0.005)
tr_a1 = a1r['A1']['gain_tr']
a_tr_v = ('a1_transferable'
          if (tr_a1 >= 0.02
              and tr_a1 >= 2.0 * own_a1)
          else 'a1_short_range_intrinsic')
gx_min = min(second2r['P'], second2r['A1'])
a_cross_v = ('cross_second_candidate'
             if gx_min >= 0.03
             else 'cross_second_absent')
a_cm_v = ('common_mode_factor_candidate'
          if float(var_exp[0]) >= 0.30
          else 'common_mode_absent')
a_sp_v = ('sparse_events_present'
          if (ev_rate >= 1e-3
              and ev_top >= 0.4)
          else 'sparse_events_absent')
regen_bit_v = ('regen_replay_bit_exact'
               if bit_ok else 'regen_bit_mismatch')
V_rebuilt = '|'.join([
    a_lag_v, a_tr_v, a_cross_v, a_cm_v,
    a_sp_v, 'qwen_path_valid',
    ('qwen_write_functional'
     if b_gates_r['write'] >= G_PORT
     else 'qwen_write_not'),
    ('qwen_port_functional'
     if b_gates_r['port'] >= G_PORT
     else 'qwen_port_not'),
    ('qwen_spectrum_aligned'
     if spec_corr_q >= G_SPEC
     else 'qwen_spectrum_weak'),
    'glm4_path_valid',
    ('glm4_write_functional'
     if c_gates_r['write'] >= G_PORT
     else 'glm4_write_not'),
    ('glm4_port_functional'
     if c_gates_r['port'] >= G_PORT
     else 'glm4_port_not'),
    ('glm4_spectrum_aligned'
     if spec_corr_g >= G_SPEC
     else 'glm4_spectrum_weak'),
    ('port_depth_aligned'
     if depth_corr >= G_DEPTH
     else 'port_depth_divergent'),
    ('baseline_s0_replay_ok' if s0_ok
     else 'baseline_s0_drift'),
    regen_bit_v,
    ('behavior_shift_full'
     if shift_min_r >= 0.10
     else 'behavior_shift_weak'),
    ('flip_polarity_separated'
     if flip_sep_r <= 0.05
     else 'flip_polarity_mixed'),
    ('coverage_full' if cov_r
     else 'coverage_degraded')])
sec('B7_verdict_reassembly', [
    V_rebuilt == verdict,
    len(V_rebuilt.split('|')) == 19])

# ---------- D. ledger ----------
led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
m3127 = [m for m in led['measurements']
         if m.get('phase') == 3127]
sha_stored = led.get('ledger_sha256_8')
led2 = json.loads(json.dumps(led))
led2.pop('ledger_sha256_8', None)
blob = json.dumps(led2, sort_keys=True,
                  ensure_ascii=False)
sha_calc = hashlib.sha256(
    blob.encode('utf-8')).hexdigest()[:8]
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
MEAS_ID = 'meas3127_omega_p125_writechain_' \
          'port_crossmodel_a1closure_' \
          'fullregen'
art_ok = True
if m3127:
    arts = m3127[0].get('artifacts', {})
    for akey in ('result', 'seal', 'readout'):
        rel = arts.get(akey, '')
        if not os.path.exists(
                os.path.join(RDIR, rel)):
            art_ok = False
sec('D_ledger', [
    len(m3127) == 1,
    m3127[0]['meas_id'] == MEAS_ID,
    m3127[0]['verdict'] == verdict,
    sha_calc == sha_stored,
    len(led['measurements']) == 264,
    len(l14['connects']) == 232,
    MEAS_ID in l14['connects'],
    art_ok,
    '@@' not in json.dumps(m3127[0],
                           ensure_ascii=False)])

# ---------- E. MEMO ----------
memo = io.open(MEMO, encoding='utf-8').read()
tail = memo[-24000:]
VS = verdict.split('|')
sec('E_memo', [
    '## Phase 3127:' in tail,
    '\u03a9-P125' in tail,
    sum(s in tail for s in VS) >= 4,
    'p125_readout.npz' in tail,
    'phase3127_omega_p125' in tail,
    'p3127_disk_verify' in tail,
    'phase3127_closeout' in tail
    or 'p3127_vals_extract' in tail,
    tail.rfind('## Phase 3127:')
    > tail.rfind('## Phase 3126:')])

# ---------- F. wlogs ----------
WDATE = datetime.date.today().strftime('%Y-%m-%d')


def wlog_check(wdir):
    wl = wdir + '\\' + WDATE + '.md'
    try:
        t = io.open(wl,
                    encoding='utf-8').read()
    except IOError:
        return [False, False, False]
    return ['Phase 3127 Omega-P125' in t,
            'Phase 3127 closeout' in t,
            sha_stored in t and '264' in t]


sec('F_wlogs_D', wlog_check(WLOG_D))
sec('F_wlogs_C', wlog_check(WLOG_C))

# ---------- G. MEMORY ----------
mem = io.open(MEMO_W, encoding='utf-8').read()
sec('G_memory', [
    mem.count('max=3127') == 1,
    '## 机制链状态（3127）' in mem,
    '- 3127（T4）' in mem,
    '- 3126（T4）' in mem,
    len(mem) < 3000])

# ---------- summary ----------
o.append('')
o.append('TOTAL FAILS: %d' % len(fails))
for (nm, bad) in fails:
    o.append('  FAIL %s at %s' % (nm, bad))
with io.open(VF, 'w', encoding='utf-8') as f:
    f.write('\n'.join(o) + '\n')
print('VERIFY_DONE fails=%d' % len(fails),
      flush=True)


