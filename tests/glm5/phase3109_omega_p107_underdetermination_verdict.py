# -*- coding: utf-8 -*-
"""Phase 3109 (Omega-P107): Underdetermination verdict.

T3 line, offline (frozen 3105 capture, no GPU).  3108 found the
ridge truth-readout is a wide, flat, degenerate solution family
(half-split Top-200 Jaccard 0.11, bootstrap solutions pairwise
|cos| 0.041, PR_norm 0.85, spectrum Gini 0.10) and proposed the
FIRST-PRINCIPLES explanation: the system is UNDERDETERMINED
(n_train=1101 < d=2560, solution space >= 1459 dims), so the L2
minimum-norm solution rotates freely under resampling.  This
phase runs the three pre-registered decisive experiments:

  (1) N-SWEEP (D1): subsample the 3105 train set to
      n in {100, 200, 400, 800, 1101}; at each n measure
      (a) bootstrap subspace floor (16 boot, top-8 singular
      subspace, 5 groupings, mean cos^2), (b) half-split
      Top-200 Jaccard (3 seeds), (c) Top-200 PR_norm.
      Prediction of the underdetermination picture: metrics
      stabilize monotonically as n approaches d.
      SUPPORT if Spearman(n, floor) >= 0.9 AND
      Spearman(n, J) >= 0.9 (5 points each, supporting not
      decisive).
  (2) RANDOM CONTROL (D2, decisive): refit the probe on
      RANDOM 100/200-coordinate subspaces (5 seeds) and
      compare test AUC against the |w|-Top-100/200 refit.
      'random_sufficient' if median random AUC >=
      |w|-Top-K AUC - 0.02 for both K -> truth readout does
      not need specific coordinates (broad distribution of
      readout information across the coordinate space);
      'specific_coordinates_needed' otherwise.
  (3) FUNCTIONAL EQUIVALENCE (D3, auxiliary): half-A |w|-Top-K
      coordinates refit on half-B, test AUC, vs half-B's own
      Top-K refit.  'transfer_ok' if A-selection AUC >=
      B-own AUC - 0.02.

VERDICT (frozen):
  D1 support AND D2 random_sufficient ->
      underdetermined_cone_confirmed
  D1 support AND D2 specific ->
      underdetermined_amplifier_only
  D1 not supported -> stability_not_n_limited
  D3 reported alongside.

SMOKE=1: n in {100, 1101}, boot 6, 2 seeds, 3 groupings.
Output: tests/glm5/result/
rdc_query_construction_20260913/phase3109/
omega_p107_underdetermination_verdict/
"""
import gc
import io
import json
import os
import time
from datetime import datetime

import numpy as np

SMOKE = os.environ.get('SMOKE', '0') == '1'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
R13 = (ROOT + r'\tests\glm5\result'
       r'\rdc_query_construction_20260913')
NAME = 'omega_p107_underdetermination_verdict'
OUT = os.path.join(R13, 'phase3109', NAME)
if SMOKE:
    OUT = os.path.join(OUT, 'smoke')
os.makedirs(OUT, exist_ok=True)
LOGF = os.path.join(OUT, 'run_log.txt')
T0 = time.time()

SEED = 31090
C5 = (R13 + r'\phase3105'
      r'\omega_p103_incontext_truth_consistency')

N_SWEEP = [100, 1101] if SMOKE else \
    [100, 200, 400, 800, 1101]
N_BOOT = 6 if SMOKE else 16
N_JSEEDS = 2 if SMOKE else 3
N_GROUPINGS = 3 if SMOKE else 5
R_SUB = 3 if SMOKE else 8
TOPN = 200
K_LIST = [50, 200] if SMOKE else [100, 200]
N_RAND = 2 if SMOKE else 5
MARGIN = 0.02
D1_RHO = 0.9

POSITIONS = ['pos0', 'crit_pred', 'crit_obj',
             'query_pred', 'query_obj', 'last']


def log(msg):
    line = '[%7.1fs] %s' % (time.time() - T0, msg)
    with io.open(LOGF, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


log('Phase 3109 Omega-P107 start; SMOKE=%d OUT=%s'
    % (SMOKE, OUT))

# ================================================================
# 0. Design seal (pre-computation)
# ================================================================
design = {
    'phase': 3109, 'name': NAME,
    'created': datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S'),
    'smoke': SMOKE, 'seed': SEED,
    'config': '3105 C1 last|L8, main records, '
              'lambda=0.01 (recorded), standardization '
              'frozen from FULL train (mu/sd of all '
              '1101 train records)',
    'n_sweep': N_SWEEP,
    'n_boot': N_BOOT, 'r_sub': R_SUB,
    'n_jseeds': N_JSEEDS,
    'n_groupings_floor': N_GROUPINGS,
    'topn': TOPN, 'k_list': K_LIST, 'n_rand': N_RAND,
    'margin': MARGIN,
    'd1_rule': 'support if Spearman(n, floor) >= 0.9 '
               'AND Spearman(n, J_half) >= 0.9 '
               '(supporting, not decisive)',
    'd2_rule': "random_sufficient iff median random "
               'AUC >= |w|-TopK AUC - 0.02 for BOTH K '
               'in {100, 200}; else '
               'specific_coordinates_needed',
    'd3_rule': "transfer_ok iff A-half Top-K refit on "
               'B-half AUC >= B-own Top-K AUC - 0.02 '
               '(auxiliary)',
    'verdict_map': {
        'D1 & D2 random_sufficient':
            'underdetermined_cone_confirmed',
        'D1 & D2 specific':
            'underdetermined_amplifier_only',
        '!D1': 'stability_not_n_limited'},
}
with io.open(os.path.join(OUT, 'design_seal.json'),
             'w', encoding='utf-8') as f:
    json.dump(design, f, indent=1)
log('design sealed (pre-computation)')

# ================================================================
# 1. Load frozen capture, build full-train Z space
# ================================================================
z = np.load(os.path.join(C5, 'capture.npz'),
            allow_pickle=False)
X = z['X'][:, POSITIONS.index('last'), 8, :] \
    .astype(np.float32)
y = z['truth'].astype(np.int64)
tag = [x.decode() if isinstance(x, bytes) else x
       for x in z['tag']]
keep = np.array([t == 'main' for t in tag])
X, y = X[keep], y[keep]
sp = [x.decode() if isinstance(x, bytes) else x
      for x in z['split'][keep]]
tr_full = np.where(np.array(sp) == 'train')[0]
te = np.where(np.array(sp) == 'test')[0]
lam = 0.01
mu = X[tr_full].mean(0)
sd = X[tr_full].std(0) + 1e-6
Z = (X - mu) / sd
Yv = (y * 2.0 - 1.0).astype(np.float32)
Zte, yte = Z[te], y[te]
log('capture loaded: n_train_full=%d n_test=%d '
    'lam=%g' % (len(tr_full), len(te), lam))


def ridge_fit(Zm, yv, lamv):
    n, d = Zm.shape
    A = (Zm.T @ Zm) / n \
        + lamv * np.eye(d, dtype=np.float32)
    return np.linalg.solve(A, (Zm.T @ yv) / n) \
        .astype(np.float32)


def ridge_fit_cols(Zm, yv, cols, lamv):
    Zs = Zm[:, cols]
    return ridge_fit(Zs, yv, lamv)


def auc_score(yy, s):
    order = np.argsort(s, kind='mergesort')
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    n1 = float((yy == 1).sum())
    n0 = float((yy == 0).sum())
    if n1 == 0 or n0 == 0:
        return None
    return float((ranks[yy == 1].sum()
                  - n1 * (n1 + 1) / 2.0)
                 / (n1 * n0))


def top_coords(w, k):
    return np.sort(np.argsort(-np.abs(w))[:k])


def jaccard(a, b):
    sa, sb = set(a.tolist()), set(b.tolist())
    return len(sa & sb) / max(1, len(sa | sb))


def svd_left(M, r):
    u, _, _ = np.linalg.svd(M, full_matrices=False)
    return u[:, :r].astype(np.float64)


def mean_cos2(Ua, Ub):
    s = np.linalg.svd(Ua.T @ Ub, compute_uv=False)
    return float(np.mean(s ** 2))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    if den == 0:
        return float('nan')
    return float((ra * rb).sum() / den)


# ================================================================
# 2. D1 n-sweep
# ================================================================
sweep = {}
for si, n in enumerate(N_SWEEP):
    rng = np.random.RandomState(SEED + 7 * si)
    sub = rng.choice(tr_full, size=min(n, len(tr_full)),
                     replace=False)
    Zs, Ys = Z[sub], Yv[sub]
    w_full = ridge_fit(Zs, Ys, lam)
    v = np.sort(np.abs(w_full))[::-1][:TOPN] ** 2
    prn = float(v.sum() ** 2 / (v ** 2).sum()) / TOPN
    # half-split Jaccard
    Js = []
    for s in range(N_JSEEDS):
        rj = np.random.RandomState(
            SEED + 100 + 11 * s + si)
        perm = rj.permutation(len(sub))
        h = len(sub) // 2
        iA, iB = sub[perm[:h]], sub[perm[h:2 * h]]
        wA = ridge_fit(Z[iA], Yv[iA], lam)
        wB = ridge_fit(Z[iB], Yv[iB], lam)
        Js.append(jaccard(top_coords(wA, TOPN),
                          top_coords(wB, TOPN)))
    Jmean = float(np.mean(Js))
    # bootstrap floor
    rb = np.random.RandomState(SEED + 500 + si)
    cols = []
    for b in range(N_BOOT):
        idx = rb.choice(sub, size=len(sub),
                        replace=True)
        cols.append(ridge_fit(Z[idx], Yv[idx], lam))
    Wb = np.stack(cols, axis=1)
    fl = []
    for g in range(N_GROUPINGS):
        rg = np.random.RandomState(
            SEED + 900 + 17 * g + si)
        perm = rg.permutation(N_BOOT)
        ha = N_BOOT // 2
        Ua = svd_left(Wb[:, perm[:ha]], R_SUB)
        Ub = svd_left(Wb[:, perm[ha:2 * ha]], R_SUB)
        fl.append(mean_cos2(Ua, Ub))
    floor = float(np.mean(fl))
    sweep['n%d' % n] = {
        'pr_norm': round(prn, 4),
        'j_half_mean': round(Jmean, 4),
        'j_half_values': [round(x, 4) for x in Js],
        'floor': round(floor, 4)}
    log('D1 n=%d: PR_norm=%.4f J=%.4f floor=%.4f'
        % (n, prn, Jmean, floor))
    del Wb
    gc.collect()

ns = np.array(N_SWEEP, dtype=np.float64)
fl_v = [sweep['n%d' % n]['floor'] for n in N_SWEEP]
j_v = [sweep['n%d' % n]['j_half_mean']
       for n in N_SWEEP]
rho_floor = spearman(ns, fl_v)
rho_j = spearman(ns, j_v)
D1 = (rho_floor >= D1_RHO) and (rho_j >= D1_RHO)
log('D1: rho_floor=%.3f rho_J=%.3f -> %s'
    % (rho_floor, rho_j, D1))

# ================================================================
# 3. D2 random control (decisive)
# ================================================================
w_full = ridge_fit(Z[tr_full], Yv[tr_full], lam)
d2 = {}
for K in K_LIST:
    sel = top_coords(w_full, K)
    wk = ridge_fit_cols(Z[tr_full], Yv[tr_full],
                        sel, lam)
    auc_top = auc_score(yte, Zte[:, sel] @ wk)
    rand_aucs = []
    for r in range(N_RAND):
        rr = np.random.RandomState(
            SEED + 2000 + 31 * r + K)
        rsel = np.sort(rr.choice(2560, size=K,
                                 replace=False))
        wr = ridge_fit_cols(Z[tr_full], Yv[tr_full],
                            rsel, lam)
        rand_aucs.append(auc_score(
            yte, Zte[:, rsel] @ wr))
    med = float(np.median(rand_aucs))
    ok = med >= auc_top - MARGIN
    d2['K%d' % K] = {
        'auc_topk': round(auc_top, 4),
        'rand_aucs': [round(a, 4) for a in rand_aucs],
        'rand_median': round(med, 4),
        'random_sufficient': bool(ok)}
    log('D2 K=%d: topk=%.4f rand_median=%.4f '
        '(%s) -> %s'
        % (K, auc_top, med,
           ['%.3f' % a for a in rand_aucs], ok))
D2 = all(d2['K%d' % K]['random_sufficient']
         for K in K_LIST)

# ================================================================
# 4. D3 functional equivalence (auxiliary)
# ================================================================
d3 = {}
for K in K_LIST:
    rj = np.random.RandomState(SEED + 3000 + K)
    perm = rj.permutation(len(tr_full))
    h = len(tr_full) // 2
    iA, iB = tr_full[perm[:h]], tr_full[perm[h:]]
    wA = ridge_fit(Z[iA], Yv[iA], lam)
    sA = top_coords(wA, K)
    w_xfer = ridge_fit_cols(Z[iB], Yv[iB], sA, lam)
    auc_xfer = auc_score(yte, Zte[:, sA] @ w_xfer)
    wB = ridge_fit(Z[iB], Yv[iB], lam)
    sB = top_coords(wB, K)
    w_own = ridge_fit_cols(Z[iB], Yv[iB], sB, lam)
    auc_own = auc_score(yte, Zte[:, sB] @ w_own)
    ok = auc_xfer >= auc_own - MARGIN
    d3['K%d' % K] = {
        'auc_xfer': round(auc_xfer, 4),
        'auc_own': round(auc_own, 4),
        'jaccard_sA_sB': round(jaccard(sA, sB), 4),
        'transfer_ok': bool(ok)}
    log('D3 K=%d: xfer=%.4f own=%.4f '
        'J(sA,sB)=%.4f -> %s'
        % (K, auc_xfer, auc_own,
           d3['K%d' % K]['jaccard_sA_sB'], ok))

# ================================================================
# 5. Verdict & save
# ================================================================
if D1 and D2:
    verdict = 'underdetermined_cone_confirmed'
elif D1:
    verdict = 'underdetermined_amplifier_only'
else:
    verdict = 'stability_not_n_limited'
log('VERDICT: %s (D1=%s D2=%s)' % (verdict, D1, D2))

results = {
    'verdict': verdict,
    'gates': {
        'D1_nsweep': {
            'per_n': sweep,
            'rho_floor': round(rho_floor, 4),
            'rho_j': round(rho_j, 4),
            'supported': bool(D1)},
        'D2_random_control': d2,
        'D3_functional_transfer': d3},
    'created': datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S'),
}
with io.open(os.path.join(OUT, 'result.json'), 'w',
             encoding='utf-8') as f:
    json.dump(results, f, indent=1)
log('result.json written; verdict=%s' % verdict)
print('PHASE3109_DONE verdict=%s D1=%s D2=%s'
      % (verdict, D1, D2))
