# -*- coding: utf-8 -*-
"""Phase 3110 (Omega-P108): K-sweep minimum readout-port
dimension + cross-variable specificity control.

T3 line, offline (frozen 3105 capture, no GPU).  3109 showed
the truth variable is DIFFUSELY REDUNDANT: random 100/200-dim
subspaces read it out at AUC ~1.0, indistinguishable from
|w|-TopK.  Two open items from the 3109 memo:

  (1) K-SWEEP (quantitative diffuseness): random K-dim
      subspaces for K in {5,10,20,50,100,200,400,800,1600,
      2560}, R seeds each, ridge refit at lambda=0.01, test
      AUC.  d_min = smallest K whose MEDIAN AUC >= 0.95 x
      full-dim AUC.  Curve shape bands (frozen):
        sharp_core : d_min <= 20   (cone has a wide core;
                                    even tiny subspaces work)
        broad      : 20 < d_min <= 200
        full_space : d_min > 200   (needs near-full space)
      Seed-wise spread reported (10 seeds) so 'every
      subspace works' is checked, not just the median.
  (2) CROSS-VARIABLE SPECIFICITY (frozen control): the SAME
      random subspaces probe a DIFFERENT variable, crit_rel
      (8-way relation identity, one-vs-rest mean AUC).
      'truth_specific_diffuseness' if d_min_rel >= 3 x
      d_min_truth OR the full-dim crit_rel baseline at this
      position has mean AUC < 0.9; else
      'diffuseness_general' (descriptive either way).

METHOD: 3105 C1 (last|L8, main records), lambda=0.01
(recorded), per-dim standardization frozen from full train.
The SAME random subsets feed both targets per (K, seed).
K=2560 is the deterministic full-dimensional fit.

SMOKE=1: K in {10,100,2560}, R=3.  Output: tests/glm5/
result/rdc_query_construction_20260913/phase3110/
omega_p108_ksweep_minport/
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
NAME = 'omega_p108_ksweep_minport'
OUT = os.path.join(R13, 'phase3110', NAME)
if SMOKE:
    OUT = os.path.join(OUT, 'smoke')
os.makedirs(OUT, exist_ok=True)
LOGF = os.path.join(OUT, 'run_log.txt')
T0 = time.time()

SEED = 31100
C5 = (R13 + r'\phase3105'
      r'\omega_p103_incontext_truth_consistency')
K_LIST = [10, 100, 2560] if SMOKE else \
    [5, 10, 20, 50, 100, 200, 400, 800, 1600, 2560]
N_RAND = 3 if SMOKE else 10
D_MIN_FRAC = 0.95
BANDS = {'sharp_core': 20, 'broad': 200}
SPEC_RATIO = 3.0
SPEC_BASELINE = 0.9

POSITIONS = ['pos0', 'crit_pred', 'crit_obj',
             'query_pred', 'query_obj', 'last']


def log(msg):
    line = '[%7.1fs] %s' % (time.time() - T0, msg)
    with io.open(LOGF, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


log('Phase 3110 Omega-P108 start; SMOKE=%d OUT=%s'
    % (SMOKE, OUT))

# ================================================================
# 0. Design seal (pre-computation)
# ================================================================
design = {
    'phase': 3110, 'name': NAME,
    'created': datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S'),
    'smoke': SMOKE, 'seed': SEED,
    'config': '3105 C1 last|L8, main records, '
              'lambda=0.01 (recorded), standardization '
              'frozen from FULL train (same as 3108/3109)',
    'k_list': K_LIST, 'n_rand': N_RAND,
    'd_min_rule': 'smallest K with median AUC >= 0.95 x '
                  'full-dim AUC',
    'curve_bands': {'sharp_core': '<=20', 'broad':
                    '20-200', 'full_space': '>200'},
    'shared_subsets': 'same random subsets feed truth '
                      'and crit_rel targets per (K, seed)',
    'cross_var_rule': 'truth_specific_diffuseness iff '
                      'd_min_rel >= 3 x d_min_truth OR '
                      'crit_rel full-dim baseline mean '
                      'AUC < 0.9; else '
                      'diffuseness_general (descriptive)',
}
with io.open(os.path.join(OUT, 'design_seal.json'),
             'w', encoding='utf-8') as f:
    json.dump(design, f, indent=1)
log('design sealed (pre-computation)')

# ================================================================
# 1. Load frozen capture, build Z space (3105 last|L8)
# ================================================================
z = np.load(os.path.join(C5, 'capture.npz'),
            allow_pickle=False)
X = z['X'][:, POSITIONS.index('last'), 8, :] \
    .astype(np.float32)
y = z['truth'].astype(np.int64)
cr = z['crit_rel'].astype(np.int64)
tag = [x.decode() if isinstance(x, bytes) else x
       for x in z['tag']]
keep = np.array([t == 'main' for t in tag])
X, y, cr = X[keep], y[keep], cr[keep]
sp = [x.decode() if isinstance(x, bytes) else x
      for x in z['split'][keep]]
tr = np.where(np.array(sp) == 'train')[0]
te = np.where(np.array(sp) == 'test')[0]
lam = 0.01
mu = X[tr].mean(0)
sd = X[tr].std(0) + 1e-6
Z = (X - mu) / sd
Yv = (y * 2.0 - 1.0).astype(np.float32)
Yrel = np.eye(8, dtype=np.float32)[cr]
Zte, yte, crte = Z[te], y[te], cr[te]
log('capture loaded: n_train=%d n_test=%d lam=%g'
    % (len(tr), len(te), lam))


def ridge_fit(Zm, Ym, lamv):
    n, d = Zm.shape
    A = (Zm.T @ Zm) / n \
        + lamv * np.eye(d, dtype=np.float32)
    return np.linalg.solve(A, (Zm.T @ Ym) / n) \
        .astype(np.float32)


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


def mean_ovr_auc(y_multi, S):
    """S: (n, 8) scores; one-vs-rest mean AUC."""
    aucs = []
    for c in range(S.shape[1]):
        a = auc_score((y_multi == c).astype(np.int64),
                      S[:, c])
        if a is not None:
            aucs.append(a)
    return float(np.mean(aucs)) if aucs else None


# ================================================================
# 2. Full-dim baselines
# ================================================================
w_full = ridge_fit(Z[tr], Yv[tr], lam)
auc_full_truth = auc_score(yte, Zte @ w_full)
assert auc_full_truth >= 0.99, auc_full_truth
Wrel_full = ridge_fit(Z[tr], Yrel[tr], lam)
auc_full_rel = mean_ovr_auc(crte, Zte @ Wrel_full)
log('full-dim: truth AUC=%.4f  crit_rel mean OVR '
    'AUC=%s' % (auc_full_truth,
                '%.4f' % auc_full_rel
                if auc_full_rel is not None else 'n/a'))

# ================================================================
# 3. K-sweep (shared random subsets for both targets)
# ================================================================
def band_of(dmin):
    if dmin is None:
        return 'full_space'
    if dmin <= BANDS['sharp_core']:
        return 'sharp_core'
    if dmin <= BANDS['broad']:
        return 'broad'
    return 'full_space'


ks_truth = {}
ks_rel = {}
d_min_truth = None
d_min_rel = None
for K in K_LIST:
    ta, ra = [], []
    for r in range(N_RAND):
        rr = np.random.RandomState(
            SEED + K + 97 * r)
        if K >= 2560:
            sel = np.arange(2560)
        else:
            sel = np.sort(rr.choice(2560, size=K,
                                    replace=False))
        Wt = ridge_fit(Z[tr][:, sel], Yv[tr], lam)
        ta.append(auc_score(yte, Zte[:, sel] @ Wt))
        Wr = ridge_fit(Z[tr][:, sel], Yrel[tr], lam)
        ra.append(mean_ovr_auc(crte,
                               Zte[:, sel] @ Wr))
        del Wt, Wr
        gc.collect()
    med_t = float(np.median([a for a in ta
                             if a is not None])) \
        if any(a is not None for a in ta) else None
    med_r = float(np.median([a for a in ra
                             if a is not None])) \
        if any(a is not None for a in ra) else None
    ks_truth['K%d' % K] = {
        'aucs': [round(a, 4) if a is not None
                 else None for a in ta],
        'median': round(med_t, 4)
        if med_t is not None else None}
    ks_rel['K%d' % K] = {
        'aucs': [round(a, 4) if a is not None
                 else None for a in ra],
        'median': round(med_r, 4)
        if med_r is not None else None}
    if d_min_truth is None and med_t is not None \
            and med_t >= D_MIN_FRAC * auc_full_truth:
        d_min_truth = K
    if d_min_rel is None and med_r is not None \
            and auc_full_rel is not None \
            and med_r >= D_MIN_FRAC * auc_full_rel:
        d_min_rel = K
    log('K=%4d: truth med=%s rel med=%s'
        % (K, '%.4f' % med_t if med_t is not None
           else 'n/a',
           '%.4f' % med_r if med_r is not None
           else 'n/a'))

shape_truth = band_of(d_min_truth)
shape_rel = band_of(d_min_rel)
log('d_min: truth=%s (%s)  crit_rel=%s (%s)'
    % (d_min_truth, shape_truth, d_min_rel,
       shape_rel))

# ================================================================
# 4. Cross-variable specificity verdict
# ================================================================
if auc_full_rel is None or auc_full_rel < SPEC_BASELINE:
    specificity = 'truth_specific_diffuseness'
    why = 'crit_rel full-dim baseline < 0.9'
elif d_min_rel is None:
    specificity = 'truth_specific_diffuseness'
    why = 'crit_rel never reaches 0.95 x baseline'
elif d_min_truth is not None \
        and d_min_rel >= SPEC_RATIO * d_min_truth:
    specificity = 'truth_specific_diffuseness'
    why = ('d_min_rel %d >= 3 x d_min_truth %s'
           % (d_min_rel, d_min_truth))
else:
    specificity = 'diffuseness_general'
    why = ('d_min_truth=%s d_min_rel=%d '
           'baseline_rel=%.4f'
           % (d_min_truth, d_min_rel, auc_full_rel))
log('SPECIFICITY: %s (%s)' % (specificity, why))

# ================================================================
# 5. Save
# ================================================================
results = {
    'verdict': '%s|%s' % (shape_truth, specificity),
    'curve_shape_truth': shape_truth,
    'd_min_truth': d_min_truth,
    'd_min_rel': d_min_rel,
    'specificity': specificity,
    'specificity_reason': why,
    'gates': {
        'full_auc_truth': round(auc_full_truth, 4),
        'full_auc_crit_rel_ovr':
            round(auc_full_rel, 4)
            if auc_full_rel is not None else None,
        'ksweep_truth': ks_truth,
        'ksweep_crit_rel': ks_rel},
    'created': datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S'),
}
with io.open(os.path.join(OUT, 'result.json'), 'w',
             encoding='utf-8') as f:
    json.dump(results, f, indent=1)
log('result.json written; verdict=%s'
    % results['verdict'])
print('PHASE3110_DONE shape=%s d_min_truth=%s '
      'd_min_rel=%s spec=%s'
      % (shape_truth, d_min_truth, d_min_rel,
         specificity))
