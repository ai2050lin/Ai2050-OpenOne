# -*- coding: utf-8 -*-
"""Phase 3111 (Omega-P109): Broadcast-signal verdict.

T3 line, offline (frozen 3105 capture, no GPU).  3110 found the
truth variable's readout information is DIFFUSELY REDUNDANT and
TRUTH-SPECIFIC (d_min=5 vs crit_rel d_min=400), and proposed a
candidate physical carrier: a GLOBAL FIRST-MOMENT signal -
per-record overall activation level / integration state
correlated with truth - which any subspace can read.  This
phase runs the three pre-registered decisive experiments:

  (1) DIRECT TEST (D-A): per-record scalars ||h||, mean(h),
      median(h) as single-feature truth scores.  AUC computed
      direction-free as max(auc, 1-auc).  In BOTH raw X space
      and per-dim-standardized Z space.  First-moment
      broadcast 'present' iff max scalar AUC >= 0.90.
  (2) CENTERED K-SWEEP (D-B, decisive): per-record centering
      Z_c[i] = Z[i] - mean_coords(Z[i]) removes the DC
      component; centered+normalized Z_cn additionally divides
      by the record norm (removes the magnitude channel).
      Rerun the 3110 K-sweep on all three spaces (raw /
      centered / centered+normalized), same K list, same seed
      stream.  d_min per space (0.95 x that space's full-dim
      AUC).
  (3) PER-COORDINATE AUC HISTOGRAM (D-C): single-dim truth
      AUC for all 2560 coordinates (direction-free), report
      median, fraction > 0.7, fraction > 0.9.

VERDICT (frozen):
  D-A present AND (d_min_cent None or > max(20, 10*d_min_raw))
      -> broadcast_first_moment
  D-A absent AND d_min_cent <= 20
      -> distributed_higher_order
  D-A present otherwise
      -> mixed_broadcast_plus_distribution
  D-A absent otherwise
      -> mixed_first_moment_minor

SMOKE=1: K in {10,100,2560}, R=3.  Output: tests/glm5/result/
rdc_query_construction_20260913/phase3111/
omega_p109_broadcast_verdict/
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
NAME = 'omega_p109_broadcast_verdict'
OUT = os.path.join(R13, 'phase3111', NAME)
if SMOKE:
    OUT = os.path.join(OUT, 'smoke')
os.makedirs(OUT, exist_ok=True)
LOGF = os.path.join(OUT, 'run_log.txt')
T0 = time.time()

SEED = 31110
C5 = (R13 + r'\phase3105'
      r'\omega_p103_incontext_truth_consistency')
K_LIST = [10, 100, 2560] if SMOKE else \
    [5, 10, 20, 50, 100, 200, 400, 800, 1600, 2560]
N_RAND = 3 if SMOKE else 10
D_MIN_FRAC = 0.95
DA_THRESH = 0.90

POSITIONS = ['pos0', 'crit_pred', 'crit_obj',
             'query_pred', 'query_obj', 'last']


def log(msg):
    line = '[%7.1fs] %s' % (time.time() - T0, msg)
    with io.open(LOGF, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


log('Phase 3111 Omega-P109 start; SMOKE=%d OUT=%s'
    % (SMOKE, OUT))

# ================================================================
# 0. Design seal (pre-computation)
# ================================================================
design = {
    'phase': 3111, 'name': NAME,
    'created': datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S'),
    'smoke': SMOKE, 'seed': SEED,
    'config': '3105 C1 last|L8, main records, '
              'lambda=0.01, per-dim standardization '
              'frozen from FULL train (same as '
              '3108-3110)',
    'da_rule': 'per-record scalars ||h||/mean/median in '
               'raw and Z space, direction-free AUC = '
               'max(auc, 1-auc); present iff max >= '
               '0.90',
    'db_rule': 'K-sweep rerun on raw / centered (per-'
               'record coordinate mean removed) / '
               'centered+normalized (also divided by '
               'record norm); d_min = smallest K with '
               'median AUC >= 0.95 x that space full '
               'AUC; same K list and seed stream as '
               '3110',
    'dc_rule': 'per-coordinate direction-free single-dim '
               'AUC over all 2560; median, frac>0.7, '
               'frac>0.9 (descriptive)',
    'verdict_map': {
        'DA & (d_min_cent None or > max(20,10*d_min_raw))':
            'broadcast_first_moment',
        '!DA & d_min_cent <= 20':
            'distributed_higher_order',
        'DA otherwise': 'mixed_broadcast_plus_'
                        'distribution',
        '!DA otherwise': 'mixed_first_moment_minor'},
}
with io.open(os.path.join(OUT, 'design_seal.json'),
             'w', encoding='utf-8') as f:
    json.dump(design, f, indent=1)
log('design sealed (pre-computation)')

# ================================================================
# 1. Load capture, build spaces
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
tr = np.where(np.array(sp) == 'train')[0]
te = np.where(np.array(sp) == 'test')[0]
lam = 0.01
mu = X[tr].mean(0)
sd = X[tr].std(0) + 1e-6
Z = (X - mu) / sd
yte = y[te]
log('capture loaded: n_train=%d n_test=%d lam=%g'
    % (len(tr), len(te), lam))


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


def auc_free(yy, s):
    a = auc_score(yy, s)
    if a is None:
        return None
    return max(a, 1.0 - a)


def ridge_fit(Zm, Ym, lamv):
    n, d = Zm.shape
    A = (Zm.T @ Zm) / n \
        + lamv * np.eye(d, dtype=np.float32)
    return np.linalg.solve(A, (Zm.T @ Ym) / n) \
        .astype(np.float32)


# ================================================================
# 2. D-A direct scalar test
# ================================================================
spaces = {
    'raw': X, 'Z': Z}
da = {}
for sname, S in spaces.items():
    Ste, Str = S[te], S[tr]
    norms_te = np.linalg.norm(Ste, axis=1)
    norms_tr = np.linalg.norm(Str, axis=1)
    items = {}
    for vname, v in (('norm', norms_te),
                     ('mean', Ste.mean(1)),
                     ('median', np.median(Ste, 1))):
        a = auc_free(yte, v)
        items[vname] = round(a, 4) if a is not None \
            else None
    mx = max(v for v in items.values()
             if v is not None)
    da[sname] = {'scalars': items, 'max': round(mx, 4)}
    log('D-A %s: %s max=%.4f'
        % (sname, items, mx))
DA = max(da['Z']['max'], da['raw']['max']) >= DA_THRESH
log('D-A present: %s' % DA)

# ================================================================
# 3. D-B centered / normalized K-sweep
# ================================================================
Zc = Z - Z.mean(axis=1, keepdims=True)
nc = np.linalg.norm(Zc, axis=1, keepdims=True) + 1e-6
Zcn = Zc / nc
SPACES3 = {'raw': Z, 'centered': Zc,
           'centered_norm': Zcn}
K_LIST_FULL = K_LIST
sweep3 = {}
d_mins = {}
for sname, S in SPACES3.items():
    Str, Ste = S[tr], S[te]
    w_full = ridge_fit(Str, (y[tr] * 2.0 - 1.0)
                       .astype(np.float32), lam)
    auc_full = auc_score(yte, Ste @ w_full)
    dmin = None
    per_k = {}
    for K in K_LIST_FULL:
        aa = []
        for r in range(N_RAND):
            rr = np.random.RandomState(
                SEED + K + 97 * r)
            if K >= 2560:
                sel = np.arange(2560)
            else:
                sel = np.sort(rr.choice(
                    2560, size=K, replace=False))
            wk = ridge_fit(Str[:, sel],
                           (y[tr] * 2.0 - 1.0)
                           .astype(np.float32), lam)
            aa.append(auc_score(yte,
                                Ste[:, sel] @ wk))
            del wk
        aa2 = [a for a in aa if a is not None]
        med = float(np.median(aa2)) if aa2 else None
        per_k['K%d' % K] = {
            'median': round(med, 4)
            if med is not None else None,
            'aucs': [round(a, 4) if a is not None
                     else None for a in aa]}
        if dmin is None and med is not None \
                and auc_full is not None \
                and med >= D_MIN_FRAC * auc_full:
            dmin = K
        del aa
        gc.collect()
    sweep3[sname] = {'full_auc': round(auc_full, 4)
                     if auc_full is not None else None,
                     'per_k': per_k}
    d_mins[sname] = dmin
    log('D-B %s: full=%.4f d_min=%s'
        % (sname, auc_full
           if auc_full is not None else -1,
           dmin))
d_raw = d_mins['raw']
d_cent = d_mins['centered']

# ================================================================
# 4. D-C per-coordinate AUC histogram
# ================================================================
Zte = Z[te]
single = np.zeros(2560, dtype=np.float64)
for d in range(2560):
    a = auc_free(yte, Zte[:, d])
    single[d] = a if a is not None else 0.5
dc = {
    'median': round(float(np.median(single)), 4),
    'p90': round(float(np.percentile(single, 90)), 4),
    'max': round(float(single.max()), 4),
    'frac_gt_0.7': round(float((single > 0.7).mean()), 4),
    'frac_gt_0.9': round(float((single > 0.9).mean()), 4),
    'argmax': int(single.argmax())}
log('D-C per-coordinate AUC: %s' % json.dumps(dc))

# ================================================================
# 5. Verdict
# ================================================================
if DA and (d_cent is None
           or d_cent > max(20, 10 * (d_raw or 0))):
    verdict = 'broadcast_first_moment'
elif (not DA) and d_cent is not None and d_cent <= 20:
    verdict = 'distributed_higher_order'
elif DA:
    verdict = 'mixed_broadcast_plus_distribution'
else:
    verdict = 'mixed_first_moment_minor'
log('VERDICT: %s (DA=%s d_raw=%s d_cent=%s)'
    % (verdict, DA, d_raw, d_cent))

# ================================================================
# 6. Save
# ================================================================
results = {
    'verdict': verdict,
    'gates': {
        'DA_scalar_test': da,
        'DB_ksweep_3spaces': sweep3,
        'd_mins': d_mins,
        'DC_per_coordinate': dc},
    'created': datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S'),
}
with io.open(os.path.join(OUT, 'result.json'), 'w',
             encoding='utf-8') as f:
    json.dump(results, f, indent=1)
log('result.json written; verdict=%s' % verdict)
print('PHASE3111_DONE verdict=%s DA=%s d_raw=%s '
      'd_cent=%s' % (verdict, DA, d_raw, d_cent))
