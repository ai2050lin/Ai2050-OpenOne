# -*- coding: utf-8 -*-
"""Phase 3108 (Omega-P106): Degeneracy separation & subspace angles.

T3 line, offline (frozen 3105/3106 captures, no GPU).  3107 found
the truth signal lives in a compact ~100-dim subspace (G1 5/5) but
the Top-200 coordinate basis rotates across materials/layers/positions
(G2 Jaccard 0.053 ~ random 0.042), while the model unembed direction
and learned probes are near-orthogonal yet functionally coupled
(G3 split).  Three explanations for the low cross-material Jaccard:

  (a) ridge coordinate-SELECTION noise (same material would not
      reproduce its own Top-200 either);
  (b) genuinely material-specific coordinate sets;
  (c) degenerate rotation: the SUBSPACE is shared but the coordinate
      basis within it is arbitrary (ridge L2 spreads load over
      near-degenerate directions).

This phase separates (a) from (b)/(c) and tests (c) directly with
rotation-invariant metrics.  Pre-registered: design_seal.json is
written before any statistic is computed.

PART 1 (M1) NOISE FLOOR: for each frozen config, split train into
random halves A/B (N_SEEDS seeds), ridge-fit each half at the
recorded lambda (standardization frozen from full train), Top-200
of each, Jaccard.  Bands (per-config mean J):
  < 0.10        -> coordinate_selection_noise_dominant
  0.10 - 0.30   -> partially_reproducible
  >= 0.30       -> coordinates_reproducible_within_material

PART 2 (M2) SUBSPACE ANGLES: per config, B bootstrap replicates
(train resampled with replacement), ridge-fit each, collect
W_boot (2560 x B), take top-r left singular vectors as the
config's readout subspace U.  For each registered pair, principal
angles via SVD(U_A^T U_B); metric = mean_i cos^2(theta_i).
Floor control: within-config random split of the B bootstrap
columns into two halves, same metric, repeated groupings - pure
sampling-noise floor of the metric.  Pair 'confirmed' iff
mean cos^2 >= 0.50 AND >= max(floor_A, floor_B) + 0.15.

Registered pairs (frozen):
  P1 cross-material last|L8:  C1(3105) vs C5(3106)  [PRIMARY;
      direct re-examination of the 3107 G2 object]
  P2 cross-material qobj|L8:  C2 vs C6
  P3 cross-material mixed:    C1 vs C6
  P4 within-3105 positions:   C1 vs C2   [sanity]
  P5 within-3106 positions:   C5 vs C6   [sanity]
  P6 within-3105 layers:      C3(last|L5) vs C4(last|L6) [sanity]

VERDICT (frozen):
  P1 confirmed AND >= 2 of {P2, P3} confirmed ->
      subspace_reuse_confirmed
  P1 confirmed or partial -> subspace_reuse_partial
  P1 < 0.30 -> per_material_readout
  Sanity (descriptive): >= 2 of {P4, P5, P6} should confirm;
      otherwise flag internal_inconsistency (reported; does not
      change the primary verdict).

PART 3 (M3) SPECTRUM (descriptive): participation ratio and Gini
of the Top-200 |w| spectrum of the full-train ridge solution per
config.  PR near 1 = flat within Top-200 (supports ridge-spread
explanation of low Jaccard); PR near 0 = few coordinates dominate.

SMOKE=1: subsample 400 records, B=6 boot (r=3), 2 seeds, 3
groupings, same pairs.  Output: tests/glm5/result/
rdc_query_construction_20260913/phase3108/
omega_p106_degeneracy_subspace/
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
NAME = 'omega_p106_degeneracy_subspace'
OUT = os.path.join(R13, 'phase3108', NAME)
if SMOKE:
    OUT = os.path.join(OUT, 'smoke')
os.makedirs(OUT, exist_ok=True)
LOGF = os.path.join(OUT, 'run_log.txt')
T0 = time.time()

SEED = 31080
N_SUB = 400 if SMOKE else None
N_BOOT = 6 if SMOKE else 16
N_SEEDS = 2 if SMOKE else 5
N_GROUPINGS = 3 if SMOKE else 5
R_SUB = 3 if SMOKE else 8
TOPN = 200
THR_CONF = 0.50
THR_PART = 0.30
FLOOR_MARGIN = 0.15

C5 = (R13 + r'\phase3105'
      r'\omega_p103_incontext_truth_consistency')
C6 = (R13 + r'\phase3106'
      r'\omega_p104_composition_dose_depth')
RES5 = C5 + r'\result.json'
RES6 = C6 + r'\result.json'


def log(msg):
    line = '[%7.1fs] %s' % (time.time() - T0, msg)
    with io.open(LOGF, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


log('Phase 3108 Omega-P106 start; SMOKE=%d OUT=%s'
    % (SMOKE, OUT))

# ================================================================
# 0. Design seal (pre-computation)
# ================================================================
design = {
    'phase': 3108, 'name': NAME,
    'created': datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S'),
    'smoke': SMOKE, 'seed': SEED,
    'configs_frozen': [
        'C1 3105 last|L8', 'C2 3105 query_obj|L8',
        'C3 3105 last|L5', 'C4 3105 last|L6',
        'C5 3106 last|L8', 'C6 3106 query_obj|L8'],
    'pairs_frozen': [
        'P1 C1-C5 cross-material last|L8 (PRIMARY)',
        'P2 C2-C6 cross-material qobj|L8',
        'P3 C1-C6 cross-material mixed',
        'P4 C1-C2 within-3105 positions (sanity)',
        'P5 C5-C6 within-3106 positions (sanity)',
        'P6 C3-C4 within-3105 layers L5/L6 (sanity)'],
    'lambda_source': 'recorded lam in each phase '
                     'result.json (no re-selection)',
    'standardization': 'per-dim mu/sd frozen from FULL '
                       'train of each config; halves/'
                       'bootstraps fit in that fixed Z '
                       'space (noise isolated to sample '
                       'resampling)',
    'n_boot': N_BOOT, 'r_sub': R_SUB,
    'n_seeds_m1': N_SEEDS, 'n_groupings_floor': N_GROUPINGS,
    'topn': TOPN,
    'm1_bands': {'<0.10': 'coordinate_selection_noise_'
                          'dominant',
                 '0.10-0.30': 'partially_reproducible',
                 '>=0.30': 'coordinates_reproducible_'
                           'within_material'},
    'm2_pair_rule': "confirmed iff mean cos^2 >= 0.50 "
                    'AND >= max(floor_A, floor_B) + '
                    '0.15; partial iff mean cos^2 >= '
                    '0.30',
    'm2_floor': 'within-config random split of bootstrap '
                'columns into two halves, mean cos^2, '
                'repeated groupings averaged',
    'verdict_map': {
        'P1 confirmed & >=2 of P2/P3 confirmed':
            'subspace_reuse_confirmed',
        'P1 confirmed or partial':
            'subspace_reuse_partial',
        'P1 < 0.30': 'per_material_readout'},
    'sanity_rule': '>=2 of P4/P5/P6 should confirm; '
                   'else flag internal_inconsistency '
                   '(descriptive)',
    'm3': 'participation ratio + Gini of Top-200 |w| '
          'spectrum per config (descriptive); PR of '
          'k=200 values lies in [1,200], normalized '
          'PR_norm=PR/200 reported (1=flat, small=few '
          'coords dominate)',
}
with io.open(os.path.join(OUT, 'design_seal.json'),
             'w', encoding='utf-8') as f:
    json.dump(design, f, indent=1)
log('design sealed (pre-computation)')

# ================================================================
# 1. Load frozen captures + recorded lambdas
# ================================================================
z5 = np.load(os.path.join(C5, 'capture.npz'),
             allow_pickle=False)
z6 = np.load(os.path.join(C6, 'capture.npz'),
             allow_pickle=False)
res5 = json.load(io.open(RES5, encoding='utf-8'))
res6 = json.load(io.open(RES6, encoding='utf-8'))
log('captures loaded: 3105 X=%s, 3106 X=%s'
    % (str(z5['X'].shape), str(z6['X'].shape)))

POSITIONS = ['pos0', 'crit_pred', 'crit_obj',
             'query_pred', 'query_obj', 'last']


def lam_of(res, name):
    v = res['T2_ctx'].get(name) or \
        res['T1_ctx'].get(name)
    assert v is not None, name
    return float(v['lam'])


def auc_score(y, s):
    order = np.argsort(s, kind='mergesort')
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    srt = s[order]
    i = 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) \
                and srt[j + 1] == srt[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = \
                (i + 1 + j + 1) / 2.0
        i = j + 1
    n1 = float((y == 1).sum())
    n0 = float((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return None
    return float((ranks[y == 1].sum()
                  - n1 * (n1 + 1) / 2.0)
                 / (n1 * n0))


def top_coords(w, k):
    return np.sort(np.argsort(-np.abs(w))[:k])


def jaccard(a, b):
    sa, sb = set(a.tolist()), set(b.tolist())
    return len(sa & sb) / max(1, len(sa | sb))


def ridge_fit(Zm, yv, lam):
    """Ridge on a (possibly resampled) row subset.
    Same closed form as 3107 ridge_solve; do NOT try to
    shortcut via a precomputed coordinate Gram - sample
    indices do not index the coordinate Gram."""
    n = Zm.shape[0]
    d = Zm.shape[1]
    A = (Zm.T @ Zm) / n \
        + lam * np.eye(d, dtype=np.float32)
    B = (Zm.T @ yv) / n
    return np.linalg.solve(A, B).astype(np.float32)


def svd_left(M, r):
    u, _, _ = np.linalg.svd(M, full_matrices=False)
    return u[:, :r].astype(np.float64)


def mean_cos2(Ua, Ub):
    s = np.linalg.svd(Ua.T @ Ub, compute_uv=False)
    return float(np.mean(s ** 2))


def gini(x):
    x = np.sort(np.abs(x))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


# ================================================================
# 2. Build the 6 frozen configs
# ================================================================
CFG_DEFS = [
    ('C1_3105_last_L8', z5, '3105', 'last', 8),
    ('C2_3105_qobj_L8', z5, '3105', 'query_obj', 8),
    ('C3_3105_last_L5', z5, '3105', 'last', 5),
    ('C4_3105_last_L6', z5, '3105', 'last', 6),
    ('C5_3106_last_L8', z6, '3106', 'last', 8),
    ('C6_3106_qobj_L8', z6, '3106', 'query_obj', 8)]

CFGS = {}
for ci, (nm, z, ph, pos, li) in enumerate(CFG_DEFS):
    meta = {'truth': z['truth'], 'split': z['split']}
    if ph == '3105':
        meta['tag'] = z['tag']
    X = z['X'][:, POSITIONS.index(pos), li, :] \
        .astype(np.float32)
    y = meta['truth'].astype(np.int64)
    if ph == '3105':
        tag = [x.decode() if isinstance(x, bytes)
               else x for x in meta['tag']]
        keep = np.array([t == 'main' for t in tag])
    else:
        keep = np.ones(len(y), dtype=bool)
    if N_SUB:
        rs = np.random.RandomState(SEED)
        idx_all = np.where(keep)[0]
        sub = rs.choice(idx_all,
                        min(N_SUB, len(idx_all)),
                        replace=False)
        keep = np.zeros(len(y), dtype=bool)
        keep[sub] = True
    X = X[keep]
    y = y[keep]
    sp = [x.decode() if isinstance(x, bytes) else x
          for x in meta['split'][keep]]
    tr = np.where(np.array(sp) == 'train')[0]
    te = np.where(np.array(sp) == 'test')[0]
    assert len(tr) > 0 and len(te) > 0, nm
    assert len(set(y[tr].tolist())) == 2, \
        ('train single class', nm)
    mu = X[tr].mean(0)
    sd = X[tr].std(0) + 1e-6
    Z = (X - mu) / sd
    lam = lam_of(res5 if ph == '3105' else res6,
                 '%s|L%d' % (pos, li))
    Yv = (y * 2.0 - 1.0).astype(np.float32)
    w_full = ridge_fit(Z[tr], Yv[tr], lam)
    fa = auc_score(y[te], Z[te] @ w_full)
    CFGS[nm] = {
        'ci': ci, 'phase': ph, 'pos': pos, 'L': li,
        'lam': lam, 'Z': Z, 'y': y, 'Yv': Yv,
        'tr': tr, 'te': te, 'w_full': w_full,
        'full_auc_test': fa, 'n_train': int(len(tr)),
        'n_test': int(len(te))}
    log('%s: n_train=%d n_test=%d lam=%g '
        'full_auc_test=%s'
        % (nm, len(tr), len(te), lam,
           '%.4f' % fa if fa is not None else 'n/a'))
    del X
    gc.collect()

# sanity vs 3107 recorded full AUC (3105 last|L8 = 0.9999)
assert CFGS['C1_3105_last_L8']['full_auc_test'] >= 0.99, \
    'C1 sanity fail vs 3107'

# ================================================================
# 3. M1 noise floor (half-split Top-200 Jaccard)
# ================================================================
m1 = {}
for nm, c in CFGS.items():
    tr, Z, Yv, lam = c['tr'], c['Z'], c['Yv'], \
        c['lam']
    Js = []
    for s in range(N_SEEDS):
        rng = np.random.RandomState(
            SEED + 11 * s + c['ci'])
        perm = rng.permutation(len(tr))
        h = len(tr) // 2
        iA, iB = tr[perm[:h]], tr[perm[h:2 * h]]
        wA = ridge_fit(Z[iA], Yv[iA], lam)
        wB = ridge_fit(Z[iB], Yv[iB], lam)
        Js.append(jaccard(top_coords(wA, TOPN),
                          top_coords(wB, TOPN)))
    meanJ = float(np.mean(Js))
    band = ('coordinate_selection_noise_dominant'
            if meanJ < 0.10 else
            'partially_reproducible'
            if meanJ < 0.30 else
            'coordinates_reproducible_within_material')
    m1[nm] = {'values': [round(v, 4) for v in Js],
              'mean': round(meanJ, 4), 'band': band}
    log('M1 %s: J=%s mean=%.4f -> %s'
        % (nm, ['%.3f' % v for v in Js], meanJ, band))

# ================================================================
# 4. M2 bootstrap subspaces + principal angles
# ================================================================
subs = {}
floors = {}
for nm, c in CFGS.items():
    tr, Z, Yv, lam = c['tr'], c['Z'], c['Yv'], \
        c['lam']
    rngb = np.random.RandomState(SEED + 500 + c['ci'])
    cols = []
    for b in range(N_BOOT):
        idx = rngb.choice(tr, size=len(tr),
                          replace=True)
        cols.append(ridge_fit(Z[idx], Yv[idx], lam))
    Wb = np.stack(cols, axis=1)
    subs[nm] = svd_left(Wb, R_SUB)
    fl = []
    for g in range(N_GROUPINGS):
        rngg = np.random.RandomState(
            SEED + 900 + 17 * g + c['ci'])
        perm = rngg.permutation(N_BOOT)
        ha = N_BOOT // 2
        Ua = svd_left(Wb[:, perm[:ha]], R_SUB)
        Ub = svd_left(Wb[:, perm[ha:2 * ha]], R_SUB)
        fl.append(mean_cos2(Ua, Ub))
    floors[nm] = float(np.mean(fl))
    log('M2 %s: floor=%.4f (groupings %s)'
        % (nm, floors[nm],
           ['%.3f' % v for v in fl]))
    del Wb
    gc.collect()

PAIRS = [
    ('P1_cross_mat_last_L8', 'C1_3105_last_L8',
     'C5_3106_last_L8'),
    ('P2_cross_mat_qobj_L8', 'C2_3105_qobj_L8',
     'C6_3106_qobj_L8'),
    ('P3_cross_mat_mixed', 'C1_3105_last_L8',
     'C6_3106_qobj_L8'),
    ('P4_within_3105_pos', 'C1_3105_last_L8',
     'C2_3105_qobj_L8'),
    ('P5_within_3106_pos', 'C5_3106_last_L8',
     'C6_3106_qobj_L8'),
    ('P6_within_3105_layers', 'C3_3105_last_L5',
     'C4_3105_last_L6')]

m2 = {}
for (pn, ca, cb) in PAIRS:
    mc = mean_cos2(subs[ca], subs[cb])
    fl = max(floors[ca], floors[cb])
    if mc >= THR_CONF and mc >= fl + FLOOR_MARGIN:
        status = 'confirmed'
    elif mc >= THR_PART:
        status = 'partial'
    else:
        status = 'fail'
    m2[pn] = {'a': ca, 'b': cb,
              'mean_cos2': round(mc, 4),
              'floor_max': round(fl, 4),
              'status': status}
    log('M2 %s: mean_cos2=%.4f floor=%.4f -> %s'
        % (pn, mc, fl, status))

p1 = m2['P1_cross_mat_last_L8']['status']
aux_ok = sum(m2[p]['status'] == 'confirmed'
             for p in ('P2_cross_mat_qobj_L8',
                       'P3_cross_mat_mixed'))
if p1 == 'confirmed' and aux_ok >= 2:
    verdict = 'subspace_reuse_confirmed'
elif p1 in ('confirmed', 'partial'):
    verdict = 'subspace_reuse_partial'
else:
    verdict = 'per_material_readout'
san_ok = sum(m2[p]['status'] == 'confirmed'
             for p in ('P4_within_3105_pos',
                       'P5_within_3106_pos',
                       'P6_within_3105_layers'))
sanity = 'ok' if san_ok >= 2 else 'internal_inconsistency'
log('VERDICT: %s (P1=%s aux_ok=%d sanity=%s %d/3)'
    % (verdict, p1, aux_ok, sanity, san_ok))

# ================================================================
# 5. M3 spectrum (descriptive)
# ================================================================
m3 = {}
for nm, c in CFGS.items():
    v = np.sort(np.abs(c['w_full']))[::-1][:TOPN] ** 2
    pr = float(v.sum() ** 2 / (v ** 2).sum())
    # PR of k=200 values lies in [1, 200]; normalize by k
    # so 1 = perfectly flat, small = few coords dominate
    prn = pr / TOPN
    gi = gini(c['w_full'][top_coords(c['w_full'],
                                     TOPN)])
    m3[nm] = {'pr_top200': round(pr, 4),
              'pr_norm': round(prn, 4),
              'gini_top200': round(gi, 4)}
    log('M3 %s: PR=%.1f PR_norm=%.4f Gini=%.4f'
        % (nm, pr, prn, gi))
m3_flag = ('flat_within_top200'
           if all(x['pr_norm'] > 0.5
                  for x in m3.values()) else
           'dominated_by_few_coords'
           if any(x['pr_norm'] < 0.2
                  for x in m3.values()) else 'mixed')
log('M3 flag: %s' % m3_flag)

# ================================================================
# 6. Save
# ================================================================
results = {
    'verdict': verdict,
    'sanity': sanity,
    'gates': {
        'M1_noise_floor': m1,
        'M2_subspace': {
            'pairs': m2,
            'floors': {k: round(v, 4)
                       for k, v in floors.items()},
            'r_sub': R_SUB, 'n_boot': N_BOOT},
        'M3_spectrum': {'per_config': m3,
                        'flag': m3_flag}},
    'config_meta': {
        nm: {'phase': c['phase'], 'pos': c['pos'],
             'L': c['L'], 'lam': c['lam'],
             'n_train': c['n_train'],
             'n_test': c['n_test'],
             'full_auc_test':
                 round(c['full_auc_test'], 4)}
        for nm, c in CFGS.items()},
    'created': datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S'),
}
with io.open(os.path.join(OUT, 'result.json'), 'w',
             encoding='utf-8') as f:
    json.dump(results, f, indent=1)
log('result.json written; verdict=%s' % verdict)
print('PHASE3108_DONE verdict=%s sanity=%s P1=%s '
      'aux_ok=%d' % (verdict, sanity, p1, aux_ok))
