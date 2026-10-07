# -*- coding: utf-8 -*-
"""Phase 3112 (Omega-P110): Broadcast emergence layer +
coordinate correlation structure + cross-material replication.

T3 line, offline (frozen 3105 + 3106 captures, no GPU).
3111 established holographic redundancy: every coordinate is
nearly a complete truth readout (median single-coordinate AUC
0.922).  Three pre-registered questions:

  (1) EMERGENCE LAYER: per-layer single-coordinate truth AUC
      median at the LAST position across L0..L8 (3105).  The
      broadcast 'emerges' at the first layer whose median >=
      0.90.  3106 found binding peaks at L3 and truth readout
      at L4 - prediction: emergence near L4.
  (2) COORDINATE CORRELATION STRUCTURE: 2560x2560 Pearson
      correlation matrix of standardized last-position states
      over ALL main records.  Spectrum: lambda1 energy share
      >= 0.50 -> single_broadcast_source; 0.20-0.50 ->
      few_channels; < 0.20 -> distributed_channels.
      Plus: PC1 score vs truth AUC (does the dominant common
      factor carry the truth variable?).
  (3) CROSS-MATERIAL REPLICATION (3106 chain/scatter):
      single-coordinate truth AUC median at last|L8 test
      records (>= 0.70 -> replicated) and the same per-layer
      emergence curve on 3106.

All single-coordinate AUCs are direction-free on TEST records
only (no fitting).  Layer standardization uses each layer's
own train mu/sd (frozen).

SMOKE=1: layers {0,4,8}, corr-matrix on a 400-coordinate
subset.  Output: tests/glm5/result/
rdc_query_construction_20260913/phase3112/
omega_p110_broadcast_emergence/
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
NAME = 'omega_p110_broadcast_emergence'
OUT = os.path.join(R13, 'phase3112', NAME)
if SMOKE:
    OUT = os.path.join(OUT, 'smoke')
os.makedirs(OUT, exist_ok=True)
LOGF = os.path.join(OUT, 'run_log.txt')
T0 = time.time()

SEED = 31120
C5 = (R13 + r'\phase3105'
      r'\omega_p103_incontext_truth_consistency')
C6 = (R13 + r'\phase3106'
      r'\omega_p104_composition_dose_depth')
LAYERS = [0, 4, 8] if SMOKE else list(range(9))
EMERGE_THR = 0.90
REPL_THR = 0.70
L1_SINGLE = 0.50
L1_FEW = 0.20
POS = 5  # 'last'

POSITIONS = ['pos0', 'crit_pred', 'crit_obj',
             'query_pred', 'query_obj', 'last']


def log(msg):
    line = '[%7.1fs] %s' % (time.time() - T0, msg)
    with io.open(LOGF, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


log('Phase 3112 Omega-P110 start; SMOKE=%d OUT=%s'
    % (SMOKE, OUT))

# ================================================================
# 0. Design seal (pre-computation)
# ================================================================
design = {
    'phase': 3112, 'name': NAME,
    'created': datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S'),
    'smoke': SMOKE, 'seed': SEED,
    'emergence_rule': 'first layer with single-coordinate '
                      'median AUC >= 0.90 at LAST position',
    'corr_rule': 'Pearson corr of standardized states over '
                 'ALL main records; lambda1 share >= 0.50 '
                 '-> single_broadcast_source, 0.20-0.50 -> '
                 'few_channels, < 0.20 -> distributed_'
                 'channels; PC1 score vs truth AUC '
                 'reported',
    'replication_rule': '3106 last|L8 single-coordinate '
                        'median AUC >= 0.70 -> replicated',
    'auc': 'direction-free (max(auc,1-auc)), TEST records '
           'only, no fitting',
    'layer_norm': 'per-layer train mu/sd (frozen)',
}
with io.open(os.path.join(OUT, 'design_seal.json'),
             'w', encoding='utf-8') as f:
    json.dump(design, f, indent=1)
log('design sealed (pre-computation)')


def auc_score(yy, s):
    if s.ndim == 1:
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
    # 2-D: per-column ranks (fast path; continuous
    # values, ties negligible for single coordinates)
    order = np.argsort(s, axis=0, kind='mergesort')
    ranks = np.empty(s.shape, dtype=np.float64)
    cols = np.arange(s.shape[1])
    ranks[order, cols] = \
        np.arange(1, s.shape[0] + 1)[:, None]
    n1 = float((yy == 1).sum())
    n0 = float((yy == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.full(s.shape[1], np.nan)
    return (ranks[yy == 1].sum(axis=0)
            - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def auc_free_cols(Xte, yy):
    """Direction-free AUC per column (vectorized)."""
    a = auc_score(yy, Xte)
    a = np.asarray(a, dtype=np.float64)
    return np.maximum(a, 1.0 - a)


def layer_single_med(z, meta_keep_mask, te_mask, li,
                     tag='main'):
    """Median single-coordinate truth AUC at layer li,
    LAST position, direction-free, on test records."""
    X = z['X'][:, POS, li, :].astype(np.float32)
    y = z['truth'].astype(np.int64)
    X, y = X[meta_keep_mask], y[meta_keep_mask]
    yte = y[te_mask]
    sp = [x.decode() if isinstance(x, bytes) else x
          for x in z['split'][meta_keep_mask]]
    trm = np.array([s == 'train'
                    for s in sp]) & np.ones(
        len(y), dtype=bool)
    mu = X[trm].mean(0)
    sd = X[trm].std(0) + 1e-6
    Zte = (X[te_mask] - mu) / sd
    a = auc_free_cols(Zte, yte)
    a = a[~np.isnan(a)]
    return float(np.median(a)), float(a.max()), \
        float((a > 0.7).mean())


# ================================================================
# 1. 3105 emergence curve (per-layer last position)
# ================================================================
z5 = np.load(os.path.join(C5, 'capture.npz'),
             allow_pickle=False)
tag5 = [x.decode() if isinstance(x, bytes) else x
        for x in z5['tag']]
keep5 = np.array([t == 'main' for t in tag5])
sp5 = [x.decode() if isinstance(x, bytes) else x
       for x in z5['split'][keep5]]
te5 = np.array([s == 'test' for s in sp5])
curve5 = {}
emerge_L = None
for li in LAYERS:
    med, mx, f7 = layer_single_med(
        z5, keep5, te5, li)
    curve5['L%d' % li] = {'median': round(med, 4),
                          'max': round(mx, 4),
                          'frac_gt_0.7': round(f7, 4)}
    if emerge_L is None and med >= EMERGE_THR:
        emerge_L = li
    log('3105 L%d: single-coord median=%.4f max=%.4f '
        'frac>0.7=%.3f' % (li, med, mx, f7))
    gc.collect()
log('3105 emergence layer (median >= %.2f): %s'
    % (EMERGE_THR, emerge_L))

# ================================================================
# 2. Coordinate correlation structure (3105, last|L8)
# ================================================================
X8 = z5['X'][:, POS, 8, :].astype(np.float32)
y5 = z5['truth'].astype(np.int64)
X8, y5 = X8[keep5], y5[keep5]
trm5 = np.array([s == 'train' for s in sp5])
mu8 = X8[trm5].mean(0)
sd8 = X8[trm5].std(0) + 1e-6
Zall = (X8 - mu8) / sd8
if SMOKE:
    rng = np.random.RandomState(SEED)
    sub = rng.choice(2560, size=400, replace=False)
    Zc = Zall[:, sub]
else:
    Zc = Zall
C = np.corrcoef(Zc, rowvar=False)
C = np.clip(C, -1, 1)
iu = np.triu_indices(C.shape[0], 1)
ev = np.linalg.eigvalsh(C)[::-1]
ev = np.clip(ev, 0, None)
l1_share = float(ev[0] / ev.sum())
pr = float(ev.sum() ** 2 / (ev ** 2).sum())
if l1_share >= L1_SINGLE:
    corr_struct = 'single_broadcast_source'
elif l1_share >= L1_FEW:
    corr_struct = 'few_channels'
else:
    corr_struct = 'distributed_channels'
# PC1 score vs truth (projection on top eigenvector)
w0, v0 = np.linalg.eigh(C)
pc1 = v0[:, -1]
pc1_score = Zc @ pc1
from numpy import argsort
ra = np.argsort(pc1_score)
ranks = np.empty(len(ra), dtype=np.float64)
ranks[ra] = np.arange(1, len(ra) + 1)
n1 = float((y5 == 1).sum())
n0 = float((y5 == 0).sum())
a1 = float((ranks[y5 == 1].sum()
            - n1 * (n1 + 1) / 2.0) / (n1 * n0))
auc_pc1 = max(a1, 1.0 - a1)
mpc = float(np.abs(C[iu]).mean())
n_coords_c = int(C.shape[0])
log('corr matrix (%d coords): lambda1 share=%.4f '
    'PR=%.1f mean|corr|=%.4f PC1-truth AUC=%.4f -> %s'
    % (n_coords_c, l1_share, pr, mpc, auc_pc1,
       corr_struct))
del C, Zc
gc.collect()

# ================================================================
# 3. Cross-material replication (3106)
# ================================================================
z6 = np.load(os.path.join(C6, 'capture.npz'),
             allow_pickle=False)
keep6 = np.ones(z6['truth'].shape[0], dtype=bool)
sp6 = [x.decode() if isinstance(x, bytes) else x
       for x in z6['split'][keep6]]
te6 = np.array([s == 'test' for s in sp6])
curve6 = {}
emerge_L6 = None
for li in LAYERS:
    med, mx, f7 = layer_single_med(
        z6, keep6, te6, li)
    curve6['L%d' % li] = {'median': round(med, 4),
                          'max': round(mx, 4),
                          'frac_gt_0.7': round(f7, 4)}
    if emerge_L6 is None and med >= EMERGE_THR:
        emerge_L6 = li
    log('3106 L%d: single-coord median=%.4f max=%.4f '
        'frac>0.7=%.3f' % (li, med, mx, f7))
    gc.collect()
med68 = curve6['L8']['median']
replicated = med68 >= REPL_THR
log('3106 last|L8 median=%.4f -> replicated=%s'
    % (med68, replicated))

# ================================================================
# 4. Verdict & save
# ================================================================
verdict = 'emerge_%s|%s|%s' % (
    ('L%d' % emerge_L) if emerge_L is not None
    else 'none',
    corr_struct,
    'replicated' if replicated else 'not_replicated')
log('VERDICT: %s' % verdict)

results = {
    'verdict': verdict,
    'emergence_layer_3105': emerge_L,
    'emergence_layer_3106': emerge_L6,
    'corr_structure': corr_struct,
    'replicated_3106': bool(replicated),
    'gates': {
        'curve_3105_last_per_layer': curve5,
        'curve_3106_last_per_layer': curve6,
        'corr_spectrum': {
            'n_coords': n_coords_c,
            'lambda1_share': round(l1_share, 4),
            'participation_ratio': round(pr, 2),
            'mean_abs_corr': round(mpc, 4),
            'pc1_truth_auc': round(auc_pc1, 4)},
        'replication_median_L8_3106':
            round(med68, 4)},
    'created': datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S'),
}
with io.open(os.path.join(OUT, 'result.json'), 'w',
             encoding='utf-8') as f:
    json.dump(results, f, indent=1)
log('result.json written; verdict=%s' % verdict)
print('PHASE3112_DONE verdict=%s' % verdict)
