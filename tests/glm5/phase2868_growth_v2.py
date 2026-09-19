"""Phase 2868 (LPF MA2 cont): mechanism-base growth curve v2 --
non-circulating block combination B23 + eta2-weighted spectrum.

Blocks (80 words, SEED=2855 vocab; all read from immutable npz, zero
forward pass):
  B2  causal spectrum        (1152)  [2867 npz = 2846 drops_all]
  B3  mlp response spectrum  (10)    [2867 npz = 2861 g_direct L26-35]
  B2w eta2-weighted B2       (1152)  w_h = eta2_h^2 / max(eta2^2) (2864)
  B2s significant-head B2    (n_sig) heads with p_per_head < 0.01 (2864)
Combos (2867 protocol: rows unit-normed, hstack, unit-normed again):
  B23  = [B2, B3]                (key uncirculated increment point)
  B23z = [zscore(B2), zscore(B3)]  (block-balance control vs dim imbalance)

Prereg (frozen before any readout):
  G1  (main) paired delta: d_obs = acc(B23) - acc(B3);
      causal_true_gain iff d_obs > 0 AND d_obs > null-delta p95
      (200 label permutations, SEED=2868; paired null delta =
      acc(B23|perm) - acc(B3|perm)); else causal_no_gain.
  G2  eta2 weighting: acc(B2w) > acc(B2) AND acc(B2w) > null-acc p95
      => eta2_weighting_helps; else eta2_weighting_neutral.
  G3  sparsity: acc(B2s) >= acc(B2) => sparse_sufficient
      (descriptive, with null p95 for reference).
  growth_v2 = (acc(B23) - acc(B3)) / max(1 - acc(B3), 1e-9)
      (remaining-headroom utilisation); < 0.1 => sublinear_reuse,
      >= 0.1 => additive_gain.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2868' / 'growth_v2'
SRC_2867 = BASE / 'phase2867' / 'word_coords_v1' / 'word_coords_v1.npz'
SRC_2864 = BASE / 'phase2864' / 'class_variance' / 'class_variance.npz'
SEED = 2868
N_PERM = 200

PREREG = {
    'G1': 'paired delta acc(B23)-acc(B3) > 0 and > null-delta p95 '
          '(200 perms, SEED=2868) => causal_true_gain else causal_no_gain',
    'G2': 'acc(B2w) > acc(B2) and > null-acc p95 => '
          'eta2_weighting_helps else eta2_weighting_neutral',
    'G3': 'acc(B2s) >= acc(B2) => sparse_sufficient (descriptive)',
    'growth_v2': '(acc(B23)-acc(B3))/max(1-acc(B3),1e-9); <0.1 '
                 'sublinear_reuse, >=0.1 additive_gain',
}


def unit(x):
    return x / max(float(np.linalg.norm(x)), 1e-30)


def zscore_cols(B):
    mu = B.mean(axis=0, keepdims=True)
    sd = B.std(axis=0, keepdims=True)
    return (B - mu) / np.maximum(sd, 1e-12)


def loo_nn_acc(C, labels):
    n = len(labels)
    C = C.copy()
    np.fill_diagonal(C, -2.0)
    a = 0
    for i in range(n):
        j = int(np.argmax(C[i]))
        a += int(labels[j] == labels[i])
    return a / n


def main():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)

    execution_path = OUT / 'execution.json'
    if not execution_path.exists():
        execution = {'timestamp': fc.stamp(),
                     'source': cc.snapshot(__file__),
                     'prereg': PREREG, 'seed': SEED,
                     'design': 'growth curve v2: non-circulating combo '
                               'B23 (causal+mlp) vs B3 paired delta, '
                               'eta2-weighted spectrum B2w, sparse '
                               'significant-head subset B2s; zero forward'}
        fc.save(execution_path, execution)

    z67 = np.load(SRC_2867, allow_pickle=True)
    B1 = z67['B1'].astype(np.float64)
    B2 = z67['B2'].astype(np.float64)
    B3 = z67['B3'].astype(np.float64)
    labels = z67['labels'].astype(np.int64)
    n_words = len(labels)

    z64 = np.load(SRC_2864, allow_pickle=True)
    eta2 = z64['eta2'].astype(np.float64)
    p_per_head = z64['p_per_head'].astype(np.float64)
    assert eta2.shape == (1152,) and B2.shape == (n_words, 1152)

    # ---------- blocks ----------
    B2u = np.stack([unit(B2[i]) for i in range(n_words)])
    B3u = np.stack([unit(B3[i]) for i in range(n_words)])

    w = eta2 ** 2 / max(float((eta2 ** 2).max()), 1e-30)
    B2w_raw = B2u * w[None, :]
    B2w = np.stack([unit(B2w_raw[i]) for i in range(n_words)])

    sig = p_per_head < 0.01
    n_sig = int(sig.sum())
    B2s_raw = B2u[:, sig]
    B2s = np.stack([unit(B2s_raw[i]) for i in range(n_words)])

    B23_raw = np.hstack([B2u, B3u])
    B23 = np.stack([unit(B23_raw[i]) for i in range(n_words)])

    B23z_raw = np.hstack([zscore_cols(B2u), zscore_cols(B3u)])
    B23z = np.stack([unit(B23z_raw[i]) for i in range(n_words)])

    blocks = {
        'B2': B2u, 'B3': B3u, 'B2w': B2w, 'B2s': B2s,
        'B23': B23, 'B23z': B23z,
        'B2ws': np.stack([unit(np.hstack([B2w, B3u])[i])
                          for i in range(n_words)]),
    }

    # ---------- accuracies ----------
    accs = {}
    Cmats = {}
    for name, B in blocks.items():
        C = B @ B.T
        Cmats[name] = C
        accs[name] = loo_nn_acc(C, labels)

    # ---------- G1 paired null ----------
    rng = np.random.default_rng(SEED)
    C23, C3 = Cmats['B23'], Cmats['B3']
    d_obs = accs['B23'] - accs['B3']
    null_d = []
    null_acc23 = []
    for _ in range(N_PERM):
        pl = rng.permutation(labels)
        null_d.append(loo_nn_acc(C23, pl) - loo_nn_acc(C3, pl))
        null_acc23.append(null_d[-1] + loo_nn_acc(C3, pl))
    nd_p95 = float(np.percentile(null_d, 95))
    g1 = bool(d_obs > 0 and d_obs > nd_p95)
    g1_label = 'causal_true_gain' if g1 else 'causal_no_gain'

    # ---------- G2 / G3 nulls ----------
    null_acc_w = []
    null_acc_s = []
    for _ in range(N_PERM):
        pl = rng.permutation(labels)
        null_acc_w.append(loo_nn_acc(Cmats['B2w'], pl))
        null_acc_s.append(loo_nn_acc(Cmats['B2s'], pl))
    nw_p95 = float(np.percentile(null_acc_w, 95))
    ns_p95 = float(np.percentile(null_acc_s, 95))
    g2 = bool(accs['B2w'] > accs['B2'] and accs['B2w'] > nw_p95)
    g2_label = 'eta2_weighting_helps' if g2 else 'eta2_weighting_neutral'
    g3 = bool(accs['B2s'] >= accs['B2'])
    g3_label = 'sparse_sufficient' if g3 else 'sparse_lossy'

    # ---------- growth v2 ----------
    headroom = max(1.0 - accs['B3'], 1e-9)
    growth = d_obs / headroom
    growth_label = 'sublinear_reuse' if growth < 0.1 else 'additive_gain'

    v = {
        'n_words': n_words,
        'n_sig_heads': n_sig,
        'acc_B2': round(accs['B2'], 4),
        'acc_B3': round(accs['B3'], 4),
        'acc_B2w': round(accs['B2w'], 4),
        'acc_B2s': round(accs['B2s'], 4),
        'acc_B23': round(accs['B23'], 4),
        'acc_B23z': round(accs['B23z'], 4),
        'acc_B2ws': round(accs['B2ws'], 4),
        'G1_delta_obs': round(d_obs, 4),
        'G1_null_delta_p95': round(nd_p95, 4),
        'G1': g1,
        'G1_label': g1_label,
        'G2_null_acc_p95': round(nw_p95, 4),
        'G2': g2,
        'G2_label': g2_label,
        'G3_null_acc_p95': round(ns_p95, 4),
        'G3': g3,
        'G3_label': g3_label,
        'growth_v2': round(growth, 4),
        'growth_label': growth_label,
        'final_verdict': 'G1=%s(%s)/G2=%s(%s)/G3=%s(%s)/growth=%s(%s)'
                         % (g1, g1_label, g2, g2_label, g3, g3_label,
                            round(growth, 4), growth_label),
    }

    result = {'phase': 2868, 'prereg': PREREG, 'verdict': v,
              'seed_null': SEED, 'n_perm': N_PERM}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'growth_v2.npz',
           B2w=B2w.astype(np.float32),
           B2s=B2s.astype(np.float32),
           B23=B23.astype(np.float32),
           sig_mask=sig.astype(np.bool_),
           eta2_weight=w.astype(np.float32),
           labels=labels.astype(np.int64))

    elapsed = time.monotonic() - t0
    cc.ledger('phase2868', elapsed)
    print('P2868 VERDICT %s' % json.dumps(v), flush=True)
    print('P2868 elapsed %.1fs' % elapsed, flush=True)


if __name__ == '__main__':
    main()
