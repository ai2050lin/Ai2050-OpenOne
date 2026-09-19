"""Phase 2869 (LPF MA2 cont): density-matched fusion -- effective point
of the mechanism-base growth curve.

Context: 2868 showed naive hstack of the low-dim dense block B3
(10-d, acc .875) with the high-dim sparse-signal block B2s (43-d, acc
.425) is interference-dominated (acc B23 .7375 < B3).  The growth
curve needs density-matched fusion: F(a) = unit([B3, a * B2s]) with a
sweep.

Blocks (80 words, zero forward; all from immutable npz):
  B3  mlp response spectrum (10)   [2867 npz]
  B2s significant-head causal spectrum (43) [rebuild: 2867 B2 x mask 2868]
a grid (frozen): {0.25, 0.5, 1.0, 2.0, 4.0}; a=0 reduces to B3.

Prereg (frozen before any readout):
  H1  (main) let a* = argmax_a acc(F(a)); fusion_gain iff
      acc(F(a*)) > acc(B3) AND acc(F(a*)) > max-alpha null p95
      (200 label permutations, SEED=2869; each permutation takes its
      own max over the alpha grid -- corrects selection bias).
      else fusion_no_gain.
  H2  a* < 1 => density_matched_fusion (low-dim dense block should
      dominate); a* >= 1 => sparse_block_dominates.
  H3  effective growth point v3:
      growth = (acc(F(a*)) - acc(B3)) / max(1 - acc(B3), 1e-9);
      <0.1 sublinear_reuse, >=0.1 additive_gain (only meaningful if
      H1 true).
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
OUT = BASE / 'phase2869' / 'fusion_curve'
SRC_2867 = BASE / 'phase2867' / 'word_coords_v1' / 'word_coords_v1.npz'
SRC_2868 = BASE / 'phase2868' / 'growth_v2' / 'growth_v2.npz'
SEED = 2869
N_PERM = 200
ALPHAS = (0.25, 0.5, 1.0, 2.0, 4.0)

PREREG = {
    'H1': 'a* = argmax_a acc(F(a)); fusion_gain iff acc(F(a*)) > acc(B3) '
          'and > max-alpha null p95 (200 perms, SEED=2869) else '
          'fusion_no_gain',
    'H2': 'a* < 1 => density_matched_fusion; a* >= 1 => '
          'sparse_block_dominates',
    'H3': 'growth = (acc(F(a*))-acc(B3))/max(1-acc(B3),1e-9); <0.1 '
          'sublinear_reuse, >=0.1 additive_gain',
}


def unit(x):
    return x / max(float(np.linalg.norm(x)), 1e-30)


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
                     'design': 'density-matched fusion curve '
                               'F(a)=unit([B3, a*B2s]), a grid '
                               '(0.25,0.5,1,2,4), max-alpha null; '
                               'zero forward'}
        fc.save(execution_path, execution)

    z67 = np.load(SRC_2867, allow_pickle=True)
    B2 = z67['B2'].astype(np.float64)
    B3 = z67['B3'].astype(np.float64)
    labels = z67['labels'].astype(np.int64)
    n_words = len(labels)

    z68 = np.load(SRC_2868, allow_pickle=True)
    sig = z68['sig_mask'].astype(bool)
    assert sig.sum() == 43

    B3u = np.stack([unit(B3[i]) for i in range(n_words)])
    B2s = np.stack([unit(B2[i][sig]) for i in range(n_words)])

    def fused(a):
        raw = np.hstack([B3u, a * B2s])
        return np.stack([unit(raw[i]) for i in range(n_words)])

    accs = {}
    Cmats = {}
    for a in ALPHAS:
        C = fused(a) @ fused(a).T
        Cmats[a] = C
        accs[a] = loo_nn_acc(C, labels)
    C3 = B3u @ B3u.T
    acc_b3 = loo_nn_acc(C3, labels)

    a_star = max(ALPHAS, key=lambda a: accs[a])
    acc_star = accs[a_star]

    # max-alpha null (selection-bias corrected)
    rng = np.random.default_rng(SEED)
    null_max = []
    null_delta = []
    for _ in range(N_PERM):
        pl = rng.permutation(labels)
        best = -1.0
        for a in ALPHAS:
            best = max(best, loo_nn_acc(Cmats[a], pl))
        null_max.append(best)
        null_delta.append(best - loo_nn_acc(C3, pl))
    nm_p95 = float(np.percentile(null_max, 95))
    nd_p95 = float(np.percentile(null_delta, 95))

    h1 = bool(acc_star > acc_b3 and acc_star > nm_p95)
    h1_label = 'fusion_gain' if h1 else 'fusion_no_gain'
    h2_label = 'density_matched_fusion' if a_star < 1.0 \
        else 'sparse_block_dominates'
    headroom = max(1.0 - acc_b3, 1e-9)
    growth = (acc_star - acc_b3) / headroom
    h3_label = 'sublinear_reuse' if growth < 0.1 else 'additive_gain'

    v = {
        'n_words': n_words,
        'acc_B3': round(acc_b3, 4),
        'acc_by_alpha': {str(a): round(accs[a], 4) for a in ALPHAS},
        'alpha_star': a_star,
        'acc_star': round(acc_star, 4),
        'null_max_p95': round(nm_p95, 4),
        'null_delta_p95': round(nd_p95, 4),
        'H1': h1,
        'H1_label': h1_label,
        'H2_label': h2_label,
        'growth_v3': round(growth, 4),
        'H3_label': h3_label,
        'final_verdict': 'H1=%s(%s)/H2=%s/growth=%s(%s)'
                         % (h1, h1_label, h2_label,
                            round(growth, 4), h3_label),
    }

    result = {'phase': 2869, 'prereg': PREREG, 'verdict': v,
              'seed_null': SEED, 'n_perm': N_PERM, 'alphas': ALPHAS}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'fusion_curve.npz',
           B3=B3u.astype(np.float32),
           B2s=B2s.astype(np.float32),
           labels=labels.astype(np.int64))

    elapsed = time.monotonic() - t0
    cc.ledger('phase2869', elapsed)
    print('P2869 VERDICT %s' % json.dumps(v), flush=True)
    print('P2869 elapsed %.1fs' % elapsed, flush=True)


if __name__ == '__main__':
    main()
