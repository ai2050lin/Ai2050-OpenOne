"""Phase 2864 (LPF TMA-1b): word-level variance decomposition of the
2846 census -- does class signal exist at all, and in which heads?

2863 showed 8-word class slices are noise-dominated (J1/J2/J3 false).
This phase decomposes per-head variance directly, zero-forward:

For each head h, over the 80 word rows x_c(w):
  eta2[h] = SS_between / (SS_between + SS_within)
  SS_between = 8 * sum_c (m_c - m)^2 ;  SS_within = sum_c sum_w (x - m_c)^2
Tensor-level: R2_class = sum_h SS_between / sum_h SS_total.

Null: 200 random relabelings of the 80 rows into 10x8 (SEED=2864).
Prereg (frozen before readout):
  V1  class_variance_present iff max_h eta2[h] > p95 of null max_h
      eta2 (family-wise via max-statistic); report count of heads
      with eta2 > per-head null p95 vs expected-false 1152*0.05.
  V2  descriptive: Spearman(eta2, mean_drop) over 1152 heads; layer
      distribution of eta2-top20; overlap of eta2-top20 with 2846
      top-64 causal front edge.
  V3  class_signal_in_tensor iff R2_class > p95 of null R2_class.
      This is the verdict 2863 could not give: is class signal
      present-but-diluted, or absent?
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
OUT = BASE / 'phase2864' / 'class_variance'
SRC_CENSUS = BASE / 'phase2846' / 'fullhead_census' / 'census_full.npz'
NL, NH = 36, 32
N_NULL = 200
NULL_SEED = 2864

PREREG = {
    'V1': 'class_variance_present iff max_h eta2[h] > p95 of null '
          'max-statistic (200 relabelings); report significant-head '
          'count vs expected-false 1152*0.05',
    'V2': 'descriptive: Spearman(eta2, mean_drop); eta2-top20 layer '
          'mix; eta2-top20 vs 2846 top-64 overlap',
    'V3': 'class_signal_in_tensor iff R2_class > p95 of null R2_class '
          '(200 relabelings) -- present-but-diluted vs absent',
}


def eta2_per_head(mat, labels):
    """mat (80, H) rows=words; labels (80,) in 0..9, 8 rows each."""
    H = mat.shape[1]
    m_all = mat.mean(0)
    ss_b = np.zeros(H)
    ss_w = np.zeros(H)
    for c in range(10):
        rows = mat[labels == c]
        m_c = rows.mean(0)
        ss_b += rows.shape[0] * (m_c - m_all) ** 2
        ss_w += ((rows - m_c) ** 2).sum(0)
    return ss_b / np.maximum(ss_b + ss_w, 1e-30)


def main():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)

    execution_path = OUT / 'execution.json'
    if not execution_path.exists():
        execution = {'timestamp': fc.stamp(),
                     'source': cc.snapshot(__file__),
                     'prereg': PREREG, 'null_seed': NULL_SEED,
                     'design': 'per-head eta2 class-variance '
                               'decomposition of 2846 drops_all, '
                               '200 relabeling nulls'}
        fc.save(execution_path, execution)

    z = np.load(SRC_CENSUS)
    mat = z['drops_all'].astype(np.float64).reshape(80, -1)  # (80, 1152)
    mean_drop = z['mean_drop'].astype(np.float64)

    labels = np.repeat(np.arange(10), 8)
    eta2 = eta2_per_head(mat, labels)

    # R2_class = sum_h SS_b / sum_h SS_total, computed directly:
    m_all = mat.mean(0)
    ss_b_vec = np.zeros(mat.shape[1])
    ss_t_vec = ((mat - m_all) ** 2).sum(0)
    for c in range(10):
        rows = mat[labels == c]
        ss_b_vec += rows.shape[0] * (rows.mean(0) - m_all) ** 2
    r2_obs = float(ss_b_vec.sum() / max(ss_t_vec.sum(), 1e-30))

    rng = np.random.default_rng(NULL_SEED)
    max_null, r2_null = [], []
    for _ in range(N_NULL):
        perm = rng.permutation(80)
        lab_n = perm // 8          # 10 groups of 8 in permuted order
        e_n = eta2_per_head(mat, lab_n)
        max_null.append(float(e_n.max()))
        mb = np.zeros(mat.shape[1])
        mt = ((mat[perm] - mat[perm].mean(0)) ** 2).sum(0)
        for c in range(10):
            rows = mat[perm][lab_n == c]
            mb += rows.shape[0] * (rows.mean(0)
                                   - mat[perm].mean(0)) ** 2
        r2_null.append(float(mb.sum() / max(mt.sum(), 1e-30)))
    max_p95 = float(np.percentile(max_null, 95))
    r2_p95 = float(np.percentile(r2_null, 95))

    # per-head significance count (approx, vs per-head null p95)
    sig_per_head = np.zeros(mat.shape[1])
    for _ in range(N_NULL):
        perm = rng.permutation(80)
        e_n = eta2_per_head(mat, perm // 8)
        sig_per_head += (e_n > eta2).astype(np.int64)
    p_per_head = sig_per_head / N_NULL
    n_sig = int((p_per_head >= 0.95).sum())

    v1 = bool(float(eta2.max()) > max_p95)
    v3 = bool(r2_obs > r2_p95)

    order = np.argsort(eta2)[::-1]
    top20 = order[:20]
    top64 = set(np.argsort(mean_drop)[::-1][:64].tolist())
    v2 = {
        'spearman_eta2_vs_meandrop': round(float(np.corrcoef(
            np.argsort(np.argsort(eta2)).astype(np.float64),
            np.argsort(np.argsort(mean_drop)).astype(np.float64)
        )[0, 1]), 4),
        'eta2_top20_layers': sorted(set((top20 // NH).tolist())),
        'eta2_top20_overlap_top64': int(len(set(top20.tolist())
                                            & top64)),
        'eta2_top20_heads': ['L%dH%d(%.3f)' % (h // NH, h % NH, eta2[h])
                             for h in top20[:8]],
    }

    v = {
        'V1_class_variance_present': v1,
        'eta2_max_obs': round(float(eta2.max()), 4),
        'eta2_max_null_p95': round(max_p95, 4),
        'n_sig_heads_p95': n_sig,
        'expected_false': int(mat.shape[1] * 0.05),
        'V3_class_signal_in_tensor': v3,
        'r2_class_obs': round(r2_obs, 4),
        'r2_class_null_p95': round(r2_p95, 4),
        'V2_structure': v2,
        'final_verdict': 'present%s/diluted-check_V3=%s'
                         % ('' if v1 else '_absent', v3),
    }
    v['final_verdict'] = ('class_variance_present' if v1
                          else 'class_variance_absent') \
        + ('|tensor_signal' if v3 else '|tensor_noise')

    result = {'phase': 2864, 'prereg': PREREG, 'verdict': v}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'class_variance.npz',
           eta2=eta2.astype(np.float32),
           p_per_head=p_per_head.astype(np.float32))

    elapsed = time.monotonic() - t0
    cc.ledger('phase2864', elapsed)
    print('P2864 VERDICT %s' % json.dumps(v), flush=True)
    print('P2864 elapsed %.1fs' % elapsed, flush=True)


if __name__ == '__main__':
    main()
