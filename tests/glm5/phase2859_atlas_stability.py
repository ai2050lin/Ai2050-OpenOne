"""Phase 2859 (MA / Atlas wrap): head-atlas sampling stability.

Zero-forward analysis over the immutable phase2846 census.  2856-2858
quantified vocab sensitivity at the cdir level (prototypicality
gradient); this phase pushes it to the atlas level: how stable are the
head rankings / front-edge identities / C1-C3 judgements under word
subsampling?

Data: phase2846 fullhead_census drops_all (80, 36, 32) + per-layer
census_L{li}.npz s0/s1 (80, 32) reassembled into (80, 36, 32).

Arms:
  A  split half: mean drop per head over words 0-39 vs 40-79 ->
     Spearman over 1152 heads, top-10 identity overlap,
     per-half C2 (frac_top64 load)
  B  bootstrap: 200 draws of 40 words without replacement ->
     rank distribution of the 2846 top-10 heads, Jaccard(top-64),
     C2 share, C3 Spearman(direct write, drop) per draw

Judgements (frozen, SEED=2859):
  S1  atlas_stable iff Spearman(front40, back40) >= 0.7
  S2  frontedge_stable iff median bootstrap rank of the 2846 top-10
      heads <= 16
  S3  c2_robust iff >= 90% of draws have frac_top64 >= 0.25
  S4  c3_robust iff >= 90% of draws have Spearman(direct, drop) < 0.3
      (bipolar orthogonality preserved)
  verdict: atlas_robust = S1 and S2 and S3 and S4
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
OUT = BASE / 'phase2859' / 'atlas_stability'
REF = BASE / 'phase2846' / 'fullhead_census'
SEED = 2859
NL, NH = 36, 32
NHEADS = NL * NH
B = 200
SUB = 40

PREREG = {
    'S1': 'atlas_stable iff Spearman(mean_drop_front40, '
          'mean_drop_back40) over 1152 heads >= 0.7',
    'S2': 'frontedge_stable iff median bootstrap rank of the 2846 '
          'top-10 heads <= 16',
    'S3': 'c2_robust iff >=90% of 200 draws have frac_top64 >= 0.25',
    'S4': 'c3_robust iff >=90% of 200 draws have '
          'Spearman(direct_write, drop) < 0.3 (orthogonality)',
    'verdict': 'atlas_robust = S1 and S2 and S3 and S4',
}


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    sa, sb = np.std(ra), np.std(rb)
    if sa <= 1e-12 or sb <= 1e-12:
        return float('nan')
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)

    execution_path = OUT / 'execution.json'
    if not execution_path.exists():
        execution = {'timestamp': fc.stamp(),
                     'source': cc.snapshot(__file__),
                     'prereg': PREREG, 'seed': SEED,
                     'design': 'head-atlas word-subsampling stability '
                               'over immutable phase2846 census '
                               '(zero forward)'}
        fc.save(execution_path, execution)

    z = np.load(REF / 'census_full.npz')
    drops = z['drops_all'].astype(np.float64)      # (80, 36, 32)
    assert drops.shape == (80, NL, NH), drops.shape
    s0_all = np.stack([np.load(REF / ('census_L%d.npz' % li))['s0']
                       for li in range(NL)], axis=1).astype(np.float64)
    s1_all = np.stack([np.load(REF / ('census_L%d.npz' % li))['s1']
                       for li in range(NL)], axis=1).astype(np.float64)
    assert s0_all.shape == (80, NL, NH)

    flat = drops.reshape(80, NHEADS)
    mean_full = flat.mean(0)
    load_full = np.maximum(mean_full, 0.0)
    order_full = np.argsort(-load_full)
    top10_full = order_full[:10]

    def metrics(rows):
        md = flat[rows].mean(0)
        load = np.maximum(md, 0.0)
        total = float(load.sum())
        o = np.argsort(-load)
        frac64 = float(load[o[:64]].sum()) / max(total, 1e-30)
        direct = (s0_all[rows] + s1_all[rows]) \
            .reshape(len(rows), NHEADS).mean(0)
        c3 = spearman(direct, md)
        return md, frac64, c3, o

    # ---------- Arm A: split half ----------
    md_f, c2_f, c3_f, o_f = metrics(np.arange(40))
    md_b, c2_b, c3_b, o_b = metrics(np.arange(40, 80))
    s1_val = spearman(md_f, md_b)
    top10_overlap = len(set(o_f[:10]) & set(o_b[:10]))
    s1 = bool(np.isfinite(s1_val) and s1_val >= 0.7)

    # ---------- Arm B: bootstrap ----------
    rng = np.random.default_rng(SEED)
    ranks_of_top10 = np.zeros((B, 10))
    frac64s = np.zeros(B)
    c3s = np.zeros(B)
    jac64s = np.zeros(B)
    set_full64 = set(order_full[:64].tolist())
    for b in range(B):
        rows = rng.choice(80, size=SUB, replace=False)
        md, frac64, c3, o = metrics(rows)
        r = np.empty(NHEADS)
        r[o] = np.arange(NHEADS)
        ranks_of_top10[b] = r[top10_full]
        frac64s[b] = frac64
        c3s[b] = c3
        jac64s[b] = len(set_full64 & set(o[:64].tolist())) / \
            len(set_full64 | set(o[:64].tolist()))
    med_rank = float(np.median(ranks_of_top10.mean(1)))
    s2 = bool(med_rank <= 16.0)
    s3 = bool((frac64s >= 0.25).mean() >= 0.90)
    s4 = bool((c3s < 0.3).mean() >= 0.90)

    verdict = {
        'S1_atlas_stable': s1,
        'split_spearman_1152': round(float(s1_val), 4),
        'split_top10_overlap_of_10': top10_overlap,
        'split_c2_front': round(c2_f, 4),
        'split_c2_back': round(c2_b, 4),
        'split_c3_front': round(c3_f, 4),
        'split_c3_back': round(c3_b, 4),
        'S2_frontedge_stable': s2,
        'median_bootstrap_rank_top10': round(med_rank, 2),
        'top10_bootstrap_rank_medians': [
            round(float(np.median(ranks_of_top10[:, k])), 1)
            for k in range(10)],
        'S3_c2_robust': s3,
        'c2_frac64_dist': {'p05': round(float(np.percentile(frac64s, 5)), 4),
                           'p50': round(float(np.percentile(frac64s, 50)), 4),
                           'p95': round(float(np.percentile(frac64s, 95)), 4)},
        'S4_c3_robust': s4,
        'c3_dist': {'p05': round(float(np.percentile(c3s, 5)), 4),
                    'p50': round(float(np.percentile(c3s, 50)), 4),
                    'p95': round(float(np.percentile(c3s, 95)), 4)},
        'jac64_top64_dist': {'p05': round(float(np.percentile(jac64s, 5)), 4),
                             'p50': round(float(np.percentile(jac64s, 50)), 4)},
        'final_verdict': 'atlas_robust=%s' % (s1 and s2 and s3 and s4),
    }

    result = {'phase': 2859, 'prereg': PREREG, 'verdict': verdict}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'atlas_stability.npz',
           ranks_of_top10=ranks_of_top10.astype(np.float32),
           frac64s=frac64s.astype(np.float64),
           c3s=c3s.astype(np.float64),
           jac64s=jac64s.astype(np.float64),
           mean_full_drop=mean_full.astype(np.float64),
           top10_full=top10_full.astype(np.int64))

    elapsed = time.monotonic() - t0
    cc.ledger('phase2859', elapsed)
    print('P2859 VERDICT %s' % json.dumps(verdict), flush=True)
    print('P2859 elapsed %.1fs' % elapsed, flush=True)


if __name__ == '__main__':
    main()
