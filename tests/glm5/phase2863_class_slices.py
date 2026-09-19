"""Phase 2863 (LPF TMA-1): per-class slice atlas of the 2846 census.

Zero-forward.  drops_all (80=10 classes x 8 words, 36, 32) already
contains ALL ten classes; every prior verdict was pooled over words.
This phase slices the tensor per class and tests, against a
row-permutation null (200 draws, seed frozen), whether class-level
mechanism structure exists at all:

  J1  class_signal_above_noise iff median_c Spearman(S_c, mean of
      other 9 class slices) > p95 of the same statistic under random
      row grouping (10 groups x 8 words).  Tests: do class slices
      share signal beyond word-sampling noise?
  J2  shared_core_real iff #{heads in >=7 of 10 class top-64s} >= 1
      AND > p95 of the same count under random grouping.  Tests: is
      there a cross-class front-edge core?
  J3  formatter_role_persistent iff mean_c fraction of global
      formatters (drop>=p75_all & direct<=p50_all) that stay
      drop>=p75 within the class slice > p95 under random grouping.

If all three hold: overall running picture = shared mechanism base x
per-class coordinates (H-shared + H-specific).  Descriptives: core
head list (layer, head, class-freq), per-class overlap with global
top-64, pairwise class similarity matrix.
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
OUT = BASE / 'phase2863' / 'class_slices'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SRC_CENSUS = BASE / 'phase2846' / 'fullhead_census' / 'census_full.npz'
NL, NH = 36, 32
N_NULL = 200
NULL_SEED = 2863
CORE_MIN_CLASSES = 7

PREREG = {
    'J1': 'class_signal_above_noise iff median_c Spearman(S_c, '
          'mean_other9) > p95 null (200 row-permutation groupings)',
    'J2': 'shared_core_real iff #{heads in >=7/10 class top-64s} >= 1 '
          'AND > p95 null',
    'J3': 'formatter_role_persistent iff mean_c frac of global '
          'formatters (drop>=p75_all & direct<=p50_all) staying '
          'drop>=p75_c > p95 null',
    'verdict': 'base_plus_coordinates iff J1 & J2 & J3 all true; '
               'else per-judgment flags',
}


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def stats_for_groups(mat, direct, glob_form):
    """mat: (80, 1152) rows = words.  10 groups x 8 rows in order."""
    n_g = mat.shape[0] // 10
    slices = [mat[g * n_g:(g + 1) * n_g].mean(0) for g in range(10)]
    # J1: leave-one-out
    j1s = []
    for g in range(10):
        others = np.mean([slices[k] for k in range(10) if k != g], axis=0)
        j1s.append(spearman(slices[g], others))
    # J2: core count
    freq = np.zeros(mat.shape[1])
    for g in range(10):
        top = np.argsort(slices[g])[::-1][:64]
        freq[top] += 1
    j2 = int((freq >= CORE_MIN_CLASSES).sum())
    # J3: formatter persistence
    fracs = []
    for g in range(10):
        p75 = float(np.percentile(slices[g], 75))
        keep = slices[g][glob_form] >= p75
        fracs.append(float(keep.mean()))
    return float(np.median(j1s)), j2, float(np.mean(fracs))


def main():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    exec2806 = json.loads(SRC_2806_EXEC.read_text(encoding='utf-8'))
    CAT_WORDS = list(exec2806['cats'].keys())

    execution_path = OUT / 'execution.json'
    if not execution_path.exists():
        execution = {'timestamp': fc.stamp(),
                     'source': cc.snapshot(__file__),
                     'prereg': PREREG, 'null_seed': NULL_SEED,
                     'design': 'per-class slice atlas of 2846 census, '
                               '10 classes x 8 words, three '
                               'preregistered judgments vs '
                               'row-permutation null'}
        fc.save(execution_path, execution)

    z = np.load(SRC_CENSUS)
    drops = z['drops_all'].astype(np.float64)      # (80, 36, 32)
    mean_s0 = z['mean_s0'].astype(np.float64)
    mean_s1 = z['mean_s1'].astype(np.float64)
    mat = drops.reshape(drops.shape[0], -1)        # (80, 1152)
    n_heads = mat.shape[1]
    direct = mean_s0 + mean_s1

    S_all = mat.mean(0)
    p75_all = float(np.percentile(S_all, 75))
    p50_dir = float(np.percentile(direct, 50))
    glob_form = np.where((S_all >= p75_all) & (direct <= p50_dir))[0]
    g_top64 = set(np.argsort(S_all)[::-1][:64].tolist())

    # ---------- observed ----------
    j1_obs, j2_obs, j3_obs = stats_for_groups(mat, direct, glob_form)

    slices = {c: mat[g * 8:(g + 1) * 8].mean(0)
              for g, c in enumerate(CAT_WORDS)}
    pair = {}
    offdiag = []
    for i, ca in enumerate(CAT_WORDS):
        for cb in CAT_WORDS[i + 1:]:
            r = spearman(slices[ca], slices[cb])
            pair['%s~%s' % (ca, cb)] = round(r, 4)
            offdiag.append(r)
    ov64 = {c: len(g_top64 & set(np.argsort(slices[c])[::-1][:64]
                                 .tolist()))
            for c in CAT_WORDS}
    freq = np.zeros(n_heads)
    for c in CAT_WORDS:
        top = np.argsort(slices[c])[::-1][:64]
        freq[top] += 1
    core_idx = np.where(freq >= CORE_MIN_CLASSES)[0]
    core_list = sorted(core_idx.tolist(),
                       key=lambda h: -freq[h])[:24]

    # ---------- null ----------
    rng = np.random.default_rng(NULL_SEED)
    j1_n, j2_n, j3_n = [], [], []
    for _ in range(N_NULL):
        perm = rng.permutation(mat.shape[0])
        a, b, c_ = stats_for_groups(mat[perm], direct, glob_form)
        j1_n.append(a)
        j2_n.append(b)
        j3_n.append(c_)
    j1_p95 = float(np.percentile(j1_n, 95))
    j2_p95 = float(np.percentile(j2_n, 95))
    j3_p95 = float(np.percentile(j3_n, 95))

    J1 = bool(j1_obs > j1_p95)
    J2 = bool(j2_obs >= 1 and j2_obs > j2_p95)
    J3 = bool(j3_obs > j3_p95)
    verdict = ('base_plus_coordinates' if (J1 and J2 and J3)
               else '/'.join(['J1=%s' % J1, 'J2=%s' % J2, 'J3=%s' % J3]))

    v = {
        'n_classes': len(CAT_WORDS),
        'J1_class_signal_above_noise': J1,
        'j1_obs_median': round(j1_obs, 4),
        'j1_null_p95': round(j1_p95, 4),
        'J2_shared_core_real': J2,
        'j2_core_count': j2_obs,
        'j2_null_p95': round(j2_p95, 4),
        'J3_formatter_role_persistent': J3,
        'j3_obs_mean_keep': round(j3_obs, 4),
        'j3_null_p95': round(j3_p95, 4),
        'n_global_formatters': int(len(glob_form)),
        'final_verdict': verdict,
        'core_heads_top24': ['L%dH%d(x%d)' % (h // NH, h % NH, freq[h])
                             for h in core_list],
        'overlap64_per_class': ov64,
        'pairwise_spearman': pair,
        'offdiag_mean': round(float(np.mean(offdiag)), 4),
        'offdiag_min': round(float(np.min(offdiag)), 4),
        'offdiag_max': round(float(np.max(offdiag)), 4),
    }

    result = {'phase': 2863, 'prereg': PREREG, 'verdict': v}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'class_slices.npz',
           slices=np.stack([slices[c] for c in CAT_WORDS])
           .astype(np.float32),
           core_freq=freq.astype(np.int32),
           glob_form=glob_form.astype(np.int64))

    elapsed = time.monotonic() - t0
    cc.ledger('phase2863', elapsed)
    print('P2863 VERDICT %s' % json.dumps(v), flush=True)
    print('P2863 elapsed %.1fs' % elapsed, flush=True)


if __name__ == '__main__':
    main()
