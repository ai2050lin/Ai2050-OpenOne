"""Phase 2871 (LPF MA2 cont): significant-head triple identity --
unifying three independent "important head" definitions (zero forward).

Sets (1152 heads = 36 layers x 32 heads):
  A  causal load      : top-64 by mean_drop (2846 census_full)
  B  class variance   : eta2-significant, p_per_head < 0.01 (2864)
                        [= 2868 B2s mask, n=43]
  C  stable frontedge : top10_full bootstrap-stable (2859)

Note: 2864's n_sig_heads_p95=102 used (p_per_head >= 0.95), the wrong
tail (descriptive only, never entered a verdict); erratum noted. The
correct significant-large set is p_per_head < 0.01.

Prereg (frozen before any readout):
  U1  intersection enrichment: |A^B|, |A^C|, |B^C|, |A^B^C| each
      judged vs hypergeometric tail (population 1152) at alpha=0.05
      (Bonferroni x4); core_confirmed iff all pairwise intersections
      enriched AND triple non-empty.
  U2  role mix of the consensus core (heads in >=2 sets): formatter /
      amplifier / other fractions with 2862 frozen quantiles
      (formatter drop>=p75 & direct<=p50; amplifier direct>=p75 &
      drop<=p50; direct = mean_s0+mean_s1). Descriptive.
  U3  load curve: cumulative mean_drop share of heads sorted desc,
      reported at |A u B u C| and at the >=2 core; unity_share_core
      iff core (>=2) carries >= 0.30 of total causal load.
  U4  descriptive: head lists per set and per intersection to npz/json.
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
OUT = BASE / 'phase2871' / 'sig_heads_triple'
SRC_CENSUS = BASE / 'phase2846' / 'fullhead_census' / 'census_full.npz'
SRC_2864 = BASE / 'phase2864' / 'class_variance' / 'class_variance.npz'
SRC_2859 = BASE / 'phase2859' / 'atlas_stability' / 'atlas_stability.npz'
NH, NL = 32, 36
N_POP = NH * NL

PREREG = {
    'U1': 'pairwise intersections hypergeometric-enriched (Bonferroni '
          'x4, alpha 0.05) and triple non-empty => core_confirmed',
    'U2': 'descriptive role mix of >=2 core with 2862 frozen quantiles',
    'U3': '>=2 core carries >= 0.30 of total causal load => '
          'unity_share_core',
    'U4': 'descriptive head lists to npz/json',
}


def hyper_p(k, K, n, N):
    """P(X >= k) for Hypergeometric(N, K, n)."""
    from math import comb
    if K == 0 or n == 0:
        return 1.0
    hi = min(K, n)
    lo = max(0, n + K - N)
    tot = comb(N, n)
    num = sum(comb(K, i) * comb(N - K, n - i)
              for i in range(max(lo, k), hi + 1))
    return num / tot


def hname(h):
    return 'L%dH%d' % (h // NH, h % NH)


def main():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)

    execution_path = OUT / 'execution.json'
    if not execution_path.exists():
        execution = {'timestamp': fc.stamp(),
                     'source': cc.snapshot(__file__),
                     'prereg': PREREG, 'seed': None,
                     'design': 'triple identity of significant heads: '
                               '2846 top-64 causal x 2864 eta2-sig43 x '
                               '2859 top-10 stable; enrichment + role '
                               'mix + load curve; zero forward'}
        fc.save(execution_path, execution)

    z46 = np.load(SRC_CENSUS, allow_pickle=True)
    mean_drop = z46['mean_drop'].astype(np.float64)
    direct = (z46['mean_s0'].astype(np.float64)
              + z46['mean_s1'].astype(np.float64))
    z64 = np.load(SRC_2864, allow_pickle=True)
    p_per_head = z64['p_per_head'].astype(np.float64)
    z59 = np.load(SRC_2859, allow_pickle=True)
    top10 = z59['top10_full'].astype(int)

    A = set(np.argsort(mean_drop)[::-1][:64].tolist())
    B = set(np.where(p_per_head < 0.01)[0].tolist())
    C = set(top10.tolist())

    inter = {
        'A_and_B': A & B, 'A_and_C': A & C, 'B_and_C': B & C,
        'A_and_B_and_C': A & B & C,
    }
    # populations: A=64, B=43, C=10
    u1_ps = {
        'A_and_B': hyper_p(len(A & B), len(A), len(B), N_POP),
        'A_and_C': hyper_p(len(A & C), len(A), len(C), N_POP),
        'B_and_C': hyper_p(len(B & C), len(B), len(C), N_POP),
    }
    u1 = bool(all(p < 0.05 / 4 for p in u1_ps.values())
              and len(inter['A_and_B_and_C']) > 0)
    u1_label = 'core_confirmed' if u1 else 'core_not_confirmed'

    core2 = (A & B) | (A & C) | (B & C)

    drop_p75 = float(np.percentile(mean_drop, 75))
    drop_p50 = float(np.percentile(mean_drop, 50))
    dir_p75 = float(np.percentile(direct, 75))
    dir_p50 = float(np.percentile(direct, 50))
    is_form = (mean_drop >= drop_p75) & (direct <= dir_p50)
    is_ampl = (direct >= dir_p75) & (mean_drop <= drop_p50)

    cl = sorted(core2)
    n_form = int(is_form[cl].sum())
    n_ampl = int(is_ampl[cl].sum())
    n_other = len(cl) - n_form - n_ampl

    total_load = float(mean_drop.sum())
    order = np.argsort(mean_drop)[::-1]
    cum = np.cumsum(mean_drop[order]) / total_load
    rank_of = {h: i + 1 for i, h in enumerate(order.tolist())}
    union = A | B | C
    share_union = float(sum(mean_drop[h] for h in union) / total_load)
    share_core = float(sum(mean_drop[h] for h in cl) / total_load)
    # alternative: rank-based cumulative at max rank of core members
    u3 = bool(share_core >= 0.30)
    u3_label = 'unity_share_core' if u3 else 'core_minor_load'

    v = {
        'n_A': len(A), 'n_B': len(B), 'n_C': len(C),
        'n_A_and_B': len(A & B), 'n_A_and_C': len(A & C),
        'n_B_and_C': len(B & C), 'n_triple': len(A & B & C),
        'U1_hyper_p': {k: round(p, 6) for k, p in u1_ps.items()},
        'U1': u1,
        'U1_label': u1_label,
        'n_core2': len(cl),
        'U2_roles': {'formatter': n_form, 'amplifier': n_ampl,
                     'other': n_other},
        'U3_share_union': round(share_union, 4),
        'U3_share_core2': round(share_core, 4),
        'U3': u3,
        'U3_label': u3_label,
        'core2_heads': [hname(h) for h in cl],
        'triple_heads': [hname(h) for h in sorted(A & B & C)],
        'final_verdict': 'U1=%s(%s)/core2=%d/roles=f%d-a%d-o%d/'
                         'share_core2=%.3f(%s)'
                         % (u1, u1_label, len(cl), n_form, n_ampl,
                            n_other, share_core, u3_label),
    }

    result = {'phase': 2871, 'prereg': PREREG, 'verdict': v}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'sig_heads_triple.npz',
           mean_drop=mean_drop.astype(np.float32),
           direct=direct.astype(np.float32),
           p_per_head=p_per_head.astype(np.float32),
           set_A=np.array(sorted(A), dtype=np.int64),
           set_B=np.array(sorted(B), dtype=np.int64),
           set_C=np.array(sorted(C), dtype=np.int64),
           core2=np.array(cl, dtype=np.int64))

    elapsed = time.monotonic() - t0
    cc.ledger('phase2871', elapsed)
    print('P2871 VERDICT %s' % json.dumps(v), flush=True)
    print('P2871 elapsed %.1fs' % elapsed, flush=True)


if __name__ == '__main__':
    main()
