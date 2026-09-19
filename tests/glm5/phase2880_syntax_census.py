# -*- coding: utf-8 -*-
"""Phase 2880: syntax-axis causal census on the legal vocabulary from
phase2878 (3 axes / 48 words / 22 pairs).

Goal: complete the channel-dissociation quadrant table.  The class axis
is drop-spectrum ORGANIZED (43-head frontedge) + mlp carrier (0.875);
the attribute axis is drop-spectrum UNORGANIZED (2875/2876 double
negative) + mlp carrier (0.5, 2877).  The syntax axis already has an
mlp carrier (2879 E1 true 0.7083) - this phase measures its drop-spectrum
organization with the identical census machinery.

Protocol: phase2875 verbatim (2872 protocol via SyntaxCensus subclass
of AtlasCensus), vocabulary from the frozen 2878 artifact.
Prereg (frozen before any readout):
  Y1  syntax frontedge: heads in >= 3 of 3 per-axis top-32 lists
      (full overlap; 2875 used >=6/8 = 75%, with 3 axes the graded
      equivalent 2/3 is NOT used - full overlap is the stricter
      preregistered choice); significant iff count > null p95
      (200 random 3x32-head subsets, SEED=2880)
      => syntax_frontedge_exists.
  Y2  mechanism-side separation: |Spearman rho(syntax spectrum, 2846
      class spectrum)| < 0.3 => mechanism_side_separate.
  Y3  population independence: hypergeometric p of
      |syntax_top64 ^ class_top64|; shared_substrate iff p < 0.05.
  Y4  descriptive: per-axis top-8 head lists; rho(syntax, eta2).
  Y5  power check: per-word LOO-NN axis retrieval acc on the 48-word
      causal spectrum vs full-pipeline-mirrored permutation null
      (200, SEED=2880) - the channel-dissociation verdict for the
      syntax quadrant.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc
from rdc_atlas_census import AtlasCensus

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2880' / 'syntax_census'
MODEL_DIR = ROOT / 'models' / 'hf' / 'qwen3-4b'
SRC_2846 = BASE / 'phase2846' / 'fullhead_census' / 'census_full.npz'
SRC_2864 = BASE / 'phase2864' / 'class_variance' / 'class_variance.npz'
SRC_2878 = BASE / 'phase2878' / 'syntax_trans_vocab' / \
    'syntax_trans_vocab.npz'
SEED = 2880
N_NULL = 200
NH, NL = 32, 36
LAST = 35

PREREG = {
    'Y1': 'heads in >=3/3 per-axis top-32 lists > null p95 (200 random '
          '3x32-head subsets, SEED=2880) => syntax_frontedge_exists',
    'Y2': '|Spearman rho(syntax spectrum, 2846 class spectrum)| < 0.3 '
          '=> mechanism_side_separate',
    'Y3': 'hypergeometric p of |syntax_top64 ^ class_top64|; '
          'shared_substrate iff p < 0.05',
    'Y4': 'descriptive per-axis top-8 lists; rho(syntax, eta2)',
    'Y5': 'LOO-NN axis retrieval on 48-word spectrum; acc > null p95 '
          '(full-pipeline-mirrored, 200 perms, SEED=2880) => '
          'axis_signal_detected',
}


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    d = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def hyper_p(k, K, n, N):
    from math import comb
    if K == 0 or n == 0:
        return 1.0
    hi = min(K, n)
    lo = max(0, n + K - N)
    tot = comb(N, n)
    num = sum(comb(K, i) * comb(N - K, n - i)
              for i in range(max(lo, k), hi + 1))
    return num / tot


class SyntaxCensus(AtlasCensus):
    """2846 machinery with axis-structured vocabulary (2872 verbatim)."""

    def conds2_for(self, i, axis, w):
        w_tid = self.tid(w)
        others = [x for x in self.targets[axis] if x != w]
        assert others, 'axis with single word: %s' % axis
        same = min(others, key=lambda x: self.tid(x))
        return {'same': [self.tid(same), w_tid],
                'func': [self.func_tid, w_tid],
                'null': [self.null_tids[i], w_tid]}


def zmat(M):
    mu = M.mean(axis=0, keepdims=True)
    sd = M.std(axis=0, keepdims=True)
    Z = (M - mu) / np.maximum(sd, 1e-30)
    n = np.linalg.norm(Z, axis=1, keepdims=True)
    return Z / np.maximum(n, 1e-30)


def loo_acc(C, lab):
    S = C @ C.T
    np.fill_diagonal(S, -2.0)
    nn = S.argmax(axis=1)
    return float(np.mean(lab[nn] == lab))


def main():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)

    execution_path = OUT / 'execution.json'
    if not execution_path.exists():
        execution = {'timestamp': fc.stamp(),
                     'source': cc.snapshot(__file__),
                     'prereg': PREREG, 'seed': SEED,
                     'design': 'syntax-axis census: 3 axes / 48 words '
                               'from 2878 legal vocab; 2875 protocol; '
                               'Y1/Y2/Y3 gates + Y5 channel verdict'}
        fc.save(execution_path, execution)

    zv = np.load(SRC_2878, allow_pickle=True)
    axis_words = json.loads(str(zv['targets']))
    AXES = json.loads(str(zv['pairs_json']))
    axes = list(zv['axes'])
    target_list = [(a, w) for a in axes for w in axis_words[a]]
    n_words = len(target_list)
    dW_unit = zv['dW_unit'].astype(np.float64)
    null_tids = {i: int(zv['null_tids'][i]) for i in range(n_words)}
    func_tid = int(zv['func_tid'])
    tid_map = json.loads(str(zv['tid_map']))
    print('P2880 axes=%d words=%d' % (len(axes), n_words), flush=True)

    import torch
    from phase2662_symmetric_mapping_contract import load_native
    model, _ = load_native('qwen4')
    model.eval()

    vocab = {'targets': axis_words, 'target_list': target_list,
             'dW_unit': dW_unit, 'null_tids': null_tids,
             'func_tid': func_tid, 'single_tok': sorted(
                 {w for ws in axis_words.values() for w in ws}),
             'tid_map': tid_map, 'n_words': n_words}

    census = SyntaxCensus(model, vocab, AXES, axes, last=LAST)

    drops = np.zeros((n_words, NL, NH))
    for li in range(NL):
        dL, s0L, s1L, cr = census.measure_layer(li)
        drops[:, li, :] = dL
        print('P2880 L%d done max_cr=%.4f' % (li, max(cr)), flush=True)

    per_axis = {}
    for ai, a in enumerate(axes):
        rows = [i for i, (aa, _) in enumerate(target_list) if aa == a]
        per_axis[a] = drops[rows].mean(axis=0).flatten()
    spec_syntax = np.stack([per_axis[a] for a in axes]).mean(axis=0)

    z46 = np.load(SRC_2846, allow_pickle=True)
    spec_class = z46['mean_drop'].astype(np.float64)
    z64 = np.load(SRC_2864, allow_pickle=True)
    eta2 = z64['eta2'].astype(np.float64)

    # ---------- Y1 ----------
    top32_per_axis = [set(np.argsort(per_axis[a])[::-1][:32].tolist())
                      for a in axes]
    counts = np.zeros(NL * NH)
    for s in top32_per_axis:
        for h in s:
            counts[h] += 1
    obs_max = int(counts.max())
    obs_n3 = int((counts >= 3).sum())
    rng2 = np.random.default_rng(SEED)
    null_n3 = []
    for _ in range(N_NULL):
        c = np.zeros(NL * NH)
        for _ in axes:
            for h in rng2.integers(0, NL * NH, size=32):
                c[h] += 1
        null_n3.append(int((c >= 3).sum()))
    n3_p95 = float(np.percentile(null_n3, 95))
    y1 = bool(obs_n3 > n3_p95)
    y1_label = 'syntax_frontedge_exists' if y1 \
        else 'syntax_frontedge_absent'

    # ---------- Y2 / Y3 ----------
    rho_class = spearman(spec_syntax, spec_class)
    rho_eta2 = spearman(spec_syntax, eta2)
    y2 = bool(abs(rho_class) < 0.3)
    y2_label = 'mechanism_side_separate' if y2 \
        else 'mechanism_side_aligned'

    syn_top64 = set(np.argsort(spec_syntax)[::-1][:64].tolist())
    class_top64 = set(np.argsort(spec_class)[::-1][:64].tolist())
    inter = len(syn_top64 & class_top64)
    p3 = hyper_p(inter, 64, 64, NL * NH)
    y3_label = ('shared_substrate' if p3 < 0.05
                else 'independent_populations')

    # ---------- Y5: word-level retrieval at 48 words ----------
    X = drops.reshape(n_words, -1)
    lab = np.array([axes.index(a) for a, _ in target_list])
    acc_full = loo_acc(zmat(X), lab)
    rng3 = np.random.default_rng(SEED)
    null_acc = []
    for _ in range(N_NULL):
        pl = rng3.permutation(lab)
        null_acc.append(loo_acc(zmat(X), pl))
    na = np.array(null_acc)
    y5 = bool(acc_full > float(np.percentile(na, 95)))
    y5_label = 'axis_signal_detected' if y5 else 'axis_signal_absent'

    v = {
        'n_axes': len(axes), 'n_words': n_words,
        'axis_word_counts': {a: len(axis_words[a]) for a in axes},
        'Y1_max_axis_overlap': obs_max,
        'Y1_n_heads_ge3': obs_n3,
        'Y1_null_n3_p95': n3_p95,
        'Y1': y1, 'Y1_label': y1_label,
        'Y2_rho_class': round(rho_class, 4),
        'Y2_rho_eta2': round(rho_eta2, 4),
        'Y2': y2, 'Y2_label': y2_label,
        'Y3_intersection': inter,
        'Y3_hyper_p': round(p3, 6),
        'Y3_label': y3_label,
        'Y5_acc_full': round(acc_full, 4),
        'Y5_null_p95': round(float(np.percentile(na, 95)), 4),
        'Y5_null_mean': round(float(na.mean()), 4),
        'Y5': y5, 'Y5_label': y5_label,
        'per_axis_top8': {a: ['L%dH%d' % (h // NH, h % NH)
                              for h in np.argsort(
                                  per_axis[a])[::-1][:8].tolist()]
                          for a in axes},
        'final_verdict': 'Y1=%s(%s)/Y2=%s(%s,rho=%.3f)/Y3=%s/Y5=%s(%s)'
                         % (y1, y1_label, y2, y2_label, rho_class,
                            y3_label, y5, y5_label),
    }

    result = {'phase': 2880, 'prereg': PREREG, 'verdict': v,
              'seed_null': SEED}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'syntax_census.npz',
           drops=drops.astype(np.float32),
           per_axis=np.stack([per_axis[a] for a in axes])
           .astype(np.float32),
           spec_syntax=spec_syntax.astype(np.float32),
           axis_names=np.array(axes, dtype=object),
           target_list=np.array(['%s:%s' % t for t in target_list],
                                dtype=object))

    elapsed = time.monotonic() - t0
    cc.ledger('phase2880', elapsed)
    print('P2880 VERDICT %s' % json.dumps(v), flush=True)
    print('P2880 elapsed %.1fs' % elapsed, flush=True)

    del model


if __name__ == '__main__':
    main()
