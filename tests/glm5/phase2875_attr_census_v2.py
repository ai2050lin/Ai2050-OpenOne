# -*- coding: utf-8 -*-
"""Phase 2875: attribute-axis causal census v2 on the expanded legal
vocabulary from phase2874 (8 axes / 42 words / 26 pairs).

Protocol: phase2872 unchanged (2846 machinery via AttrCensus subclass),
except the vocabulary comes from the frozen 2874 artifact (VG-gated).
Prereg (frozen before any readout), gates identical in form to 2872:
  X1  attr frontedge: heads in >= 6 of 8 per-axis top-32 lists;
      significant iff count > null p95 (200 random 32-head subsets per
      axis, SEED=2875) => attr_frontedge_exists.
  X2  mechanism-side separation: |Spearman rho(cross-axis mean attr
      spectrum, 2846 class spectrum)| < 0.3 => mechanism_side_separate.
  X3  population independence: hypergeometric p of
      |attr_top64 ^ class_top64|; independent_populations iff p >= 0.05.
  X4  descriptive: per-axis top-8 head lists; rho(attr, eta2).
  X5  power check vs 2873: per-word LOO-NN axis retrieval acc on the
      42-word causal spectrum vs full-pipeline-mirrored permutation null
      (200, SEED=2875) -- the 2873 C1 gate re-run at higher word count.
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
OUT = BASE / 'phase2875' / 'attr_census_v2'
MODEL_DIR = ROOT / 'models' / 'hf' / 'qwen3-4b'
SRC_2846 = BASE / 'phase2846' / 'fullhead_census' / 'census_full.npz'
SRC_2864 = BASE / 'phase2864' / 'class_variance' / 'class_variance.npz'
SRC_2874 = BASE / 'phase2874' / 'attr_vocab_v2' / 'attr_vocab_v2.npz'
SEED = 2875
N_NULL = 200
NH, NL = 32, 36
LAST = 35

PREREG = {
    'X1': 'heads in >=6/8 per-axis top-32 lists > null p95 (200 random '
          '32-head subsets per axis, SEED=2875) => attr_frontedge_exists',
    'X2': '|Spearman rho(attr spectrum, 2846 class spectrum)| < 0.3 => '
          'mechanism_side_separate',
    'X3': 'hypergeometric p of |attr_top64 ^ class_top64|; '
          'independent_populations iff p >= 0.05',
    'X4': 'descriptive per-axis top-8 lists; rho(attr, eta2)',
    'X5': 'LOO-NN axis retrieval on 42-word spectrum; acc > null p95 '
          '(full-pipeline-mirrored, 200 perms, SEED=2875) => '
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


class AttrCensus(AtlasCensus):
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
                     'design': 'attribute-axis census v2: 8 axes / 42 words '
                               'from 2874 legal vocab; 2872 protocol; '
                               'X1/X2/X3 gates + X5 power re-check'}
        fc.save(execution_path, execution)

    zv = np.load(SRC_2874, allow_pickle=True)
    axis_words = json.loads(str(zv['targets']))
    AXES = json.loads(str(zv['pairs_json']))
    axes = list(zv['axes'])
    target_list = [(a, w) for a in axes for w in axis_words[a]]
    n_words = len(target_list)
    dW_unit = zv['dW_unit'].astype(np.float64)
    null_tids = {i: int(zv['null_tids'][i]) for i in range(n_words)}
    func_tid = int(zv['func_tid'])
    tid_map = json.loads(str(zv['tid_map']))
    print('P2875 axes=%d words=%d' % (len(axes), n_words), flush=True)

    import torch
    from phase2662_symmetric_mapping_contract import load_native
    model, _ = load_native('qwen4')
    model.eval()

    vocab = {'targets': axis_words, 'target_list': target_list,
             'dW_unit': dW_unit, 'null_tids': null_tids,
             'func_tid': func_tid, 'single_tok': sorted(
                 {w for ws in axis_words.values() for w in ws}),
             'tid_map': tid_map, 'n_words': n_words}

    census = AttrCensus(model, vocab, AXES, axes, last=LAST)

    drops = np.zeros((n_words, NL, NH))
    for li in range(NL):
        dL, s0L, s1L, cr = census.measure_layer(li)
        drops[:, li, :] = dL
        print('P2875 L%d done max_cr=%.4f' % (li, max(cr)), flush=True)

    per_axis = {}
    for ai, a in enumerate(axes):
        rows = [i for i, (aa, _) in enumerate(target_list) if aa == a]
        per_axis[a] = drops[rows].mean(axis=0).flatten()
    spec_attr = np.stack([per_axis[a] for a in axes]).mean(axis=0)

    z46 = np.load(SRC_2846, allow_pickle=True)
    spec_class = z46['mean_drop'].astype(np.float64)
    z64 = np.load(SRC_2864, allow_pickle=True)
    eta2 = z64['eta2'].astype(np.float64)

    # ---------- X1 ----------
    top32_per_axis = [set(np.argsort(per_axis[a])[::-1][:32].tolist())
                      for a in axes]
    counts = np.zeros(NL * NH)
    for s in top32_per_axis:
        for h in s:
            counts[h] += 1
    obs_max = int(counts.max())
    obs_n6 = int((counts >= 6).sum())
    rng2 = np.random.default_rng(SEED)
    null_n6 = []
    for _ in range(N_NULL):
        c = np.zeros(NL * NH)
        for _ in axes:
            for h in rng2.integers(0, NL * NH, size=32):
                c[h] += 1
        null_n6.append(int((c >= 6).sum()))
    n6_p95 = float(np.percentile(null_n6, 95))
    x1 = bool(obs_n6 > n6_p95)
    x1_label = 'attr_frontedge_exists' if x1 else 'attr_frontedge_absent'

    # ---------- X2 / X3 ----------
    rho_class = spearman(spec_attr, spec_class)
    rho_eta2 = spearman(spec_attr, eta2)
    x2 = bool(abs(rho_class) < 0.3)
    x2_label = 'mechanism_side_separate' if x2 \
        else 'mechanism_side_aligned'

    attr_top64 = set(np.argsort(spec_attr)[::-1][:64].tolist())
    class_top64 = set(np.argsort(spec_class)[::-1][:64].tolist())
    inter = len(attr_top64 & class_top64)
    p3 = hyper_p(inter, 64, 64, NL * NH)
    x3_label = ('shared_substrate' if p3 < 0.05
                else 'independent_populations')

    # ---------- X5: word-level retrieval at 42 words ----------
    X = drops.reshape(n_words, -1)
    lab = np.array([axes.index(a) for a, _ in target_list])
    acc_full = loo_acc(zmat(X), lab)
    rng3 = np.random.default_rng(SEED)
    null_acc = []
    for _ in range(N_NULL):
        pl = rng3.permutation(lab)
        null_acc.append(loo_acc(zmat(X), pl))
    na = np.array(null_acc)
    x5 = bool(acc_full > float(np.percentile(na, 95)))
    x5_label = 'axis_signal_detected' if x5 else 'axis_signal_absent'

    v = {
        'n_axes': len(axes), 'n_words': n_words,
        'axis_word_counts': {a: len(axis_words[a]) for a in axes},
        'X1_max_axis_overlap': obs_max,
        'X1_n_heads_ge6': obs_n6,
        'X1_null_n6_p95': n6_p95,
        'X1': x1, 'X1_label': x1_label,
        'X2_rho_class': round(rho_class, 4),
        'X2_rho_eta2': round(rho_eta2, 4),
        'X2': x2, 'X2_label': x2_label,
        'X3_intersection': inter,
        'X3_hyper_p': round(p3, 6),
        'X3_label': x3_label,
        'X5_acc_full': round(acc_full, 4),
        'X5_null_p95': round(float(np.percentile(na, 95)), 4),
        'X5_null_mean': round(float(na.mean()), 4),
        'X5': x5, 'X5_label': x5_label,
        'per_axis_top8': {a: ['L%dH%d' % (h // NH, h % NH)
                              for h in np.argsort(
                                  per_axis[a])[::-1][:8].tolist()]
                          for a in axes},
        'final_verdict': 'X1=%s(%s)/X2=%s(%s,rho=%.3f)/X3=%s/X5=%s(%s)'
                         % (x1, x1_label, x2, x2_label, rho_class,
                            x3_label, x5, x5_label),
    }

    result = {'phase': 2875, 'prereg': PREREG, 'verdict': v,
              'seed_null': SEED}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'attr_census_v2.npz',
           drops=drops.astype(np.float32),
           per_axis=np.stack([per_axis[a] for a in axes])
           .astype(np.float32),
           spec_attr=spec_attr.astype(np.float32),
           axis_names=np.array(axes, dtype=object),
           target_list=np.array(['%s:%s' % t for t in target_list],
                                dtype=object))

    elapsed = time.monotonic() - t0
    cc.ledger('phase2875', elapsed)
    print('P2875 VERDICT %s' % json.dumps(v), flush=True)
    print('P2875 elapsed %.1fs' % elapsed, flush=True)

    del model


if __name__ == '__main__':
    main()
