"""Phase 2872 (LPF MA2 cont): attribute-axis causal census -- where does
the attributional axis live in mechanism space?  (2846 protocol port)

Word family: 11 attribute axes (28 single-token polar words) from the
2870 pair table, merged within axis:
  size{big,small,large,little,huge,tiny} speed{fast,slow,quick}
  temp{hot,cold,warm,cool} age{old,new,young} weight{heavy,light}
  strength{strong,weak} brightness{bright,dark} moisture{wet,dry}
  height{tall,short} fullness{full,empty}
cdir per axis = unit(mean of unit(E(w+)-E(w-)) over its 2870 pairs).
same-context word (2846 'same' analog) = deterministic: first other
word of the same axis by tid order (polarity-agnostic; limitation
logged).  func/null conditions identical to 2846 (SEED=2872 nulls).

Measurement: rdc_atlas_census.AtlasCensus machinery unchanged except
conds2_for overridden (AttrCensus subclass).  Full 36-layer x 28-word
census -> drops_attr (28,36,32); per-axis spectra averaged over words.

Prereg (frozen before any readout):
  X1  (main) attr frontedge: heads appearing in >= 6 of 11 per-axis
      top-32 lists; significant iff count > null p95 (200 random
      32-head subsets per axis, SEED=2872) => attr_frontedge_exists.
  X2  mechanism-side separation: Spearman rho between cross-axis mean
      attr drop spectrum (1152) and 2846 class drop spectrum < 0.3
      => mechanism_side_separate.
  X3  population independence: hypergeometric p of
      |attr_top64 ^ class_top64|; independent_populations iff
      p >= 0.05 (no enrichment), shared_substrate iff p < 0.05.
  X4  descriptive: per-axis top-8 head lists; rho(attr, eta2).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc
from rdc_atlas_census import AtlasCensus, unit, single_token_id

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2872' / 'attr_census'
MODEL_DIR = ROOT / 'models' / 'hf' / 'qwen3-4b'
SRC_2846 = BASE / 'phase2846' / 'fullhead_census' / 'census_full.npz'
SRC_2864 = BASE / 'phase2864' / 'class_variance' / 'class_variance.npz'
SEED = 2872
N_NULL = 200
NH, NL = 32, 36
LAST = 35

AXES = {
    'size': [('size1', 'big', 'small'), ('size2', 'large', 'little'),
             ('size3', 'huge', 'tiny')],
    'speed': [('speed1', 'fast', 'slow'), ('speed2', 'quick', 'slow')],
    'temp': [('temp1', 'hot', 'cold'), ('temp2', 'warm', 'cool')],
    'age': [('age1', 'old', 'new'), ('age2', 'young', 'old')],
    'weight': [('weight', 'heavy', 'light')],
    'strength': [('strength', 'strong', 'weak')],
    'brightness': [('brightness', 'bright', 'dark')],
    'moisture': [('moisture', 'wet', 'dry')],
    'height': [('height', 'tall', 'short')],
    'fullness': [('fullness', 'full', 'empty')],
}
# note: 2870 'temp2' warm/cool counted; axes merged as declared.

PREREG = {
    'X1': 'heads in >=6/11 per-axis top-32 lists > null p95 (200 '
          'random 32-head subsets per axis, SEED=2872) => '
          'attr_frontedge_exists',
    'X2': 'Spearman rho(cross-axis mean attr spectrum, 2846 class '
          'spectrum) < 0.3 => mechanism_side_separate',
    'X3': 'hypergeometric p of |attr_top64 ^ class_top64|; '
          'independent_populations iff p >= 0.05',
    'X4': 'descriptive per-axis top-8 lists; rho(attr, eta2)',
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
    """2846 machinery with axis-structured vocabulary."""

    def conds2_for(self, i, axis, w):
        w_tid = self.tid(w)
        others = [x for x in self.targets[axis] if x != w]
        assert others, 'axis with single word: %s' % axis
        same = min(others, key=lambda x: self.tid(x))
        return {'same': [self.tid(same), w_tid],
                'func': [self.func_tid, w_tid],
                'null': [self.null_tids[i], w_tid]}


def main():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)

    execution_path = OUT / 'execution.json'
    if not execution_path.exists():
        execution = {'timestamp': fc.stamp(),
                     'source': cc.snapshot(__file__),
                     'prereg': PREREG, 'seed': SEED,
                     'design': 'attribute-axis causal census: 11 axes / '
                               '28 words, 2846 machinery via AttrCensus '
                               'subclass, full 36-layer spectrum; '
                               'X1 frontedge / X2 mechanism separation '
                               '/ X3 population independence'}
        fc.save(execution_path, execution)

    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        str(MODEL_DIR), local_files_only=True,
        trust_remote_code=True, use_fast=True)
    from phase2662_symmetric_mapping_contract import load_native
    model, _ = load_native('qwen4')
    model.eval()

    W_U = model.lm_head.weight.detach().float().cpu().numpy()

    # ---------- vocabulary (2846 construction order) ----------
    tc = {}
    axis_words = {}
    for axis, pairs in AXES.items():
        ws = []
        for _, wp, wm in pairs:
            for w in (wp, wm):
                if w not in ws:
                    ws.append(w)
        kept = []
        for w in ws:
            try:
                single_token_id(tok, w, tc)
                kept.append(w)
            except AssertionError:
                pass
        axis_words[axis] = kept
    axes = [a for a in AXES if axis_words[a]]
    target_list = [(a, w) for a in axes for w in axis_words[a]]
    n_words = len(target_list)
    print('P2872 axes=%d words=%d' % (len(axes), n_words), flush=True)

    axis_dirs = []
    for a in axes:
        dirs = [unit(W_U[single_token_id(tok, wp, tc)].astype(np.float64)
                     - W_U[single_token_id(tok, wm, tc)].astype(np.float64))
                for _, wp, wm in AXES[a]
                if wp in axis_words[a] and wm in axis_words[a]]
        axis_dirs.append(unit(np.stack(dirs).mean(axis=0)))
    dW_unit = np.stack(axis_dirs)

    rng = np.random.default_rng(SEED)
    vocab_size = W_U.shape[0]
    word_tids = set(tc.values())
    null_tids = {}
    while len(null_tids) < n_words:
        r = int(rng.integers(0, vocab_size))
        if r not in word_tids and r > 0:
            null_tids[len(null_tids)] = r
    func_tid = single_token_id(tok, 'the', tc)

    vocab = {'targets': axis_words, 'target_list': target_list,
             'dW_unit': dW_unit, 'null_tids': null_tids,
             'func_tid': func_tid,
             'single_tok': sorted({w for ws in axis_words.values()
                                   for w in ws}),
             'tid_map': dict(tc), 'n_words': n_words}

    census = AttrCensus(model, vocab, AXES, axes, last=LAST)

    # ---------- full census ----------
    drops = np.zeros((n_words, NL, NH))
    for li in range(NL):
        dL, s0L, s1L, cr = census.measure_layer(li)
        drops[:, li, :] = dL
        print('P2872 L%d done max_cr=%.4f' % (li, max(cr)), flush=True)

    per_axis = {}
    for ai, a in enumerate(axes):
        rows = [i for i, (aa, _) in enumerate(target_list) if aa == a]
        per_axis[a] = drops[rows].mean(axis=0).flatten()  # 1152

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
    null_max = []
    for _ in range(N_NULL):
        c = np.zeros(NL * NH)
        for _ in axes:
            for h in rng2.integers(0, NL * NH, size=32):
                c[h] += 1
        null_max.append(int(c.max()))
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

    v = {
        'n_axes': len(axes), 'n_words': n_words,
        'axis_word_counts': {a: len(axis_words[a]) for a in axes},
        'X1_max_axis_overlap': obs_max,
        'X1_n_heads_ge6': obs_n6,
        'X1_null_n6_p95': n6_p95,
        'X1': x1,
        'X1_label': x1_label,
        'X2_rho_class': round(rho_class, 4),
        'X2_rho_eta2': round(rho_eta2, 4),
        'X2': x2,
        'X2_label': x2_label,
        'X3_intersection': inter,
        'X3_hyper_p': round(p3, 6),
        'X3_label': x3_label,
        'per_axis_top8': {a: ['L%dH%d' % (h // NH, h % NH)
                              for h in np.argsort(
                                  per_axis[a])[::-1][:8].tolist()]
                          for a in axes},
        'final_verdict': 'X1=%s(%s)/X2=%s(%s,rho=%.3f)/X3=%s'
                         % (x1, x1_label, x2, x2_label, rho_class,
                            x3_label),
    }

    result = {'phase': 2872, 'prereg': PREREG, 'verdict': v,
              'seed_null': SEED}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'attr_census.npz',
           drops=drops.astype(np.float32),
           per_axis=np.stack([per_axis[a] for a in axes])
           .astype(np.float32),
           spec_attr=spec_attr.astype(np.float32),
           axis_names=np.array(axes, dtype=object),
           target_list=np.array(['%s:%s' % t for t in target_list],
                                dtype=object))

    elapsed = time.monotonic() - t0
    cc.ledger('phase2872', elapsed)
    print('P2872 VERDICT %s' % json.dumps(v), flush=True)
    print('P2872 elapsed %.1fs' % elapsed, flush=True)

    del model


if __name__ == '__main__':
    main()
