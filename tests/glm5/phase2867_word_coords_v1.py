"""Phase 2867 (LPF MA2 cont): per-word coordinates v1 -- three-block
complementarity + first point of the mechanism-base growth curve.

Blocks (80 words, SEED=2855 vocab):
  B1 unembed projection      (10 dims)   proj[w,c] = E_w . dW_unit[c]
  B2 causal spectrum         (1152 dims) drops_all[w].flatten()   (2846)
  B3 mlp response spectrum   (10 dims)   g_direct_true[w, L26-35]  (2861)
each unit-normed; combined blocks are concatenated then unit-normed.

Prereg (frozen before any readout):
  T1  complementarity (descriptive): rho12 / rho13 / rho23 = Spearman
      of upper-triangle cosine matrices; three_block_complementary iff
      all three < 0.3
  T2  growth curve (main): leave-one-out nearest-neighbour same-class
      accuracy for B1, B2, B3, B1+B2, B1+B2+B3; final combo judged
      against null p95 (200 label permutations, SEED=2867); delta-acc
      per added block reported -- sublinear iff final acc <= max(single
      acc) + 0.05 (reuse image), additive if clearly above
  T3  B3 word-level class structure: mean margin > null p95 (same
      permutations) => mlp_spectrum_class_structure
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
OUT = BASE / 'phase2867' / 'word_coords_v1'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SRC_CENSUS = BASE / 'phase2846' / 'fullhead_census' / 'census_full.npz'
SRC_2861 = BASE / 'phase2861' / 'g_true' / 'g_true.npz'
SEED = 2855
MAX_WORDS = 8

PREREG = {
    'T1': 'complementarity descriptive: rho12/rho13/rho23 < 0.3 all -> '
          'three_block_complementary',
    'T2': 'growth curve: LOO-NN same-class acc for B1/B2/B3/B12/B123; '
          'final combo vs null p95 (200 perms, SEED=2867); sublinear '
          'iff final acc <= max(single)+0.05',
    'T3': 'B3 mean margin > null p95 => mlp_spectrum_class_structure',
}


def unit(x):
    return x / max(float(np.linalg.norm(x)), 1e-30)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    d = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / d) if d > 0 else 0.0


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
    exec2806 = json.loads(SRC_2806_EXEC.read_text(encoding='utf-8'))
    CATS = exec2806['cats']
    CAT_WORDS = list(CATS.keys())

    execution_path = OUT / 'execution.json'
    if not execution_path.exists():
        execution = {'timestamp': fc.stamp(),
                     'source': cc.snapshot(__file__),
                     'prereg': PREREG, 'seed': SEED,
                     'design': 'three-block word coordinates (B1 unembed '
                               '10-d, B2 causal 1152-d, B3 mlp response '
                               '10-d), complementarity + growth curve '
                               'first point'}
        fc.save(execution_path, execution)

    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        str(ROOT / 'models' / 'hf' / 'qwen3-4b'), local_files_only=True,
        trust_remote_code=True, use_fast=True)
    from phase2662_symmetric_mapping_contract import load_native
    model, _ = load_native('qwen4')
    model.eval()
    W_U = model.lm_head.weight.detach().float().cpu().numpy()

    tc = {}

    def tid(t):
        if t not in tc:
            ids = tok(' ' + t, add_special_tokens=False)['input_ids']
            if len(ids) != 1:
                ids = tok(t, add_special_tokens=False)['input_ids']
            assert len(ids) == 1, '%s -> %s' % (t, ids)
            tc[t] = int(ids[0])
        return tc[t]

    all_words = [w for v in CATS.values() for w in v]
    single_tok = []
    for w in all_words:
        try:
            tid(w)
            single_tok.append(w)
        except AssertionError:
            pass
    targets = {}
    for cat in CAT_WORDS:
        targets[cat] = [w for w in CATS[cat]
                        if w in single_tok][:MAX_WORDS]
    target_list = [(cat, w) for cat in CAT_WORDS for w in targets[cat]]
    words = [w for _, w in target_list]
    labels = np.array([CAT_WORDS.index(c) for c, _ in target_list])
    n_words = len(words)
    assert n_words == 80, n_words

    Erows = {w: W_U[tid(w)].astype(np.float64) for w in single_tok}
    cents = []
    for cat in CAT_WORDS:
        ws = [w for w in CATS[cat] if w in single_tok]
        cents.append(np.stack([Erows[w] for w in ws]).mean(0))
    Cm = np.stack(cents)
    dW = Cm - (Cm.sum(0, keepdims=True) - Cm) / 9.0
    dW_unit = np.stack([unit(dW[i]) for i in range(10)])

    B1 = np.stack([unit(np.array([Erows[w] @ dW_unit[c]
                                  for c in range(10)]))
                   for w in words])
    z = np.load(SRC_CENSUS)
    B2 = np.stack([unit(z['drops_all'][i].astype(np.float64).flatten())
                   for i in range(n_words)])
    z61 = np.load(SRC_2861)
    B3 = np.stack([unit(z61['g_direct'][i].astype(np.float64))
                   for i in range(n_words)])

    def cosmat(B):
        return B @ B.T

    C1, C2, C3 = cosmat(B1), cosmat(B2), cosmat(B3)
    iu = np.triu_indices(n_words, 1)
    rho12 = spearman(C1[iu], C2[iu])
    rho13 = spearman(C1[iu], C3[iu])
    rho23 = spearman(C2[iu], C3[iu])
    t1 = bool(max(abs(rho12), abs(rho13), abs(rho23)) < 0.3)

    combos = {
        'B1': B1, 'B2': B2, 'B3': B3,
        'B12': np.hstack([B1, B2]),
        'B123': np.hstack([B1, B2, B3]),
    }
    accs = {}
    Cmats = {}
    for name, B in combos.items():
        Bu = np.stack([unit(B[i]) for i in range(n_words)])
        Cmats[name] = cosmat(Bu)
        accs[name] = loo_nn_acc(Cmats[name], labels)

    rng = np.random.default_rng(2867)
    Cfin = Cmats['B123']
    null_acc = []
    for _ in range(200):
        pl = rng.permutation(labels)
        null_acc.append(loo_nn_acc(Cfin, pl))
    null_p95 = float(np.percentile(null_acc, 95))
    single_max = max(accs['B1'], accs['B2'], accs['B3'])
    growth = accs['B123'] - single_max
    t2_sig = bool(accs['B123'] > null_p95)
    t2_sublinear = bool(growth <= 0.05)
    t2_label = 'sublinear_reuse' if (t2_sig and t2_sublinear) \
        else ('additive_info' if t2_sig else 'no_signal')

    same = (labels[:, None] == labels[None, :])
    offm = ~np.eye(n_words, dtype=bool)

    def mean_margin(C):
        m = []
        for i in range(n_words):
            m.append(C[i][same[i] & offm[i]].mean()
                     - C[i][(~same[i]) & offm[i]].mean())
        return float(np.mean(m))

    def null_margin(C):
        vals = []
        for _ in range(200):
            pl = rng.permutation(labels)
            plm = (pl[:, None] == pl[None, :])
            sm = plm & offm
            df = (~plm) & offm
            mm = []
            for i in range(n_words):
                mm.append(C[i][sm[i]].mean() - C[i][df[i]].mean())
            vals.append(float(np.mean(mm)))
        return vals

    m3 = mean_margin(C3)
    nm3 = null_margin(C3)
    t3 = bool(m3 > float(np.percentile(nm3, 95)))

    v = {
        'n_words': n_words,
        'T1_three_block_complementary': t1,
        'rho12': round(rho12, 4),
        'rho13': round(rho13, 4),
        'rho23': round(rho23, 4),
        'acc_B1': round(accs['B1'], 4),
        'acc_B2': round(accs['B2'], 4),
        'acc_B3': round(accs['B3'], 4),
        'acc_B12': round(accs['B12'], 4),
        'acc_B123': round(accs['B123'], 4),
        'acc_null_p95': round(null_p95, 4),
        'growth_final_vs_best_single': round(growth, 4),
        'T2_signal': t2_sig,
        'T2_growth_label': t2_label,
        'T3_mlp_class_structure': t3,
        't3_margin_mean': round(m3, 5),
        't3_null_p95': round(float(np.percentile(nm3, 95)), 5),
        'final_verdict': 'T1=%s/T2=%s(%s)/T3=%s'
                         % (t1, t2_sig, t2_label, t3),
    }

    result = {'phase': 2867, 'prereg': PREREG, 'verdict': v,
              'words': words, 'labels': labels.tolist()}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'word_coords_v1.npz',
           B1=B1.astype(np.float32),
           B2=B2.astype(np.float32),
           B3=B3.astype(np.float32),
           C1=C1.astype(np.float32),
           C2=C2.astype(np.float32),
           C3=C3.astype(np.float32),
           labels=labels.astype(np.int64))

    elapsed = time.monotonic() - t0
    cc.ledger('phase2867', elapsed)
    print('P2867 VERDICT %s' % json.dumps(v), flush=True)
    print('P2867 elapsed %.1fs' % elapsed, flush=True)

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
