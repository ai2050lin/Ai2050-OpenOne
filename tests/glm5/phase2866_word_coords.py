"""Phase 2866 (LPF MA2 cont): per-word mechanism coordinates v0 --
minimal entity of the word-family atlas (TMA L2).

Question: does the per-word CAUSAL spectrum (2846 drops_all, 1152
head-dims) carry class structure at the WORD level, or is it word-
specific noise (2863 class-mean verdict suggests the latter)?  And do
the unembed projection block (B1, class-structure positive control per
2857) and the mechanism block (B2) carry DIFFERENT information?

Blocks (80 words, SEED=2855 vocab rebuilt deterministically):
  B1 proj[w,c] = E_w . dW_unit[c]          (10 dims, unit-normed)
  B2 drop[w]  = drops_all[w].flatten()     (1152 dims, unit-normed)

Prereg (frozen before any readout):
  W1 word_mechanism_class_structure (B2):
     margin[w] = mean cos(w, same-class) - mean cos(w, diff-class);
     TRUE iff mean_w margin > null p95 (200 random label permutations,
     SEED=2866)
  W2 unembed_class_structure (B1): same test (positive control;
     2857 predicts TRUE)
  W3 block_complementarity: Spearman(upper-tri cos_B1, upper-tri cos_B2);
     descriptive (low rho => different information)
  W4 word_specificity_floor (B2): per-word max cos to any OTHER word
     (1 - that = nearest-neighbour mechanism distance); descriptive
     distribution (user's token-speciality principle: no two words
     share a mechanism)
  W5 leave-one-out retrieval (B2): nearest neighbour same-class iff;
     TRUE iff accuracy > null p95 (same permutations)
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
OUT = BASE / 'phase2866' / 'word_coords'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SRC_CENSUS = BASE / 'phase2846' / 'fullhead_census' / 'census_full.npz'
SEED = 2855
MAX_WORDS = 8

PREREG = {
    'W1': 'B2 (1152-d causal spectrum): mean_w margin > null p95 '
          '(200 label permutations, SEED=2866) => '
          'word_mechanism_class_structure',
    'W2': 'B1 (10-d unembed projection): same test, positive control',
    'W3': 'descriptive: Spearman(upper-tri cos_B1, cos_B2)',
    'W4': 'descriptive: per-word max cos to other words (B2), '
          'nearest-neighbour mechanism distance floor',
    'W5': 'B2 leave-one-out nearest-neighbour same-class accuracy '
          '> null p95',
}


def unit(x):
    n = np.linalg.norm(x)
    return x / max(float(n), 1e-30)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    d = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / d) if d > 0 else 0.0


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
                     'design': 'per-word mechanism coordinates v0: B1 '
                               'unembed projection (10-d) vs B2 causal '
                               'spectrum (1152-d), class structure at '
                               'word level, permutation nulls'}
        fc.save(execution_path, execution)

    # --- rebuild SEED=2855 target_list verbatim (2860 protocol) ---
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

    # --- blocks ---
    B1 = np.stack([unit(np.array([Erows[w] @ dW_unit[c]
                                  for c in range(10)]))
                   for w in words])
    z = np.load(SRC_CENSUS)
    drops_all = z['drops_all'].astype(np.float64)   # (80, 36, 32)
    B2 = np.stack([unit(drops_all[i].flatten())
                   for i in range(n_words)])

    def cosmat(B):
        return B @ B.T

    C1 = cosmat(B1)
    C2 = cosmat(B2)
    same = (labels[:, None] == labels[None, :])
    off = ~np.eye(n_words, dtype=bool)

    def mean_margin(C):
        m = []
        for i in range(n_words):
            s = C[i][same[i] & off[i]].mean()
            d = C[i][(~same[i]) & off[i]].mean()
            m.append(s - d)
        return np.array(m)

    def null_margins(C, rng):
        vals = []
        for _ in range(200):
            pl = rng.permutation(labels)
            plm = (pl[:, None] == pl[None, :])
            sm = plm & off
            df = (~plm) & off
            mm = []
            for i in range(n_words):
                mm.append(C[i][sm[i]].mean() - C[i][df[i]].mean())
            vals.append(float(np.mean(mm)))
        return vals

    rng = np.random.default_rng(2866)
    m1 = mean_margin(C2)
    null1 = null_margins(C2, rng)
    w1 = bool(m1.mean() > float(np.percentile(null1, 95)))
    m2 = mean_margin(C1)
    null2 = null_margins(C1, rng)
    w2 = bool(m2.mean() > float(np.percentile(null2, 95)))

    iu = np.triu_indices(n_words, 1)
    w3 = spearman(C1[iu], C2[iu])

    C2o = C2.copy()
    np.fill_diagonal(C2o, -2.0)
    maxcos = C2o.max(1)
    w4_min = float(maxcos.min())
    w4_p50 = float(np.median(maxcos))

    nn_same = 0
    for i in range(n_words):
        j = int(np.argmax(C2o[i]))
        nn_same += int(labels[j] == labels[i])
    acc = nn_same / n_words
    null_acc = []
    for _ in range(200):
        pl = rng.permutation(labels)
        a = 0
        for i in range(n_words):
            j = int(np.argmax(C2o[i]))
            a += int(pl[j] == pl[i])
        null_acc.append(a / n_words)
    w5 = bool(acc > float(np.percentile(null_acc, 95)))

    v = {
        'n_words': n_words,
        'W1_word_mechanism_class_structure': w1,
        'w1_margin_mean': round(float(m1.mean()), 5),
        'w1_null_p95': round(float(np.percentile(null1, 95)), 5),
        'W2_unembed_class_structure': w2,
        'w2_margin_mean': round(float(m2.mean()), 5),
        'w2_null_p95': round(float(np.percentile(null2, 95)), 5),
        'W3_block_rho': round(w3, 4),
        'W4_nn_cos_min': round(w4_min, 4),
        'W4_nn_cos_p50': round(w4_p50, 4),
        'W5_retrieval_accuracy': round(acc, 4),
        'w5_null_p95': round(float(np.percentile(null_acc, 95)), 4),
        'W5_same_class_retrieval': w5,
        'final_verdict': 'W1=%s/W2=%s/W5=%s/W3=%.3f'
                         % (w1, w2, w5, w3),
    }

    result = {'phase': 2866, 'prereg': PREREG, 'verdict': v,
              'words': words, 'labels': labels.tolist()}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'word_coords.npz',
           B1=B1.astype(np.float32),
           B2=B2.astype(np.float32),
           C1=C1.astype(np.float32),
           C2=C2.astype(np.float32),
           labels=labels.astype(np.int64),
           margin_B2=m1.astype(np.float32),
           margin_B1=m2.astype(np.float32))

    elapsed = time.monotonic() - t0
    cc.ledger('phase2866', elapsed)
    print('P2866 VERDICT %s' % json.dumps(v), flush=True)
    print('P2866 elapsed %.1fs' % elapsed, flush=True)

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
