"""Phase 2870 (LPF MA2 cont): attribute-axis pilot -- antonym-pair
directions vs the class axis (zero forward, unembed geometry only).

Motivation: 2864 established the class-variance axis as a 4th
mechanism axis; the TMA atlas needs the ATTRIBUTIONAL axis as a
separate word-family axis (user's "attributes of various objects").
This pilot measures, purely in unembed space, whether antonym-pair
attribute directions are (a) independent of the 10 class directions
(P1: attribute is NOT an 11th class) and (b) separable from each
other (P2: axis structure, not one blob).

Method:
  pairs (frozen, 15): size/speed/temp/age/weight/strength/brightness/
  value/moisture/height/fullness/width/depth axes, 2 words each.
  single-token filter via tokenizer (both raw and leading-space form;
  prefer space form if both single).
  d_attr(p) = unit(E(w+) - E(w-)), E = embed_tokens (tie=True -> W_U).
  Class directions dW_unit[c] rebuilt from 2806 execution.json cats
  (mean of unit embeddings of the 80 SEED=2855 census words, same
  construction as 2867 B1).

Prereg (frozen before any readout):
  P1  (main) for every pair p: max_c |cos(d_attr(p), dW_unit[c])| < 0.3
      => attribute_axis_independent; else attribute_class_entangled.
      Null: max-c |cos| distribution of 200 random token-pair
      difference directions (SEED=2870), p95 reported.
  P2  mean pairwise |cos| among the 15 d_attr < null p95 of random
      pairs AND max pairwise |cos| < 0.6 => axis_separable;
      else axis_collapsed.
  P3  residual-energy readout: for each pair, energy fraction of
      d_attr orthogonal to the 10-d class subspace (projector on
      span{dW_unit}); mean fraction > 0.8 => class_subspace_clean.
      (continuous version of P1)
  P4  descriptive: full |cos| table d_attr x dW_unit (10 cols) and
      pair x pair |cos| matrix saved to npz; speed pairs highlighted
      in report for future 400b hookup.
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
OUT = BASE / 'phase2870' / 'attr_axis_pilot'
MODEL_DIR = Path(r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b')
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SEED = 2870
N_NULL = 200
THRESH_CLASS_COS = 0.3
THRESH_PAIR_COS = 0.6

PAIRS = [
    ('size1', 'big', 'small'),
    ('size2', 'large', 'little'),
    ('size3', 'huge', 'tiny'),
    ('speed1', 'fast', 'slow'),
    ('speed2', 'quick', 'slow'),
    ('temp1', 'hot', 'cold'),
    ('temp2', 'warm', 'cool'),
    ('age1', 'old', 'new'),
    ('age2', 'young', 'old'),
    ('weight', 'heavy', 'light'),
    ('strength', 'strong', 'weak'),
    ('brightness', 'bright', 'dark'),
    ('moisture', 'wet', 'dry'),
    ('height', 'tall', 'short'),
    ('fullness', 'full', 'empty'),
]

PREREG = {
    'P1': 'every pair: max_c |cos(d_attr, dW_unit[c])| < 0.3 => '
          'attribute_axis_independent; null = 200 random token-pair '
          'diff directions (SEED=2870), p95 reported',
    'P2': 'mean pairwise |cos| < null p95 and max < 0.6 => '
          'axis_separable else axis_collapsed',
    'P3': 'mean orthogonal-energy fraction vs 10-d class subspace '
          '> 0.8 => class_subspace_clean',
    'P4': 'descriptive |cos| tables to npz; speed pairs highlighted',
}


def unit(x):
    return x / max(float(np.linalg.norm(x)), 1e-30)


def main():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)

    execution_path = OUT / 'execution.json'
    if not execution_path.exists():
        execution = {'timestamp': fc.stamp(),
                     'source': cc.snapshot(__file__),
                     'prereg': PREREG, 'seed': SEED,
                     'design': 'attribute-axis pilot: 15 antonym-pair '
                               'unembed directions vs 10 class '
                               'directions, orthogonality + axis '
                               'structure; zero forward'}
        fc.save(execution_path, execution)

    from transformers import AutoTokenizer
    from safetensors import safe_open

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    def single_form(word):
        for form in (' %s' % word, word):
            ids = tok.encode(form, add_special_tokens=False)
            if len(ids) == 1:
                return form, int(ids[0])
        return None, None

    # class directions (2867 B1 construction)
    exec2806 = json.loads(SRC_2806_EXEC.read_text(encoding='utf-8'))
    CATS = exec2806['cats']
    cat_words = list(CATS.keys())
    SEED_VOCAB = 2855

    # rebuild the 80-word census vocab (SEED=2855, same as 2867/2846)
    rng0 = np.random.default_rng(SEED_VOCAB)
    words = []
    labels = []
    for c, w in enumerate(cat_words):
        pool = list(CATS[w])
        rng0.shuffle(pool)
        take = pool[:8]
        words.extend(take)
        labels.extend([c] * len(take))
    labels = np.array(labels, dtype=np.int64)
    n_words = len(words)
    assert n_words == 80, n_words

    # ---------- locate embed shard ----------
    idx = json.loads((MODEL_DIR / 'model.safetensors.index.json')
                     .read_text(encoding='utf-8'))
    shard_name = idx['weight_map']['model.embed_tokens.weight']

    word_rows = {}
    for w in words:
        form, tid = single_form(w)
        assert tid is not None, 'multi-token word: %s' % w
        word_rows[w] = tid
    pair_words = {}
    for name, wp, wm in PAIRS:
        fid = single_form(wp)
        mid = single_form(wm)
        assert fid[1] is not None, 'multi-token: %s' % wp
        assert mid[1] is not None, 'multi-token: %s' % wm
        pair_words[name] = (fid[1], mid[1])

    need_tids = sorted(set(word_rows.values())
                       | {t for pr in pair_words.values() for t in pr})
    tid_index = {t: i for i, t in enumerate(need_tids)}

    with safe_open(str(MODEL_DIR / shard_name), framework='pt',
                   device='cpu') as f:
        emb = f.get_tensor('model.embed_tokens.weight')
    rows = emb[need_tids].float().numpy()

    E_word = np.stack([unit(rows[tid_index[word_rows[w]]])
                       for w in words])
    dW = np.stack([unit(E_word[labels == c].mean(axis=0))
                   for c in range(10)])

    d_attr = {}
    for name, (tp, tm) in pair_words.items():
        d_attr[name] = unit(rows[tid_index[tp]] - rows[tid_index[tm]])

    names = [p[0] for p in PAIRS]
    D = np.stack([d_attr[n] for n in names])

    # ---------- P1 ----------
    cos_attr_class = np.abs(D @ dW.T)          # 15 x 10
    max_class_cos = cos_attr_class.max(axis=1)  # per pair
    rng = np.random.default_rng(SEED)
    vocab_size = 151936
    null_max = []
    for _ in range(N_NULL):
        ta, tb = rng.integers(0, vocab_size, size=2)
        while ta == tb:
            tb = rng.integers(0, vocab_size)
        da = emb[int(ta)].float().numpy()
        db = emb[int(tb)].float().numpy()
        dd = unit(da - db)
        null_max.append(float(np.abs(dd @ dW.T).max()))
    null_p95 = float(np.percentile(null_max, 95))
    p1 = bool(max_class_cos.max() < THRESH_CLASS_COS)
    p1_label = ('attribute_axis_independent' if p1
                else 'attribute_class_entangled')

    # ---------- P2 ----------
    C_pp = np.abs(D @ D.T)
    iu = np.triu_indices(len(names), 1)
    pp_vals = C_pp[iu]
    mean_pp = float(pp_vals.mean())
    max_pp = float(pp_vals.max())
    null_pp = []
    for _ in range(N_NULL):
        idxs = rng.integers(0, vocab_size, size=(30, 2))
        dirs = []
        for ta, tb in idxs:
            if ta == tb:
                tb = (tb + 1) % vocab_size
            da = emb[int(ta)].float().numpy()
            db = emb[int(tb)].float().numpy()
            dirs.append(unit(da - db))
        Dn = np.stack(dirs)
        Cn = np.abs(Dn @ Dn.T)
        j = np.triu_indices(len(dirs), 1)
        null_pp.append(float(Cn[j].mean()))
    del emb
    null_pp_p95 = float(np.percentile(null_pp, 95))
    p2 = bool(mean_pp < null_pp_p95 and max_pp < THRESH_PAIR_COS)
    p2_label = 'axis_separable' if p2 else 'axis_collapsed'

    # ---------- P3 ----------
    Q, _ = np.linalg.qr(dW.T)                   # 2560 x 10
    proj = (D @ Q) @ Q.T
    orth_energy = 1.0 - (proj * proj).sum(axis=1)  # rows unit-normed
    p3 = bool(float(orth_energy.mean()) > 0.8)
    p3_label = 'class_subspace_clean' if p3 else 'class_subspace_leaky'

    v = {
        'n_pairs': len(names),
        'pairs_kept': len(names),
        'max_class_cos_by_pair': {n: round(float(m), 4)
                                  for n, m in zip(names, max_class_cos)},
        'P1_max_of_max': round(float(max_class_cos.max()), 4),
        'P1_threshold': THRESH_CLASS_COS,
        'P1_null_p95': round(null_p95, 4),
        'P1': p1,
        'P1_label': p1_label,
        'P2_mean_pairwise': round(mean_pp, 4),
        'P2_max_pairwise': round(max_pp, 4),
        'P2_null_mean_p95': round(null_pp_p95, 4),
        'P2': p2,
        'P2_label': p2_label,
        'P3_mean_orth_energy': round(float(orth_energy.mean()), 4),
        'P3': p3,
        'P3_label': p3_label,
        'final_verdict': 'P1=%s(%s)/P2=%s(%s)/P3=%s(%s)'
                         % (p1, p1_label, p2, p2_label, p3, p3_label),
    }

    result = {'phase': 2870, 'prereg': PREREG, 'verdict': v,
              'pairs': PAIRS, 'seed_null': SEED}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'attr_axis_pilot.npz',
           D=D.astype(np.float32),
           dW=dW.astype(np.float32),
           cos_attr_class=cos_attr_class.astype(np.float32),
           C_pp=C_pp.astype(np.float32),
           orth_energy=orth_energy.astype(np.float32),
           pair_names=np.array(names, dtype=object),
           labels=labels.astype(np.int64))

    elapsed = time.monotonic() - t0
    cc.ledger('phase2870', elapsed)
    print('P2870 VERDICT %s' % json.dumps(v), flush=True)
    print('P2870 elapsed %.1fs' % elapsed, flush=True)


if __name__ == '__main__':
    main()
