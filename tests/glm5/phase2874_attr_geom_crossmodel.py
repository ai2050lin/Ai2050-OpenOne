"""Phase 2874 (LPF MA2 cont): attribute-vs-class geometry prereq,
cross-model -- does the 2870 prerequisite (attribute axis independent
of class axis in unembed geometry) generalize beyond qwen3-4b?

Zero forward; per model (sequential, memory freed between models):
  qwen14 = Qwen3-14B, glm4 = glm4-9b-chat-hf
  - embed_tokens read directly from safetensors (tie=True -> W_U space)
  - class directions dW[c] = unit(mean of unit(E_w)) over single-token
    class words (same 10 categories / 10-word pools as 2806, <=8 kept)
  - attribute directions d_attr(p) = unit(E(w+) - E(w-)) over the same
    frozen 15 antonym pairs (2870), single-token filtered per model

Prereg (frozen before any readout):
  Q1  (main, per model) every kept pair: max_c |cos(d_attr, dW[c])|
      < 0.3 => attribute_axis_independent; else entangled.  Null: 200
      random token-pair diff directions (SEED=2874 qwen14 / 28741
      glm4), p95 reported.
  Q2  (per model) mean orthogonal-energy fraction of d_attr vs 10-d
      class subspace > 0.8 => class_subspace_clean.
  Q3  (per model, descriptive) LOO-NN class acc of the 15-d attr
      profile B_attr = unit(E_w).D vs null p95 (200 perms, same seed):
      tests whether the 2873 A3 class-leak finding (acc 0.25 > null
      0.15 on qwen4) replicates cross-model.
  Q4  cross-model comparability note: kept word/pair counts recorded;
      verdicts conditioned on >= 6 classes with >= 3 words and >= 10
      pairs kept, else degraded_for_<model>.
"""
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2874' / 'attr_geom_crossmodel'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
HF = Path(r'D:\AI2050\Ai2050-OpenOne\models\hf')
MODELS = [('qwen14', 'Qwen3-14B', 2874), ('glm4', 'glm4-9b-chat-hf', 28741)]
N_NULL = 200
THRESH_CLASS_COS = 0.3
THRESH_ORTH = 0.8

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
    'Q1': 'per model: every kept pair max_c |cos(d_attr, dW)| < 0.3 => '
          'attribute_axis_independent; null 200 random pairs '
          '(SEED=2874/28741)',
    'Q2': 'per model: mean orth-energy fraction vs class subspace > 0.8 '
          '=> class_subspace_clean',
    'Q3': 'per model descriptive: LOO-NN class acc of 15-d attr profile '
          'vs null p95 (2873 A3 cross-check)',
    'Q4': '>=6 classes with >=3 words and >=10 pairs kept else '
          'degraded_for_model',
}


def unit(x):
    return x / max(float(np.linalg.norm(x)), 1e-30)


def loo_nn_acc(C, labels):
    n = len(labels)
    C = C.copy()
    np.fill_diagonal(C, -2.0)
    a = 0
    for i in range(n):
        j = int(np.argmax(C[i]))
        a += int(labels[j] == labels[i])
    return a / n


def embed_key_candidates():
    return ['model.embed_tokens.weight',
            'transformer.embedding.word_embeddings.weight']


def run_model(key, name, seed):
    from transformers import AutoTokenizer
    from safetensors import safe_open

    mdir = HF / name
    tok = AutoTokenizer.from_pretrained(str(mdir), local_files_only=True,
                                        trust_remote_code=True,
                                        use_fast=True)

    def single_form(word):
        for form in (' %s' % word, word):
            ids = tok.encode(form, add_special_tokens=False)
            if len(ids) == 1:
                return form, int(ids[0])
        return None, None

    exec2806 = json.loads(SRC_2806_EXEC.read_text(encoding='utf-8'))
    CATS = exec2806['cats']
    cat_words = list(CATS.keys())

    kept = {}
    for c, cat in enumerate(cat_words):
        ws = []
        for w in CATS[cat]:
            form, tid = single_form(w)
            if tid is not None:
                ws.append((w, tid))
        kept[cat] = ws[:8]
    n_classes_ok = sum(1 for ws in kept.values() if len(ws) >= 3)
    n_words = sum(len(ws) for ws in kept.values())

    pairs_kept = []
    for pname, wp, wm in PAIRS:
        fid = single_form(wp)
        mid = single_form(wm)
        if fid[1] is not None and mid[1] is not None:
            pairs_kept.append((pname, fid[1], mid[1]))

    idx = json.loads((mdir / 'model.safetensors.index.json')
                     .read_text(encoding='utf-8'))
    emb_key = None
    for k in embed_key_candidates():
        if k in idx['weight_map']:
            emb_key = k
            break
    assert emb_key, 'embed key not found for %s' % name
    shard = idx['weight_map'][emb_key]
    with safe_open(str(mdir / shard), framework='pt', device='cpu') as f:
        emb = f.get_tensor(emb_key).half().numpy()
    vocab_size = emb.shape[0]

    word_tids = [(w, tid, ci) for ci, cat in enumerate(cat_words)
                 for (w, tid) in kept[cat]]
    need = sorted(set(t for _, t, _ in word_tids)
                  | {t for _, a, b in pairs_kept for t in (a, b)})
    ti = {t: i for i, t in enumerate(need)}
    rows = emb[need].astype(np.float64)

    labels = np.array([ci for _, _, ci in word_tids], dtype=np.int64)
    E_unit = np.stack([unit(rows[ti[t]]) for _, t, _ in word_tids])
    dW = np.stack([unit(E_unit[labels == c].mean(axis=0))
                   for c in range(len(cat_words))
                   if (labels == c).sum() >= 1])
    dW = np.stack([unit(E_unit[labels == c].mean(axis=0))
                   for c in range(len(cat_words))])

    D = np.stack([unit(rows[ti[a]] - rows[ti[b]])
                  for _, a, b in pairs_kept])
    names = [p[0] for p in pairs_kept]

    cos_ac = np.abs(D @ dW.T)
    max_class_cos = cos_ac.max(axis=1)

    rng = np.random.default_rng(seed)
    null_max = []
    for _ in range(N_NULL):
        ta, tb = rng.integers(0, vocab_size, size=2)
        while ta == tb:
            tb = rng.integers(0, vocab_size)
        dd = unit(emb[int(ta)].astype(np.float64)
                  - emb[int(tb)].astype(np.float64))
        null_max.append(float(np.abs(dd @ dW.T).max()))
    null_p95 = float(np.percentile(null_max, 95))

    Q, _ = np.linalg.qr(dW.T)
    proj = (D @ Q) @ Q.T
    orth_energy = 1.0 - (proj * proj).sum(axis=1)

    # Q3 attr-profile class readout (kept words only)
    B_attr = np.stack([unit(E_unit[i] @ D.T) for i in range(len(labels))])
    C_at = B_attr @ B_attr.T
    acc_at = loo_nn_acc(C_at, labels)
    null_at = [loo_nn_acc(C_at, rng.permutation(labels))
               for _ in range(N_NULL)]
    acc_at_p95 = float(np.percentile(null_at, 95))

    ok = n_classes_ok >= 6 and len(pairs_kept) >= 10
    q1 = bool(max_class_cos.max() < THRESH_CLASS_COS)
    q2 = bool(float(orth_energy.mean()) > THRESH_ORTH)
    v = {
        'model': key,
        'comparable': ok,
        'n_words_kept': n_words,
        'n_classes_ok': n_classes_ok,
        'n_pairs_kept': len(pairs_kept),
        'pairs_kept_names': names,
        'Q1_max_of_max': round(float(max_class_cos.max()), 4),
        'Q1_null_p95': round(null_p95, 4),
        'Q1': q1,
        'Q1_label': 'attribute_axis_independent' if q1
                    else 'attribute_class_entangled',
        'Q2_mean_orth_energy': round(float(orth_energy.mean()), 4),
        'Q2': q2,
        'Q2_label': 'class_subspace_clean' if q2
                    else 'class_subspace_leaky',
        'Q3_acc_attr_alone': round(acc_at, 4),
        'Q3_null_p95': round(acc_at_p95, 4),
        'Q3_leak': bool(acc_at > acc_at_p95),
    }
    arrays = {
        key + '_D': D.astype(np.float32),
        key + '_dW': dW.astype(np.float32),
        key + '_cos_ac': cos_ac.astype(np.float32),
        key + '_orth': orth_energy.astype(np.float32),
        key + '_labels': labels,
    }
    del emb, rows
    gc.collect()
    return v, arrays


def main():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'result.json').exists():
        raise RuntimeError('result.json exists; delete execution.json and '
                           'result.json before re-run (immutability rule)')

    execution_path = OUT / 'execution.json'
    if not execution_path.exists():
        execution = {'timestamp': fc.stamp(),
                     'source': cc.snapshot(__file__),
                     'prereg': PREREG, 'seed': 2874,
                     'design': 'cross-model attr-vs-class unembed geometry '
                               'prereq (qwen14, glm4; sequential, zero '
                               'forward, safetensors only)'}
        fc.save(execution_path, execution)

    verdicts = {}
    arrays = {}
    for key, name, seed in MODELS:
        t1 = time.monotonic()
        v, arr = run_model(key, name, seed)
        verdicts[key] = v
        arrays.update(arr)
        print('P2874 %s %s elapsed %.1fs'
              % (key, v['final_verdict'] if 'final_verdict' in v
                 else json.dumps(v), time.monotonic() - t1), flush=True)

    summary = {
        'Q1_all_independent': all(verdicts[k]['Q1']
                                  and verdicts[k]['comparable']
                                  for k, _, _ in MODELS),
        'Q2_all_clean': all(verdicts[k]['Q2']
                            and verdicts[k]['comparable']
                            for k, _, _ in MODELS),
        'Q3_leak_pattern': {k: verdicts[k]['Q3_leak']
                            for k, _, _ in MODELS},
        'Q1_max_by_model': {k: verdicts[k]['Q1_max_of_max']
                            for k, _, _ in MODELS},
    }
    fc.save(OUT / 'result.json',
            {'phase': 2874, 'prereg': PREREG, 'verdicts': verdicts,
             'summary': summary, 'seed_nulls': {k: s for k, _, s in MODELS},
             'n_null': N_NULL})
    fc.npz(OUT / 'attr_geom_crossmodel.npz', **arrays)

    elapsed = time.monotonic() - t0
    cc.ledger('phase2874', elapsed)
    print('P2874 SUMMARY %s' % json.dumps(summary), flush=True)
    print('P2874 elapsed %.1fs' % elapsed, flush=True)


if __name__ == '__main__':
    main()
