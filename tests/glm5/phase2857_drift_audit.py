"""Phase 2857 (MA / Atlas-E cont): E2 direction-drift audit + refill rerun.

2856 left atlas_e200_ready=False with two candidate explanations for the
direction drift (cos 0.847-0.936):
  (a) polysemous/weak member pollution dragging class centres;
  (b) sampling noise of the 8-10 word dW_unit baseline itself.
This phase separates them at the unembed level (no forward):

  A1  jackknife on the 200a vocab (rebuild of 2856, seed=2856):
      per class, drop one new word w -> dW'_c -> cos(dW'_c, dW80_c);
      delta_cos(w) = cos' - cos_base.  Large positive delta marks a
      dragging word.
  A2  jackknife inside the 80-protocol vocab (full single-token set,
      10/class): directional wobble scale of the baseline itself,
      delta80_c = mean/max over w of (1 - cos(dW80[-w], dW80)).
  A3  new-only vs old centre (Cm80) cosine per class.

Judgements (frozen):
  J1   pollution_dominant iff >= 2/10 classes have a word with
       delta_cos >= 0.05
  J2   subset_compatible iff median_c cos(Cm_new, Cm_old) >= 0.90
  J3   drift_source = pollution_dominant if J1; subset_incompatibility
       if not J2; else distributed_noise (delta80 reported, descriptive)
  E1v2 e200b_coverage iff >= 9/10 classes reach 10 new single-token
       words after cleaning (drop delta_cos >= 0.05 words) + refill
       from extended pools (CAND + CAND2, ordered)
  E2v2 direction_stable_v2 iff >= 9/10 classes cos(dW_200b, dW80) >= 0.9
  E3v2 geometry_retained_v2 iff off-diagonal centre-cosine matrix
       r(200b vs 80) >= 0.95
  verdict: atlas_e200_ready_v2 = E1v2 and E2v2 and E3v2
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc
from rdc_atlas_census import build_vocab, single_token_id

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2857' / 'drift_audit'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SEED_80 = 2846          # baseline vocab protocol
SEED_A = 2856           # 200a rebuild (must match phase2856)
SEED_B = 2857           # 200b vocab
MAX_WORDS_OLD = 8
MAX_WORDS_NEW = 20
J1_THRESH = 0.05

# phase2856 pools (verbatim) + refill pools (CAND2, appended in order)
CAND = {
    'fruit': ['plum', 'melon', 'kiwi', 'fig', 'papaya', 'guava',
              'apricot', 'coconut', 'lime', 'pineapple',
              'pomegranate', 'watermelon'],
    'animal': ['pig', 'sheep', 'goat', 'deer', 'bear', 'fox',
               'monkey', 'snake', 'frog', 'duck', 'camel', 'chicken'],
    'metal': ['zinc', 'lead', 'platinum', 'titanium', 'tungsten',
              'cobalt', 'mercury', 'lithium', 'uranium', 'cadmium',
              'chrome', 'alloy'],
    'vehicle': ['van', 'jeep', 'scooter', 'tractor', 'subway',
                'ferry', 'canoe', 'wagon', 'yacht', 'sled',
                'glider', 'trailer'],
    'country': ['Spain', 'Norway', 'Sweden', 'Poland', 'Mexico',
                'Korea', 'Chile', 'Peru', 'Kenya', 'Greece',
                'Turkey', 'Nigeria'],
    'food': ['salt', 'sugar', 'flour', 'noodle', 'tofu', 'salad',
             'bacon', 'sausage', 'cream', 'curry', 'pie', 'candy'],
    'nature': ['valley', 'hill', 'beach', 'island', 'storm',
               'thunder', 'fog', 'frost', 'breeze', 'glacier',
               'volcano', 'canyon'],
    'furniture': ['couch', 'dresser', 'drawer', 'crib', 'mattress',
                  'armchair', 'bookshelf', 'cupboard', 'recliner',
                  'hutch', 'ottoman', 'nightstand'],
    'tool': ['pliers', 'screwdriver', 'chisel', 'lathe', 'rake',
             'hatchet', 'crowbar', 'tweezers', 'scissors', 'grinder',
             'anvil', 'sickle'],
    'clothing': ['skirt', 'jeans', 'boots', 'vest', 'blouse',
                 'sweater', 'hoodie', 'shorts', 'helmet', 'mittens',
                 'pajamas', 'sneaker'],
}
CAND2 = {
    'fruit': ['date', 'olive', 'raisin', 'prune', 'tangerine',
              'nectarine'],
    'animal': ['hen', 'owl', 'eagle', 'seal', 'crab', 'shark'],
    'metal': ['barium', 'cesium', 'osmium', 'iridium', 'palladium',
              'rhodium'],
    'vehicle': ['metro', 'limo', 'bike', 'cart', 'buggy'],
    'country': ['Chad', 'Mali', 'Togo', 'Cuba', 'Nepal', 'Qatar'],
    'food': ['stew', 'broth', 'toast', 'sushi', 'taco', 'jam'],
    'nature': ['mist', 'dune', 'cave', 'reef', 'swamp'],
    'furniture': ['cradle', 'bureau', 'settee', 'divan', 'rocker',
                  'bunk', 'pew'],
    'tool': ['mallet', 'trowel', 'spanner', 'awl', 'adze', 'wedge',
             'clamp', 'vise', 'hoe', 'shears'],
    'clothing': ['robe', 'gown', 'apron', 'belt', 'cap', 'cape'],
}

PREREG = {
    'J1': 'pollution_dominant iff >=2/10 classes have a leave-one-out '
          'word with delta_cos = cos(dW200a[-w], dW80) - '
          'cos(dW200a, dW80) >= 0.05',
    'J2': 'subset_compatible iff median_c cos(Cm_new_c, Cm_old_c) '
          '>= 0.90',
    'J3': 'drift_source = pollution_dominant if J1; '
          'subset_incompatibility if not J2; else distributed_noise '
          '(delta80 wobble scale reported, descriptive)',
    'E1v2': 'e200b_coverage iff >=9/10 classes reach 10 new '
            'single-token words after cleaning + refill',
    'E2v2': 'direction_stable_v2 iff >=9/10 classes '
            'cos(dW_200b, dW80) >= 0.9',
    'E3v2': 'geometry_retained_v2 iff off-diagonal centre-cosine '
            'matrix r(200b vs 80) >= 0.95',
    'verdict': 'atlas_e200_ready_v2 = E1v2 and E2v2 and E3v2',
}


def unit(x):
    return x / max(np.linalg.norm(x), 1e-30)


def pearson(a, b):
    sa, sb = np.std(a), np.std(b)
    if sa <= 1e-12 or sb <= 1e-12:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def survivors(tok, pools, CAT_WORDS, exclude=None):
    """ordered single-token survivors of the candidate pools."""
    chk = {}
    out = {}
    for cat in CAT_WORDS:
        ex = set(exclude[cat]) if exclude else set()
        s = []
        for w in pools[cat]:
            if w in ex:
                continue
            try:
                single_token_id(tok, w, chk)
                s.append(w)
            except AssertionError:
                pass
        out[cat] = s
    return out


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
                     'prereg': PREREG, 'seed': SEED_B,
                     'design': 'E2 drift audit: jackknife (200a + 80) '
                               '+ subset compatibility + refill rerun '
                               '(E1v2/E2v2/E3v2), unembed level'}
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

    # ---------- baseline (80 protocol, seed=2846) ----------
    vocab80 = build_vocab(tok, W_U, CATS, CAT_WORDS,
                          SEED_80, MAX_WORDS_OLD)
    dW80 = vocab80['dW_unit']
    Cm80 = vocab80['Cm']
    tmap = vocab80['tid_map']
    E80 = {cat: np.stack([W_U[tmap[w]] for w in CATS[cat]
                          if w in tmap]) for cat in CAT_WORDS}

    # ---------- 200a rebuild (must match phase2856) ----------
    surv_a = survivors(tok, CAND, CAT_WORDS)
    cats200a = {c: list(CATS[c]) + surv_a[c][:10] for c in CAT_WORDS}
    vocab200a = build_vocab(tok, W_U, cats200a, CAT_WORDS,
                            SEED_A, MAX_WORDS_NEW)
    assert vocab200a['n_words'] == 184, \
        '200a rebuild mismatch: %d' % vocab200a['n_words']
    tmap_a = vocab200a['tid_map']
    dW200a = vocab200a['dW_unit']
    Cm200a = vocab200a['Cm']
    cos_base = np.array([float(dW200a[k] @ dW80[k])
                         for k in range(10)])

    # ---------- A1: jackknife on 200a (new words only) ----------
    jack = {}
    worst = {}
    for ci, cat in enumerate(CAT_WORDS):
        tgt = vocab200a['targets'][cat]
        E = np.stack([W_U[tmap_a[w]] for w in tgt])
        n = E.shape[0]
        others = (Cm200a.sum(0) - Cm200a[ci]) / 9.0
        deltas = {}
        for wi in range(10, n):          # new words only
            w = tgt[wi]
            Cm2 = (E.sum(0) - E[wi]) / (n - 1)
            dW2 = unit(Cm2 - others)
            deltas[w] = float(dW2 @ dW80[ci]) - float(cos_base[ci])
        jack[cat] = deltas
        bw = max(deltas, key=deltas.get)
        worst[cat] = {'word': bw, 'delta': round(deltas[bw], 4)}
    n_poll = sum(1 for cat in CAT_WORDS
                 if max(jack[cat].values()) >= J1_THRESH)
    j1 = bool(n_poll >= 2)

    # ---------- A2: 80-protocol wobble scale ----------
    delta80 = {}
    for ci, cat in enumerate(CAT_WORDS):
        E = E80[cat]
        n = E.shape[0]
        others = (Cm80.sum(0) - Cm80[ci]) / 9.0
        vals = []
        for wi in range(n):
            Cm2 = (E.sum(0) - E[wi]) / (n - 1)
            dW2 = unit(Cm2 - others)
            vals.append(1.0 - float(dW2 @ dW80[ci]))
        delta80[cat] = {'mean': round(float(np.mean(vals)), 4),
                        'max': round(float(np.max(vals)), 4)}
    d80_med = float(np.median([delta80[c]['mean']
                               for c in CAT_WORDS]))

    # ---------- A3: subset compatibility ----------
    sub_cos = {}
    for ci, cat in enumerate(CAT_WORDS):
        neww = vocab200a['targets'][cat][10:]
        Enew = np.stack([W_U[tmap_a[w]] for w in neww])
        sub_cos[cat] = round(float(unit(Enew.mean(0)) @ unit(Cm80[ci])),
                             4)
    sub_med = float(np.median(list(sub_cos.values())))
    j2 = bool(sub_med >= 0.90)

    if j1:
        drift = 'pollution_dominant'
    elif not j2:
        drift = 'subset_incompatibility'
    else:
        drift = 'distributed_noise'
    print('P2857 audit: J1=%s J2=%s drift=%s' % (j1, j2, drift),
          flush=True)
    print('P2857 worst %s' % json.dumps(worst), flush=True)
    print('P2857 subset_cos %s' % json.dumps(sub_cos), flush=True)
    print('P2857 delta80_med %.5f' % d80_med, flush=True)

    # ---------- refill + clean -> 200b ----------
    cleaned = {cat: sorted([w for w, d in jack[cat].items()
                            if d >= J1_THRESH]) for cat in CAT_WORDS}
    surv_b = survivors(tok, {c: CAND[c] + CAND2[c] for c in CAT_WORDS},
                       CAT_WORDS, exclude=cleaned)
    cats200b = {}
    e1v2_counts = {}
    for cat in CAT_WORDS:
        take = surv_b[cat][:10]
        e1v2_counts[cat] = len(take)
        cats200b[cat] = list(CATS[cat]) + take
    n_full = sum(1 for cat in CAT_WORDS if e1v2_counts[cat] >= 10)
    e1v2 = bool(n_full >= 9)

    vocab200b = build_vocab(tok, W_U, cats200b, CAT_WORDS,
                            SEED_B, MAX_WORDS_NEW)
    dW200b = vocab200b['dW_unit']
    Cm200b = vocab200b['Cm']
    cos_b = np.array([float(dW200b[k] @ dW80[k]) for k in range(10)])
    e2v2 = bool((cos_b >= 0.9).sum() >= 9)

    def offdiag_cos(Cm):
        Cn = Cm / np.maximum(np.linalg.norm(Cm, axis=1,
                                            keepdims=True), 1e-30)
        M = Cn @ Cn.T
        return M[~np.eye(10, dtype=bool)]

    c80 = offdiag_cos(Cm80)
    c200b = offdiag_cos(Cm200b)
    e3v2_val = pearson(c80, c200b)
    e3v2 = bool(np.isfinite(e3v2_val) and e3v2_val >= 0.95)

    new_b = {cat: vocab200b['targets'][cat][10:]
             for cat in CAT_WORDS}

    verdict = {
        'n_words_200a': vocab200a['n_words'],
        'J1_pollution_dominant': j1,
        'n_poll_classes': n_poll,
        'worst_word_per_class': worst,
        'J2_subset_compatible': j2,
        'subset_cos_per_class': sub_cos,
        'subset_cos_median': round(sub_med, 4),
        'drift_source': drift,
        'delta80_mean_per_class': {c: delta80[c]['mean']
                                   for c in CAT_WORDS},
        'delta80_median': round(d80_med, 5),
        'cos_base_200a_per_class': [round(float(x), 4)
                                    for x in cos_base],
        'cleaned_words': cleaned,
        'E1v2_e200b_coverage': e1v2,
        'e1v2_counts': e1v2_counts,
        'new_words_200b': new_b,
        'n_words_200b': vocab200b['n_words'],
        'E2v2_direction_stable': e2v2,
        'cos_dW200b_dW80_per_class': [round(float(x), 4)
                                      for x in cos_b],
        'min_cos_200b': round(float(cos_b.min()), 4),
        'E3v2_geometry_retained': e3v2,
        'E3v2_offdiag_r': round(e3v2_val, 5),
        'final_verdict': 'atlas_e200_ready_v2=%s/drift=%s'
                         % (e1v2 and e2v2 and e3v2, drift),
    }

    result = {'phase': 2857, 'prereg': PREREG, 'verdict': verdict}
    fc.save(OUT / 'result.json', result)

    nn_max = max(len(vocab200a['targets'][c]) - 10
                 for c in CAT_WORDS)
    jmat = np.full((10, nn_max), np.nan)
    for ci, cat in enumerate(CAT_WORDS):
        tgt = vocab200a['targets'][cat][10:]
        for wi, w in enumerate(tgt):
            jmat[ci, wi] = jack[cat][w]
    fc.npz(OUT / 'drift_audit.npz',
           dW_unit_80=dW80.astype(np.float64),
           dW_unit_200a=dW200a.astype(np.float64),
           dW_unit_200b=dW200b.astype(np.float64),
           cos_base_200a=cos_base.astype(np.float64),
           cos_200b=cos_b.astype(np.float64),
           jackknife_delta=jmat.astype(np.float64),
           delta80_mean=np.array([delta80[c]['mean']
                                  for c in CAT_WORDS]),
           subset_cos=np.array([sub_cos[c] for c in CAT_WORDS]))

    elapsed = time.monotonic() - t0
    cc.ledger('phase2857', elapsed)
    print('P2857 VERDICT %s' % json.dumps(verdict), flush=True)
    print('P2857 elapsed %.1fs' % elapsed, flush=True)

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
