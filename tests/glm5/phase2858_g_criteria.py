"""Phase 2858 (MA / Atlas-E cont): legality criteria redesign (G) + 200-hit.

2857 diagnosed the E2 drift as subset_incompatibility (prototypicality
gradient): the class direction dW_unit is a function of vocab choice,
and class-comparison geometry stays robust (r~0.94).  Conclusion: the
legality of an extended vocab must be judged by ITS OWN geometric
health, not by agreement with the 80-word baseline.  This phase
freezes the G criteria and completes the vocab to 200 words with a
third refill pool (fruit/metal/tool shortfalls):

  G1  hit200 iff >= 9/10 classes reach 10 new single-token words
      (CAND + CAND2 + CAND3 ordered, no exclusions: 2857 found no
      dragging words)
  G2  separable iff max off-diag centre cosine of Cm200c
      <= max of Cm80 + 0.10 AND mean off-diag of Cm200c
      <= mean of Cm80 + 0.05  (baseline-relative thresholds,
      frozen formula, values computed from data)
  G3  non-degenerate iff min_c |dW200c| >= 0.5 * min_c |dW80|
      (dW = class centre minus mean of the other nine, unnormalised)

verdict: atlas_e200_legal = G1 and G2 and G3.
The resulting e200c vocab (word lists + dW_unit) is persisted for the
phase2859 census, which rebuilds it deterministically and asserts
n_words == 200.
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
OUT = BASE / 'phase2858' / 'g_criteria'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SEED_80 = 2846
SEED_C = 2858
MAX_WORDS_OLD = 8
MAX_WORDS_NEW = 20

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
CAND3 = {
    'fruit': ['durian', 'lychee', 'quince', 'plantain', 'mulberry'],
    'animal': [],
    'metal': ['ingot', 'pewter', 'nugget', 'ore', 'foil'],
    'vehicle': [],
    'country': [],
    'food': [],
    'nature': [],
    'furniture': [],
    'tool': ['tongs', 'rasp', 'gouge', 'auger', 'bit'],
    'clothing': [],
}

PREREG = {
    'G1': 'hit200 iff >=9/10 classes reach 10 new single-token words '
          '(CAND+CAND2+CAND3 ordered, no exclusions)',
    'G2': 'separable iff max_offdiag_cos(Cm200c) <= '
          'max_offdiag_cos(Cm80) + 0.10 AND mean_offdiag_cos(Cm200c) '
          '<= mean_offdiag_cos(Cm80) + 0.05 (baseline-relative, '
          'frozen formula)',
    'G3': 'non-degenerate iff min_c |dW200c| >= 0.5 * min_c |dW80| '
          '(unnormalised class-contrast norms)',
    'verdict': 'atlas_e200_legal = G1 and G2 and G3',
}


def unit(x):
    return x / max(np.linalg.norm(x), 1e-30)


def survivors(tok, pools, CAT_WORDS):
    chk = {}
    out = {}
    for cat in CAT_WORDS:
        s = []
        for w in pools[cat]:
            try:
                single_token_id(tok, w, chk)
                s.append(w)
            except AssertionError:
                pass
        out[cat] = s
    return out


def offdiag_stats(Cm):
    Cn = Cm / np.maximum(np.linalg.norm(Cm, axis=1,
                                        keepdims=True), 1e-30)
    M = Cn @ Cn.T
    off = M[~np.eye(len(Cm), dtype=bool)]
    return {'max': float(off.max()), 'mean': float(off.mean())}


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
                     'prereg': PREREG, 'seed': SEED_C,
                     'design': 'G criteria (vocab-self geometric '
                               'health) + CAND3 refill to 200 words; '
                               'e200c vocab persisted for phase2859'}
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

    # baseline
    vocab80 = build_vocab(tok, W_U, CATS, CAT_WORDS,
                          SEED_80, MAX_WORDS_OLD)
    Cm80 = vocab80['Cm']
    dW80_raw = []
    for k in range(10):
        others = (Cm80.sum(0) - Cm80[k]) / 9.0
        dW80_raw.append(Cm80[k] - others)
    dW80_raw = np.stack(dW80_raw)

    # e200c: full pooled refill, no exclusions
    pools = {c: CAND[c] + CAND2[c] + CAND3[c] for c in CAT_WORDS}
    surv_c = survivors(tok, pools, CAT_WORDS)
    cats200c = {}
    g1_counts = {}
    new_c = {}
    for cat in CAT_WORDS:
        take = surv_c[cat][:10]
        new_c[cat] = take
        g1_counts[cat] = len(take)
        cats200c[cat] = list(CATS[cat]) + take
    n_full = sum(1 for c in CAT_WORDS if g1_counts[c] >= 10)
    g1 = bool(n_full >= 9)

    vocab200c = build_vocab(tok, W_U, cats200c, CAT_WORDS,
                            SEED_C, MAX_WORDS_NEW)
    Cm200c = vocab200c['Cm']
    dW200c_raw = []
    for k in range(10):
        others = (Cm200c.sum(0) - Cm200c[k]) / 9.0
        dW200c_raw.append(Cm200c[k] - others)
    dW200c_raw = np.stack(dW200c_raw)

    s80 = offdiag_stats(Cm80)
    s200 = offdiag_stats(Cm200c)
    g2 = bool(s200['max'] <= s80['max'] + 0.10
              and s200['mean'] <= s80['mean'] + 0.05)

    n80 = float(np.linalg.norm(dW80_raw, axis=1).min())
    n200 = float(np.linalg.norm(dW200c_raw, axis=1).min())
    g3 = bool(n200 >= 0.5 * n80)

    verdict = {
        'G1_hit200': g1,
        'g1_counts': g1_counts,
        'new_words_200c': new_c,
        'n_words_200c': vocab200c['n_words'],
        'G2_separable': g2,
        'offdiag_80': {k: round(v, 4) for k, v in s80.items()},
        'offdiag_200c': {k: round(v, 4) for k, v in s200.items()},
        'G3_non_degenerate': g3,
        'min_dW_norm_80': round(n80, 4),
        'min_dW_norm_200c': round(n200, 4),
        'dW_norm_ratio': round(n200 / max(n80, 1e-30), 4),
        'final_verdict': 'atlas_e200_legal=%s'
                         % (g1 and g2 and g3),
    }

    result = {'phase': 2858, 'prereg': PREREG, 'verdict': verdict}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'g_criteria.npz',
           dW_unit_80=vocab80['dW_unit'].astype(np.float64),
           dW_unit_200c=vocab200c['dW_unit'].astype(np.float64),
           Cm_80=Cm80.astype(np.float64),
           Cm_200c=Cm200c.astype(np.float64),
           dW_norm_80=np.linalg.norm(dW80_raw, axis=1),
           dW_norm_200c=np.linalg.norm(dW200c_raw, axis=1),
           words_200c=np.array(
               [' '.join(vocab200c['targets'][c]) for c in CAT_WORDS]))

    elapsed = time.monotonic() - t0
    cc.ledger('phase2858', elapsed)
    print('P2858 VERDICT %s' % json.dumps(verdict), flush=True)
    print('P2858 elapsed %.1fs' % elapsed, flush=True)

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
