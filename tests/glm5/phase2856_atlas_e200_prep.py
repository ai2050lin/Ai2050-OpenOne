"""Phase 2856 (MA / Atlas-E prep): vocab extension preresearch (80->200)
+ dual-spectrum census pipeline verification against phase2846.

Arm A (no forward, CPU): extend the 2806 vocab by a fixed ordered
12-candidate pool per class; single-token filter keeps the first 10
survivors -> 20 words/class target.  Judgements:
  E1  e200_coverage iff >= 9/10 classes reach 10 new single-token words
  E2  direction_stable iff >= 9/10 classes have
      cos(dW_unit_200, dW_unit_80) >= 0.9
  E3  geometry_retained iff Pearson r(off-diagonal class-centre
      cosine matrices, 200 vs 80) >= 0.95
  (dW-cosine matrix correlation reported descriptively)

Arm B (forward, GPU): rerun the packaged census pipeline
(rdc_atlas_census, phase2846 protocol verbatim) on L29-31 for the
first 40 target words with the seed=2846 vocab construction, and
compare with the registered phase2846 census_L{29,30,31}.npz drops:
  B1  pipeline_verified iff all 3 layers have
      Pearson r(drops_new, drops_ref) >= 0.99 over the 40x32 flat
      array AND median |diff| <= 0.002

verdict: atlas_e200_ready iff E1 and E2 and E3;
         pipeline_verified = B1 (independent flag).
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc
from rdc_atlas_census import AtlasCensus, build_vocab, single_token_id

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2856' / 'atlas_e200_prep'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
REF_2846 = BASE / 'phase2846' / 'fullhead_census'
SEED_A = 2856
SEED_B = 2846          # must match phase2846 for null_tids reproduction
MAX_WORDS_OLD = 8
MAX_WORDS_NEW = 20
SMOKE_WORDS = 40
SMOKE_LAYERS = [29, 30, 31]

# ordered candidate pools: first 10 single-token survivors are kept
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

PREREG = {
    'E1': 'e200_coverage iff >= 9/10 classes reach 10 new '
          'single-token words (first 10 ordered survivors)',
    'E2': 'direction_stable iff >= 9/10 classes have '
          'cos(dW_unit_200, dW_unit_80) >= 0.9',
    'E3': 'geometry_retained iff Pearson r(off-diagonal class-centre '
          'cosine matrix, Cm200 vs Cm80) >= 0.95',
    'B1': 'pipeline_verified iff all of L29-31 have Pearson '
          'r(drops_new, drops_2846) >= 0.99 (40x32 flat) AND '
          'median |diff| <= 0.002',
    'verdict': 'atlas_e200_ready iff E1 and E2 and E3; '
               'pipeline_verified = B1',
}


def pearson(a, b):
    sa, sb = np.std(a), np.std(b)
    if sa <= 1e-12 or sb <= 1e-12:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


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
                     'prereg': PREREG, 'seed': SEED_A,
                     'design': 'Atlas-E200 vocab preresearch (Arm A, '
                               'unembed-level) + census pipeline '
                               'reproduction check vs phase2846 '
                               '(Arm B, L29-31, 40 words)'}
        fc.save(execution_path, execution)

    lib_sha = hashlib.sha256(
        (Path(__file__).parent / 'rdc_atlas_census.py')
        .read_bytes()).hexdigest()[:16]

    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        str(ROOT / 'models' / 'hf' / 'qwen3-4b'), local_files_only=True,
        trust_remote_code=True, use_fast=True)
    from phase2662_symmetric_mapping_contract import load_native
    model, _ = load_native('qwen4')
    model.eval()
    W_U = model.lm_head.weight.detach().float().cpu().numpy()

    # ---------------- Arm A: vocab extension ----------------
    vocab80 = build_vocab(tok, W_U, CATS, CAT_WORDS,
                          SEED_B, MAX_WORDS_OLD)
    dW80 = vocab80['dW_unit']
    Cm80 = vocab80['Cm']

    chk = {}
    cats200 = {}
    new_words = {}
    e1_counts = {}
    for cat in CAT_WORDS:
        surv = []
        for w in CAND[cat]:
            try:
                single_token_id(tok, w, chk)
                surv.append(w)
            except AssertionError:
                pass
        take = surv[:10]
        new_words[cat] = take
        e1_counts[cat] = len(take)
        cats200[cat] = list(CATS[cat]) + take
    n_full = sum(1 for cat in CAT_WORDS if e1_counts[cat] >= 10)
    e1 = bool(n_full >= 9)

    vocab200 = build_vocab(tok, W_U, cats200, CAT_WORDS,
                           SEED_A, MAX_WORDS_NEW)
    dW200 = vocab200['dW_unit']
    Cm200 = vocab200['Cm']

    cos_dw = np.array([float(dW200[k] @ dW80[k]) for k in range(10)])
    e2 = bool((cos_dw >= 0.9).sum() >= 9)

    def offdiag_cos(Cm):
        Cn = Cm / np.maximum(np.linalg.norm(Cm, axis=1,
                                            keepdims=True), 1e-30)
        M = Cn @ Cn.T
        return M[~np.eye(10, dtype=bool)]

    c80 = offdiag_cos(Cm80)
    c200 = offdiag_cos(Cm200)
    e3_val = pearson(c80, c200)
    e3 = bool(np.isfinite(e3_val) and e3_val >= 0.95)
    cdw = offdiag_cos(np.stack([dW80[k] / max(np.linalg.norm(dW80[k]),
                                              1e-30)
                                for k in range(10)]))
    cdw200 = offdiag_cos(np.stack([dW200[k] / max(np.linalg.norm(dW200[k]),
                                                  1e-30)
                                   for k in range(10)]))
    e3_dw_val = pearson(cdw, cdw200)

    print('P2856 Arm A done: E1=%s E2=%s E3=%s' % (e1, e2, e3),
          flush=True)
    print('P2856 e1_counts %s' % json.dumps(e1_counts), flush=True)
    print('P2856 cos_dw %s'
          % json.dumps([round(float(x), 4) for x in cos_dw]), flush=True)
    print('P2856 E3 offdiag r centre=%.4f dW(desc)=%.4f'
          % (e3_val, e3_dw_val), flush=True)

    # ---------------- Arm B: pipeline reproduction ----------------
    census = AtlasCensus(model, vocab80, CATS, CAT_WORDS)
    rows = list(range(min(SMOKE_WORDS, vocab80['n_words'])))
    b_layers = {}
    resid_all = []
    drops_new = np.zeros((len(SMOKE_LAYERS), len(rows), 32))
    s0_new = np.zeros((len(SMOKE_LAYERS), len(rows), 32))
    s1_new = np.zeros((len(SMOKE_LAYERS), len(rows), 32))
    drops_ref = np.zeros_like(drops_new)
    for k, li in enumerate(SMOKE_LAYERS):
        dL, s0L, s1L, cr = census.measure_layer(li, rows)
        drops_new[k] = dL
        s0_new[k] = s0L
        s1_new[k] = s1L
        resid_all.extend(cr)
        z = np.load(REF_2846 / ('census_L%d.npz' % li))
        ref = z['drops'][:len(rows)]
        drops_ref[k] = ref
        d = dL - ref
        r = pearson(dL.reshape(-1).astype(np.float64),
                    ref.reshape(-1).astype(np.float64))
        med = float(np.median(np.abs(d)))
        b_layers[li] = {'r': round(r, 5), 'med_abs_diff': round(med, 6),
                        'mean_new': round(float(dL.mean()), 6),
                        'mean_ref': round(float(ref.mean()), 6)}
        print('P2856 Arm B L%d r=%.5f med|d|=%.6f' % (li, r, med),
              flush=True)
    b1 = bool(all(v['r'] >= 0.99 and v['med_abs_diff'] <= 0.002
                  for v in b_layers.values()))

    verdict = {
        'lib_sha256_16': lib_sha,
        'n_words_80': vocab80['n_words'],
        'E1_e200_coverage': e1,
        'e1_new_word_counts': e1_counts,
        'new_words': new_words,
        'n_words_200': vocab200['n_words'],
        'E2_direction_stable': e2,
        'cos_dW200_dW80_per_class': [round(float(x), 4)
                                     for x in cos_dw],
        'min_cos': round(float(cos_dw.min()), 4),
        'E3_geometry_retained': e3,
        'E3_offdiag_centre_cos_r': round(e3_val, 5),
        'E3_offdiag_dW_cos_r_desc': round(e3_dw_val, 5),
        'B1_pipeline_verified': b1,
        'b1_per_layer': b_layers,
        'clamp_max_resid': round(float(np.max(resid_all)), 5)
            if resid_all else None,
        'final_verdict': 'atlas_e200_ready=%s/pipeline_verified=%s'
                         % (e1 and e2 and e3, b1),
    }

    result = {'phase': 2856, 'prereg': PREREG, 'verdict': verdict}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'atlas_e200.npz',
           dW_unit_80=dW80.astype(np.float64),
           dW_unit_200=dW200.astype(np.float64),
           Cm_80=Cm80.astype(np.float64),
           Cm_200=Cm200.astype(np.float64),
           cos_dw=cos_dw.astype(np.float64),
           smoke_layers=np.array(SMOKE_LAYERS),
           drops_new=drops_new.astype(np.float32),
           drops_ref=drops_ref.astype(np.float32),
           s0_new=s0_new.astype(np.float32),
           s1_new=s1_new.astype(np.float32))

    elapsed = time.monotonic() - t0
    cc.ledger('phase2856', elapsed)
    print('P2856 VERDICT %s' % json.dumps(verdict), flush=True)
    print('P2856 elapsed %.1fs' % elapsed, flush=True)

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
