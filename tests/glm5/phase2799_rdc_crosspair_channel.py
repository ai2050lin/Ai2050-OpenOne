"""Phase 2799 (LPF-12): cross-category difference-vector control +
carrier-dim channel portraits.  Pure RE-ANALYSIS (E30/H30 + 2797 npz
+ weights; NO forward pass).

2798 left two open points:
  (i)  P-C2 found same-category near-pair difference vectors AVOID
       category dims (sig frac 0.20 < random 0.25-0.40), explained
       as: a-b cancels the shared category component.  The missing
       CONTROL: cross-category pairs (apple/dog) should ENRICH
       category dims -- the category component is NOT cancelled.
  (ii) What do carrier dims point to downstream?  For a fixed word
       and dim, delta z(t) = -W_U[t,d]*g[d]*E[w,d]/rms_w, so the
       channel portrait of dim d is its W_U column: which vocab
       tokens receive the readout through this dim?  Are they the
       category's own member words?

Prereg (frozen before any readout):
  P-A3  cross_pair_carries_shared_components iff >= 5/6 preregistered
        cross-category pairs have top-20 |dH| dim sig-frac >= 0.471
        (1.5 x base 31.4%) AND >= 5/6 pairs frac > their paired
        random-20-dim control (seed 2799).
  P-B3  carrier_dims_point_to_class_tokens iff for >= 7/10
        categories, the union of top-20 |W_U[:,d]| tokens over the
        category's top-3 carrier dims contains >= 1 member word of
        that category (lowercase substring match, 2795 rule).
  D     descriptive: per-pair top-20 dims with eta2/carrier values;
        same-category 3 pairs re-measured as within-design control;
        per-category top-3 carrier dims' token portraits (top-10
        decoded each).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2799' / 'qwen4_crosspair_channel'
SRC_NPZ = BASE / 'phase2795' / 'qwen4_semantic_atlas' / 'atlas.npz'
SRC_2797 = BASE / 'phase2797' / 'qwen4_dim_attribution' / 'dim_attribution.npz'
SRC_RESULT = BASE / 'phase2795' / 'qwen4_semantic_atlas' / 'result.json'

CATS = {
    'fruit': ['apple', 'banana', 'orange', 'grape', 'lemon', 'peach',
              'pear', 'mango', 'cherry', 'berry'],
    'animal': ['dog', 'cat', 'horse', 'cow', 'lion', 'tiger', 'wolf',
               'rabbit', 'bird', 'fish'],
    'metal': ['gold', 'silver', 'iron', 'copper', 'steel', 'bronze',
              'brass', 'tin', 'aluminum', 'nickel'],
    'vehicle': ['car', 'bus', 'truck', 'train', 'ship', 'boat',
                'plane', 'bicycle', 'taxi', 'tram'],
    'country': ['Japan', 'China', 'France', 'Germany', 'Brazil',
                'India', 'Canada', 'Russia', 'Italy', 'Egypt'],
    'food': ['bread', 'rice', 'cheese', 'egg', 'meat', 'soup',
             'pasta', 'pizza', 'honey', 'butter'],
    'nature': ['ocean', 'river', 'mountain', 'forest', 'desert',
               'lake', 'rain', 'snow', 'cloud', 'wind'],
    'furniture': ['chair', 'table', 'bed', 'desk', 'sofa', 'shelf',
                  'cabinet', 'bench', 'stool', 'wardrobe'],
    'tool': ['hammer', 'knife', 'file', 'wrench', 'drill', 'saw',
             'axe', 'nail', 'rope', 'shovel'],
    'clothing': ['shirt', 'pants', 'dress', 'coat', 'shoe', 'sock',
                 'hat', 'glove', 'scarf', 'jacket'],
}
WORDS = [w for v in CATS.values() for w in v]
CAT_OF = {w: c for c, v in CATS.items() for w in v}
CAT_WORDS = list(CATS.keys())
CROSS_PAIRS = [
    ('apple', 'dog'),
    ('car', 'apple'),
    ('Japan', 'hammer'),
    ('ocean', 'chair'),
    ('bread', 'dog'),
    ('gold', 'shirt'),
]
SAME_PAIRS = [
    ('apple', 'banana'),
    ('dog', 'cat'),
    ('car', 'bus'),
]
SEED = 2799

PREREG = {
    'P-A3': 'cross_pair_carries_shared_components iff >= 5/6 cross '
            'pairs top-20 |dH| dim sig-frac >= 0.471 AND >= 5/6 '
            'frac > paired random-20 control',
    'P-B3': 'carrier_dims_point_to_class_tokens iff >= 7/10 cats: '
            'union of top-20 |W_U[:,d]| tokens over top-3 carrier '
            'dims contains >= 1 member word (lowercase substring)',
    'verdict': 'shared_component_picture iff P-A3',
}


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    data = np.load(SRC_NPZ, allow_pickle=True)
    words = [str(w) for w in data['words']]
    H30 = data['H30'].astype(np.float64)
    assert words == WORDS
    d27 = np.load(SRC_2797, allow_pickle=True)
    eta2 = d27['eta2_E'].astype(np.float64)
    q995 = d27['q995'].astype(np.float64)
    carrier = d27['carrier'].astype(np.float64)
    sig = eta2 > q995
    base_rate = float(sig.mean())
    assert abs(base_rate - 0.314) < 0.01

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG, 'cats': CATS, 'words': WORDS,
                 'cross_pairs': CROSS_PAIRS, 'same_pairs': SAME_PAIRS,
                 'seed': SEED, 'base_rate': base_rate,
                 'src_npz': str(SRC_NPZ), 'src_2797': str(SRC_2797)}
    fc.save(OUT / 'execution.json', execution)

    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        str(ROOT / 'models' / 'hf' / 'qwen3-4b'), local_files_only=True,
        trust_remote_code=True, use_fast=True)
    from phase2662_symmetric_mapping_contract import load_native
    model, _ = load_native('qwen4')
    model.eval()
    W_U = model.lm_head.weight.detach().float().cpu().numpy()
    src = json.loads(Path(SRC_RESULT).read_text(encoding='utf-8'))
    mce = src['margin_cat_emb']

    tc = {}

    def tid(t):
        if t not in tc:
            ids = tok(' ' + t, add_special_tokens=False)['input_ids']
            if len(ids) != 1:
                ids = tok(t, add_special_tokens=False)['input_ids']
            assert len(ids) == 1, t
            tc[t] = int(ids[0])
        return tc[t]
    for t in WORDS + CAT_WORDS:
        tid(t)

    # ---------- Arm A: cross vs same difference-vector dims ----------
    rng = np.random.default_rng(SEED)
    labels = np.array([CAT_WORDS.index(CAT_OF[w]) for w in WORDS])

    def pair_row(a, b, kind):
        ia, ib = WORDS.index(a), WORDS.index(b)
        dH = np.abs(H30[ia] - H30[ib])
        topd = np.argsort(-dH)[:20]
        frac = float(sig[topd].mean())
        rnd = rng.choice(2560, 20, replace=False)
        frac_rnd = float(sig[rnd].mean())
        row = {'pair': '%s/%s' % (a, b), 'kind': kind,
               'top20_sig_frac': frac, 'rand20_sig_frac': frac_rnd,
               'mean_eta2': float(eta2[topd].mean()),
               'top20_dims': [int(d) for d in topd],
               'enriched': bool(frac >= 0.471),
               'above_rand': bool(frac > frac_rnd)}
        print('P2799 %s %s/%s: sig_frac=%.3f (rand %.3f) '
              'mean_eta2=%.4f enriched=%s above_rand=%s'
              % (kind, a, b, frac, frac_rnd, row['mean_eta2'],
                 row['enriched'], row['above_rand']), flush=True)
        return row

    rows = []
    for (a, b) in CROSS_PAIRS:
        rows.append(pair_row(a, b, 'cross'))
    n_enrich = sum(int(r['enriched']) for r in rows
                   if r['kind'] == 'cross')
    n_above = sum(int(r['above_rand']) for r in rows
                  if r['kind'] == 'cross')
    p_a3 = bool(n_enrich >= 5 and n_above >= 5)
    print('P2799 P-A3=%s (enriched %d/6, above_rand %d/6)'
          % (p_a3, n_enrich, n_above), flush=True)
    same_rows = [pair_row(a, b, 'same') for (a, b) in SAME_PAIRS]
    rows += same_rows

    # ---------- Arm B: carrier-dim channel portraits ----------
    portrait = {}
    n_hit = 0
    for c in range(10):
        cat = CAT_WORDS[c]
        members = [m.lower() for m in CATS[cat]]
        topd = np.argsort(-carrier[c])[:3]
        union_tokens = []
        hits = []
        for d in topd:
            col = np.abs(W_U[:, int(d)])
            top_t = np.argsort(-col)[:20]
            toks = [tok.decode([int(t)]).strip().lower()
                    for t in top_t]
            union_tokens += toks
        for m in members:
            if any(m == t or m in t for t in union_tokens):
                hits.append(m)
        hit = len(hits) >= 1
        n_hit += int(hit)
        sample = []
        for d in topd:
            col = np.abs(W_U[:, int(d)])
            top_t = np.argsort(-col)[:10]
            sample.append([tok.decode([int(t)]).strip()
                           for t in top_t])
        portrait[cat] = {'top3_dims': [int(d) for d in topd],
                         'member_hits': hits, 'hit': hit,
                         'token_sample': sample}
        print('P2799 cat %s dims=%s member_hits=%s'
              % (cat, [int(d) for d in topd], hits), flush=True)
        print('P2799 cat %s W_U col top10 samples: %s'
              % (cat, json.dumps(sample, ensure_ascii=False)),
              flush=True)
    p_b3 = bool(n_hit >= 7)
    print('P2799 P-B3=%s (%d/10 cats)' % (p_b3, n_hit), flush=True)

    verdict = {
        'cross_pair_carries_shared_components': p_a3,
        'n_enriched': n_enrich, 'n_above_rand': n_above,
        'base_rate': base_rate,
        'carrier_dims_point_to_class_tokens': p_b3, 'n_hit_cats': n_hit,
        'shared_component_picture': bool(p_a3),
    }
    result = {'phase': 2799, 'prereg': PREREG, 'verdict': verdict,
              'pair_rows': rows, 'channel_portrait': portrait}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'crosspair.npz',
           eta2=eta2, sig=sig, carrier=carrier, labels=labels)
    print('P2799 VERDICT %s' % json.dumps(verdict), flush=True)


if __name__ == '__main__':
    main()
