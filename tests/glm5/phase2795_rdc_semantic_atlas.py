"""Phase 2795 (LPF-8): noun semantic atlas + reuse/differentiation.

User directive: if the mechanism holds, the key next step is to MAP
the noun structure -- what does the population of nouns look like in
the word-position dwelling space, and how do near pairs like
apple/banana REUSE shared features (fruit) vs DIFFERENTIATE unique
ones (red vs yellow)?

Protocol (enabled by 2793/2794 word-position self-sufficiency):
100 nouns x 10 categories, one forward each on "The {w}", word
position readout at L30 (= layer-29 output, l* of 2790-2794).

Measures per word:
  h30        word-position residual at L30
  lens top50 top-50 lens tokens at L30
  margin_cat z(cat_word) - mean(z(other 9 category words))
  margin_emb same in the pure-embedding lens (prior version)

Atlas analyses:
  C      100x100 cos matrix of unit(h30); category structure.
  Pairs  3 preregistered near pairs:
         apple/banana  shared=fruit  unique red vs yellow
         dog/cat       shared=animal unique bark vs fur
         car/bus       shared=vehicle unique driver vs passenger

Prereg (frozen before any forward):
  P-S  category_clustering iff mean within-category cos (different
       words, same category) - mean between-category cos >= 0.05
       on unit(h30).
  P-T  category_property_reuse iff >= 8/10 categories have their
       category word present in >= 8/10 member words' lens top-50.
  P-U  differentiation_in_difference_vector iff for >= 2/3 near
       pairs: mean(|cos(d, W_U[uniq1])|, |cos(d, W_U[uniq2])|) >=
       |cos(d, W_U[shared])| + 0.05, where d = unit(h_a - h_b).
  D    descriptive: full cos matrix; per-word margin_cat (L30 and
       emb); per-word lens top-10; background |cos(d, random W_U
       rows)| for each pair.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2795' / 'qwen4_semantic_atlas'

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
PAIRS = [
    ('apple', 'banana', 'fruit', 'red', 'yellow'),
    ('dog', 'cat', 'animal', 'bark', 'fur'),
    ('car', 'bus', 'vehicle', 'driver', 'passenger'),
]
N_BG = 300
BG_SEED = 2795
L_READ = 30

PREREG = {
    'P-S': 'category_clustering iff mean within-cat cos (diff words, '
           'same cat) - mean between-cat cos >= 0.05 on unit(h30)',
    'P-T': 'category_property_reuse iff >= 8/10 categories have '
           'their cat word in >= 8/10 member words lens top-50',
    'P-U': 'differentiation_in_difference_vector iff >= 2/3 near '
           'pairs satisfy mean(|cos(d,uniq1)|,|cos(d,uniq2)|) >= '
           '|cos(d,shared)| + 0.05, d = unit(h_a - h_b)',
    'verdict': 'atlas_established iff P-S AND P-T; '
               'reuse_differentiation_geometric iff P-U',
}


def main():
    import torch
    from transformers import AutoTokenizer
    OUT.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(
        str(ROOT / 'models' / 'hf' / 'qwen3-4b'), local_files_only=True,
        trust_remote_code=True, use_fast=True)

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG, 'cats': CATS, 'words': WORDS,
                 'pairs': PAIRS, 'l_read': L_READ}
    fc.save(OUT / 'execution.json', execution)

    from phase2662_symmetric_mapping_contract import load_native
    model, tok2 = load_native('qwen4')
    model.eval()
    device = next(model.parameters()).device
    W_U = model.lm_head.weight.detach().float().cpu().numpy()
    final_norm = model.model.norm
    E = model.model.embed_tokens.weight.detach().float().cpu().numpy()

    def unit(x):
        return x / max(np.linalg.norm(x), 1e-9)

    tc = {}

    def tid(t):
        if t not in tc:
            ids = tok(' ' + t, add_special_tokens=False)['input_ids']
            if len(ids) != 1:
                ids = tok(t, add_special_tokens=False)['input_ids']
            assert len(ids) == 1, t
            tc[t] = int(ids[0])
        return tc[t]

    for t in WORDS + CAT_WORDS + ['red', 'yellow', 'bark', 'fur',
                                  'driver', 'passenger']:
        tid(t)

    def lens_np(h):
        with torch.inference_mode():
            hn = final_norm(
                torch.tensor(h, device=device).unsqueeze(0))
            return W_U @ hn[0].float().cpu().numpy()

    # ---------- forward pass: one per word ----------
    H30 = []
    top50 = {}
    top10 = {}
    margin_cat = {}
    margin_cat_emb = {}
    for w in WORDS:
        ids = tok('The', add_special_tokens=False)['input_ids'] + \
            tok(' ' + w, add_special_tokens=False)['input_ids']
        p = 1
        with torch.inference_mode():
            o = model(torch.tensor([ids], device=device),
                      output_hidden_states=True)
        h30 = o.hidden_states[L_READ][0, p].float().cpu().numpy().copy()
        H30.append(h30)
        z = lens_np(h30)
        top = np.argsort(-z)[:50]
        top50[w] = [tok.decode([int(t)]).strip().lower() for t in top]
        top10[w] = [tok.decode([int(t)]).strip()
                    for t in np.argsort(-z)[:10]]
        cat = CAT_OF[w]
        others = [c for c in CAT_WORDS if c != cat]
        margin_cat[w] = (float(z[tid(cat)])
                         - float(np.mean([z[tid(c)] for c in others])))
        ze = lens_np(E[tid(w)])
        margin_cat_emb[w] = (float(ze[tid(cat)])
                             - float(np.mean([ze[tid(c)]
                                              for c in others])))
    H30 = np.stack(H30)
    print('P2795 forward done: %d words' % len(WORDS), flush=True)

    # ---------- Arm C: atlas cos matrix ----------
    U = np.stack([unit(h) for h in H30])
    C = U @ U.T
    words_a = np.array(WORDS)
    cats_a = np.array([CAT_OF[w] for w in WORDS])
    iu = np.triu_indices(len(WORDS), k=1)
    same_w = cats_a[iu[0]] == cats_a[iu[1]]
    within = float(np.mean(C[iu][same_w]))
    between = float(np.mean(C[iu][~same_w]))
    gap = within - between
    p_s = bool(gap >= 0.05)
    print('P2795 within=%.4f between=%.4f gap=%.4f P-S=%s'
          % (within, between, gap, p_s), flush=True)
    per_cat_within = {}
    for c in CATS:
        idx = np.where(cats_a == c)[0]
        sub = C[np.ix_(idx, idx)]
        n = len(idx)
        tri = np.triu_indices(n, k=1)
        per_cat_within[c] = float(np.mean(sub[tri]))

    # ---------- P-T: category property reuse ----------
    reuse = {}
    for c, members in CATS.items():
        hits = [w for w in members
                if any(t == c or c in t for t in top50[w])]
        reuse[c] = (len(hits), hits)
    n_cats_ok = sum(1 for c, (k, _) in reuse.items() if k >= 8)
    p_t = bool(n_cats_ok >= 8)
    print('P2795 reuse per cat: %s' % json.dumps(
        {c: k for c, (k, _) in reuse.items()}), flush=True)
    print('P2795 P-T=%s (%d/10 cats >= 8/10 words)'
          % (p_t, n_cats_ok), flush=True)

    # ---------- P-U: differentiation in difference vector ----------
    rng = np.random.default_rng(BG_SEED)
    banned = set(tc.values())
    bg_ids = []
    while len(bg_ids) < N_BG:
        i = int(rng.integers(0, W_U.shape[0]))
        if i not in banned:
            bg_ids.append(i)
            banned.add(i)
    bg_dirs = np.stack([unit(W_U[i]) for i in bg_ids])

    pair_res = {}
    n_u_ok = 0
    for (a, b, shared, u1, u2) in PAIRS:
        ia, ib = WORDS.index(a), WORDS.index(b)
        d = unit(H30[ia] - H30[ib])
        ca = abs(float(d @ unit(W_U[tid(u1)])))
        cb = abs(float(d @ unit(W_U[tid(u2)])))
        cs = abs(float(d @ unit(W_U[tid(shared)])))
        cw1 = abs(float(d @ unit(W_U[tid(a)])))
        cw2 = abs(float(d @ unit(W_U[tid(b)])))
        bgc = float(np.mean(np.abs(bg_dirs @ d)))
        uniq_mean = (ca + cb) / 2
        ok = uniq_mean >= cs + 0.05
        n_u_ok += int(ok)
        pair_res['%s/%s' % (a, b)] = {
            'uniq_align': [ca, cb], 'uniq_mean': uniq_mean,
            'shared_align': cs, 'word_align': [cw1, cw2],
            'bg_align': bgc, 'diff_margin_shared':
                float(margin_cat[a] - margin_cat[b]),
            'pass': bool(ok)}
        print('P2795 pair %s/%s uniq=%.4f shared=%.4f bg=%.4f '
              'pass=%s' % (a, b, uniq_mean, cs, bgc, ok), flush=True)
    p_u = bool(n_u_ok >= 2)

    verdict = {
        'category_clustering': p_s, 'within_cat_cos': within,
        'between_cat_cos': between, 'gap': gap,
        'per_cat_within': per_cat_within,
        'category_property_reuse': p_t, 'n_cats_reuse': n_cats_ok,
        'reuse_detail': {c: k for c, (k, _) in reuse.items()},
        'differentiation_in_difference_vector': p_u,
        'pairs': pair_res,
        'atlas_established': bool(p_s and p_t),
        'reuse_differentiation_geometric': p_u,
    }
    result = {'phase': 2795, 'prereg': PREREG, 'verdict': verdict,
              'top10': top10, 'top50_has_cat': {w: reuse[CAT_OF[w]][0]
                                                for w in []},
              'margin_cat': margin_cat,
              'margin_cat_emb': margin_cat_emb}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'atlas.npz',
           words=words_a.astype(np.str_),
           cats=cats_a.astype(np.str_),
           cos=C, H30=H30, E30=np.stack([E[tid(w)] for w in WORDS]))
    print('P2795 VERDICT %s' % json.dumps(
        {k: v for k, v in verdict.items()
         if k != 'pairs'}), flush=True)


if __name__ == '__main__':
    main()
