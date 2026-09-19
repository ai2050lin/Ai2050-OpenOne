"""Phase 2796 (LPF-9): full-atlas retain spectrum + difference-vector
archive + atlas clustering.  Pure RE-ANALYSIS of phase2795 atlas.npz
(H30/E30 stored): load weights once, recompute lens readouts offline,
NO new forward passes.

2795 findings to generalize into laws:
  (i)  category-direction reuse is heterogeneous (apple keeps 41% of
       its fruit-direction prior, dog suppresses animal to 3%) -- is
       this a LAW over the full 100-word atlas?
  (ii) near-pair differentiation lives in readout-channel content
       selection, not in the difference vector (P-U FALSE) -- does the
       difference vector's OWN lens archive z(d) contain the unique
       attribute words?  (predicted ABSENT -> d = word-identity
       carrier, not property carrier)
  (iii) does average-linkage clustering on the 100x100 cos matrix
       recover the 10 preregistered categories?

Prereg (frozen before any readout):
  P-V  retain_spectrum_heterogeneous iff
       max_c(r_c) - min_c(r_c) >= 0.30 AND r_fruit - r_animal >= 0.15,
       where r_c = mean over member words of retain(w, own_c),
       retain(w,c) = margin_H(w,c) / margin_E(w,c),
       margin(w,c) = z(c) - mean(z(other 9 category words)) computed
       at the same readout for the word-position state (H) and the
       pure embedding (E).  Words with margin_E(own) < 0.5 are
       excluded from the mean (small-denominator guard), count
       reported.  margin formulas identical to 2795; sanity gate:
       |M_H[w,own] - margin_cat_2795[w]| < 0.05 for all 100 words
       (same for E) -- phase aborts if the re-analysis does not
       reproduce 2795's own-channel margins.
  P-W  prior_readout_law iff Pearson rho over the 100 own-channel
       pairs (margin_E, margin_H) >= 0.5 AND slope_fruit > slope_animal
       (per-class OLS slope on own-channel points).
  P-X  diff_vector_property_archive iff >= 2/3 near pairs have >= 1
       unique attribute word (red/yellow, bark/fur, driver/passenger)
       inside the top-20 of z(d), d = unit(h_a - h_b)  (matched-
       substring rule as in 2795 P-T).  Falsifiable both ways;
       2795 P-U predicts ABSENT (n_x <= 1).
  P-Y  clustering_recovers_categories iff ARI(average linkage k=10 on
       1-cos distance) >= 0.30 vs the preregistered labels.
  D    descriptive: full 100x10 margin_H/margin_E matrices; own
       retain per word; 10x10 category-centroid cos matrix + nearest
       other category; z(d) top-20 per pair; linkage tree.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2796' / 'qwen4_retain_spectrum'
SRC_NPZ = BASE / 'phase2795' / 'qwen4_semantic_atlas' / 'atlas.npz'
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
PAIRS = [
    ('apple', 'banana', 'fruit', 'red', 'yellow'),
    ('dog', 'cat', 'animal', 'bark', 'fur'),
    ('car', 'bus', 'vehicle', 'driver', 'passenger'),
]
L_READ = 30

PREREG = {
    'P-V': 'retain_spectrum_heterogeneous iff max(r_c)-min(r_c) >= 0.30 '
           'AND r_fruit - r_animal >= 0.15; r_c = mean retain(w, own_c), '
           'retain = margin_H/margin_E, margin = z(c)-mean(z(other9)); '
           'guard: words with margin_E(own) < 0.5 excluded (counted)',
    'P-W': 'prior_readout_law iff Pearson rho(margin_E, margin_H) over '
           '100 own-channel pairs >= 0.5 AND slope_fruit > slope_animal',
    'P-X': 'diff_vector_property_archive iff >= 2/3 pairs contain >= 1 '
           'unique attribute word in z(d) top-20 (2795 predicts ABSENT)',
    'P-Y': 'clustering_recovers_categories iff ARI(average linkage, '
           'k=10, 1-cos) >= 0.30',
    'sanity': 'M_H[w,own] vs 2795 margin_cat and M_E[w,own] vs '
              'margin_cat_emb must match < 0.05 for all 100 words',
    'verdict': 'spectrum_law iff P-V AND P-W; '
               'diff_vector_is_identity_carrier iff NOT P-X',
}


def unit(x):
    return x / max(np.linalg.norm(x), 1e-9)


def ari(la, lb):
    n = len(la)
    cont = {}
    for a, b in zip(la, lb):
        cont[(a, b)] = cont.get((a, b), 0) + 1
    c2 = lambda k: k * (k - 1) / 2.0
    sa = {}
    sb = {}
    for (a, b), v in cont.items():
        sa[a] = sa.get(a, 0) + v
        sb[b] = sb.get(b, 0) + v
    sum_ij = sum(c2(v) for v in cont.values())
    sum_a = sum(c2(v) for v in sa.values())
    sum_b = sum(c2(v) for v in sb.values())
    expected = sum_a * sum_b / c2(n)
    max_index = (sum_a + sum_b) / 2.0
    return (sum_ij - expected) / (max_index - expected)


def average_linkage(D, k):
    """Naive average-linkage on full square distance D (n x n).
    Returns (Z linkage rows, labels at k clusters)."""
    n = D.shape[0]
    big = np.full((2 * n - 1, 2 * n - 1), np.inf)
    big[:n, :n] = D
    np.fill_diagonal(big[:n, :n], np.inf)
    members = {i: [i] for i in range(n)}
    active = list(range(n))
    Z = []
    next_id = n
    while len(active) > k:
        sub = big[np.ix_(active, active)]
        p, q = np.unravel_index(int(np.argmin(sub)), sub.shape)
        a, b = active[int(p)], active[int(q)]
        na, nb = len(members[a]), len(members[b])
        Z.append([float(a), float(b), float(big[a, b]),
                  float(na + nb)])
        members[next_id] = members[a] + members[b]
        for c in active:
            if c == a or c == b:
                continue
            nd = (na * big[a, c] + nb * big[b, c]) / (na + nb)
            big[next_id, c] = big[c, next_id] = nd
        for c in (a, b):
            active.remove(c)
            big[c, :] = np.inf
            big[:, c] = np.inf
        active.append(next_id)
        next_id += 1
    labels = [None] * n
    for lab, root in enumerate(active):
        for i in members[root]:
            labels[i] = lab
    return np.array(Z), labels


def main():
    import torch
    from transformers import AutoTokenizer
    OUT.mkdir(parents=True, exist_ok=True)

    data = np.load(SRC_NPZ, allow_pickle=True)
    words = [str(w) for w in data['words']]
    cats = [str(c) for c in data['cats']]
    H30 = data['H30'].astype(np.float32)
    E30 = data['E30'].astype(np.float32)
    C2795 = data['cos'].astype(np.float32)
    assert words == WORDS, 'word order mismatch vs 2795 npz'
    assert cats == [CAT_OF[w] for w in WORDS]
    src = json.loads(Path(SRC_RESULT).read_text(encoding='utf-8'))
    mc_2795 = src['margin_cat']
    mce_2795 = src['margin_cat_emb']

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG, 'cats': CATS, 'words': WORDS,
                 'pairs': PAIRS, 'l_read': L_READ,
                 'src_npz': str(SRC_NPZ)}
    fc.save(OUT / 'execution.json', execution)

    tok = AutoTokenizer.from_pretrained(
        str(ROOT / 'models' / 'hf' / 'qwen3-4b'), local_files_only=True,
        trust_remote_code=True, use_fast=True)
    from phase2662_symmetric_mapping_contract import load_native
    model, _ = load_native('qwen4')
    model.eval()
    device = next(model.parameters()).device
    W_U = model.lm_head.weight.detach().float().cpu().numpy()
    final_norm = model.model.norm

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

    # ---------- Arm A: retain spectrum (100 x 10) ----------
    idx_cat = [tid(c) for c in CAT_WORDS]
    Z_H = np.zeros((100, 10), dtype=np.float64)
    Z_E = np.zeros((100, 10), dtype=np.float64)
    for i in range(100):
        Z_H[i] = lens_np(H30[i])[idx_cat]
        Z_E[i] = lens_np(E30[i])[idx_cat]
    M_H = Z_H - (Z_H.sum(1, keepdims=True) - Z_H) / 9.0
    M_E = Z_E - (Z_E.sum(1, keepdims=True) - Z_E) / 9.0

    own_idx = [CAT_WORDS.index(CAT_OF[w]) for w in WORDS]
    sH = max(abs(float(M_H[i, own_idx[i]]) - float(mc_2795[w]))
             for i, w in enumerate(WORDS))
    sE = max(abs(float(M_E[i, own_idx[i]]) - float(mce_2795[w]))
             for i, w in enumerate(WORDS))
    print('P2796 sanity vs 2795: max|M_H-MC|=%.6f max|M_E-MCE|=%.6f'
          % (sH, sE), flush=True)
    assert sH < 0.05 and sE < 0.05, 're-analysis does not reproduce 2795'

    retain = np.full(100, np.nan)
    excl = []
    for i, w in enumerate(WORDS):
        me = float(M_E[i, own_idx[i]])
        if me < 0.5:
            excl.append(w)
        else:
            retain[i] = float(M_H[i, own_idx[i]]) / me
    r_by_cat = {}
    for c in CATS:
        vals = [float(retain[i]) for i, w in enumerate(WORDS)
                if CAT_OF[w] == c and not np.isnan(retain[i])]
        if vals:
            r_by_cat[c] = float(np.mean(vals))
    spec = sorted(r_by_cat.items(), key=lambda kv: -kv[1])
    r_span = max(r_by_cat.values()) - min(r_by_cat.values())
    p_v = bool(r_span >= 0.30
               and r_by_cat['fruit'] - r_by_cat['animal'] >= 0.15)
    print('P2796 retain spectrum (desc): %s  span=%.4f  excl=%s'
          % (json.dumps({c: round(v, 4) for c, v in spec}),
             r_span, excl), flush=True)
    print('P2796 P-V=%s' % p_v, flush=True)

    me_own = np.array([float(M_E[i, own_idx[i]]) for i in range(100)])
    mh_own = np.array([float(M_H[i, own_idx[i]]) for i in range(100)])
    rho = float(np.corrcoef(me_own, mh_own)[0, 1])

    def slope(c):
        ii = [i for i, w in enumerate(WORDS) if CAT_OF[w] == c]
        return float(np.polyfit(me_own[ii], mh_own[ii], 1)[0])

    s_fruit, s_animal = slope('fruit'), slope('animal')
    p_w = bool(rho >= 0.5 and s_fruit > s_animal)
    print('P2796 own-channel rho=%.4f slope_fruit=%.4f '
          'slope_animal=%.4f P-W=%s'
          % (rho, s_fruit, s_animal, p_w), flush=True)

    # ---------- Arm B: difference-vector archive ----------
    dx = {}
    n_x = 0
    for (a, b, shared, u1, u2) in PAIRS:
        ia, ib = WORDS.index(a), WORDS.index(b)
        d = unit(H30[ia] - H30[ib])
        zd = lens_np(d)
        top = [tok.decode([int(t)]).strip().lower()
               for t in np.argsort(-zd)[:20]]
        h1 = any(t == u1 or u1 in t for t in top)
        h2 = any(t == u2 or u2 in t for t in top)
        hs = any(t == shared or shared in t for t in top)
        wa = any(t == a or a in t for t in top)
        wb = any(t == b or b in t for t in top)
        dx['%s/%s' % (a, b)] = {'top20': top, 'hit_uniq': [bool(h1),
                                  bool(h2)], 'hit_shared': bool(hs),
                                'hit_word': [bool(wa), bool(wb)]}
        n_x += int(h1 or h2)
        print('P2796 z(d) %s/%s uniq_hit=%s shared_hit=%s word_hit=%s'
              % (a, b, [bool(h1), bool(h2)], bool(hs),
                 [bool(wa), bool(wb)]), flush=True)
        print('P2796 z(d) top20: %s' % json.dumps(top), flush=True)
    p_x = bool(n_x >= 2)
    print('P2796 P-X=%s (%d/3 pairs)' % (p_x, n_x), flush=True)

    # ---------- Arm C: clustering ----------
    U = np.stack([unit(h) for h in H30])
    Dd = (1.0 - C2795).astype(np.float64)
    np.fill_diagonal(Dd, np.inf)
    Zlink, labels = average_linkage(Dd, 10)
    truth = [CAT_OF[w] for w in WORDS]
    ari_v = float(ari(labels, truth))
    p_y = bool(ari_v >= 0.30)
    cont = {}
    for l, t in zip(labels, truth):
        cont['%d|%s' % (l, t)] = cont.get('%d|%s' % (l, t), 0) + 1
    print('P2796 clustering ARI=%.4f P-Y=%s' % (ari_v, p_y), flush=True)
    print('P2796 cluster x truth: %s' % json.dumps(cont), flush=True)

    cent = np.stack([
        unit(np.mean(H30[[i for i, w in enumerate(WORDS)
                          if CAT_OF[w] == c]], axis=0))
        for c in CAT_WORDS])
    ccos = cent @ cent.T
    nearest = {}
    for j, c in enumerate(CAT_WORDS):
        row = np.where(np.arange(10) == j, -2.0, ccos[j])
        nearest[c] = {'cat': CAT_WORDS[int(np.argmax(row))],
                      'cos': float(np.max(row))}
    print('P2796 nearest other cat: %s' % json.dumps(
        {c: v['cat'] for c, v in nearest.items()}), flush=True)

    verdict = {
        'retain_spectrum_heterogeneous': p_v,
        'r_by_cat': r_by_cat, 'r_span': float(r_span),
        'n_excluded_guard': excl,
        'prior_readout_law': p_w, 'rho_own': rho,
        'slope_fruit': s_fruit, 'slope_animal': s_animal,
        'diff_vector_property_archive': p_x, 'n_pairs_hit': n_x,
        'diff_vector_is_identity_carrier': bool(not p_x),
        'clustering_recovers_categories': p_y, 'ari': ari_v,
        'nearest_other_cat': nearest,
        'spectrum_law': bool(p_v and p_w),
    }
    mh_dict = {WORDS[i] + '|' + CAT_WORDS[j]: float(M_H[i, j])
               for i in range(100) for j in range(10)}
    me_dict = {WORDS[i] + '|' + CAT_WORDS[j]: float(M_E[i, j])
               for i in range(100) for j in range(10)}
    retain_dict = {w: (None if np.isnan(retain[i]) else float(retain[i]))
                   for i, w in enumerate(WORDS)}
    result = {'phase': 2796, 'prereg': PREREG, 'verdict': verdict,
              'margin_H': mh_dict, 'margin_E': me_dict,
              'retain_own': retain_dict, 'dx_top20': dx,
              'cluster_labels': {w: int(labels[i])
                                 for i, w in enumerate(WORDS)}}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'retain_spectrum.npz',
           words=np.array(WORDS).astype(np.str_),
           cats=np.array(cats).astype(np.str_),
           M_H=M_H, M_E=M_E, retain=retain,
           Zlink=Zlink, ccos=ccos,
           labels=np.array(labels))
    print('P2796 VERDICT %s' % json.dumps(verdict), flush=True)


if __name__ == '__main__':
    main()
