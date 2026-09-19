"""Phase 2800 (LPF-13): carrier SUBSPACE joint discrimination + signed
pooled channel profile + category-channel geometry.  Pure RE-ANALYSIS
(atlas.npz E30/H30 + 2797 dim_attribution.npz + weights; NO forward).

2798 established: carrier selection (not eta^2) predicts causal
specificity (10/10 vs 5/10); 2799 established single carrier dims have
polysemantic W_U columns (1/10 category-relevant).  Open questions:
  A  Is a TOP-K CARRIER subspace jointly sufficient for readout
     discriminability?  At what K does it saturate?  Does it beat
     eta^2-selected and random subspaces of the same size?
  B  Does a SIGNED, dW-weighted POOLED profile over the top-40
     carrier dims (the subspace's channel direction mapped into
     vocab space) surface category semantics that single dims hide?
     (direction-level vs dim-level semantics)
  C  Category channel geometry: are the 10 category readout channels
     (dW rows) near-orthogonal?  How much of each channel does its
     top-40 carrier subspace capture?  Do carrier sets overlap?

Prereg (frozen before any readout):
  P-A4  carrier_subspace_sufficient iff for >= 8/10 categories:
        d'(top-40 carrier dims) >= 0.7 * d'(full 2560) AND
        d'(top-40 carrier) > d'(top-40 eta2) AND
        d'(top-40 carrier) > mean d'(random-40, 10 draws).
        d' = (mean s_members - mean s_nonmembers) / pooled sd,
        s(w) = lens margin of w toward category c computed with all
        non-selected dims zeroed (exact additive decomposition).
  P-B4  signed_pool_profile_semantic iff >= 5/10 categories have >= 1
        category-relevant token (instance word / category word /
        preregistered attribute list) among top-30 tokens of the
        signed pooled profile p_c = sum_{d in top40} dW[c,d]*W_U[:,d]
        (union of positive and negative side).
  P-C4  channels_near_orthogonal_fruitfood_lead iff
        mean |cos(dW_c, dW_c')| over 45 pairs < 0.5 AND fruit-food is
        among the top-3 pairs by |cos| (echoes 2795 centroid finding).
  D     descriptive: K* (smallest K reaching 80% of full d') per cat;
        full carrier/eta2/random K-curves; subspace-capture
        cos(dW_c|top40, dW_c); top-40 carrier-dim overlap census
        (45 pairs + random baseline 40*40/2560); sample profiles.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2800' / 'qwen4_carrier_subspace'
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
CAT_WORDS = list(CATS.keys())
# preregistered attribute lists (frozen before readout)
ATTRS = {
    'fruit': ['sweet', 'juice', 'tree', 'red', 'orchard', 'ripe',
              'seed', 'taste'],
    'animal': ['fur', 'tail', 'pet', 'wild', 'zoo', 'bark', 'meow',
               'creature'],
    'metal': ['shiny', 'hard', 'ore', 'mine', 'metal', 'alloy',
              'rust', 'precious'],
    'vehicle': ['drive', 'road', 'wheel', 'engine', 'speed', 'travel',
                'ride'],
    'country': ['capital', 'flag', 'language', 'population', 'city',
                'nation', 'government'],
    'food': ['eat', 'tasty', 'cook', 'meal', 'dish', 'kitchen',
             'flavor', 'delicious'],
    'nature': ['water', 'land', 'earth', 'wild', 'landscape',
               'natural', 'outdoor'],
    'furniture': ['room', 'sit', 'wood', 'home', 'house', 'interior',
                  'seat'],
    'tool': ['work', 'fix', 'build', 'hand', 'hardware', 'repair',
             'construct'],
    'clothing': ['wear', 'fabric', 'fashion', 'body', 'style', 'worn',
                 'apparel'],
}
K_GRID = [5, 10, 20, 40, 80, 160, 320]
K_CRIT = 40
N_RAND = 10
SEED = 2800

PREREG = {
    'P-A4': 'carrier_subspace_sufficient iff >= 8/10 cats: '
            "d'(top-40 carrier) >= 0.7*d'(full) AND > d'(top-40 eta2) "
            "AND > mean d'(random-40 x10)",
    'P-B4': 'signed_pool_profile_semantic iff >= 5/10 cats have >= 1 '
            'relevant token (instance/category/attribute) in top-30 '
            'signed pooled profile tokens (either sign)',
    'P-C4': 'channels_near_orthogonal_fruitfood_lead iff '
            'mean|cos(dW rows)| over 45 pairs < 0.5 AND fruit-food in '
            'top-3 pairs by |cos|',
    'verdict': 'subspace_carrier_model iff P-A4 AND P-C4',
}


def margins_from_E(Ew, W_U, g, eps, idx_cat):
    rms = np.sqrt(np.mean(Ew.astype(np.float64) ** 2, axis=1) + eps)
    Hn = (Ew.astype(np.float64) / rms[:, None]) * g[None, :]
    Z = Hn @ W_U[idx_cat].T.astype(np.float64)
    return Z - (Z.sum(1, keepdims=True) - Z) / 9.0


def dprime(s, members):
    """s: scores over 100 words for one category; members: bool."""
    a = s[members]
    b = s[~members]
    pool = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    return float((a.mean() - b.mean()) / max(pool, 1e-9))


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    data = np.load(SRC_NPZ, allow_pickle=True)
    words = [str(w) for w in data['words']]
    E30 = data['E30'].astype(np.float64)
    assert words == WORDS
    d27 = np.load(SRC_2797, allow_pickle=True)
    eta2 = d27['eta2_E'].astype(np.float64)
    carrier = d27['carrier'].astype(np.float64)
    labels = d27['labels'].astype(int)
    members = [labels == c for c in range(10)]

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG, 'cats': CATS, 'attrs': ATTRS,
                 'k_grid': K_GRID, 'k_crit': K_CRIT, 'n_rand': N_RAND,
                 'seed': SEED, 'src_npz': str(SRC_NPZ),
                 'src_2797': str(SRC_2797)}
    fc.save(OUT / 'execution.json', execution)

    src = json.loads(Path(SRC_RESULT).read_text(encoding='utf-8'))
    mce = src['margin_cat_emb']
    own = labels

    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        str(ROOT / 'models' / 'hf' / 'qwen3-4b'), local_files_only=True,
        trust_remote_code=True, use_fast=True)
    from phase2662_symmetric_mapping_contract import load_native
    model, _ = load_native('qwen4')
    model.eval()
    W_U = model.lm_head.weight.detach().float().cpu().numpy()
    final_norm = model.model.norm
    g = final_norm.weight.detach().float().cpu().numpy().astype(np.float64)
    eps = float(final_norm.variance_epsilon)

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
    idx_cat = [tid(c) for c in CAT_WORDS]
    assert len(set(idx_cat)) == 10

    M0 = margins_from_E(E30, W_U, g, eps, idx_cat)
    gate = max(abs(float(M0[i, own[i]]) - float(mce[w]))
               for i, w in enumerate(WORDS))
    print('P2800 gate numpy lens vs 2795 max|d|=%.6f' % gate, flush=True)
    assert gate < 0.05

    Wcat = W_U[idx_cat].astype(np.float64)
    dW = Wcat - (Wcat.sum(0, keepdims=True) - Wcat) / 9.0

    # ---------- Arm A: carrier-subspace K-saturation ----------
    rng = np.random.default_rng(SEED)
    eta2_order = np.argsort(-eta2)
    curves = {'carrier': {}, 'eta2': {}, 'rand': {}}
    kstar = {}
    a_ok = 0
    a_rows = {}
    for c in range(10):
        carr_order = np.argsort(-carrier[c])
        dp_full = dprime(M0[:, c], members[c])
        row = {'dp_full': dp_full, 'carrier': {}, 'eta2': {}, 'rand': {}}
        for K in K_GRID:
            sets = {'carrier': carr_order[:K], 'eta2': eta2_order[:K]}
            for name, dims in sets.items():
                Em = np.zeros_like(E30)
                Em[:, dims] = E30[:, dims]
                MK = margins_from_E(Em, W_U, g, eps, idx_cat)
                dp = dprime(MK[:, c], members[c])
                row[name][K] = dp
                curves[name].setdefault(K, []).append(dp)
            rdp = []
            for _ in range(N_RAND):
                rd = rng.choice(2560, K, replace=False)
                Em = np.zeros_like(E30)
                Em[:, rd] = E30[:, rd]
                MK = margins_from_E(Em, W_U, g, eps, idx_cat)
                rdp.append(dprime(MK[:, c], members[c]))
            row['rand'][K] = {'mean': float(np.mean(rdp)),
                              'max': float(np.max(rdp))}
            curves['rand'].setdefault(K, []).append(float(np.mean(rdp)))
        dp40 = row['carrier'][K_CRIT]
        ok = bool(dp40 >= 0.7 * dp_full
                  and dp40 > row['eta2'][K_CRIT]
                  and dp40 > row['rand'][K_CRIT]['mean'])
        a_ok += int(ok)
        for K in K_GRID:
            if row['carrier'][K] >= 0.8 * dp_full:
                row['kstar80'] = K
                break
        else:
            row['kstar80'] = None
        kstar[CAT_WORDS[c]] = row['kstar80']
        a_rows[CAT_WORDS[c]] = {
            'dp_full': round(dp_full, 4),
            'dp_carrier40': round(dp40, 4),
            'dp_eta2_40': round(row['eta2'][K_CRIT], 4),
            'dp_rand40_mean': round(row['rand'][K_CRIT]['mean'], 4),
            'frac_of_full': round(dp40 / max(dp_full, 1e-9), 3),
            'kstar80': row['kstar80'], 'pass': ok}
        print('P2800 cat %-9s dp_full=%.3f carrier40=%.3f (%.0f%%) '
              'eta2_40=%.3f rand40=%.3f K*80=%s pass=%s'
              % (CAT_WORDS[c], dp_full, dp40,
                 100 * dp40 / max(dp_full, 1e-9), row['eta2'][K_CRIT],
                 row['rand'][K_CRIT]['mean'], row['kstar80'], ok),
              flush=True)
    p_a4 = bool(a_ok >= 8)
    print('P2800 P-A4=%s (%d/10 cats)' % (p_a4, a_ok), flush=True)

    # ---------- Arm B: signed pooled channel profile ----------
    dec = {}

    def dec_tok(i):
        if i not in dec:
            dec[i] = tok.decode([int(i)]).strip().lower()
        return dec[i]
    b_ok = 0
    b_rows = {}
    profile_top = {}
    for c in range(10):
        topd = np.argsort(-carrier[c])[:K_CRIT]
        p = (dW[c, topd][:, None] * W_U[:, topd].T.astype(np.float64)
             ).sum(0)
        pos = np.argsort(-p)[:30]
        neg = np.argsort(p)[:30]
        rel = set(w.lower() for w in CATS[CAT_WORDS[c]])
        rel.add(CAT_WORDS[c])
        rel.update(ATTRS[CAT_WORDS[c]])
        toks_pos = [dec_tok(i) for i in pos]
        toks_neg = [dec_tok(i) for i in neg]
        hit_pos = sorted(set(t for t in toks_pos
                             for r in rel if r in t))
        hit_neg = sorted(set(t for t in toks_neg
                             for r in rel if r in t))
        hits = sorted(set(hit_pos) | set(hit_neg))
        ok = bool(len(hits) >= 1)
        b_ok += int(ok)
        profile_top[CAT_WORDS[c]] = {
            'pos30': toks_pos[:15], 'neg30': toks_neg[:15],
            'hits': hits, 'pass': ok}
        b_rows[CAT_WORDS[c]] = {'n_hits': len(hits), 'pass': ok}
        print('P2800 cat %-9s signed-pool hits=%s pass=%s | pos15=%s'
              % (CAT_WORDS[c], hits if hits else '-', ok,
                 toks_pos[:8]), flush=True)
    p_b4 = bool(b_ok >= 5)
    print('P2800 P-B4=%s (%d/10 cats)' % (p_b4, b_ok), flush=True)

    # ---------- Arm C: category-channel geometry ----------
    Wn = dW / np.linalg.norm(dW, axis=1, keepdims=True)
    C = Wn @ Wn.T
    iu = np.triu_indices(10, 1)
    absc = np.abs(C[iu])
    pair_names = ['%s/%s' % (CAT_WORDS[i], CAT_WORDS[j])
                  for i, j in zip(*iu)]
    mean_abs_cos = float(absc.mean())
    order = np.argsort(-absc)
    top3 = [(pair_names[k], round(float(absc[k]), 4))
            for k in order[:3]]
    ff_idx = pair_names.index('fruit/food')
    ff_rank = int((absc > absc[ff_idx]).sum()) + 1
    p_c4 = bool(mean_abs_cos < 0.5 and ff_rank <= 3)
    capture = {}
    for c in range(10):
        topd = np.argsort(-carrier[c])[:K_CRIT]
        v = dW[c]
        cap = float(np.sqrt((v[topd] ** 2).sum()
                            / max((v ** 2).sum(), 1e-9)))
        capture[CAT_WORDS[c]] = round(cap, 4)
    overlap = np.zeros((10, 10), dtype=int)
    top40 = {c: set(np.argsort(-carrier[c])[:K_CRIT].tolist())
             for c in range(10)}
    for i in range(10):
        for j in range(10):
            if i != j:
                overlap[i, j] = len(top40[i] & top40[j])
    ov_iu = overlap[iu]
    print('P2800 channel geometry: mean|cos|=%.4f top3=%s '
          'fruit/food rank=%d P-C4=%s'
          % (mean_abs_cos, top3, ff_rank, p_c4), flush=True)
    print('P2800 subspace capture (top-40): %s' % json.dumps(capture),
          flush=True)
    print('P2800 top-40 overlap: max=%d mean=%.1f rand_expect=%.1f'
          % (int(ov_iu.max()), float(ov_iu.mean()), 40 * 40 / 2560),
          flush=True)

    verdict = {
        'carrier_subspace_sufficient': p_a4, 'n_cats_ok': a_ok,
        'a_rows': a_rows,
        'signed_pool_profile_semantic': p_b4, 'n_cats_hits': b_ok,
        'channels_near_orthogonal_fruitfood_lead': p_c4,
        'mean_abs_cos': mean_abs_cos, 'top3_pairs': top3,
        'fruitfood_rank': ff_rank,
        'subspace_capture': capture,
        'subspace_carrier_model': bool(p_a4 and p_c4),
    }
    result = {'phase': 2800, 'prereg': PREREG, 'verdict': verdict,
              'b_rows': b_rows,
              'profiles': {k: {kk: vv for kk, vv in v.items()
                               if kk != 'pass'}
                           for k, v in profile_top.items()}}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'subspace.npz',
           dW=dW, cosmat=C, overlap=overlap, labels=labels,
           eta2=eta2, carrier=carrier,
           dp_carrier=np.array([[curves['carrier'][K][c]
                                 for K in K_GRID]
                                for c in range(10)]),
           dp_eta2=np.array([[curves['eta2'][K][c] for K in K_GRID]
                             for c in range(10)]),
           dp_rand=np.array([[curves['rand'][K][c] for K in K_GRID]
                             for c in range(10)]),
           k_grid=np.array(K_GRID))
    print('P2800 VERDICT %s' % json.dumps(
        {k: v for k, v in verdict.items() if k != 'a_rows'}),
        flush=True)


if __name__ == '__main__':
    main()
