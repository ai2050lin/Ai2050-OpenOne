"""Phase 2798 (LPF-11): discriminate-vs-carry plane + carrier-side
causality + difference-vector dim decomposition.  Pure RE-ANALYSIS
(atlas.npz E30/H30 + 2797 dim_attribution.npz + weights; NO forward).

2797 established: margin is exactly additive over embedding dims
(C(d;w,c)=E[w,d]*g[d]*dW[c,d]/rms_w); 805/2560 dims are significant
category-discriminating dims (eta^2 over perm null); but top-eta^2
dims are NOT necessarily readout-carrier dims (P-B 5/10).

Phase 2798 questions:
  A  What does the full (eta^2, carrier) plane look like over all
     2560 dims?  Are the two roles correlated-but-distinct?
  B  Do TOP-CARRIER dims (selected purely by carrier, ignoring
     eta^2) pass the causal specificity test that top-eta^2 dims
     failed?  (carrier-side completion of 2797 P-B)
  C  Where do near-pair difference vectors live -- in discriminating
     dims (word-identity geometry) or carrier dims?

Prereg (frozen before any readout):
  P-A2  plane_correlated_not_identical iff Spearman rho(eta^2,
        carrier_own) over the 805 significant dims is in [0.20,
        0.95) -- positively correlated but not the same ranking
        (rho < 0.20 would mean fully independent roles; >= 0.95
        would mean discriminating = carrying).
  P-B2  carrier_dims_causally_specific iff for >= 8/10 categories,
        zeroing each of the top-10 carrier dims (ranked by
        carrier(c) over ALL 2560 dims, eta^2 ignored) changes the
        own-category margin of that category's members MORE
        (group-mean |delta|) than of non-members.
  P-C2  diff_vector_lives_in_discriminating_dims iff for 3/3 near
        pairs, the top-20 dims by |H30[a,d]-H30[b,d]| contain >=
        47.1% significant dims (1.5 x the 31.4% base rate).
  D     descriptive: quadrant census of the plane (sig x carrier-
        high); per-category dual/pure-discriminating/pure-carrying
        dim counts; per-pair top-20 dim lists with eta^2/carrier;
        random-20-dim control enrichment.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2798' / 'qwen4_discriminate_carry'
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
PAIRS = [
    ('apple', 'banana', 'fruit', 'red', 'yellow'),
    ('dog', 'cat', 'animal', 'bark', 'fur'),
    ('car', 'bus', 'vehicle', 'driver', 'passenger'),
]
SEED = 2798

PREREG = {
    'P-A2': 'plane_correlated_not_identical iff 0.20 <= Spearman '
            'rho(eta2, carrier_own) over 805 sig dims < 0.95',
    'P-B2': 'carrier_dims_causally_specific iff >= 8/10 categories: '
            'top-10 carrier dims (by carrier(c), eta2 ignored) '
            'zeroed -> member |dMargin| > non-member (group means)',
    'P-C2': 'diff_vector_lives_in_discriminating_dims iff 3/3 pairs '
            'have >= 47.1% sig dims (1.5 x base 31.4%) in top-20 '
            'dims by |H30[a]-H30[b]|',
    'verdict': 'role_split_confirmed iff P-A2 AND P-B2',
}


def unit(x):
    return x / max(np.linalg.norm(x), 1e-9)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def margins_from_E(Ew, W_U, g, eps, idx_cat):
    rms = np.sqrt(np.mean(Ew.astype(np.float64) ** 2, axis=1) + eps)
    Hn = (Ew.astype(np.float64) / rms[:, None]) * g[None, :]
    Z = Hn @ W_U[idx_cat].T.astype(np.float64)
    return Z - (Z.sum(1, keepdims=True) - Z) / 9.0


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    data = np.load(SRC_NPZ, allow_pickle=True)
    words = [str(w) for w in data['words']]
    H30 = data['H30'].astype(np.float64)
    E30 = data['E30'].astype(np.float64)
    assert words == WORDS
    d27 = np.load(SRC_2797, allow_pickle=True)
    eta2 = d27['eta2_E'].astype(np.float64)
    q995 = d27['q995'].astype(np.float64)
    dim_cat = d27['dim_cat']
    carrier = d27['carrier'].astype(np.float64)
    labels = d27['labels'].astype(int)
    sig = eta2 > q995
    n_sig = int(sig.sum())
    assert n_sig == 805, 'unexpected sig count %d' % n_sig

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG, 'cats': CATS, 'words': WORDS,
                 'pairs': PAIRS, 'seed': SEED,
                 'src_npz': str(SRC_NPZ), 'src_2797': str(SRC_2797)}
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
    print('P2798 gate numpy lens vs 2795 max|d|=%.6f' % gate,
          flush=True)
    assert gate < 0.05

    # ---------- Arm A: the (eta^2, carrier) plane ----------
    carr_max = carrier.max(0)
    carrier_own_sig = np.array([carrier[dim_cat[d], d]
                                for d in range(2560)])
    rho = spearman(eta2[sig], carrier_own_sig[sig])
    p_a2 = bool(0.20 <= rho < 0.95)
    quad_carrier_q = float(np.quantile(carr_max[sig], 0.75))
    qDual = int(((eta2 > q995) & (carr_max >= quad_carrier_q)).sum())
    qPureD = int(((eta2 > q995) & (carr_max < quad_carrier_q)).sum())
    qPureC = int(((eta2 <= q995) & (carr_max >= quad_carrier_q)).sum())
    qRest = 2560 - qDual - qPureD - qPureC
    print('P2798 rho(eta2,carrier)=%.4f P-A2=%s' % (rho, p_a2),
          flush=True)
    print('P2798 quadrants: dual=%d pure_discrim=%d pure_carry=%d '
          'rest=%d (carrier-high = sig-dim Q75 of carrier max)'
          % (qDual, qPureD, qPureC, qRest), flush=True)
    per_cat = {}
    for c in range(10):
        cc_dims = np.argsort(-carrier[c])
        top50 = cc_dims[:50]
        nd = int(sig[top50].sum())
        per_cat[CAT_WORDS[c]] = {
            'top50_carry_sig_frac': nd / 50.0,
            'top10_carry': [int(d) for d in cc_dims[:10]],
            'top10_carry_sig': int(sig[cc_dims[:10]].sum())}
    print('P2798 per-cat top50 carrier sig frac: %s' % json.dumps(
        {c: round(v['top50_carry_sig_frac'], 2)
         for c, v in per_cat.items()}), flush=True)

    # ---------- Arm B: carrier-side causal specificity ----------
    b_ok = 0
    b_rows = {}
    for c in range(10):
        topd = np.argsort(-carrier[c])[:10]
        res_c = []
        ok_c = 0
        for d in topd:
            Ez = E30.copy()
            Ez[:, int(d)] = 0.0
            Mz = margins_from_E(Ez, W_U, g, eps, idx_cat)
            dM = np.abs(M0[np.arange(100), own]
                        - Mz[np.arange(100), own])
            din = float(dM[labels == c].mean())
            dout = float(dM[labels != c].mean())
            ok = bool(din > dout)
            ok_c += int(ok)
            res_c.append({'dim': int(d), 'eta2': round(float(eta2[d]),
                                                          3),
                          'sig': bool(sig[d]), 'delta_in': round(din, 4),
                          'delta_out': round(dout, 4),
                          'specific': ok})
        b_ok += int(ok_c >= 8)
        b_rows[CAT_WORDS[c]] = {'n_specific': ok_c, 'dims': res_c}
        print('P2798 cat %s: %d/10 carrier dims specific'
              % (CAT_WORDS[c], ok_c), flush=True)
    p_b2 = bool(b_ok >= 8)
    print('P2798 P-B2=%s (%d/10 cats)' % (p_b2, b_ok), flush=True)

    # ---------- Arm C: difference-vector dim decomposition ----------
    rng = np.random.default_rng(SEED)
    c_rows = []
    n_c2 = 0
    for (a, b, shared, u1, u2) in PAIRS:
        ia, ib = WORDS.index(a), WORDS.index(b)
        dH = np.abs(H30[ia] - H30[ib])
        topd = np.argsort(-dH)[:20]
        frac = float(sig[topd].mean())
        rnd = rng.choice(2560, 20, replace=False)
        frac_rnd = float(sig[rnd].mean())
        mean_eta2 = float(eta2[topd].mean())
        mean_eta2_rnd = float(eta2[rnd].mean())
        ok = bool(frac >= 0.471)
        n_c2 += int(ok)
        c_rows.append({
            'pair': '%s/%s' % (a, b), 'top20_sig_frac': frac,
            'rand20_sig_frac': frac_rnd,
            'mean_eta2_top20': mean_eta2,
            'mean_eta2_rand20': mean_eta2_rnd,
            'top20_dims': [int(d) for d in topd],
            'pass': ok})
        print('P2798 diff %s/%s: top20 sig frac=%.3f (rand %.3f) '
              'mean eta2=%.4f (rand %.4f) pass=%s'
              % (a, b, frac, frac_rnd, mean_eta2, mean_eta2_rnd, ok),
              flush=True)
    p_c2 = bool(n_c2 == 3)
    print('P2798 P-C2=%s (%d/3)' % (p_c2, n_c2), flush=True)

    verdict = {
        'plane_correlated_not_identical': p_a2, 'rho_plane': rho,
        'quadrants': {'dual': qDual, 'pure_discriminating': qPureD,
                      'pure_carrying': qPureC, 'rest': qRest},
        'carrier_dims_causally_specific': p_b2, 'n_cats_ok': b_ok,
        'b_rows': b_rows,
        'diff_vector_lives_in_discriminating_dims': p_c2,
        'n_pairs_ok': n_c2, 'c_rows': c_rows,
        'role_split_confirmed': bool(p_a2 and p_b2),
    }
    result = {'phase': 2798, 'prereg': PREREG, 'verdict': verdict,
              'per_cat_carry': per_cat}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'plane.npz',
           eta2=eta2, carr_max=carr_max, sig=sig,
           carrier_own_sig=carrier_own_sig, labels=labels)
    print('P2798 VERDICT %s' % json.dumps(
        {k: v for k, v in verdict.items()
         if k not in ('b_rows', 'c_rows')}), flush=True)


if __name__ == '__main__':
    main()
