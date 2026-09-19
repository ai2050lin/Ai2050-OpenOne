"""Phase 2801 (LPF-14): layer-wise rotation trajectory + per-layer
channel profiles.  REQUIRES FORWARD (100 words x 1 forward each,
collecting ALL 37 hidden states at the word position).

2797 found "information retained, carrier rotates" from the two
endpoints only (eta2(E) vs eta2(H30): rho=0.009, top-30 overlap 1/30,
while category structure gap is preserved).  2800 established the
carrier direction is monosemantic (signed pooled profile hits the
category word 10/10).  Open: WHERE in the 36-layer pipeline does the
carrier switch happen, is it gradual or sudden, and does channel
semantics survive the rotation?

Protocol (identical to 2795 for gating): one forward per word,
ids = tok('The') + tok(' '+w), read position p=1, hidden_states[l].
Lens margin at ANY layer l: M_l[w,c] = final_norm(h_l)[w] . dW[c]
(final_norm = h/rms(h) * g is layer-independent), so the exact
additive decomposition C(d;w,c)=h_l[w,d]*g[d]*dW[c,d]/rms(h_l[w])
holds at EVERY layer.

Prereg (frozen before any readout):
  P-A5  switch_mid_structure_kept iff exists l* in [2,30] with
        Spearman rho(eta2_l, eta2_0) < 0.30 AND
        min gap_l over l in [2,35] >= 0.7 * gap_0
        (gap_l = within-class minus between-class centroid-cos mean,
        the 2795 structure metric computed on layer-l states).
  P-B5  rotation_gradual iff min over l in [1,36] of
        O_l >= 0.25, O_l = |top40-carrier_l  n  top40-carrier_{l-1}|/40
        (no single layer loses more than 75% of its carrier set).
  P-C5  channel_semantics_survives iff mean over the 10 sampled
        layers {0,4,...,36} of (#categories whose signed pooled
        profile from that layer's top-40 carrier dims hits a
        category-relevant token) >= 8.0.
  D     descriptive: full curves rho_l, overlap30_l, gap_l, O_l;
        lens d' per layer per category; member-margin formation
        curves; eta2-l0 gate vs 2797 npz; H30/E30 gates vs atlas.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2801' / 'qwen4_rotation_trajectory'
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
L_READ = 30
NL = 37
K_CARR = 40
SAMPLED = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36]
SEED = 2801

PREREG = {
    'P-A5': 'switch_mid_structure_kept iff exists l* in [2,30]: '
            'rho(eta2_l, eta2_0) < 0.30 AND min gap_l [2,35] >= '
            '0.7*gap_0',
    'P-B5': 'rotation_gradual iff min O_l over l in [1,36] >= 0.25 '
            '(O_l = top40-carrier overlap between adjacent layers)',
    'P-C5': 'channel_semantics_survives iff mean over 10 sampled '
            'layers of (#cats with profile hit) >= 8.0',
    'verdict': 'layer_dynamics_mapped iff P-A5 AND P-B5',
}


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    data = np.load(SRC_NPZ, allow_pickle=True)
    words = [str(w) for w in data['words']]
    E30 = data['E30'].astype(np.float64)
    H30 = data['H30'].astype(np.float64)
    assert words == WORDS
    d27 = np.load(SRC_2797, allow_pickle=True)
    eta2_E_2797 = d27['eta2_E'].astype(np.float64)
    labels = d27['labels'].astype(int)

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG, 'cats': CATS, 'attrs': ATTRS,
                 'nl': NL, 'l_read': L_READ, 'k_carr': K_CARR,
                 'sampled': SAMPLED, 'seed': SEED,
                 'src_npz': str(SRC_NPZ), 'src_2797': str(SRC_2797)}
    fc.save(OUT / 'execution.json', execution)

    src = json.loads(Path(SRC_RESULT).read_text(encoding='utf-8'))
    mce = src['margin_cat_emb']

    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        str(ROOT / 'models' / 'hf' / 'qwen3-4b'), local_files_only=True,
        trust_remote_code=True, use_fast=True)
    from phase2662_symmetric_mapping_contract import load_native
    model, _ = load_native('qwen4')
    model.eval()
    device = next(model.parameters()).device
    W_U = model.lm_head.weight.detach().float().cpu().numpy()
    final_norm = model.model.norm
    g = final_norm.weight.detach().float().cpu().numpy().astype(np.float64)
    eps = float(final_norm.variance_epsilon)
    Etab = model.model.embed_tokens.weight.detach().float().cpu().numpy()

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
    Wcat = W_U[idx_cat].astype(np.float64)
    dW = Wcat - (Wcat.sum(0, keepdims=True) - Wcat) / 9.0

    # ---------- forward: one per word, collect all 37 layers ----------
    HS = np.zeros((NL, len(WORDS), 2560), dtype=np.float32)
    for i, w in enumerate(WORDS):
        ids = tok('The', add_special_tokens=False)['input_ids'] + \
            tok(' ' + w, add_special_tokens=False)['input_ids']
        p = 1
        with torch.inference_mode():
            o = model(torch.tensor([ids], device=device),
                      output_hidden_states=True)
        for l in range(NL):
            HS[l, i] = o.hidden_states[l][0, p].float().cpu().numpy()
    print('P2801 forward done: %d words x %d layers' % (len(WORDS), NL),
          flush=True)

    # ---------- gates vs 2795 ----------
    gateE = float(np.abs(HS[0].astype(np.float64) - E30).max())
    gateH = float(np.abs(HS[L_READ].astype(np.float64) - H30).max())
    hn0 = HS[0].astype(np.float64)
    rms0 = np.sqrt((hn0 ** 2).mean(1) + eps)
    M0 = ((hn0 / rms0[:, None]) * g[None, :]) @ Wcat.T
    Mg0 = M0 - (M0.sum(1, keepdims=True) - M0) / 9.0
    gateM = max(abs(float(Mg0[i, CAT_WORDS.index(CAT_OF[w])])
                    - float(mce[w]))
                for i, w in enumerate(WORDS))
    print('P2800 gates: E=%.6f H30=%.6f lens=%.6f'
          % (gateE, gateH, gateM), flush=True)
    assert gateE < 0.05 and gateH < 0.05 and gateM < 0.05

    dec = {}

    def dec_tok(i):
        if i not in dec:
            dec[i] = tok.decode([int(i)]).strip().lower()
        return dec[i]
    rel_sets = []
    for c in range(10):
        rel = set(w.lower() for w in CATS[CAT_WORDS[c]])
        rel.add(CAT_WORDS[c])
        rel.update(ATTRS[CAT_WORDS[c]])
        rel_sets.append(rel)

    # ---------- per-layer trajectories ----------
    eta2_l = np.zeros((NL, 2560))
    gap_l = np.zeros(NL)
    within_l = np.zeros(NL)
    rho_l = np.zeros(NL)
    ov30_l = np.zeros(NL)
    O_l = np.zeros(NL)
    dprime_l = np.zeros((NL, 10))
    mem_margin_l = np.zeros((NL, 10))
    top40_sets = [None] * NL
    members = [labels == c for c in range(10)]
    eta2_0 = None
    for l in range(NL):
        X = HS[l].astype(np.float64)
        mu = X.mean(0)
        gm = np.stack([X[members[c]].mean(0) for c in range(10)])
        ss_tot = ((X - mu) ** 2).sum(0)
        ss_bet = (10.0 * (gm - mu) ** 2).sum(0)
        eta2_l[l] = ss_bet / np.maximum(ss_tot, 1e-9)
        if l == 0:
            eta2_0 = eta2_l[0].copy()
        U = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True),
                           1e-9)
        Cm = U @ U.T
        same = (labels[:, None] == labels[None, :])
        np.fill_diagonal(same, False)
        diffm = ~same
        within_l[l] = float(Cm[same].mean())
        between_mean = float(Cm[diffm].mean())
        gap_l[l] = within_l[l] - between_mean
        rho_l[l] = spearman(eta2_l[l], eta2_0)
        ov30_l[l] = len(set(np.argsort(-eta2_l[l])[:30])
                        & set(np.argsort(-eta2_0)[:30])) / 30.0
        rms_l = np.sqrt((X ** 2).mean(1) + eps)
        Hn = (X / rms_l[:, None]) * g[None, :]
        Ml = Hn @ Wcat.T
        Mgl = Ml - (Ml.sum(1, keepdims=True) - Ml) / 9.0
        for c in range(10):
            s = Mgl[:, c]
            a, b = s[members[c]], s[~members[c]]
            dprime_l[l, c] = (a.mean() - b.mean()) / max(
                np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2), 1e-9)
            mem_margin_l[l, c] = float(Mgl[members[c], c].mean())
        contrib = np.abs(X / rms_l[:, None] * g[None, :]
                         )[:, None, :] * np.abs(dW)[None, :, :]
        carrier_l = np.zeros((10, 2560))
        for c in range(10):
            carrier_l[c] = contrib[members[c], c].mean(0)
        top40_sets[l] = [set(np.argsort(-carrier_l[c])[:K_CARR]
                             .tolist()) for c in range(10)]
        if l > 0:
            O_l[l] = float(np.mean([
                len(top40_sets[l][c] & top40_sets[l - 1][c])
                for c in range(10)]) / K_CARR)
    gate_eta = float(np.abs(eta2_0 - eta2_E_2797).max())
    gate_gap = abs(float(gap_l[L_READ]) - 0.127)
    print('P2801 gates: eta2_l0 vs 2797 max|d|=%.2e ; gap_30 vs 2795 '
          '|d|=%.4f' % (gate_eta, gate_gap), flush=True)
    assert gate_eta < 1e-6 and gate_gap < 0.02

    # ---------- prereg evaluations ----------
    rho_seg = rho_l[2:31]
    l_star = int(np.argmax(rho_seg < 0.30)) + 2 if bool(
        (rho_seg < 0.30).any()) else None
    gap_ok = bool(gap_l[2:36].min() >= 0.7 * gap_l[0])
    p_a5 = bool(l_star is not None and gap_ok)
    p_b5 = bool(O_l[1:].min() >= 0.25)
    hit_rows = []
    for l in SAMPLED:
        n_hit = 0
        for c in range(10):
            topd = np.argsort(
                -(np.abs(HS[l].astype(np.float64)
                         / np.sqrt((HS[l].astype(np.float64) ** 2)
                                   .mean(1) + eps)[:, None]
                         * g[None, :])[:, None, :]
                  * np.abs(dW)[None, :, :])[
                      members[c], c].mean(0))[:K_CARR]
            p = (dW[c, topd][:, None]
                 * W_U[:, topd].T.astype(np.float64)).sum(0)
            toks = [dec_tok(i) for i in
                    np.argsort(-p)[:30]] + [dec_tok(i) for i in
                                            np.argsort(p)[:30]]
            if any(any(r in t for r in rel_sets[c]) for t in toks):
                n_hit += 1
        hit_rows.append({'layer': l, 'n_cats_hit': n_hit})
        print('P2801 layer %2d profile hits: %d/10' % (l, n_hit),
              flush=True)
    p_c5 = bool(np.mean([r['n_cats_hit'] for r in hit_rows]) >= 8.0)
    print('P2801 P-A5=%s (l*=%s gap_ok=%s) P-B5=%s (min O=%.2f) '
          'P-C5=%s (mean hits %.1f)'
          % (p_a5, l_star, gap_ok, p_b5, float(O_l[1:].min()), p_c5,
             float(np.mean([r['n_cats_hit'] for r in hit_rows]))),
          flush=True)
    print('P2801 curves rho: %s' % json.dumps(
        [round(float(v), 3) for v in rho_l]), flush=True)
    print('P2801 curves gap: %s' % json.dumps(
        [round(float(v), 3) for v in gap_l]), flush=True)
    print('P2801 curves O_l: %s' % json.dumps(
        [round(float(v), 2) for v in O_l]), flush=True)
    print('P2801 curves ov30: %s' % json.dumps(
        [round(float(v), 2) for v in ov30_l]), flush=True)
    print('P2801 dprime per layer (mean over cats): %s' % json.dumps(
        [round(float(v), 2) for v in dprime_l.mean(1)]), flush=True)
    print('P2801 member-margin formation (cat fruit): %s'
          % json.dumps([round(float(v), 2)
                        for v in mem_margin_l[:, 0]]), flush=True)

    verdict = {
        'switch_mid_structure_kept': p_a5, 'l_star': l_star,
        'gap_ok': gap_ok, 'rotation_gradual': p_b5,
        'min_O': float(O_l[1:].min()),
        'channel_semantics_survives': p_c5,
        'profile_hits': hit_rows,
        'layer_dynamics_mapped': bool(p_a5 and p_b5),
    }
    result = {'phase': 2801, 'prereg': PREREG, 'verdict': verdict,
              'gates': {'E': gateE, 'H30': gateH, 'lens': gateM,
                        'eta2_l0': gate_eta, 'gap30': gate_gap}}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'trajectory.npz',
           HS=HS, eta2_l=eta2_l, gap_l=gap_l, within_l=within_l,
           rho_l=rho_l, ov30_l=ov30_l, O_l=O_l,
           dprime_l=dprime_l, mem_margin_l=mem_margin_l,
           labels=labels, dW=dW)
    print('P2801 VERDICT %s' % json.dumps(verdict), flush=True)


if __name__ == '__main__':
    main()
