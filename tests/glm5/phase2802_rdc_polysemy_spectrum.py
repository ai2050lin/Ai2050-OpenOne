"""Phase 2802 (LPF-15): POLYSEMY SPECTRAL DECOMPOSITION of a single
word -- "apple" -- across four senses (fruit / food / plant / company).

User anchor: can we precisely identify WHICH parameters of apple's
word embedding express the fruit sense, which the food sense, which
the plant sense, which the tech-company sense?

Machinery (established 2797-2800): lens margin is EXACTLY additive
over embedding dims; the sense-contrast direction
  dW_s = W_s - mean(W_others)          (4 sense word-sets x 2560)
gives per-dim sense contribution for word w:
  C_s(d; w) = E[w,d] * g[d] * dW_s[d] / rms_w
So apple's 4x2560 contribution matrix CM is exact, zero forward.

Arms:
  A  sense-channel readability: pooled signed profile per sense
     (top-40 carrier dims, p_s = sum dW_s[d]*W_U[:,d]) -> vocab top
     tokens vs preregistered relevance lists (fruit/food reuse 2800
     ATTRS; plant/company new).
  B  apple's sense spectrum: 4x4 Spearman over dims of CM;
     participation ratio per sense (effective #dims); SVD singular
     spectrum of CM (the "spectral analysis"); top-15 dims per sense;
     top-40 set overlaps; overlap with the 805 category sig dims;
     sense-vote census over all dims; apple's default sense ranking
     (sum_d C_s).
  C  reader-side disambiguation: two sentences differing ONLY after
     the apple token ("the apple pie is tasty" vs "the apple released
     the iphone").  By causal-mask immunity (2793) h[apple] must be
     bit-exact identical -> archive readout cannot disambiguate; the
     fruit-minus-company margin must move only at LATER positions
     (final-token lens).  Quantify the flip.

Prereg (frozen before any readout):
  P-A6  sense_channels_readable iff >= 3/4 senses' pooled signed
        profiles hit >= 1 sense-relevant token in top-30 (either sign).
  P-B6  sense_spectra_separated iff Spearman(C_fruit, C_company)
        over 2560 dims < 0.20 AND Spearman(C_fruit, C_food) >
        Spearman(C_fruit, C_company)  (close senses share carriers
        more than distant senses).
  P-C6  disambiguation_reader_side iff max|h1-h2| at apple position
        == 0.0 exactly AND |delta_final| >= 1.0 AND
        (fruit-comp margin at final pos) is larger in the pie
        sentence than in the released sentence.
  D     descriptive: full tables (top dims per sense), singular
        spectrum, votes, sig-dim overlap, default sense ranking.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2802' / 'qwen4_polysemy_spectrum'
SRC_NPZ = BASE / 'phase2795' / 'qwen4_semantic_atlas' / 'atlas.npz'
SRC_2797 = BASE / 'phase2797' / 'qwen4_dim_attribution' / 'dim_attribution.npz'
SRC_RESULT = BASE / 'phase2795' / 'qwen4_semantic_atlas' / 'result.json'

CATS = {
    'fruit': ['apple', 'banana', 'orange', 'grape', 'lemon', 'peach',
              'pear', 'mango', 'cherry', 'berry'],
    'food': ['bread', 'rice', 'cheese', 'egg', 'meat', 'soup',
             'pasta', 'pizza', 'honey', 'butter'],
    'plant': ['tree', 'flower', 'rose', 'leaf', 'root', 'grass',
              'oak', 'pine', 'maple', 'fern'],
    'company': ['Google', 'Microsoft', 'Amazon', 'Meta', 'Tesla',
                'Nvidia', 'Samsung', 'Sony', 'IBM', 'Intel'],
}
SENSES = list(CATS.keys())
ATTRS = {
    'fruit': ['sweet', 'juice', 'tree', 'red', 'orchard', 'ripe',
              'seed', 'taste', 'fruit'],
    'food': ['eat', 'tasty', 'cook', 'meal', 'dish', 'kitchen',
             'flavor', 'delicious', 'food'],
    'plant': ['tree', 'leaf', 'flower', 'grow', 'root', 'seed',
              'soil', 'botan', 'plant', 'garden'],
    'company': ['iphone', 'ipad', 'mac', 'steve', 'jobs', 'tech',
                'company', 'brand', 'software', 'google', 'microsoft',
                'cupertino'],
}
SENT_FRUIT = 'the apple pie is tasty'
SENT_COMP = 'the apple released the iphone'
K_PROF = 40
SEED = 2802

PREREG = {
    'P-A6': 'sense_channels_readable iff >= 3/4 senses pooled signed '
            'profile hits >= 1 relevant token in top-30 (either sign)',
    'P-B6': 'sense_spectra_separated iff Spearman(C_fruit,C_company)'
            ' < 0.20 AND Spearman(C_fruit,C_food) > '
            'Spearman(C_fruit,C_company)',
    'P-C6': 'disambiguation_reader_side iff max|h1-h2| at apple pos '
            '== 0.0 AND |delta_final| >= 1.0 AND fruit-comp margin '
            'larger in pie sentence',
    'verdict': 'polysemy_spectrum_solved iff P-A6 AND P-C6',
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
    d27 = np.load(SRC_2797, allow_pickle=True)
    sig = (d27['eta2_E'].astype(np.float64)
           > d27['q995'].astype(np.float64))
    assert words[0] == 'apple'

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG, 'cats': CATS, 'attrs': ATTRS,
                 'sent_fruit': SENT_FRUIT, 'sent_comp': SENT_COMP,
                 'k_prof': K_PROF, 'seed': SEED,
                 'src_npz': str(SRC_NPZ)}
    fc.save(OUT / 'execution.json', execution)

    src = json.loads(Path(SRC_RESULT).read_text(encoding='utf-8'))
    mce_apple = float(src['margin_cat_emb']['apple'])

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
    sense_words = [w for v in CATS.values() for w in v]
    for t in sense_words + words:
        tid(t)
    idx_s = [[tid(w) for w in CATS[s]] for s in SENSES]

    # gate: apple embedding row identical to atlas + old 10-cat margin
    ia = words.index('apple')
    gateE = float(np.abs(Etab[tid('apple')] - E30[ia]).max())
    Wcat10 = W_U[[tid(c) for c in
                  ['fruit', 'animal', 'metal', 'vehicle', 'country',
                   'food', 'nature', 'furniture', 'tool', 'clothing']]]
    dW10 = Wcat10 - (Wcat10.sum(0, keepdims=True) - Wcat10) / 9.0
    Ea = E30[ia]
    rms_a = np.sqrt((Ea ** 2).mean() + eps)
    z = (Ea / rms_a * g) @ Wcat10.T
    gateM = abs(float(z[0] - (z.sum() - z[0]) / 9.0) - mce_apple)
    print('P2802 gates: E=%.6f lens10cat=%.6f' % (gateE, gateM),
          flush=True)
    assert gateE < 0.05 and gateM < 0.05

    # sense contrast directions (rival = other 3 senses)
    Ws = np.stack([W_U[idx_s[k]].astype(np.float64).mean(0)
                   for k in range(4)])
    dW_s = Ws - (Ws.sum(0, keepdims=True) - Ws) / 3.0
    rms_s = {}
    for k, s in enumerate(SENSES):
        Erows = np.stack([Etab[tid(w)].astype(np.float64)
                          for w in CATS[s]])
        rms_s[k] = np.sqrt((Erows ** 2).mean(1) + eps)

    # ---------- Arm A: per-sense pooled signed profile ----------
    dec = {}

    def dec_tok(i):
        if i not in dec:
            dec[i] = tok.decode([int(i)]).strip().lower()
        return dec[i]
    carriers = np.zeros((4, 2560))
    for k, s in enumerate(SENSES):
        contrib = np.stack([
            np.abs(Etab[tid(w)].astype(np.float64) / rms_s[k][i] * g
                   * dW_s[k])
            for i, w in enumerate(CATS[s])])
        carriers[k] = contrib.mean(0)
    a_ok = 0
    prof_rows = {}
    for k, s in enumerate(SENSES):
        topd = np.argsort(-carriers[k])[:K_PROF]
        p = (dW_s[k, topd][:, None]
             * W_U[:, topd].T.astype(np.float64)).sum(0)
        toks = ([dec_tok(i) for i in np.argsort(-p)[:30]]
                + [dec_tok(i) for i in np.argsort(p)[:30]])
        hits = sorted(set(t for t in toks
                          for r in ATTRS[s] if r in t))
        ok = bool(len(hits) >= 1)
        a_ok += int(ok)
        prof_rows[s] = {'hits': hits, 'pass': ok,
                        'pos15': toks[:15]}
        print('P2802 sense %-8s hits=%s pass=%s | pos15=%s'
              % (s, hits if hits else '-', ok, toks[:8]), flush=True)
    p_a6 = bool(a_ok >= 3)
    print('P2802 P-A6=%s (%d/4)' % (p_a6, a_ok), flush=True)

    # ---------- Arm B: apple's sense spectrum ----------
    CM = np.stack([Ea / rms_a * g * dW_s[k] for k in range(4)])
    rho = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            rho[i, j] = spearman(CM[i], CM[j])
    pr = np.zeros(4)
    for k in range(4):
        w = CM[k] ** 2
        pr[k] = float(w.sum() ** 2 / max((w ** 2).sum(), 1e-30))
    sv = np.linalg.svd(CM, compute_uv=False)
    top_sets = [set(np.argsort(-np.abs(CM[k]))[:K_PROF].tolist())
                for k in range(4)]
    ov = {}
    for i in range(4):
        for j in range(i + 1, 4):
            ov['%s/%s' % (SENSES[i], SENSES[j])] = len(
                top_sets[i] & top_sets[j])
    sig_frac = {SENSES[k]: round(
        float(sig[list(top_sets[k])].mean()), 3) for k in range(4)}
    votes = np.argmax(np.abs(CM), axis=0)
    census = {SENSES[k]: int((votes == k).sum()) for k in range(4)}
    default = {SENSES[k]: round(float(CM[k].sum()), 3)
               for k in range(4)}
    p_b6 = bool(rho[0, 3] < 0.20 and rho[0, 1] > rho[0, 3])
    print('P2802 rho matrix: %s' % json.dumps(
        [[round(float(v), 3) for v in row] for row in rho]),
        flush=True)
    print('P2802 participation ratio: %s' % json.dumps(
        {SENSES[k]: round(float(pr[k]), 1) for k in range(4)}),
        flush=True)
    print('P2802 singular spectrum: %s' % json.dumps(
        [round(float(v), 2) for v in sv]), flush=True)
    print('P2802 top40 overlaps: %s' % json.dumps(ov), flush=True)
    print('P2802 top40 in 805-sig frac: %s' % json.dumps(sig_frac),
          flush=True)
    print('P2802 dim vote census: %s' % json.dumps(census),
          flush=True)
    print('P2802 apple default sense margin: %s' % json.dumps(default),
          flush=True)
    top15 = {}
    for k, s in enumerate(SENSES):
        td = np.argsort(-np.abs(CM[k]))[:15]
        top15[s] = [{'dim': int(d),
                     'C': round(float(CM[k, d]), 4),
                     'sig': bool(sig[d])} for d in td]
        print('P2802 %s top5: %s' % (s, [(t['dim'], t['C'])
                                         for t in top15[s][:5]]),
              flush=True)

    # ---------- Arm C: reader-side disambiguation ----------
    def encode(text):
        return tok(text, add_special_tokens=False)['input_ids']
    ids1, ids2 = encode(SENT_FRUIT), encode(SENT_COMP)
    assert ids1[:2] == ids2[:2]
    pa = 1
    with torch.inference_mode():
        o1 = model(torch.tensor([ids1], device=device),
                   output_hidden_states=True)
        o2 = model(torch.tensor([ids2], device=device),
                   output_hidden_states=True)
    h1a = o1.hidden_states[-1][0, pa].float().cpu().numpy().astype(
        np.float64)
    h2a = o2.hidden_states[-1][0, pa].float().cpu().numpy().astype(
        np.float64)
    dh = float(np.abs(h1a - h2a).max())
    def s4(h):
        hn = h / np.sqrt((h ** 2).mean() + eps) * g
        sc = hn @ dW_s.T
        return sc
    sc1a, sc2a = s4(h1a), s4(h2a)
    d_arch = abs(float((sc1a[0] - sc1a[3]) - (sc2a[0] - sc2a[3])))
    l1 = o1.hidden_states[-1][0, -1].float().cpu().numpy().astype(
        np.float64)
    l2 = o2.hidden_states[-1][0, -1].float().cpu().numpy().astype(
        np.float64)
    sc1f, sc2f = s4(l1), s4(l2)
    df1 = float(sc1f[0] - sc1f[3])
    df2 = float(sc2f[0] - sc2f[3])
    delta_final = df1 - df2
    p_c6 = bool(dh == 0.0 and abs(delta_final) >= 1.0 and df1 > df2)
    print('P2802 apple-pos: max|h1-h2|=%.6f arch d(fruit-comp)=%.6f'
          % (dh, d_arch), flush=True)
    print('P2802 final pos: fruit-comp pie=%.4f released=%.4f '
          'delta=%.4f P-C6=%s' % (df1, df2, delta_final, p_c6),
          flush=True)
    print('P2802 final-pos sense scores pie: %s' % json.dumps(
        {SENSES[k]: round(float(sc1f[k]), 3) for k in range(4)}),
        flush=True)
    print('P2802 final-pos sense scores released: %s' % json.dumps(
        {SENSES[k]: round(float(sc2f[k]), 3) for k in range(4)}),
        flush=True)

    verdict = {
        'sense_channels_readable': p_a6, 'n_senses_ok': a_ok,
        'sense_spectra_separated': p_b6, 'rho_fruit_company':
            round(float(rho[0, 3]), 4),
        'rho_fruit_food': round(float(rho[0, 1]), 4),
        'disambiguation_reader_side': p_c6, 'dh_apple_pos': dh,
        'delta_final': round(delta_final, 4),
        'polysemy_spectrum_solved': bool(p_a6 and p_c6),
    }
    result = {'phase': 2802, 'prereg': PREREG, 'verdict': verdict,
              'profiles': prof_rows, 'top15': top15,
              'overlap40': ov, 'sig_frac': sig_frac,
              'census': census, 'default': default,
              'participation_ratio': {SENSES[k]: float(pr[k])
                                      for k in range(4)},
              'singular_values': [float(v) for v in sv]}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'polysemy.npz', CM=CM, dW_s=dW_s, carriers=carriers,
           rho=rho, sv=sv, votes=votes, sig=sig)
    print('P2802 VERDICT %s' % json.dumps(verdict), flush=True)


if __name__ == '__main__':
    main()
