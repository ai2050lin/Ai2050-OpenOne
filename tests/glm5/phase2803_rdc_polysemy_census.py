"""Phase 2803 (LPF-16): POLYSEMY SPECTRUM GENERALITY CENSUS.

User directive: scale up data volume and type diversity so results
are GENERAL, not special cases.  2802 ran the polysemy protocol on 1
word (apple, 4 senses).  2803 runs it on 25 target words x 14 sense
anchor sets = 52 word-sense pairs, plus a context dose-response sweep
and a layer-localization scan.

Protocol per target word w with sense list S (k senses, each an
anchor set of 10 single-token words):
  Wmeans_s = mean W_U rows of anchor set s
  dW_s = Wmeans_s - mean(other k-1 senses)
  CM_s(d) = E[w,d] * g[d] * dW_s[d] / rms_w     (exact, zero forward)
  per word: pairwise Spearman(CM_i, CM_j); SVD spectrum (rank k-1,
  flatness = min/max over nonzero sigma); participation ratio PR_s;
  default sense ranking; top-40 sets; pooled signed profile hits
  (anchor words + preregistered attr keywords).

Arms:
  A  profile readability across 52 pairs   -> P-A7
  B  contrast encoding + flat spectrum     -> P-B7
  C  PR stratification (max/min >= 2)      -> P-C7
  D  dose-response: apple company-cue sweep "the apple [+iphone/mac/
     steve/jobs/cupertino]" x4 doses; company & fruit scores at apple
     position (context-immune) and final position (reading side)
     -> P-D7
  E  layer localization: for 2802 sentence pair (pie vs released),
     per-layer lens fruit-comp difference at final position; first
     divergence layer l_div (descriptive).

Prereg (frozen before any readout):
  P-A7  profiles_general_readable iff >= 70% of the 52 (word,sense)
        pairs hit >= 1 relevant token in signed pooled top-30.
  P-B7  contrast_encoding_general iff >= 80% of words have ALL
        pairwise sense-rho < 0 AND >= 75% of words have flatness
        (min/max nonzero sigma) >= 0.4.
  P-C7  pr_stratification_general iff >= 60% of words have
        max(PR)/min(PR) >= 2 across their senses.
  P-D7  dose_response_positive iff final-position company score
        (dose3) > (dose0) AND >= 2 of 3 successive deltas > 0.
  D     descriptive: full per-word tables; l_div; apple-position
        drift across doses (immunity check).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2803' / 'qwen4_polysemy_census'
SRC_NPZ = BASE / 'phase2795' / 'qwen4_semantic_atlas' / 'atlas.npz'
SRC_2802 = BASE / 'phase2802' / 'qwen4_polysemy_spectrum' / 'polysemy.npz'
SRC_RESULT = BASE / 'phase2795' / 'qwen4_semantic_atlas' / 'result.json'

SETS = {
    'fruit': ['apple', 'banana', 'orange', 'grape', 'lemon', 'peach',
              'pear', 'mango', 'cherry', 'berry'],
    'food': ['bread', 'rice', 'cheese', 'egg', 'meat', 'soup',
             'pasta', 'pizza', 'honey', 'butter'],
    'plant': ['tree', 'flower', 'rose', 'leaf', 'root', 'grass',
              'oak', 'pine', 'maple', 'fern'],
    'company': ['Google', 'Microsoft', 'Amazon', 'Meta', 'Tesla',
                'Nvidia', 'Samsung', 'Sony', 'IBM', 'Intel'],
    'animal': ['dog', 'cat', 'horse', 'cow', 'lion', 'tiger', 'wolf',
               'rabbit', 'bird', 'fish'],
    'vehicle': ['car', 'bus', 'truck', 'train', 'ship', 'boat',
                'plane', 'bicycle', 'taxi', 'tram'],
    'metal': ['gold', 'silver', 'iron', 'copper', 'steel', 'bronze',
              'brass', 'tin', 'aluminum', 'nickel'],
    'tool': ['hammer', 'knife', 'file', 'wrench', 'drill', 'saw',
             'axe', 'nail', 'rope', 'shovel'],
    'furniture': ['chair', 'table', 'bed', 'desk', 'sofa', 'shelf',
                  'cabinet', 'bench', 'stool', 'wardrobe'],
    'clothing': ['shirt', 'pants', 'dress', 'coat', 'shoe', 'sock',
                 'hat', 'glove', 'scarf', 'jacket'],
    'color': ['red', 'blue', 'green', 'yellow', 'purple', 'pink',
              'black', 'white', 'brown', 'gray'],
    'music': ['piano', 'violin', 'guitar', 'drums', 'flute',
              'trumpet', 'opera', 'jazz', 'melody', 'rhythm'],
    'sport': ['soccer', 'tennis', 'golf', 'boxing', 'rugby', 'hockey',
              'baseball', 'cricket', 'cycling', 'skiing'],
    'computer': ['computer', 'keyboard', 'screen', 'laptop', 'server',
                 'software', 'code', 'data', 'app', 'internet'],
}
TARGETS = {
    'apple': ['fruit', 'company', 'plant', 'food'],
    'orange': ['fruit', 'color'],
    'lime': ['fruit', 'color'],
    'rose': ['plant', 'color'],
    'violet': ['plant', 'color'],
    'iron': ['metal', 'tool', 'clothing'],
    'bass': ['animal', 'music'],
    'mouse': ['animal', 'computer'],
    'python': ['animal', 'computer'],
    'bug': ['animal', 'computer'],
    'trunk': ['plant', 'animal'],
    'bark': ['animal', 'plant'],
    'corn': ['food', 'plant'],
    'turkey': ['animal', 'food'],
    'chicken': ['animal', 'food'],
    'salmon': ['animal', 'food'],
    'lobster': ['animal', 'food'],
    'pepper': ['food', 'plant'],
    'cotton': ['clothing', 'plant'],
    'wool': ['clothing', 'animal'],
    'leather': ['clothing', 'animal'],
    'olive': ['fruit', 'food'],
    'coconut': ['fruit', 'food'],
    'coach': ['vehicle', 'sport'],
    'table': ['furniture', 'computer'],
}
ATTRS = {
    'fruit': ['sweet', 'juice', 'ripe', 'orchard'],
    'food': ['eat', 'tasty', 'meal', 'flavor'],
    'plant': ['grow', 'soil', 'garden', 'leaf'],
    'company': ['iphone', 'ipad', 'mac', 'tech', 'brand'],
    'animal': ['fur', 'tail', 'wild', 'pet'],
    'vehicle': ['drive', 'road', 'wheel', 'engine'],
    'metal': ['shiny', 'ore', 'mine', 'alloy'],
    'tool': ['fix', 'build', 'hardware', 'hand'],
    'furniture': ['room', 'wood', 'home', 'seat'],
    'clothing': ['wear', 'fabric', 'fashion', 'style'],
    'color': ['shade', 'bright', 'dark', 'pigment'],
    'music': ['sound', 'song', 'melody', 'note'],
    'sport': ['game', 'team', 'athlete', 'match'],
    'computer': ['screen', 'digital', 'program', 'data'],
}
DOSES = ['the apple', 'the apple iphone', 'the apple iphone mac steve',
         'the apple iphone mac steve jobs cupertino']
SENT_FRUIT = 'the apple pie is tasty'
SENT_COMP = 'the apple released the iphone'
K_PROF = 40
SEED = 2803

PREREG = {
    'P-A7': 'profiles_general_readable iff >= 70% of 52 pairs hit '
            '>=1 relevant token in signed pooled top-30',
    'P-B7': 'contrast_encoding_general iff >= 80% words all-pairs '
            'rho<0 AND >= 75% words flatness>=0.4',
    'P-C7': 'pr_stratification_general iff >= 60% words '
            'max(PR)/min(PR) >= 2',
    'P-D7': 'dose_response_positive iff final company(dose3)>dose0 '
            'AND >=2/3 successive deltas >0',
    'verdict': 'polysemy_general_law iff P-A7 AND P-B7 AND P-C7',
}


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    data = np.load(SRC_NPZ, allow_pickle=True)
    words_atlas = [str(w) for w in data['words']]
    E30 = data['E30'].astype(np.float64)
    d02 = np.load(SRC_2802, allow_pickle=False)
    CM02 = d02['CM'].astype(np.float64)

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG, 'sets': SETS, 'targets': TARGETS,
                 'attrs': ATTRS, 'doses': DOSES, 'k_prof': K_PROF,
                 'seed': SEED}
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
    all_sets = sorted({s for v in TARGETS.values() for s in v})
    for s in all_sets:
        for w in SETS[s]:
            tid(w)
    for w in TARGETS:
        tid(w)
    Wmeans = {s: W_U[[tid(w) for w in SETS[s]]].astype(np.float64)
              .mean(0) for s in all_sets}

    # gates
    gateE = float(np.abs(Etab[tid('apple')]
                         - E30[words_atlas.index('apple')]).max())
    cat10 = ['fruit', 'animal', 'metal', 'vehicle', 'country', 'food',
             'nature', 'furniture', 'tool', 'clothing']
    Wcat10 = W_U[[tid(c) for c in cat10]]
    dW10 = Wcat10 - (Wcat10.sum(0, keepdims=True) - Wcat10) / 9.0
    Ea = E30[words_atlas.index('apple')]
    rms_a = np.sqrt((Ea ** 2).mean() + eps)
    z = (Ea / rms_a * g) @ Wcat10.T
    gateM = abs(float(z[0] - (z.sum() - z[0]) / 9.0) - mce_apple)
    dW_apple = None
    senses_a = TARGETS['apple']
    Wm = np.stack([Wmeans[s] for s in senses_a])
    dW_apple = Wm - (Wm.sum(0, keepdims=True) - Wm) / 3.0
    CM_apple = np.stack([Ea / rms_a * g * dW_apple[k]
                         for k in range(4)])
    order02 = ['fruit', 'food', 'plant', 'company']
    perm = [senses_a.index(s) for s in order02]
    gateCM = float(np.abs(CM_apple[perm] - CM02).max())
    print('P2803 gates: E=%.6f lens=%.6f CM_vs_2802=%.2e'
          % (gateE, gateM, gateCM), flush=True)
    assert gateE < 0.05 and gateM < 0.05 and gateCM < 1e-4

    dec = {}

    def dec_tok(i):
        if i not in dec:
            dec[i] = tok.decode([int(i)]).strip().lower()
        return dec[i]
    rel = {s: set(w.lower() for w in SETS[s]) | set(ATTRS[s])
           for s in all_sets}

    # ---------- census ----------
    records = []
    n_pairs = n_hit = 0
    n_allneg = n_flat = n_prstrat = 0
    for w, senses in TARGETS.items():
        k = len(senses)
        Ew = Etab[tid(w)].astype(np.float64)
        rw = np.sqrt((Ew ** 2).mean() + eps)
        Wm = np.stack([Wmeans[s] for s in senses])
        dW = Wm - (Wm.sum(0, keepdims=True) - Wm) / (k - 1)
        CM = np.stack([Ew / rw * g * dW[j] for j in range(k)])
        rho = np.array([[spearman(CM[i], CM[j]) for j in range(k)]
                        for i in range(k)])
        iu = np.triu_indices(k, 1)
        allneg = bool((rho[iu] < 0).all())
        sv = np.linalg.svd(CM, compute_uv=False)
        svz = sv[:k - 1]
        flat = float(svz.min() / max(svz.max(), 1e-9))
        pr = np.array([float((CM[j] ** 2).sum() ** 2
                             / max((CM[j] ** 4).sum(), 1e-30))
                       for j in range(k)])
        default = CM.sum(1)
        topsets = [set(np.argsort(-np.abs(CM[j]))[:K_PROF].tolist())
                   for j in range(k)]
        pair_hits = []
        for j, s in enumerate(senses):
            n_pairs += 1
            topd = np.argsort(-np.abs(CM[j]))[:K_PROF]
            p = (dW[j, topd][:, None]
                 * W_U[:, topd].T.astype(np.float64)).sum(0)
            toks = ([dec_tok(i) for i in np.argsort(-p)[:30]]
                    + [dec_tok(i) for i in np.argsort(p)[:30]])
            hits = sorted(set(t for t in toks
                              for r in rel[s] if r in t))
            ok = len(hits) >= 1
            n_hit += int(ok)
            pair_hits.append({'sense': s, 'hits': hits, 'pass': ok})
        rec = {'word': w, 'senses': senses,
               'rho': [[round(float(v), 3) for v in row]
                       for row in rho],
               'allneg': allneg, 'flatness': round(flat, 3),
               'sv': [round(float(v), 3) for v in sv],
               'pr': [round(float(v), 1) for v in pr],
               'pr_ratio': round(float(pr.max() / max(pr.min(), 1e-9)),
                                 1),
               'default': {senses[j]: round(float(default[j]), 2)
                           for j in range(k)},
               'top5': {senses[j]: [
                   {'dim': int(d), 'C': round(float(CM[j, d]), 3)}
                   for d in np.argsort(-np.abs(CM[j]))[:5]]
                   for j in range(k)},
               'profiles': pair_hits}
        records.append(rec)
        n_allneg += int(allneg)
        n_flat += int(flat >= 0.4)
        n_prstrat += int(pr.max() / max(pr.min(), 1e-9) >= 2)
        print('P2803 %-8s flat=%.2f allneg=%s pr=%s default=%s'
              % (w, flat, allneg, [round(float(v), 0) for v in pr],
                 {senses[j]: round(float(default[j]), 1)
                  for j in range(k)}), flush=True)
    n_words = len(TARGETS)
    hit_rate = n_hit / n_pairs
    p_a7 = bool(hit_rate >= 0.70)
    p_b7 = bool(n_allneg / n_words >= 0.80
                and n_flat / n_words >= 0.75)
    p_c7 = bool(n_prstrat / n_words >= 0.60)
    print('P2803 census: pairs %d/%d hit (%.2f) | allneg %d/%d | '
          'flat %d/%d | prstrat %d/%d'
          % (n_hit, n_pairs, hit_rate, n_allneg, n_words, n_flat,
             n_words, n_prstrat, n_words), flush=True)
    print('P2803 P-A7=%s P-B7=%s P-C7=%s' % (p_a7, p_b7, p_c7),
          flush=True)

    # ---------- Arm D: dose-response ----------
    dose_rows = []
    apple_pos_scores = []
    final_scores = []
    for text in DOSES:
        ids = tok(text, add_special_tokens=False)['input_ids']
        with torch.inference_mode():
            o = model(torch.tensor([ids], device=device),
                      output_hidden_states=True)
        ha = o.hidden_states[-1][0, 1].float().cpu().numpy().astype(
            np.float64)
        hf = o.hidden_states[-1][0, -1].float().cpu().numpy().astype(
            np.float64)

        def sc(h):
            hn = h / np.sqrt((h ** 2).mean() + eps) * g
            return hn @ dW_apple.T
        sa, sf = sc(ha), sc(hf)
        apple_pos_scores.append([float(v) for v in sa])
        final_scores.append([float(v) for v in sf])
        dose_rows.append({'dose': text,
                          'apple_pos': {senses_a[j]:
                                        round(float(sa[j]), 3)
                                        for j in range(4)},
                          'final': {senses_a[j]: round(float(sf[j]), 3)
                                    for j in range(4)}})
        print('P2803 dose %-46s apple company=%.3f final company=%.3f '
              'final fruit=%.3f'
              % (text, sa[1], sf[1], sf[0]), flush=True)
    comp_final = [r['final']['company'] for r in dose_rows]
    deltas = [comp_final[i + 1] - comp_final[i] for i in range(3)]
    p_d7 = bool(comp_final[3] > comp_final[0]
                and sum(d > 0 for d in deltas) >= 2)
    drift = max(abs(apple_pos_scores[i][1] - apple_pos_scores[0][1])
                for i in range(4))
    print('P2803 P-D7=%s comp_final=%s deltas=%s apple-drift=%.6f'
          % (p_d7, [round(v, 3) for v in comp_final],
             [round(d, 3) for d in deltas], drift), flush=True)

    # ---------- Arm E: layer localization ----------
    def layers_final(text):
        ids = tok(text, add_special_tokens=False)['input_ids']
        with torch.inference_mode():
            o = model(torch.tensor([ids], device=device),
                      output_hidden_states=True)
        out = []
        for l in range(37):
            h = o.hidden_states[l][0, -1].float().cpu().numpy().astype(
                np.float64)
            hn = h / np.sqrt((h ** 2).mean() + eps) * g
            sc = hn @ dW_apple.T
            out.append(float(sc[0] - sc[1]))
        return out
    curve_f = layers_final(SENT_FRUIT)
    curve_c = layers_final(SENT_COMP)
    diff = [a - b for a, b in zip(curve_f, curve_c)]
    l_div = None
    for l, v in enumerate(diff):
        if abs(v) >= 0.5:
            l_div = l
            break
    print('P2803 layer divergence (fruit-comp diff, pie minus '
          'released): l_div=%s curve=%s'
          % (l_div, [round(v, 2) for v in diff]), flush=True)

    verdict = {
        'profiles_general_readable': p_a7, 'hit_rate': round(hit_rate, 3),
        'contrast_encoding_general': p_b7,
        'allneg_rate': round(n_allneg / n_words, 3),
        'flat_rate': round(n_flat / n_words, 3),
        'pr_stratification_general': p_c7,
        'prstrat_rate': round(n_prstrat / n_words, 3),
        'dose_response_positive': p_d7,
        'comp_final': [round(v, 3) for v in comp_final],
        'apple_drift': round(drift, 6),
        'l_div': l_div,
        'polysemy_general_law': bool(p_a7 and p_b7 and p_c7),
    }
    result = {'phase': 2803, 'prereg': PREREG, 'verdict': verdict,
              'records': records, 'dose_rows': dose_rows,
              'div_curve': [round(v, 3) for v in diff]}
    fc.save(OUT / 'result.json', result)
    kmax = 4
    rho_pad = np.full((n_words, kmax, kmax), np.nan)
    sv_pad = np.full((n_words, kmax), np.nan)
    pr_pad = np.full((n_words, kmax), np.nan)
    df_pad = np.full((n_words, kmax), np.nan)
    for i, rec in enumerate(records):
        k = len(rec['senses'])
        rho_pad[i, :k, :k] = np.array(rec['rho'])
        sv_pad[i, :k] = np.array(rec['sv']) if 'sv' in rec else np.nan
    fc.npz(OUT / 'census.npz', rho=rho_pad, sv=sv_pad, pr=pr_pad,
           default=df_pad, dose_final=np.array(final_scores),
           dose_apple=np.array(apple_pos_scores),
           div_curve=np.array(diff))
    print('P2803 VERDICT %s' % json.dumps(verdict), flush=True)


if __name__ == '__main__':
    main()
