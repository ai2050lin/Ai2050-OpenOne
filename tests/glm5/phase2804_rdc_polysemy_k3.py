"""Phase 2804 (LPF-17): K>=3 POLYSEMY STRATIP - PR stratification on
20 three-sense words, directly repairing the P-C7 design flaw.

Motivation: 2803 found the participation-ratio stratification criterion
undefined for k=2 words (dW_1 = -dW_2 antisymmetry), so P-C7 was judged
on a single word (apple).  2804 freezes a 20-word k>=3 census.

Design decisions frozen before any readout:
  - 20 target words, each exactly 3 senses drawn from 21 anchor domains
    (14 reused from 2803 + 7 new: country anatomy money weapon container
    card geometry).  All tokens prechecked single.
  - De-selfreference protocol: if a target word appears in its own sense
    anchor set, that anchor word is REMOVED before computing Wmeans_s
    (2803 contained this flaw: apple in fruit, iron in metal, olive in
    fruit, table in furniture).  Gate G3 quantifies the effect on apple.
  - Gates:
      G1 E-vs-atlas;  G2 apple CM with-selfreference reproduces 2802
      CM (|.|<1e-4, pipeline correctness);  G3 selfreference effect
      size (descriptive, reported);  G4 iron pr_ratio with-selfreference
      reproduces 2803 record 1.5 (tolerance 0.15).
  - Arms:
      A  census on 20 words x 3 senses (rho, SVD, PR, default, top5,
         pooled signed profile hits)                 -> P-A8/P-B8/P-C8
      B  named-vs-natural PR pairing: natural domains = fruit food
         plant animal metal anatomy; named = all others.  For each word
         having >=1 natural and >=1 named sense, compare PR per pair -> P-D8
      E  flatness spectrum                           -> P-E8
      C  continuity: 4 words measured in 2803 (turkey salmon olive
         trunk) - default-sense winner must remain the same after adding
         the new sense and de-selfreferencing (descriptive)

Prereg (frozen before any readout):
  P-A8  k3_profiles_readable iff >= 70% of 60 (word,sense) pairs hit
        >= 1 relevant token in signed pooled top-30.
  P-B8  k3_contrast_encoding iff >= 80% of 20 words have all-pairs
        sense-rho < 0.
  P-C8  k3_pr_stratification iff >= 60% of 20 words have
        max(PR)/min(PR) >= 2.        [direct repair of P-C7]
  P-D8  named_vs_natural_law iff >= 70% of co-occurring
        (word, named sense, natural sense) pairs have
        PR(named) < PR(natural).
  P-E8  k3_flat_spectrum iff >= 75% of words flatness >= 0.4.
  verdict: k3_generalization_law iff P-A8 AND P-B8 AND P-C8.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2804' / 'qwen4_polysemy_k3'
SRC_NPZ = BASE / 'phase2795' / 'qwen4_semantic_atlas' / 'atlas.npz'
SRC_2802 = BASE / 'phase2802' / 'qwen4_polysemy_spectrum' / 'polysemy.npz'
SRC_RESULT = BASE / 'phase2795' / 'qwen4_semantic_atlas' / 'result.json'
SRC_2803 = BASE / 'phase2803' / 'qwen4_polysemy_census' / 'result.json'

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
    'country': ['China', 'America', 'France', 'Germany', 'Japan',
                'Italy', 'Spain', 'Russia', 'Brazil', 'India'],
    'anatomy': ['heart', 'hand', 'foot', 'eye', 'ear', 'nose',
                'mouth', 'arm', 'leg', 'bone'],
    'money': ['money', 'cash', 'coin', 'dollar', 'loan', 'debt',
              'tax', 'fund', 'profit', 'salary'],
    'weapon': ['gun', 'rifle', 'pistol', 'cannon', 'bullet', 'bomb',
               'sword', 'arrow', 'spear', 'shield'],
    'container': ['box', 'bottle', 'jar', 'barrel', 'basket',
                  'bucket', 'tube', 'crate', 'kettle', 'tray'],
    'card': ['poker', 'bridge', 'king', 'queen', 'jack', 'heart',
             'deal', 'bet', 'chip', 'suit'],
    'geometry': ['square', 'circle', 'triangle', 'cube', 'sphere',
                 'cone', 'angle', 'curve', 'oval', 'prism'],
}
TARGETS = {
    'turkey': ['animal', 'food', 'country'],
    'shell': ['animal', 'company', 'computer'],
    'squash': ['food', 'sport', 'plant'],
    'mint': ['plant', 'money', 'color'],
    'gold': ['metal', 'color', 'sport'],
    'silver': ['metal', 'color', 'sport'],
    'bronze': ['metal', 'color', 'sport'],
    'salmon': ['animal', 'food', 'color'],
    'olive': ['fruit', 'food', 'color'],
    'bow': ['weapon', 'clothing', 'music'],
    'drum': ['music', 'container', 'tool'],
    'port': ['vehicle', 'computer', 'food'],
    'bench': ['furniture', 'tool', 'sport'],
    'trunk': ['plant', 'animal', 'vehicle'],
    'boot': ['clothing', 'vehicle', 'computer'],
    'horn': ['music', 'anatomy', 'vehicle'],
    'diamond': ['metal', 'card', 'geometry'],
    'club': ['weapon', 'card', 'sport'],
    'tank': ['vehicle', 'weapon', 'container'],
    'polish': ['country', 'tool', 'clothing'],
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
    'country': ['flag', 'nation', 'border', 'capital'],
    'anatomy': ['body', 'skin', 'muscle', 'skull'],
    'money': ['pay', 'price', 'rich', 'coin'],
    'weapon': ['shoot', 'war', 'fight', 'blade'],
    'container': ['lid', 'empty', 'fill', 'pour'],
    'card': ['poker', 'deal', 'bet', 'flush'],
    'geometry': ['shape', 'angle', 'flat', 'round'],
}
NATURAL = {'fruit', 'food', 'plant', 'animal', 'metal', 'anatomy'}
NAMED = set(SETS) - NATURAL
K_PROF = 40
SEED = 2804
SELFREF_2803 = {'apple': 'fruit', 'iron': 'metal', 'olive': 'fruit',
                'table': 'furniture'}

PREREG = {
    'P-A8': 'k3_profiles_readable iff >= 70% of 60 (word,sense) pairs '
            'hit >=1 relevant token in signed pooled top-30',
    'P-B8': 'k3_contrast_encoding iff >= 80% of 20 words all-pairs '
            'rho < 0',
    'P-C8': 'k3_pr_stratification iff >= 60% of 20 words '
            'max(PR)/min(PR) >= 2  [direct repair of P-C7]',
    'P-D8': 'named_vs_natural_law iff >= 70% of co-occurring '
            '(word, named, natural) pairs have PR(named) < PR(natural)',
    'P-E8': 'k3_flat_spectrum iff >= 75% of words flatness >= 0.4',
    'verdict': 'k3_generalization_law iff P-A8 AND P-B8 AND P-C8',
    'design': 'de-selfreference protocol; gates G1-G4; prereg frozen '
              'after token precheck, before any embedding readout',
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
    src2803 = json.loads(Path(SRC_2803).read_text(encoding='utf-8'))
    rec2803 = {r['word']: r for r in src2803['records']}

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG, 'sets': SETS, 'targets': TARGETS,
                 'attrs': ATTRS, 'natural': sorted(NATURAL),
                 'named': sorted(NAMED), 'k_prof': K_PROF, 'seed': SEED,
                 'selfref_2803': SELFREF_2803}
    fc.save(OUT / 'execution.json', execution)

    mce_apple = float(json.loads(
        Path(SRC_RESULT).read_text(encoding='utf-8'))['margin_cat_emb']['apple'])

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

    all_sets = sorted(SETS)
    for s in all_sets:
        for w in SETS[s]:
            tid(w)
    for w in TARGETS:
        tid(w)
    tid('apple')
    tid('iron')
    Wmeans = {s: W_U[[tid(w) for w in SETS[s]]].astype(np.float64).mean(0)
              for s in all_sets}

    def wmean_clean(s, drop):
        keep = [w for w in SETS[s] if w.lower() != drop.lower()]
        assert len(keep) == len(SETS[s]) - 1, (s, drop)
        return W_U[[tid(w) for w in keep]].astype(np.float64).mean(0)

    # ---------- gates ----------
    gateE = float(np.abs(Etab[tid('apple')]
                         - E30[words_atlas.index('apple')]).max())
    cat10 = ['fruit', 'animal', 'metal', 'vehicle', 'country', 'food',
             'nature', 'furniture', 'tool', 'clothing']
    Wcat10 = W_U[[tid(c) for c in cat10]]
    Ea = E30[words_atlas.index('apple')]
    rms_a = np.sqrt((Ea ** 2).mean() + eps)
    z = (Ea / rms_a * g) @ Wcat10.T
    gateM = abs(float(z[0] - (z.sum() - z[0]) / 9.0) - mce_apple)

    # G2: with-selfreference apple CM (2803 protocol) must reproduce 2802
    senses_a = ['fruit', 'food', 'plant', 'company']
    Wm_with = np.stack([Wmeans[s] for s in senses_a])
    dW_with = Wm_with - (Wm_with.sum(0, keepdims=True) - Wm_with) / 3.0
    CM_with = np.stack([Ea / rms_a * g * dW_with[k] for k in range(4)])
    gateCM02 = float(np.abs(CM_with - CM02).max())

    # G3: de-selfreference effect on apple (fruit anchor drops 'apple')
    Wm_clean = np.stack([wmean_clean(s, 'apple') if s == 'fruit'
                         else Wmeans[s] for s in senses_a])
    dW_clean = Wm_clean - (Wm_clean.sum(0, keepdims=True) - Wm_clean) / 3.0
    CM_clean = np.stack([Ea / rms_a * g * dW_clean[k] for k in range(4)])
    gate_selfref = float(np.abs(CM_clean - CM_with).max())

    # G4: iron pr_ratio with-selfreference must reproduce 2803 (1.5)
    senses_i = rec2803['iron']['senses']
    Wi = Etab[tid('iron')].astype(np.float64)
    ri = np.sqrt((Wi ** 2).mean() + eps)
    Wmi = np.stack([Wmeans[s] for s in senses_i])
    dWi = Wmi - (Wmi.sum(0, keepdims=True) - Wmi) / 2.0
    CMi = np.stack([Wi / ri * g * dWi[j] for j in range(3)])
    pri = np.array([float((CMi[j] ** 2).sum() ** 2
                          / max((CMi[j] ** 4).sum(), 1e-30))
                    for j in range(3)])
    gate_iron = abs(float(pri.max() / max(pri.min(), 1e-9))
                    - rec2803['iron']['pr_ratio'])

    print('P2804 gates: E=%.6f lens=%.6f CM_vs_2802=%.2e '
          'selfref=%.4f iron_pr=%.3f'
          % (gateE, gateM, gateCM02, gate_selfref, gate_iron), flush=True)
    assert gateE < 0.05 and gateM < 0.05 and gateCM02 < 1e-4
    assert gate_iron < 0.15, gate_iron

    dec = {}

    def dec_tok(i):
        if i not in dec:
            dec[i] = tok.decode([int(i)]).strip().lower()
        return dec[i]
    rel = {s: set(w.lower() for w in SETS[s]) | set(ATTRS[s])
           for s in all_sets}

    # ---------- Arm A: census (de-selfreferenced) ----------
    records = []
    n_pairs = n_hit = 0
    n_allneg = n_flat = n_prstrat = 0
    for w, senses in TARGETS.items():
        k = len(senses)
        Ew = Etab[tid(w)].astype(np.float64)
        rw = np.sqrt((Ew ** 2).mean() + eps)
        Wm = np.stack([wmean_clean(s, w) if any(
            a.lower() == w.lower() for a in SETS[s]) else Wmeans[s]
            for s in senses])
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
            pair_hits.append({'sense': s, 'hits': hits[:6],
                              'n_hits': len(hits), 'pass': ok})
        rec = {'word': w, 'senses': senses,
               'rho': [[round(float(v), 3) for v in row]
                       for row in rho],
               'allneg': allneg, 'flatness': round(flat, 3),
               'sv': [round(float(v), 3) for v in sv],
               'pr': [round(float(v), 1) for v in pr],
               'pr_ratio': round(float(pr.max() / max(pr.min(), 1e-9)), 1),
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
        print('P2804 %-8s flat=%.2f allneg=%s pr=%s ratio=%.1f '
              'default=%s'
              % (w, flat, allneg, [round(float(v), 0) for v in pr],
                 pr.max() / max(pr.min(), 1e-9),
                 {senses[j]: round(float(default[j]), 1)
                  for j in range(k)}), flush=True)
    n_words = len(TARGETS)
    hit_rate = n_hit / n_pairs
    p_a8 = bool(hit_rate >= 0.70)
    p_b8 = bool(n_allneg / n_words >= 0.80)
    p_c8 = bool(n_prstrat / n_words >= 0.60)
    p_e8 = bool(n_flat / n_words >= 0.75)
    print('P2804 census: pairs %d/%d (%.2f) | allneg %d/%d | '
          'flat %d/%d | prstrat %d/%d'
          % (n_hit, n_pairs, hit_rate, n_allneg, n_words, n_flat,
             n_words, n_prstrat, n_words), flush=True)

    # ---------- Arm B: named vs natural PR pairing ----------
    pairs = []
    for rec in records:
        pr_by = dict(zip(rec['senses'], rec['pr']))
        nat = [s for s in rec['senses'] if s in NATURAL]
        nam = [s for s in rec['senses'] if s in NAMED]
        for s1 in nam:
            for s2 in nat:
                pairs.append({'word': rec['word'], 'named': s1,
                              'natural': s2,
                              'pr_named': pr_by[s1],
                              'pr_natural': pr_by[s2],
                              'named_tighter': bool(
                                  pr_by[s1] < pr_by[s2])})
    n_nn = len(pairs)
    n_tight = sum(1 for p in pairs if p['named_tighter'])
    p_d8 = bool(n_tight / max(n_nn, 1) >= 0.70)
    print('P2804 named-vs-natural: %d/%d tighter (%.2f) P-D8=%s'
          % (n_tight, n_nn, n_tight / max(n_nn, 1), p_d8), flush=True)

    # ---------- Arm C: continuity with 2803 (4 re-measured words) ----------
    cont = []
    for w in ['turkey', 'salmon', 'olive', 'trunk']:
        old = rec2803[w]['default']
        new = dict((s, v) for s, v in
                   next(r for r in records if r['word'] == w)
                   ['default'].items() if s in old)
        same = all(sorted(old, key=old.get, reverse=True)[0]
                   == sorted(new, key=new.get, reverse=True)[0]
                   for _ in [0]) and \
            sorted(old, key=old.get, reverse=True)[0] in new and \
            sorted(new, key=new.get, reverse=True)[0] == \
            sorted(old, key=old.get, reverse=True)[0]
        cont.append({'word': w, 'default_2803': old,
                     'default_2804_shared_senses': new,
                     'winner_stable': bool(same)})
        print('P2804 continuity %-7s winner_2803=%s winner_2804=%s'
              % (w, sorted(old, key=old.get, reverse=True)[0],
                 sorted(new, key=new.get, reverse=True)[0]), flush=True)

    verdict = {
        'k3_profiles_readable': p_a8,
        'hit_rate': round(hit_rate, 3),
        'k3_contrast_encoding': p_b8,
        'allneg_rate': round(n_allneg / n_words, 3),
        'k3_pr_stratification': p_c8,
        'prstrat_rate': round(n_prstrat / n_words, 3),
        'named_vs_natural_law': p_d8,
        'named_tighter_rate': round(n_tight / max(n_nn, 1), 3),
        'k3_flat_spectrum': p_e8,
        'flat_rate': round(n_flat / n_words, 3),
        'gates': {'E_vs_atlas': round(gateE, 6),
                  'lens_vs_2797': round(gateM, 6),
                  'CM_vs_2802': '%.2e' % gateCM02,
                  'selfref_effect_apple': round(gate_selfref, 4),
                  'iron_pr_ratio_err': round(gate_iron, 3)},
        'k3_generalization_law': bool(p_a8 and p_b8 and p_c8),
    }
    result = {'phase': 2804, 'prereg': PREREG, 'verdict': verdict,
              'records': records, 'named_natural_pairs': pairs,
              'continuity': cont}
    fc.save(OUT / 'result.json', result)

    n_w = len(records)
    kmax = 3
    rho_pad = np.full((n_w, kmax, kmax), np.nan)
    sv_pad = np.full((n_w, kmax), np.nan)
    pr_pad = np.full((n_w, kmax), np.nan)
    df_pad = np.full((n_w, kmax), np.nan)
    for i, rec in enumerate(records):
        k = len(rec['senses'])
        rho_pad[i, :k, :k] = np.array(rec['rho'])
        sv_pad[i, :k] = np.array(rec['sv'][:k])
        pr_pad[i, :k] = np.array(rec['pr'][:k])
        df_pad[i, :k] = np.array([rec['default'][s]
                                  for s in rec['senses']])
    fc.npz(OUT / 'k3.npz', rho=rho_pad, sv=sv_pad, pr=pr_pad,
           default=df_pad)
    print('P2804 VERDICT %s' % json.dumps(verdict), flush=True)


if __name__ == '__main__':
    main()
