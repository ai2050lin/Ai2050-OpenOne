"""Phase 2805 (LPF-18): ENTITY-SENSE LAW LARGE-SAMPLE VERIFICATION +
DIRECTION/PROJECTION ATTRIBUTION DECOMPOSITION.

2804 found a 21/21 separation: words with a country/company entity
sense are PR-stratified, all others flat.  n=4 is too small.  2805
tests the law on the full 21-domain anchor vocabulary (210 word-sense
pairs) + the 100-word atlas as a generalization arm, and decomposes
WHERE the compactness lives:

  global contrast direction  dW~_s = Wmeans_s - mean(other 20 domains)
  projection                 CM~(w,s) = E[w]/rms_w * g * dW~_s
  participation ratio        PR(x) = (sum x^2)^2 / sum x^4

Attribution question: PR(CM~) mixes (i) direction compactness
PR(dW~_s) - how many dims the sense contrast direction itself spans -
and (ii) word-specific voting.  If entity senses are compact because
their DIRECTIONS are sparse (W_U entity rows cluster), the law is a
direction-level fact; if residuals stay entity-low after regressing
out direction compactness, it is a word-level fact.

Groups (21 domains): entity {country, company};
natural {fruit food plant animal metal anatomy};
named_cat {color music sport computer card geometry money weapon
container}; artifact {vehicle tool furniture clothing}.

Gates:
  G1 E-vs-atlas;  G2 2797 apple margin;  G3 sum_s dW~_s = 0;
  G4 within-word direction recompute of turkey matches 2804 pr
     [162.0, 85.0, 32.0] (tol 0.15).

Prereg (frozen before any readout):
  P-F1  direction level iff PR(dW~) of country and company are the
        two smallest among all 21 domains.
  P-F2  projection level iff median PR~ over entity pairs < median
        over natural pairs AND < median over named_cat pairs, each
        one-sided Mann-Whitney z < -2.33.
  P-F3  attribution: R^2 of the 21-point regression
        mean_w PR~(w,s) ~ PR(dW~_s) >= 0.5  => direction-level fact.
  verdict: entity_law_general iff P-F1 AND P-F2.
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
OUT = BASE / 'phase2805' / 'qwen4_entity_law'
SRC_NPZ = BASE / 'phase2795' / 'qwen4_semantic_atlas' / 'atlas.npz'
SRC_RESULT = BASE / 'phase2795' / 'qwen4_semantic_atlas' / 'result.json'
SRC_2804 = BASE / 'phase2804' / 'qwen4_polysemy_k3' / 'result.json'

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
GROUP = {
    'country': 'entity', 'company': 'entity',
    'fruit': 'natural', 'food': 'natural', 'plant': 'natural',
    'animal': 'natural', 'metal': 'natural', 'anatomy': 'natural',
    'color': 'named', 'music': 'named', 'sport': 'named',
    'computer': 'named', 'card': 'named', 'geometry': 'named',
    'money': 'named', 'weapon': 'named', 'container': 'named',
    'vehicle': 'artifact', 'tool': 'artifact',
    'furniture': 'artifact', 'clothing': 'artifact',
}
ATLAS_CATS = {
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
SEED = 2805

PREREG = {
    'P-F1': 'direction level iff PR(dW~) of country and company are '
            'the two smallest among all 21 domains',
    'P-F2': 'projection level iff median PR~(entity) < median '
            'PR~(natural) AND < median PR~(named), each one-sided '
            'Mann-Whitney z < -2.33',
    'P-F3': 'attribution direction-level iff R^2 of 21-point '
            'regression mean_w PR~(w,s) ~ PR(dW~_s) >= 0.5',
    'verdict': 'entity_law_general iff P-F1 AND P-F2; '
               'attribution = direction_level iff P-F3',
}


def pratio(x):
    x = np.asarray(x, dtype=np.float64)
    return float((x ** 2).sum() ** 2 / max((x ** 4).sum(), 1e-30))


def mannwhitney_lt(a, b):
    """one-sided P(a < b); returns (U_a, z, p)"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    allv = np.concatenate([a, b])
    order = np.argsort(allv)
    ranks = np.empty(len(allv), dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1, dtype=np.float64)
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    n1, n2 = len(a), len(b)
    R1 = ranks[:n1].sum()
    U1 = R1 - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    z = (U1 - mu) / sigma
    p = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return float(U1), float(z), float(p)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    data = np.load(SRC_NPZ, allow_pickle=True)
    words_atlas = [str(w) for w in data['words']]
    E30 = data['E30'].astype(np.float64)
    mce_apple = float(json.loads(
        Path(SRC_RESULT).read_text(encoding='utf-8'))['margin_cat_emb']['apple'])
    r2804 = json.loads(Path(SRC_2804).read_text(encoding='utf-8'))
    rec2804 = {r['word']: r for r in r2804['records']}

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG, 'sets': SETS, 'group': GROUP,
                 'atlas_cats': ATLAS_CATS, 'seed': SEED}
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
    for cat in ATLAS_CATS:
        for w in ATLAS_CATS[cat]:
            tid(w)
    Wmeans = {s: W_U[[tid(w) for w in SETS[s]]].astype(np.float64).mean(0)
              for s in all_sets}

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

    Wm_stack = np.stack([Wmeans[s] for s in all_sets])
    dWg = Wm_stack - (Wm_stack.sum(0, keepdims=True) - Wm_stack) / 20.0
    gateG3 = float(np.abs(dWg.sum(0)).max())

    # G4: within-word turkey recompute vs 2804
    senses_t = rec2804['turkey']['senses']
    Wt = Etab[tid('turkey')].astype(np.float64)
    rt = np.sqrt((Wt ** 2).mean() + eps)
    Wmt = np.stack([Wmeans[s] for s in senses_t])
    dWt = Wmt - (Wmt.sum(0, keepdims=True) - Wmt) / 2.0
    CMt = np.stack([Wt / rt * g * dWt[j] for j in range(3)])
    prt = [pratio(CMt[j]) for j in range(3)]
    gate_2804 = max(abs(prt[j] - rec2804['turkey']['pr'][j])
                    for j in range(3))
    print('P2805 gates: E=%.6f lens=%.6f G3sum=%.2e turkey2804=%.3f'
          % (gateE, gateM, gateG3, gate_2804), flush=True)
    assert gateE < 0.05 and gateM < 0.05 and gateG3 < 1e-8
    assert gate_2804 < 0.15, gate_2804

    dom_idx = {s: i for i, s in enumerate(all_sets)}
    pr_dir = {s: pratio(dWg[dom_idx[s]]) for s in all_sets}

    # ---------- Arm A: direction compactness by group ----------
    grp_dir = {}
    for grp in ('entity', 'natural', 'named', 'artifact'):
        grp_dir[grp] = [pr_dir[s] for s in all_sets if GROUP[s] == grp]
    ranks_dir = {s: int(sorted(pr_dir.values()).index(pr_dir[s]) + 1)
                 for s in all_sets}
    print('P2805 Arm A PR(dW~): %s'
          % json.dumps({s: round(v, 1) for s, v in
                        sorted(pr_dir.items(), key=lambda kv: kv[1])}),
          flush=True)
    p_f1 = bool(ranks_dir['country'] <= 2 and ranks_dir['company'] <= 2
                and ranks_dir['country'] != ranks_dir['company'])
    print('P2805 P-F1(direction)=%s ranks country=%d company=%d'
          % (p_f1, ranks_dir['country'], ranks_dir['company']), flush=True)

    # ---------- Arm B: projection PR~ on own-domain direction ----------
    rows = []
    for s in all_sets:
        for w in SETS[s]:
            Ew = Etab[tid(w)].astype(np.float64)
            rw = np.sqrt((Ew ** 2).mean() + eps)
            CMs = Ew / rw * g * dWg[dom_idx[s]]
            rows.append({'word': w, 'sense': s, 'group': GROUP[s],
                         'pr': pratio(CMs)})
    grp_pr = {}
    for grp in ('entity', 'natural', 'named', 'artifact'):
        grp_pr[grp] = [r['pr'] for r in rows if r['group'] == grp]
    med = {grp: float(np.median(v)) for grp, v in grp_pr.items()}
    _, z_en, p_en = mannwhitney_lt(grp_pr['entity'], grp_pr['natural'])
    _, z_ed, p_ed = mannwhitney_lt(grp_pr['entity'], grp_pr['named'])
    p_f2 = bool(med['entity'] < med['natural']
                and med['entity'] < med['named']
                and z_en < -2.33 and z_ed < -2.33)
    print('P2805 Arm B medians entity=%.1f natural=%.1f named=%.1f '
          'artifact=%.1f | z(ent<nat)=%.2f z(ent<named)=%.2f P-F2=%s'
          % (med['entity'], med['natural'], med['named'],
             med['artifact'], z_en, z_ed, p_f2), flush=True)

    # ---------- Arm C: attribution regression ----------
    pts = []
    for s in all_sets:
        xs = [r['pr'] for r in rows if r['sense'] == s]
        pts.append((pr_dir[s], float(np.mean(xs)), s))
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
    resid_entity = {}
    for i, (xd, ym, sname) in enumerate(pts):
        if GROUP[sname] == 'entity':
            resid_entity[sname] = float(y[i] - yhat[i])
    p_f3 = bool(r2 >= 0.5)
    print('P2805 Arm C R2=%.3f slope=%.3f entity_resid=%s P-F3=%s'
          % (r2, slope, {k: round(v, 1) for k, v in
                         resid_entity.items()}, p_f3), flush=True)

    # ---------- Generalization arm: atlas 100 words ----------
    gen = []
    for cat, ws in ATLAS_CATS.items():
        if cat not in dom_idx:
            continue
        grp = GROUP.get(cat, 'nature')
        for w in ws:
            Ew = E30[words_atlas.index(w)].astype(np.float64)
            rw = np.sqrt((Ew ** 2).mean() + eps)
            CMs = Ew / rw * g * dWg[dom_idx[cat]]
            gen.append({'word': w, 'cat': cat, 'group': grp,
                        'pr': pratio(CMs)})
    gen_grp = {}
    for grp in ('entity', 'natural', 'named', 'artifact', 'nature'):
        gen_grp[grp] = [r['pr'] for r in gen if r['group'] == grp]
    gen_med = {k: (round(float(np.median(v)), 1) if v else None)
               for k, v in gen_grp.items()}
    print('P2805 gen medians %s' % json.dumps(gen_med), flush=True)

    verdict = {
        'direction_level': p_f1,
        'dir_ranks': ranks_dir,
        'projection_level': p_f2,
        'medians': {k: round(v, 1) for k, v in med.items()},
        'z_entity_vs_natural': round(z_en, 2),
        'p_entity_vs_natural': round(p_en, 5),
        'z_entity_vs_named': round(z_ed, 2),
        'p_entity_vs_named': round(p_ed, 5),
        'attribution_r2': round(r2, 3),
        'direction_level_fact': p_f3,
        'entity_residual': {k: round(v, 1)
                            for k, v in resid_entity.items()},
        'gen_medians': gen_med,
        'gates': {'E_vs_atlas': round(gateE, 6),
                  'lens_vs_2797': round(gateM, 6),
                  'dWg_sum': '%.2e' % gateG3,
                  'turkey_2804_err': round(gate_2804, 3)},
        'entity_law_general': bool(p_f1 and p_f2),
    }
    result = {'phase': 2805, 'prereg': PREREG, 'verdict': verdict,
              'rows': rows, 'gen': gen,
              'pr_dir': {s: round(v, 1) for s, v in pr_dir.items()},
              'regression': {'slope': round(float(slope), 3),
                             'intercept': round(float(intercept), 1),
                             'r2': round(float(r2), 3)}}
    fc.save(OUT / 'result.json', result)
    doms = all_sets
    fc.npz(OUT / 'entity.npz',
           dWg=dWg.astype(np.float32),
           pr_dir=np.array([pr_dir[s] for s in doms]),
           pr_proj=np.array([r['pr'] for r in rows]),
           gen_pr=np.array([r['pr'] for r in gen]))
    print('P2805 VERDICT %s' % json.dumps(verdict), flush=True)


if __name__ == '__main__':
    main()
