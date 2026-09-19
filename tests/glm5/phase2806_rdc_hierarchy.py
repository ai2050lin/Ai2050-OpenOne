"""Phase 2806 (LPF-19): MULTI-LEVEL DEFINITION STRUCTURE OF THE
NOUN EMBEDDING SPACE (user anchor).

User goal: with a large set of objects, recover how the embedding
encodes RELATIVE DEFINITIONS AT MULTIPLE LEVELS - apple/banana vs
fruit-food-plant-object; tiger/lion vs apple; car/plane vs apple -
culminating in how ALL nouns distribute and differentiate overall,
and how category levels are laid out, so that types and attributes
are expressed EXTREMELY EFFICIENTLY.

All zero-forward, E-side + W_U-side, on the 100-word/10-class atlas.

Levels:
  L1 class directions   dW_c = centroid_c - mean(other 9 centroids)
  L2 domain directions  dW_nat_art (natural incl metal vs artifact),
                        dW_liv_non (living fruit/food/animal vs rest)
  L3 individual diffs   dE = E[w1] - E[w2]

Arms:
  A  nesting: domain directions recovered by the 10-dim class
     direction subspace (projection retention) + 12x12 direction
     cosine matrix (10 class + 2 domain)          -> P-G1
  B  centroid tree: average-linkage clustering of 10 class
     centroids; cophenetic vs preregistered semantic distance
     (same class 1 / same domain 2 / cross domain 3)  -> P-G2
  C  hierarchical variance split of score matrix S[w,c] =
     z(w)@centroid_c: column variance into domain share vs
     class-within-domain share                     -> P-G3
  D  efficiency: k class-direction features (k=1..10) + 2 domain
     features -> nearest-centroid 10-class accuracy vs full
     2560-dim baseline                              -> P-G4
  E  user example pairs: dE energy share on domain directions /
     class subspace / residual for same-class (apple-banana),
     cross-class-same-domain (apple-tiger), cross-domain
     (apple-car, apple-hammer), cross-realm (apple-japan) -> P-G5

Gates: E-vs-atlas; 2797 apple margin; dW_c column sums = 0.

Prereg (frozen before any readout):
  P-G1  nesting iff both domain directions have class-subspace
        projection retention >= 0.90.
  P-G2  tree consistency iff Spearman(cophenetic, semantic) >= 0.5.
  P-G3  hierarchical layout iff domain share of column variance
        >= 0.30 AND class-within share >= 0.30.
  P-G4  efficiency iff 10-dim class-direction accuracy >= 0.85 x
        full-dim accuracy.
  P-G5  level grading iff mean domain-direction energy share of
        cross-domain pairs >= 3 x that of same-class pairs.
  verdict: hierarchy_law iff P-G1 AND P-G2 AND P-G3 AND P-G4.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2806' / 'qwen4_hierarchy'
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
CAT_OF = {w: c for c, v in CATS.items() for w in v}
CAT_WORDS = list(CATS.keys())
DOMAIN_OF = {'fruit': 'natural', 'food': 'natural', 'animal': 'natural',
             'nature': 'natural', 'metal': 'natural',
             'vehicle': 'artifact', 'tool': 'artifact',
             'furniture': 'artifact', 'clothing': 'artifact',
             'country': 'abstract'}
NAT_DOM = ['fruit', 'food', 'animal', 'nature', 'metal']
ART_DOM = ['vehicle', 'tool', 'furniture', 'clothing']
LIVING = ['fruit', 'food', 'animal']
PAIRS = [('apple', 'banana', 'same_class'),
         ('tiger', 'lion', 'same_class'),
         ('car', 'bus', 'same_class'),
         ('apple', 'bread', 'cross_class_same_domain'),
         ('apple', 'tiger', 'cross_class_same_domain'),
         ('apple', 'car', 'cross_domain'),
         ('apple', 'hammer', 'cross_domain'),
         ('apple', 'japan', 'cross_realm'),
         ('car', 'tiger', 'cross_domain')]
SEED = 2806

PREREG = {
    'P-G1': 'nesting iff both domain directions have class-subspace '
            'projection retention >= 0.90',
    'P-G2': 'tree consistency iff Spearman(cophenetic, semantic '
            'distance 1/2/3) >= 0.5',
    'P-G3': 'hierarchical layout iff domain share of column variance '
            '>= 0.30 AND class-within share >= 0.30',
    'P-G4': 'efficiency iff 10-dim class-direction accuracy >= '
            '0.85 x full-dim nearest-centroid accuracy',
    'P-G5': 'level grading iff mean domain-direction energy share '
            '(cross-domain pairs) >= 3 x (same-class pairs)',
    'verdict': 'hierarchy_law iff P-G1 AND P-G2 AND P-G3 AND P-G4',
}


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def condensed_links(merges):
    """cophenetic distances from average-linkage merge list."""
    n = len(merges) + 1
    members = {i: [i] for i in range(n)}
    cop = {}
    for h, (a, b) in enumerate(merges):
        ma, mb = members[a], members[b]
        d = float(h + 1)
        for i in ma:
            for j in mb:
                cop[(min(i, j), max(i, j))] = d
        members[n + h] = ma + mb
    return cop


def average_linkage(D):
    """D: (n,n) distance matrix -> merge list [(i,j), ...]."""
    n = D.shape[0]
    d = D.astype(np.float64).copy()
    active = list(range(n))
    size = {i: 1 for i in range(n)}
    merges = []
    while len(active) > 1:
        best = None
        for ii in range(len(active)):
            for jj in range(ii + 1, len(active)):
                a, b = active[ii], active[jj]
                if best is None or d[a, b] < best[0]:
                    best = (float(d[a, b]), a, b)
        _, a, b = best
        merges.append((a, b))
        m = d.shape[0]
        newrow = np.zeros(m + 1)
        for c in active:
            if c not in (a, b):
                newrow[c] = (d[a, c] * size[a] + d[b, c] * size[b]) \
                    / (size[a] + size[b])
        d2 = np.zeros((m + 1, m + 1))
        d2[:m, :m] = d
        d2[m, :m] = newrow[:m]
        d2[:m, m] = newrow[:m]
        d = d2
        size[m] = size[a] + size[b]
        active = [c for c in active if c not in (a, b)] + [m]
    return merges


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    data = np.load(SRC_NPZ, allow_pickle=True)
    words_atlas = [str(w) for w in data['words']]
    E30 = data['E30'].astype(np.float64)
    mce_apple = float(json.loads(
        Path(SRC_RESULT).read_text(encoding='utf-8'))['margin_cat_emb']['apple'])

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG, 'cats': CATS,
                 'domain_of': DOMAIN_OF, 'nat_dom': NAT_DOM,
                 'art_dom': ART_DOM, 'living': LIVING, 'pairs': PAIRS,
                 'seed': SEED}
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

    for c in CAT_WORDS:
        tid(c)
    for v in CATS.values():
        for w in v:
            tid(w)

    # ---------- gates ----------
    gateE = float(np.abs(Etab[tid('apple')]
                         - E30[words_atlas.index('apple')]).max())
    cat10 = CAT_WORDS
    Wcat10 = W_U[[tid(c) for c in cat10]]
    Ea = E30[words_atlas.index('apple')]
    rms_a = np.sqrt((Ea ** 2).mean() + eps)
    z = (Ea / rms_a * g) @ Wcat10.T
    gateM = abs(float(z[0] - (z.sum() - z[0]) / 9.0) - mce_apple)
    print('P2806 gates: E=%.6f lens=%.6f' % (gateE, gateM), flush=True)
    assert gateE < 0.05 and gateM < 0.05

    # ---------- centroids and level directions ----------
    Erows = {w: Etab[tid(w)].astype(np.float64) for v in CATS.values()
             for w in v}
    cent = {c: np.stack([W_U[tid(w)].astype(np.float64)
                         for w in CATS[c]]).mean(0) for c in CAT_WORDS}
    Cm = np.stack([cent[c] for c in CAT_WORDS])
    dW_class = Cm - (Cm.sum(0, keepdims=True) - Cm) / 9.0
    gateG0 = float(np.abs(dW_class.sum(0)).max())
    assert gateG0 < 1e-8, gateG0

    nat_mean = np.stack([cent[c] for c in NAT_DOM]).mean(0)
    art_mean = np.stack([cent[c] for c in ART_DOM]).mean(0)
    dW_nat_art = nat_mean - art_mean
    liv_mean = np.stack([cent[c] for c in LIVING]).mean(0)
    rest = [c for c in CAT_WORDS if c not in LIVING]
    non_mean = np.stack([cent[c] for c in rest]).mean(0)
    dW_liv_non = liv_mean - non_mean
    dom_dir = {'nat_art': dW_nat_art, 'liv_non': dW_liv_non}

    def unit(x):
        return x / max(np.linalg.norm(x), 1e-30)

    # ---------- Arm A: nesting ----------
    Vc = dW_class.T  # 2560 x 10
    Qc, _ = np.linalg.qr(Vc)
    nest = {}
    for k, dv in dom_dir.items():
        proj = Qc @ (Qc.T @ dv)
        nest[k] = float(np.linalg.norm(proj) / np.linalg.norm(dv))
    cosmat = {}
    dirs = [('class:' + c, dW_class[i]) for i, c in enumerate(CAT_WORDS)] \
        + [('dom:' + k, v) for k, v in dom_dir.items()]
    for (n1, v1) in dirs:
        cosmat[n1] = {}
        for (n2, v2) in dirs:
            cosmat[n1][n2] = round(
                float(unit(v1) @ unit(v2)), 3)
    p_g1 = bool(all(v >= 0.90 for v in nest.values()))
    print('P2806 Arm A nesting retention %s P-G1=%s'
          % ({k: round(v, 3) for k, v in nest.items()}, p_g1), flush=True)

    # ---------- Arm B: centroid tree ----------
    Dm = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            Dm[i, j] = np.linalg.norm(unit(cent[CAT_WORDS[i]])
                                      - unit(cent[CAT_WORDS[j]]))
    merges = average_linkage(Dm)
    cop = condensed_links(merges)
    sem = {}
    for i in range(10):
        for j in range(i + 1, 10):
            ci, cj = CAT_WORDS[i], CAT_WORDS[j]
            if ci == cj:
                d = 1
            elif DOMAIN_OF[ci] == DOMAIN_OF[cj]:
                d = 2
            else:
                d = 3
            sem[(min(i, j), max(i, j))] = d
    keys = sorted(sem.keys())
    rho_tree = spearman([sem[k] for k in keys],
                        [cop[k] for k in keys])
    p_g2 = bool(rho_tree >= 0.5)

    def cname(i):
        return CAT_WORDS[i] if i < len(CAT_WORDS) else 'cluster%d' % i
    print('P2806 Arm B merges %s rho_tree=%.3f P-G2=%s'
          % ([(cname(a), cname(b)) for a, b in merges],
             rho_tree, p_g2), flush=True)

    # ---------- Arm C: hierarchical variance split ----------
    Zn = np.stack([E30[words_atlas.index(w)] /
                   np.sqrt((E30[words_atlas.index(w)] ** 2).mean() + eps)
                   * g for w in words_atlas])
    CentM = np.stack([cent[c] for c in CAT_WORDS])
    S = Zn @ CentM.T  # 100 x 10
    colmean = S.mean(0)
    colvar = float(colmean.var())
    dommean = {}
    for c in CAT_WORDS:
        dommean.setdefault(DOMAIN_OF[c], []).append(colmean[CAT_WORDS.index(c)])
    dmean = {k: float(np.mean(v)) for k, v in dommean.items()}
    gm = float(colmean.mean())
    ss_dom = float(sum(len(v) * (dmean[k] - gm) ** 2
                       for k, v in dommean.items()))
    ss_cls = float(sum((colmean[CAT_WORDS.index(c)] - dmean[DOMAIN_OF[c]]) ** 2
                       for c in CAT_WORDS))
    share_dom = ss_dom / max(ss_dom + ss_cls, 1e-30)
    share_cls = 1.0 - share_dom
    p_g3 = bool(share_dom >= 0.30 and share_cls >= 0.30)
    print('P2806 Arm C col-var share dom=%.3f class=%.3f P-G3=%s'
          % (share_dom, share_cls, p_g3), flush=True)

    # ---------- Arm D: efficiency ----------
    Ztr = Zn
    ytrue = np.array([CAT_WORDS.index(CAT_OF[w]) for w in words_atlas])
    feats_full = Zn @ np.stack([unit(cent[c]) for c in CAT_WORDS]).T
    # nearest centroid classification in various feature spaces
    def nca(Xf, Ccent_feats):
        pred = []
        for i in range(len(Xf)):
            dd = [np.linalg.norm(Xf[i] - cc_) for cc_ in Ccent_feats]
            pred.append(int(np.argmin(dd)))
        return float(np.mean(np.array(pred) == ytrue))

    class_feats_full = {c: np.stack(
        [feats_full[words_atlas.index(w)]
         for w in CATS[c]]).mean(0) for c in CAT_WORDS}
    acc_full = nca(feats_full, list(class_feats_full.values()))
    accs = {}
    Qlist = [unit(dW_class[i]) for i in range(10)]
    for k in range(1, 11):
        Qk = np.stack(Qlist[:k])
        Fk = Zn @ Qk.T
        cfk = {c: np.stack([Fk[words_atlas.index(w)]
                            for w in CATS[c]]).mean(0)
               for c in CAT_WORDS}
        accs[k] = nca(Fk, list(cfk.values()))
    Qdom = np.stack([unit(dW_nat_art), unit(dW_liv_non)])
    Fd = Zn @ Qdom.T
    cfd = {c: np.stack([Fd[words_atlas.index(w)]
                        for w in CATS[c]]).mean(0) for c in CAT_WORDS}
    acc_dom2 = nca(Fd, list(cfd.values()))
    p_g4 = bool(accs[10] >= 0.85 * acc_full)
    print('P2806 Arm D acc_full=%.3f accs=%s dom2=%.3f P-G4=%s'
          % (acc_full, {k: round(v, 3) for k, v in accs.items()},
             acc_dom2, p_g4), flush=True)

    # ---------- Arm E: user example pairs decomposition ----------
    Vd = np.stack([unit(dW_nat_art), unit(dW_liv_non)])
    Qall = np.concatenate([Qc, Vd.T], axis=1)
    Qa, _ = np.linalg.qr(Qall)
    pair_rows = []
    dom_shares = {'same_class': [], 'cross_class_same_domain': [],
                  'cross_domain': [], 'cross_realm': []}
    for w1, w2, tag in PAIRS:
        E1 = Etab[tid(w1)].astype(np.float64)
        E2 = Etab[tid(w2)].astype(np.float64)
        dE = E1 - E2
        e_dom = float(sum((unit(v) @ dE) ** 2 for v in Vd))
        e_cls = float(sum((q @ dE) ** 2 for q in Qlist))
        e_tot = float(dE @ dE)
        e_res = e_tot - e_cls
        row = {'pair': '%s-%s' % (w1, w2), 'tag': tag,
               'norm': round(np.linalg.norm(dE), 1),
               'dom_energy_share': round(e_dom / max(e_tot, 1e-30), 3),
               'class_energy_share': round(e_cls / max(e_tot, 1e-30), 3),
               'residual_share': round(e_res / max(e_tot, 1e-30), 3),
               'dom_proj_per_unit': round(e_dom / max(e_tot, 1e-30), 3)}
        pair_rows.append(row)
        dom_shares[tag].append(e_dom / max(e_tot, 1e-30))
        print('P2806 Arm E %-16s %-24s dom=%.3f cls=%.3f res=%.3f'
              % (row['pair'], tag, row['dom_energy_share'],
                 row['class_energy_share'], row['residual_share']),
              flush=True)
    m_same = float(np.mean(dom_shares['same_class']))
    m_xdom = float(np.mean(dom_shares['cross_domain']))
    p_g5 = bool(m_xdom >= 3.0 * m_same)
    print('P2806 P-G5 mean dom-share same=%.4f cross-domain=%.4f '
          'ratio=%.1f P-G5=%s'
          % (m_same, m_xdom, m_xdom / max(m_same, 1e-30), p_g5),
          flush=True)

    verdict = {
        'nesting_retention': {k: round(v, 3) for k, v in nest.items()},
        'hierarchy_nesting': p_g1,
        'tree_rho_semantic': round(rho_tree, 3),
        'tree_consistent': p_g2,
        'col_var_domain_share': round(share_dom, 3),
        'col_var_class_share': round(share_cls, 3),
        'hierarchical_layout': p_g3,
        'acc_full': round(acc_full, 3),
        'acc_k': {str(k): round(v, 3) for k, v in accs.items()},
        'acc_2dom': round(acc_dom2, 3),
        'efficiency': p_g4,
        'dom_share_same_class': round(m_same, 4),
        'dom_share_cross_domain': round(m_xdom, 4),
        'level_grading': p_g5,
        'gates': {'E_vs_atlas': round(gateE, 6),
                  'lens_vs_2797': round(gateM, 6),
                  'dW_class_sum': '%.2e' % gateG0},
        'hierarchy_law': bool(p_g1 and p_g2 and p_g3 and p_g4),
    }
    result = {'phase': 2806, 'prereg': PREREG, 'verdict': verdict,
              'pair_rows': pair_rows,
              'cos_matrix': cosmat,
              'merges': [[cname(a), cname(b)] for a, b in merges]}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'hierarchy.npz',
           dW_class=dW_class.astype(np.float32),
           dW_dom=np.stack([dW_nat_art, dW_liv_non]).astype(np.float32),
           Dm=Dm.astype(np.float32),
           S=S.astype(np.float32))
    print('P2806 VERDICT %s' % json.dumps(verdict), flush=True)


if __name__ == '__main__':
    main()
