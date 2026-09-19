"""Phase 2807 (LPF-20): HELD-OUT GENERALIZATION OF THE HIERARCHY LAW.

User anchor: continue -> 2806 candidate (a): held-out generalization.
2806 found the class hierarchy compressed into 10/2560 class directions
(0.39%) with in-atlas nearest-centroid accuracy 1.000 -- but directions
were built from the same 100 words used for evaluation (self-consistency
loop).  2807 removes the loop: a NEW 99-word / 10-class held-out atlas
(disjoint from the 2795 words) is classified by templates built ONLY
from the 2806 atlas words.

All zero-forward, E-side + W_U-side (clone of the 2806 protocol).

Arms:
  HI   held-out nearest-centroid accuracy, full 2560-dim features -> P-I1
  HII  held-out accuracy with k class-direction features, k=1..10,
       plus 2 domain directions                                  -> P-I2
  HIII held-out pair energy decomposition dom/cls/res on
       held-out pairs (plum-fig, whale-dolphin, couch-rug,
       plum-tractor, whale-bolt, couch-tiger, plum-Norway)       -> P-I3

Prereg (frozen before any readout):
  P-I1  generalization iff held-out full-dim accuracy >= 0.80 (chance 0.10)
  P-I2  efficiency generalizes iff 10-dim held-out accuracy >=
        0.85 x full-dim held-out accuracy
  P-I3  level grading generalizes iff mean dom-energy share of held-out
        cross-domain pairs >= 3 x that of held-out same-class pairs
  verdict: generalization_law iff P-I1 AND P-I2

Gates:
  E_vs_atlas   |Etab[apple] - atlas E30[apple]| < 0.05
  lens_vs_2797 margin clone vs 2795 margin_cat_emb[apple] < 0.05
  dW_vs_2806   recomputed f64 dW_class vs 2806 f32 npz < 1e-3
  overlap      held-out words disjoint from 2795 atlas words
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2807' / 'qwen4_heldout'
SRC_ATLAS = BASE / 'phase2795' / 'qwen4_semantic_atlas' / 'atlas.npz'
SRC_ATLAS_RES = BASE / 'phase2795' / 'qwen4_semantic_atlas' / 'result.json'
SRC_2806 = BASE / 'phase2806' / 'qwen4_hierarchy' / 'hierarchy.npz'

CAT_WORDS = ['fruit', 'animal', 'metal', 'vehicle', 'country', 'food',
             'nature', 'furniture', 'tool', 'clothing']
DOMAIN_OF = {'fruit': 'natural', 'food': 'natural', 'animal': 'natural',
             'nature': 'natural', 'metal': 'natural',
             'vehicle': 'artifact', 'tool': 'artifact',
             'furniture': 'artifact', 'clothing': 'artifact',
             'country': 'abstract'}
NAT_DOM = ['fruit', 'food', 'animal', 'nature', 'metal']
ART_DOM = ['vehicle', 'tool', 'furniture', 'clothing']
LIVING = ['fruit', 'food', 'animal']

HELD = {
    'fruit': ['plum', 'fig', 'coconut', 'olive', 'prune', 'melon',
              'date', 'almond', 'lime'],
    'animal': ['deer', 'bear', 'monkey', 'elephant', 'dolphin', 'whale',
               'snake', 'frog', 'duck', 'goat'],
    'metal': ['lead', 'zinc', 'platinum', 'titanium', 'chrome',
              'magnesium', 'calcium', 'lithium', 'sodium', 'mercury'],
    'vehicle': ['scooter', 'tractor', 'subway', 'ferry', 'canoe',
                'kayak', 'trailer', 'wagon', 'sedan', 'raft'],
    'country': ['Norway', 'Sweden', 'Finland', 'Poland', 'Portugal',
                'Greece', 'Mexico', 'Kenya', 'Chile', 'Peru'],
    'food': ['tofu', 'sausage', 'bacon', 'ham', 'salad', 'curry', 'pie',
             'cake', 'candy', 'stew'],
    'nature': ['beach', 'island', 'valley', 'storm', 'thunder', 'fog',
               'frost', 'star', 'moon', 'cave'],
    'furniture': ['couch', 'drawer', 'mattress', 'pillow', 'rug',
                  'dresser', 'cot', 'crib', 'buffet', 'mat'],
    'tool': ['router', 'grinder', 'mop', 'vice', 'screw', 'bolt', 'nut',
             'clamp', 'wedge', 'punch'],
    'clothing': ['vest', 'jeans', 'skirt', 'blouse', 'sweater', 'robe',
                 'uniform', 'boot', 'cap', 'gown'],
}
HELD_REJECTED = {
    'fruit': ['apricot 3tok', 'papaya 3tok', 'guava 2tok', 'lychee 3tok',
              'raisin 2tok', 'kiwi 2tok', 'quince 2tok', 'durian 2tok',
              'pomelo 3tok', 'pawpaw 4tok'],
    'metal': ['tungsten 3tok', 'cobalt 2tok', 'radium 2tok', 'barium 2tok',
              'cesium 2tok', 'osmium 2tok', 'iridium 3tok', 'rhodium 2tok',
              'gallium 3tok', 'arsenic 2tok', 'indium 2tok'],
    'vehicle': ['limo 2tok', 'sleigh 2tok', 'glider 2tok'],
    'food': ['noodle 2tok'],
    'nature': ['meadow 2tok'],
    'furniture': ['cradle 2tok', 'bookcase 2tok', 'ottoman 2tok',
                  'hutch 2tok'],
    'tool': ['chisel 2tok', 'lathe 2tok', 'sander 2tok', 'spanner 2tok',
             'crowbar 2tok', 'anvil 2tok', 'hatchet 2tok', 'trowel 3tok',
             'auger 2tok', 'mallet 2tok', 'adze 2tok'],
    'clothing': ['apron 2tok', 'scarf in-atlas', 'kimono 2tok',
                 'sneaker 3tok'],
}
HELD_OF = {w: c for c, v in HELD.items() for w in v}
PAIRS = [('plum', 'fig', 'same_class'),
         ('whale', 'dolphin', 'same_class'),
         ('couch', 'rug', 'same_class'),
         ('plum', 'tractor', 'cross_domain'),
         ('whale', 'bolt', 'cross_domain'),
         ('couch', 'tiger', 'cross_domain'),
         ('plum', 'Norway', 'cross_realm')]
SEED = 2807

PREREG = {
    'P-I1': 'generalization iff held-out full-dim accuracy >= 0.80 '
            '(chance 0.10)',
    'P-I2': 'efficiency generalizes iff 10-dim held-out accuracy >= '
            '0.85 x full-dim held-out accuracy',
    'P-I3': 'level grading generalizes iff mean dom-energy share of '
            'held-out cross-domain pairs >= 3 x that of held-out '
            'same-class pairs',
    'verdict': 'generalization_law iff P-I1 AND P-I2',
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    data = np.load(SRC_ATLAS, allow_pickle=True)
    words_atlas = [str(w) for w in data['words']]
    E30 = data['E30'].astype(np.float64)
    mce_apple = float(json.loads(
        Path(SRC_ATLAS_RES).read_text(encoding='utf-8'))
        ['margin_cat_emb']['apple'])
    h6 = np.load(SRC_2806)
    dW2806 = h6['dW_class'].astype(np.float64)

    CATS_2806 = {}
    for c in CAT_WORDS:
        CATS_2806[c] = None  # filled from execution below

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG, 'held': HELD,
                 'held_rejected': HELD_REJECTED, 'pairs': PAIRS,
                 'seed': SEED,
                 'note': 'class words of 2806 read from its execution.json'}
    # recover 2806 word lists for template construction
    exec2806 = json.loads((BASE / 'phase2806' / 'qwen4_hierarchy'
                           / 'execution.json').read_text(encoding='utf-8'))
    CATS = exec2806['cats']
    assert [c for c in exec2806['cats']] == CAT_WORDS
    CAT_OF = {w: c for c, v in CATS.items() for w in v}
    execution['cats_2806'] = CATS
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
    del model
    torch.cuda.empty_cache()

    tc = {}

    def tid(t):
        if t not in tc:
            ids = tok(' ' + t, add_special_tokens=False)['input_ids']
            if len(ids) != 1:
                ids = tok(t, add_special_tokens=False)['input_ids']
            assert len(ids) == 1, t
            tc[t] = int(ids[0])
        return tc[t]

    held_words = [w for v in HELD.values() for w in v]
    for c in CAT_WORDS:
        tid(c)
    for v in CATS.values():
        for w in v:
            tid(w)
    for w in held_words:
        tid(w)

    # ---------- gates ----------
    gateE = float(np.abs(Etab[tid('apple')]
                         - E30[words_atlas.index('apple')]).max())
    Wcat10 = W_U[[tid(c) for c in CAT_WORDS]]
    Ea = E30[words_atlas.index('apple')]
    rms_a = np.sqrt((Ea ** 2).mean() + eps)
    z = (Ea / rms_a * g) @ Wcat10.T
    gateM = abs(float(z[0] - (z.sum() - z[0]) / 9.0) - mce_apple)

    cent = {c: np.stack([W_U[tid(w)].astype(np.float64)
                         for w in CATS[c]]).mean(0) for c in CAT_WORDS}
    Cm = np.stack([cent[c] for c in CAT_WORDS])
    dW_class = Cm - (Cm.sum(0, keepdims=True) - Cm) / 9.0
    gate_dW = float(np.abs(dW_class - dW2806).max())

    overlap = sorted(set(held_words) & set(words_atlas))
    print('P2807 gates: E=%.6f lens=%.6f dW_vs_2806=%.2e overlap=%d'
          % (gateE, gateM, gate_dW, len(overlap)), flush=True)
    assert gateE < 0.05 and gateM < 0.05 and gate_dW < 1e-3
    assert not overlap

    def unit(x):
        return x / max(np.linalg.norm(x), 1e-30)

    nat_mean = np.stack([cent[c] for c in NAT_DOM]).mean(0)
    art_mean = np.stack([cent[c] for c in ART_DOM]).mean(0)
    dW_nat_art = nat_mean - art_mean
    liv_mean = np.stack([cent[c] for c in LIVING]).mean(0)
    rest = [c for c in CAT_WORDS if c not in LIVING]
    non_mean = np.stack([cent[c] for c in rest]).mean(0)
    dW_liv_non = liv_mean - non_mean

    Zn_atlas = np.stack([E30[words_atlas.index(w)]
                         / np.sqrt((E30[words_atlas.index(w)] ** 2).mean()
                                   + eps) * g for w in words_atlas])
    Znew = np.stack([Etab[tid(w)].astype(np.float64)
                    / np.sqrt((Etab[tid(w)].astype(np.float64) ** 2).mean()
                              + eps) * g for w in held_words])
    ytrue = np.array([CAT_WORDS.index(HELD_OF[w]) for w in held_words])

    def nca(Xf, Ccent_feats):
        pred = []
        for i in range(len(Xf)):
            dd = [np.linalg.norm(Xf[i] - cc_) for cc_ in Ccent_feats]
            pred.append(int(np.argmin(dd)))
        return pred

    # ---------- Arm HI: held-out full-dim accuracy ----------
    unitcent = np.stack([unit(cent[c]) for c in CAT_WORDS])
    feats_atlas = Zn_atlas @ unitcent.T
    feats_new = Znew @ unitcent.T
    tmpl_full = {c: np.stack([feats_atlas[words_atlas.index(w)]
                              for w in CATS[c]]).mean(0)
                 for c in CAT_WORDS}
    pred = nca(feats_new, list(tmpl_full.values()))
    acc_full = float(np.mean(np.array(pred) == ytrue))
    wrong = [(held_words[i], CAT_WORDS[pred[i]])
             for i in range(len(held_words)) if pred[i] != ytrue[i]]
    p_i1 = bool(acc_full >= 0.80)
    print('P2807 Arm HI held-out acc_full=%.3f (chance 0.10) wrong=%s '
          'P-I1=%s' % (acc_full, wrong, p_i1), flush=True)

    # ---------- Arm HII: k class-direction features ----------
    Qlist = [unit(dW_class[i]) for i in range(10)]
    accs = {}
    for k in range(1, 11):
        Qk = np.stack(Qlist[:k])
        Fatl = Zn_atlas @ Qk.T
        Fnew = Znew @ Qk.T
        cfk = {c: np.stack([Fatl[words_atlas.index(w)]
                            for w in CATS[c]]).mean(0) for c in CAT_WORDS}
        pr = nca(Fnew, list(cfk.values()))
        accs[k] = float(np.mean(np.array(pr) == ytrue))
    Qdom = np.stack([unit(dW_nat_art), unit(dW_liv_non)])
    Fatl_d = Zn_atlas @ Qdom.T
    Fnew_d = Znew @ Qdom.T
    cfd = {c: np.stack([Fatl_d[words_atlas.index(w)]
                        for w in CATS[c]]).mean(0) for c in CAT_WORDS}
    pr = nca(Fnew_d, list(cfd.values()))
    acc_dom2 = float(np.mean(np.array(pr) == ytrue))
    p_i2 = bool(accs[10] >= 0.85 * acc_full)
    print('P2807 Arm HII accs=%s dom2=%.3f P-I2=%s'
          % ({k: round(v, 3) for k, v in accs.items()}, acc_dom2, p_i2),
          flush=True)

    # ---------- Arm HIII: held-out pair energy decomposition ----------
    Vd = np.stack([unit(dW_nat_art), unit(dW_liv_non)])
    pair_rows = []
    dom_shares = {'same_class': [], 'cross_domain': [], 'cross_realm': []}
    for w1, w2, tag in PAIRS:
        E1 = Etab[tid(w1)].astype(np.float64)
        E2 = Etab[tid(w2)].astype(np.float64)
        dE = E1 - E2
        e_dom = float(sum((unit(v) @ dE) ** 2 for v in Vd))
        e_cls = float(sum((q @ dE) ** 2 for q in Qlist))
        e_tot = float(dE @ dE)
        row = {'pair': '%s-%s' % (w1, w2), 'tag': tag,
               'norm': round(float(np.linalg.norm(dE)), 1),
               'dom_energy_share': round(e_dom / max(e_tot, 1e-30), 3),
               'class_energy_share': round(e_cls / max(e_tot, 1e-30), 3),
               'residual_share': round((e_tot - e_cls)
                                       / max(e_tot, 1e-30), 3)}
        pair_rows.append(row)
        dom_shares[tag].append(row['dom_energy_share'])
        print('P2807 Arm HIII %-16s %-14s dom=%.3f cls=%.3f res=%.3f'
              % (row['pair'], tag, row['dom_energy_share'],
                 row['class_energy_share'], row['residual_share']),
              flush=True)
    m_same = float(np.mean(dom_shares['same_class']))
    m_xdom = float(np.mean(dom_shares['cross_domain']))
    p_i3 = bool(m_xdom >= 3.0 * m_same)
    print('P2807 P-I3 dom-share same=%.4f cross=%.4f ratio=%.1f P-I3=%s'
          % (m_same, m_xdom, m_xdom / max(m_same, 1e-30), p_i3), flush=True)

    verdict = {
        'acc_full_heldout': round(acc_full, 3),
        'acc_k_heldout': {str(k): round(v, 3) for k, v in accs.items()},
        'acc_2dom_heldout': round(acc_dom2, 3),
        'wrong_pairs': wrong,
        'generalization': p_i1,
        'efficiency_generalizes': p_i2,
        'dom_share_same_class': round(m_same, 4),
        'dom_share_cross_domain': round(m_xdom, 4),
        'level_grading_generalizes': p_i3,
        'n_heldout_words': len(held_words),
        'gates': {'E_vs_atlas': round(gateE, 6),
                  'lens_vs_2797': round(gateM, 6),
                  'dW_vs_2806': '%.2e' % gate_dW,
                  'overlap': len(overlap)},
        'generalization_law': bool(p_i1 and p_i2),
    }
    result = {'phase': 2807, 'prereg': PREREG, 'verdict': verdict,
              'pair_rows': pair_rows, 'held_words': HELD}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'heldout.npz',
           dW_class=dW_class.astype(np.float32),
           Znew=Znew.astype(np.float32),
           feats_new=feats_new.astype(np.float32),
           ytrue=ytrue)
    print('P2807 VERDICT %s' % json.dumps(verdict), flush=True)


if __name__ == '__main__':
    main()
