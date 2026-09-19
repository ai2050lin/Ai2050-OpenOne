"""Phase 2812 (LPF-25): ATTRIBUTE SUBSPACE HYPOTHESIS (the 91% residual)
+ metal->fruit forensics.

2811 established: noun embeddings = near-one-hot class spike (9% energy)
+ huge near-full-rank residual (91%).  Question now: is the residual an
ATTRIBUTE manifold?  And is the P-L1 cell metal->fruit a semantic
secondary membership or a generic-axis geometric artifact?

Zero-forward, 2808 safetensors protocol (tie-aware: lm_head falls back
to embed_tokens when tied, 2811 gate proved tie=True); eval battery
rebuilt identically to 2811 (gates: dW vs 2807 <1e-6, eval_words list
vs 2811 exact, Z vs 2811 battery.npz <1e-4).

Arms:
  L4  attribute axes: K bipolar adjective-anchor directions (embedding-
      side centroids, single-token filtered, disjoint from battery)
      - energy share of attribute subspace on eval battery
      - split-half axis stability (alphabetical anchor split)
      - random-token baseline
  S1  forensics: positive-projection fraction of non-fruit eval words on
      unit dW_fruit; cos(fruit; food,animal,nature) vs off-diagonal q95;
      top unembed tokens of dW_fruit (interpretive)
  S2  remediation: remove generic component g = mean non-fruit eval z
      from dW_fruit, recompute the 4 P-L1 cells with per-cell shuffle
      nulls x1000

Nulls:
  N1 pole-shuffle x1000 (reassign +/- labels within each attribute pool,
     sizes preserved) -> subspace share q95/median           (P-L4a)
  N2 per-attribute pole-shuffle x300 on split-half B -> |cos| q95
                                                             (P-L4b)
  N3 label-shuffle x1000 for remediated cells                (P-S2)

Prereg (frozen before any readout):
  P-L4a attribute_subspace_real iff mean eval share > N1 q95 AND
       > 3 x N1 median
  P-L4b axis_stability iff >= 60% of attributes have split-half |cos|
       > their N2 q95
  P-S1 fruit_generic_axis iff >= 0.60 non-fruit eval words project
       positive on unit dW_fruit AND mean cos(fruit; food,animal,nature)
       > off-diagonal |cos| q95
  P-S2 cells_survive_remediation iff metal->fruit AND food->fruit rates
       (remediated fruit direction) still >= 0.50 AND > N3 per-cell q95
  verdict: attribute_manifold_support = P-L4a AND P-L4b
       (P-S1/P-S2 = diagnosis only, feed P-L1 interpretation)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2812' / 'attribute_subspace'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SRC_2807 = BASE / 'phase2807' / 'qwen4_heldout'
SRC_2811 = BASE / 'phase2811' / 'noun_superposition'
SEED = 2812
N_SHUFFLE = 1000
N_HALF = 300
N_RAND = 250

ATTR = {
    'size': (['big', 'large', 'huge', 'tall'],
             ['small', 'little', 'tiny', 'short']),
    'weight': (['heavy', 'hefty', 'massive', 'dense'],
               ['light', 'airy', 'feathery', 'weightless']),
    'temperature': (['hot', 'warm', 'burning', 'boiling'],
                    ['cold', 'cool', 'freezing', 'icy']),
    'wetness': (['wet', 'damp', 'humid', 'soaked'],
                ['dry', 'arid', 'parched', 'dusty']),
    'hardness': (['hard', 'solid', 'firm', 'stiff'],
                 ['soft', 'fluffy', 'plush', 'silky']),
    'speed': (['fast', 'quick', 'rapid', 'swift'],
              ['slow', 'sluggish', 'lazy', 'creeping']),
    'brightness': (['bright', 'luminous', 'shiny', 'gleaming'],
                   ['dark', 'dim', 'gloomy', 'shadowy']),
    'loudness': (['loud', 'noisy', 'deafening', 'roaring'],
                 ['quiet', 'silent', 'hushed', 'muted']),
    'cleanliness': (['clean', 'sterile', 'spotless', 'pure'],
                    ['dirty', 'filthy', 'grimy', 'soiled']),
    'sweetness': (['sweet', 'sugary', 'syrupy', 'honeyed'],
                  ['bitter', 'sour', 'tart', 'bland']),
    'danger': (['dangerous', 'deadly', 'toxic', 'lethal'],
               ['safe', 'harmless', 'benign', 'gentle']),
    'value': (['expensive', 'costly', 'pricey', 'premium'],
              ['cheap', 'inexpensive', 'bargain', 'budget']),
    'age': (['new', 'modern', 'young', 'fresh'],
            ['old', 'ancient', 'antique', 'aged']),
    'strength': (['strong', 'powerful', 'mighty', 'sturdy'],
                 ['weak', 'fragile', 'feeble', 'flimsy']),
    'sharpness': (['sharp', 'jagged', 'pointed', 'spiky'],
                  ['dull', 'blunt', 'rounded', 'edgeless']),
}

PREREG = {
    'P-L4a': 'attribute_subspace_real iff mean eval attribute-subspace '
             'share > N1 pole-shuffle q95 AND > 3 x N1 median',
    'P-L4b': 'axis_stability iff >= 60% of attributes have split-half '
             '|cos| > their N2 per-attribute pole-shuffle q95',
    'P-S1': 'fruit_generic_axis iff >= 0.60 of non-fruit eval words '
            'project positive on unit dW_fruit AND mean '
            'cos(fruit;food,animal,nature) > off-diagonal |cos| q95',
    'P-S2': 'cells_survive_remediation iff metal->fruit AND food->fruit '
            'rates (remediated fruit direction) >= 0.50 AND > N3 '
            'per-cell q95',
    'verdict': 'attribute_manifold_support = P-L4a AND P-L4b; '
               'P-S1/P-S2 diagnosis only',
}


def unit(x):
    return x / max(np.linalg.norm(x), 1e-30)


def main():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    exec2806 = json.loads(SRC_2806_EXEC.read_text(encoding='utf-8'))
    CATS = exec2806['cats']
    CAT_WORDS = list(CATS.keys())
    exec2807 = json.loads((SRC_2807 / 'execution.json').read_text(
        encoding='utf-8'))
    HELD = exec2807['held']
    assert list(HELD.keys()) == CAT_WORDS
    res2811 = json.loads((SRC_2811 / 'result.json').read_text(
        encoding='utf-8'))

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG, 'seed': SEED,
                 'n_shuffle': N_SHUFFLE, 'n_half': N_HALF,
                 'n_rand': N_RAND, 'attr_candidates': ATTR,
                 'note': 'attribute anchors embedding-side; eval battery '
                         'rebuilt identical to 2811 (gate-checked)'}
    fc.save(OUT / 'execution.json', execution)

    # ---------- tensors (2808 protocol, tie-aware) ----------
    from safetensors import safe_open
    mdir = ROOT / 'models' / 'hf' / 'qwen3-4b'
    index = json.loads((mdir / 'model.safetensors.index.json')
                       .read_text(encoding='utf-8'))['weight_map']

    def read_tensor(name):
        with safe_open(str(mdir / index[name]), framework='pt') as f:
            return f.get_tensor(name).float().numpy()

    Etab = read_tensor('model.embed_tokens.weight')
    g = read_tensor('model.norm.weight').astype(np.float64)
    try:
        Wu = read_tensor('lm_head.weight')
        tie = False
    except KeyError:
        Wu = Etab
        tie = True
    cfg = json.loads((mdir / 'config.json').read_text(encoding='utf-8'))
    eps = float(cfg.get('rms_norm_eps', 1e-6))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        str(mdir), local_files_only=True, trust_remote_code=True,
        use_fast=True)

    tc = {}

    def tid(t):
        if t not in tc:
            ids = tok(' ' + t, add_special_tokens=False)['input_ids']
            if len(ids) != 1:
                ids = tok(t, add_special_tokens=False)['input_ids']
            assert len(ids) == 1, t
            tc[t] = int(ids[0])
        return tc[t]

    atlas_words = [w for v in CATS.values() for w in v]
    new_kept = res2811['verdict']['new_survivors']
    eval_pairs = [(w, c) for c in CAT_WORDS for w in HELD[c]] \
        + [(w, c) for c in CAT_WORDS for w in new_kept[c]]
    eval_words = [w for w, _ in eval_pairs]
    ytrue = np.array([CAT_WORDS.index(c) for _, c in eval_pairs])
    n_held = sum(len(v) for v in HELD.values())

    # ---------- gates ----------
    assert eval_words == res2811['eval_words'], 'eval battery drift'
    h7 = np.load(SRC_2807 / 'heldout.npz')
    cent = {c: np.stack([Wu[tid(w)].astype(np.float64)
                         for w in CATS[c]]).mean(0) for c in CAT_WORDS}
    Cm = np.stack([cent[c] for c in CAT_WORDS])
    dW = Cm - (Cm.sum(0, keepdims=True) - Cm) / 9.0
    gate_dW = float(np.abs(dW - h7['dW_class'].astype(np.float64)).max())
    b11 = np.load(SRC_2811 / 'battery.npz')

    def zrow(t):
        e = Etab[tid(t)].astype(np.float64)
        return e / np.sqrt((e ** 2).mean() + eps) * g

    def zrow_id(i):
        e = Etab[i].astype(np.float64)
        return e / np.sqrt((e ** 2).mean() + eps) * g

    Z_eval = np.stack([zrow(w) for w in eval_words])
    Z_atlas = np.stack([zrow(w) for w in atlas_words])
    gate_Z = float(np.abs(Z_eval - b11['Z_eval'].astype(np.float64)).max())
    print('P2812 gates: dW_vs_2807=%.2e Z_vs_2811=%.2e n_eval=%d tie=%s'
          % (gate_dW, gate_Z, len(eval_words), tie), flush=True)
    assert gate_dW < 1e-6 and gate_Z < 1e-4

    # ---------- attribute anchors (filtered, disjoint) ----------
    lower_known = set(w.lower() for w in atlas_words + eval_words
                      + CAT_WORDS)
    kept_attr, attr_rej = {}, {}
    used = set()
    for a, (pos, neg) in ATTR.items():
        kp, kn, rj = [], [], []
        for pole, pool in (('pos', pos), ('neg', neg)):
            for w in pool:
                if w.lower() in lower_known or w.lower() in used:
                    rj.append(w + ' dup')
                    continue
                try:
                    tid(w)
                except AssertionError:
                    rj.append(w + ' multi-tok')
                    continue
                (kp if pole == 'pos' else kn).append(w)
                used.add(w.lower())
        attr_rej[a] = rj
        if len(kp) >= 2 and len(kn) >= 2:
            kept_attr[a] = (kp, kn)
    K = len(kept_attr)
    print('P2812 attribute axes kept %d/%d %s'
          % (K, len(ATTR), list(kept_attr)), flush=True)
    assert K >= 10, 'too few attribute axes survived'

    Zanch = {a: ([zrow(w) for w in kp], [zrow(w) for w in kn])
             for a, (kp, kn) in kept_attr.items()}

    dA = np.stack([unit(np.stack(Zanch[a][0]).mean(0)
                        - np.stack(Zanch[a][1]).mean(0))
                   for a in kept_attr])
    QA, _ = np.linalg.qr(dA.T)

    def share_mean(Z, Q):
        num = (Z @ Q) ** 2
        return float((num.sum(1)
                      / np.maximum((Z ** 2).sum(1), 1e-30)).mean())

    share_eval = share_mean(Z_eval, QA)
    share_atlas = share_mean(Z_atlas, QA)

    # ---------- N1: pole-shuffle null ----------
    null_shares = np.empty(N_SHUFFLE)
    for it in range(N_SHUFFLE):
        rows = []
        for a in kept_attr:
            kp, kn = kept_attr[a]
            words = [(w, 0) for w in kp] + [(w, 1) for w in kn]
            perm = rng.permutation(len(words))
            npos = len(kp)
            p_sel = [words[i][0] for i in perm[:npos]]
            n_sel = [words[i][0] for i in perm[npos:]]
            zp = np.stack([zrow(w) for w in p_sel]).mean(0)
            zn = np.stack([zrow(w) for w in n_sel]).mean(0)
            rows.append(unit(zp - zn))
        QA_n, _ = np.linalg.qr(np.stack(rows).T)
        null_shares[it] = share_mean(Z_eval, QA_n)
    n1_med = float(np.median(null_shares))
    n1_q95 = float(np.quantile(null_shares, 0.95))
    p_l4a = bool(share_eval > n1_q95 and share_eval > 3 * n1_med)
    print('P2812 share eval=%.4f atlas=%.4f | N1 med=%.4f q95=%.4f '
          '(K/2560=%.4f) P-L4a=%s'
          % (share_eval, share_atlas, n1_med, n1_q95, K / 2560.0, p_l4a),
          flush=True)

    # ---------- random-token baseline ----------
    all_tids = set(tc.values())
    rand_ids = []
    while len(rand_ids) < N_RAND:
        r = int(rng.integers(0, Etab.shape[0]))
        if r > 0 and r not in all_tids and r not in rand_ids:
            rand_ids.append(r)
    Z_rand = np.stack([zrow_id(i) for i in rand_ids])
    share_rand = share_mean(Z_rand, QA)
    print('P2812 share random-tokens=%.4f' % share_rand, flush=True)

    # ---------- P-L4b: split-half stability ----------
    stab_rows = []
    for a, (kp, kn) in kept_attr.items():
        ps, ns = sorted(kp), sorted(kn)
        cutp, cutn = (len(ps) + 1) // 2, (len(ns) + 1) // 2
        pa, pb = ps[:cutp], ps[cutp:]
        na, nb = ns[:cutn], ns[cutn:]
        dA_a = unit(np.stack([zrow(w) for w in pa]).mean(0)
                    - np.stack([zrow(w) for w in na]).mean(0))
        dA_b = unit(np.stack([zrow(w) for w in pb]).mean(0)
                    - np.stack([zrow(w) for w in nb]).mean(0))
        cosv = float(abs(unit(dA_a) @ unit(dA_b)))
        pool = pb + nb
        null_cos = np.empty(N_HALF)
        for it in range(N_HALF):
            perm = rng.permutation(len(pool))
            p_sel = [pool[i] for i in perm[:len(pb)]]
            n_sel = [pool[i] for i in perm[len(pb):]]
            dA_s = unit(np.stack([zrow(w) for w in p_sel]).mean(0)
                        - np.stack([zrow(w) for w in n_sel]).mean(0))
            null_cos[it] = abs(float(unit(dA_a) @ unit(dA_s)))
        q95 = float(np.quantile(null_cos, 0.95))
        stab_rows.append({'attr': a, 'cos_half': round(cosv, 4),
                          'q95': round(q95, 4), 'pass': bool(cosv > q95)})
    frac_pass = float(np.mean([r['pass'] for r in stab_rows]))
    p_l4b = bool(frac_pass >= 0.60)
    print('P2812 stability pass-frac=%.2f P-L4b=%s %s'
          % (frac_pass, p_l4b, json.dumps(stab_rows)), flush=True)

    # ---------- P-S1: fruit direction forensics ----------
    unitD = np.stack([unit(dW[i]) for i in range(10)])
    fi = CAT_WORDS.index('fruit')
    dfu = unitD[fi]
    nonfruit = ytrue != fi
    proj = Z_eval @ dfu
    pos_frac = float((proj[nonfruit] > 0).mean())
    Ccos = unitD @ unitD.T
    off = np.abs(Ccos[np.triu_indices(10, 1)])
    off_q95 = float(np.quantile(off, 0.95))
    trio = [CAT_WORDS.index(c) for c in ('food', 'animal', 'nature')]
    trio_cos = float(np.mean([Ccos[fi, j] for j in trio]))
    p_s1 = bool(pos_frac >= 0.60 and trio_cos > off_q95)
    top_ids = np.argsort(-(Wu @ dfu))[:10].tolist()
    top_toks = [tok.decode([i]).strip() for i in top_ids]
    print('P2812 P-S1 pos_frac=%.3f trio_cos=%.3f off_q95=%.3f P-S1=%s '
          'top_unembed=%s'
          % (pos_frac, trio_cos, off_q95, p_s1, top_toks), flush=True)

    # ---------- P-S2: remediation ----------
    ghat = unit(Z_eval[nonfruit].mean(0))
    dW_fruit_rem = unit(dW[fi] - (dW[fi] @ ghat) * ghat)
    unitD_rem = unitD.copy()
    unitD_rem[fi] = dW_fruit_rem
    F_rem = Z_eval @ unitD_rem.T

    def sec_argmax(frow, own):
        f = frow.copy()
        f[own] = -1e30
        return int(np.argmax(f))

    cells = []
    for c1 in ('metal', 'food', 'furniture', 'clothing'):
        i1 = CAT_WORDS.index(c1)
        m = ytrue == i1
        idx1 = np.where(m)[0]
        n1_ = int(m.sum())
        rate = float(np.mean([sec_argmax(F_rem[i], i1) == fi
                              for i in idx1]))
        null_rates = np.empty(N_SHUFFLE)
        cnt = {c: int((ytrue == CAT_WORDS.index(c)).sum())
               for c in CAT_WORDS}
        pool = sum([[CAT_WORDS.index(c)] * cnt[c] for c in CAT_WORDS], [])
        for it in range(N_SHUFFLE):
            ys2 = np.empty(len(ytrue), dtype=int)
            ys2[rng.permutation(len(ytrue))] = pool
            m2 = ys2 == i1
            if m2.sum() == 0:
                null_rates[it] = 0.0
                continue
            null_rates[it] = np.mean(
                [sec_argmax(F_rem[i], int(ys2[i])) == fi
                 for i in np.where(m2)[0]])
        q95 = float(np.quantile(null_rates, 0.95))
        cells.append({'cell': '%s->fruit' % c1, 'n': n1_,
                      'rate_rem': round(rate, 3), 'q95_rem': round(q95, 3),
                      'pass': bool(rate >= 0.50 and rate > q95)})
    p_s2 = bool(cells[0]['pass'] and cells[1]['pass'])
    print('P2812 P-S2 remediated cells %s P-S2=%s'
          % (json.dumps(cells), p_s2), flush=True)

    # ---------- measurement extras ----------
    class_Q, _ = np.linalg.qr(unitD.T)
    attr_in_class = float(np.mean(np.sum((dA @ class_Q) ** 2, axis=1)))
    per_attr_share = {}
    top_words = {}
    for a in kept_attr:
        d = unit(np.stack(Zanch[a][0]).mean(0) - np.stack(Zanch[a][1]).mean(0))
        per_attr_share[a] = round(float(np.mean(
            (Z_eval @ d) ** 2 / np.maximum((Z_eval ** 2).sum(1), 1e-30))), 5)
        top_words[a] = [eval_words[i] for i in np.argsort(
            -(Z_eval @ d))[:8]]

    verdict = {
        'n_eval': len(eval_words), 'n_held': n_held,
        'n_attr_axes': K, 'kept_attr': list(kept_attr),
        'attr_rejected': attr_rej,
        'share_eval': round(share_eval, 5),
        'share_atlas': round(share_atlas, 5),
        'share_random_tokens': round(share_rand, 5),
        'n1_median': round(n1_med, 5), 'n1_q95': round(n1_q95, 5),
        'theoretical_random_share': round(K / 2560.0, 5),
        'stability': stab_rows, 'stability_pass_frac': round(frac_pass, 3),
        'attribute_subspace_real': p_l4a, 'axis_stability': p_l4b,
        'ps1_pos_frac': round(pos_frac, 4),
        'ps1_trio_cos': round(trio_cos, 4),
        'ps1_offdiag_q95': round(off_q95, 4),
        'ps1_top_unembed_fruit': top_toks, 'fruit_generic_axis': p_s1,
        'ps2_cells': cells, 'cells_survive_remediation': p_s2,
        'attr_energy_inside_class_subspace': round(attr_in_class, 4),
        'per_attr_share': per_attr_share, 'top_words_per_attr': top_words,
        'attribute_manifold_support': bool(p_l4a and p_l4b),
    }
    result = {'phase': 2812, 'prereg': PREREG, 'verdict': verdict,
              'null_shares_summary': {'median': n1_med, 'q95': n1_q95},
              'rand_ids': rand_ids, 'eval_words': eval_words}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'axes.npz', dA=dA.astype(np.float32),
           QA=QA.astype(np.float32),
           dW_fruit_rem=dW_fruit_rem.astype(np.float32),
           Z_eval=Z_eval.astype(np.float32), ytrue=ytrue)

    elapsed = time.monotonic() - t0
    cc.ledger('phase2812', elapsed)
    print('P2812 VERDICT %s' % json.dumps(verdict), flush=True)
    print('P2812 elapsed %.1fs' % elapsed, flush=True)


if __name__ == '__main__':
    main()
