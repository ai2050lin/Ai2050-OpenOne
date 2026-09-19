"""Phase 2808 (LPF-21): CROSS-MODEL VALIDATION OF THE HIERARCHY LAW.

User anchor: "test different models, verify whether the law holds
across models".  The 2806 hierarchy findings (nesting 1.000, 0.39%
efficiency, 74x level grading) are re-tested on four further models
with a STRICTLY IDENTICAL 100-word/10-class word list (tokenizer
precheck passed on all four):

  ds7    deepseek-r1-distill-qwen-7b   (Qwen2ForCausalLM, distilled reasoner)
  glm4   glm4-9b-chat-hf               (GlmForCausalLM, different family)
  qwen17 qwen3-1.7b                    (Qwen3 small scale point)
  qwen25 qwen2.5-3b-instruct           (previous generation family)

All-zero-forward protocol: only three tensors per model are read
directly from safetensors (lm_head or embed for W_U when tied,
model.norm for g, config rms_norm_eps); no model loading, no forward.

Arms (clone of 2806):
  A nesting     domain directions in 10-class-direction subspace -> P-G1
  B tree        average-linkage centroid tree, cophenetic rho    -> P-G2
  C var split   domain vs class-within share of column variance  -> P-G3
  D efficiency  k class-direction features nearest-centroid      -> P-G4
  E grading     pair energy decomposition dom/cls/res            -> P-G5

Prereg (frozen before any readout):
  P-H1  nesting_general iff every model has both projection
        retentions >= 0.90 (P-G1)
  P-H2  efficiency_general iff every model has 10-dim accuracy >=
        0.85 x full-dim accuracy (P-G4)
  P-H3  grading_general iff every model has mean dom-energy share of
        cross-domain pairs >= 3 x same-class pairs (P-G5)
  verdict: hierarchy_law_general iff P-H1 AND P-H2 AND P-H3

Gates per model: W_U/E shape agreement, dW_class zero-sum < 1e-8,
tid resolution for all 100 words + 10 class words.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2808' / 'crossmodel_hierarchy'
SRC_ATLAS = BASE / 'phase2795' / 'qwen4_semantic_atlas' / 'atlas.npz'
SRC_2806 = BASE / 'phase2806' / 'qwen4_hierarchy' / 'hierarchy.npz'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'

MODELS = {
    'ds7': 'deepseek-r1-distill-qwen-7b',
    'glm4': 'glm4-9b-chat-hf',
    'qwen17': 'qwen3-1.7b',
    'qwen25': 'qwen2.5-3b-instruct',
}
PAIRS = [('apple', 'banana', 'same_class'),
         ('tiger', 'lion', 'same_class'),
         ('car', 'bus', 'same_class'),
         ('apple', 'bread', 'cross_class_same_domain'),
         ('apple', 'tiger', 'cross_class_same_domain'),
         ('apple', 'car', 'cross_domain'),
         ('apple', 'hammer', 'cross_domain'),
         ('apple', 'japan', 'cross_realm'),
         ('car', 'tiger', 'cross_domain')]
SEED = 2808

PREREG = {
    'P-H1': 'nesting_general iff every model has both domain-direction '
            'projection retentions >= 0.90',
    'P-H2': 'efficiency_general iff every model has 10-dim class-direction '
            'accuracy >= 0.85 x full-dim accuracy',
    'P-H3': 'grading_general iff every model has mean dom-energy share of '
            'cross-domain pairs >= 3 x same-class pairs',
    'verdict': 'hierarchy_law_general iff P-H1 AND P-H2 AND P-H3',
}


def load_tensors(dirname):
    """Read W_U (f32), E (f32), g (f64), eps from safetensors only."""
    import torch
    from safetensors import safe_open
    d = Path(ROOT) / 'models' / 'hf' / dirname
    cfg = json.loads((d / 'config.json').read_text(encoding='utf-8'))
    eps = float(cfg.get('rms_norm_eps', 1e-6))
    tie = bool(cfg.get('tie_word_embeddings', False))
    idxp = d / 'model.safetensors.index.json'
    if idxp.exists():
        wm = json.loads(idxp.read_text(encoding='utf-8'))['weight_map']
    else:
        wm = {'model.embed_tokens.weight': 'model.safetensors',
              'model.norm.weight': 'model.safetensors',
              'lm_head.weight': 'model.safetensors'}
    out = {}
    want = {'model.embed_tokens.weight', 'model.norm.weight',
            'lm_head.weight'}
    need_files = sorted(set(wm[t] for t in want if t in wm))
    for fn in need_files:
        with safe_open(d / fn, framework='pt') as f:
            for t in want:
                if t in wm and wm[t] == fn and t not in out:
                    out[t] = f.get_tensor(t).float().numpy()
    W_U = out.get('lm_head.weight')
    E = out['model.embed_tokens.weight']
    if W_U is None:
        W_U = E
    assert W_U.shape == E.shape, (W_U.shape, E.shape)
    g = out['model.norm.weight'].astype(np.float64)
    return W_U, E, g, eps, tie, str(cfg.get('architectures'))


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    exec2806 = json.loads(SRC_2806_EXEC.read_text(encoding='utf-8'))
    CATS = exec2806['cats']
    CAT_WORDS = list(CATS.keys())
    assert CAT_WORDS == ['fruit', 'animal', 'metal', 'vehicle', 'country',
                         'food', 'nature', 'furniture', 'tool', 'clothing']
    CAT_OF = {w: c for c, v in CATS.items() for w in v}
    DOMAIN_OF = exec2806['domain_of']
    NAT_DOM = exec2806['nat_dom']
    ART_DOM = exec2806['art_dom']
    LIVING = exec2806['living']

    words_all = [w for v in CATS.values() for w in v] + CAT_WORDS

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG, 'models': MODELS, 'cats': CATS,
                 'pairs': PAIRS, 'seed': SEED}
    fc.save(OUT / 'execution.json', execution)

    from transformers import AutoTokenizer

    per_model = {}
    for key, dirname in MODELS.items():
        print('P2808 [%s] loading tensors' % key, flush=True)
        tok = AutoTokenizer.from_pretrained(
            str(ROOT / 'models' / 'hf' / dirname), local_files_only=True,
            trust_remote_code=True, use_fast=True)
        W_U, E, g, eps, tie, arch = load_tensors(dirname)

        tc = {}

        def tid(t):
            if t not in tc:
                ids = tok(' ' + t, add_special_tokens=False)['input_ids']
                if len(ids) != 1:
                    ids = tok(t, add_special_tokens=False)['input_ids']
                assert len(ids) == 1, (key, t)
                tc[t] = int(ids[0])
            return tc[t]

        for w in words_all:
            tid(w)

        gateE = float(np.abs(
            E[tid('apple')] - W_U[tid('apple')]).max()) if tie else -1.0

        cent = {c: np.stack([W_U[tid(w)].astype(np.float64)
                             for w in CATS[c]]).mean(0) for c in CAT_WORDS}
        Cm = np.stack([cent[c] for c in CAT_WORDS])
        dW_class = Cm - (Cm.sum(0, keepdims=True) - Cm) / 9.0
        gateG0 = float(np.abs(dW_class.sum(0)).max())
        assert gateG0 < 1e-8, (key, gateG0)

        def unit(x):
            return x / max(np.linalg.norm(x), 1e-30)

        nat_mean = np.stack([cent[c] for c in NAT_DOM]).mean(0)
        art_mean = np.stack([cent[c] for c in ART_DOM]).mean(0)
        dW_nat_art = nat_mean - art_mean
        liv_mean = np.stack([cent[c] for c in LIVING]).mean(0)
        rest = [c for c in CAT_WORDS if c not in LIVING]
        non_mean = np.stack([cent[c] for c in rest]).mean(0)

        # gate vs 2806 qwen4: identical word list -> dW only comparable
        # structurally; hidden sizes differ across models so no
        # elementwise reference diff is possible.
        hidden = int(W_U.shape[1])

        # Arm A: nesting
        Vc = dW_class.T
        Qc, _ = np.linalg.qr(Vc)
        nest = {}
        for nm, dv in (('nat_art', dW_nat_art), ('liv_non', liv_mean
                                                - non_mean)):
            proj = Qc @ (Qc.T @ dv)
            nest[nm] = float(np.linalg.norm(proj)
                             / max(np.linalg.norm(dv), 1e-30))
        p_g1 = bool(all(v >= 0.90 for v in nest.values()))
        print('P2808 [%s] Arm A nest=%s P-G1=%s'
              % (key, {k: round(v, 3) for k, v in nest.items()}, p_g1),
              flush=True)

        # Arm B: centroid tree
        import phase2806_rdc_hierarchy as h6mod
        Dm = np.zeros((10, 10))
        for i in range(10):
            for j in range(10):
                Dm[i, j] = np.linalg.norm(
                    unit(cent[CAT_WORDS[i]]) - unit(cent[CAT_WORDS[j]]))
        merges = h6mod.average_linkage(Dm)
        cop = h6mod.condensed_links(merges)
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
        ra = np.argsort(np.argsort([sem[k] for k in keys]))
        rb = np.argsort(np.argsort([cop[k] for k in keys]))
        rho_tree = float(np.corrcoef(ra, rb)[0, 1])
        p_g2 = bool(rho_tree >= 0.5)
        print('P2808 [%s] Arm B rho_tree=%.3f P-G2=%s' % (key, rho_tree,
                                                          p_g2), flush=True)

        # Arm C: column variance split (E-side features)
        def rmsz(row):
            r = row.astype(np.float64)
            return r / np.sqrt((r ** 2).mean() + eps) * g

        Zn = np.stack([rmsz(E[tid(w)]) for w in
                       [w for v in CATS.values() for w in v]])
        watlas = [w for v in CATS.values() for w in v]
        ytrue = np.array([CAT_WORDS.index(CAT_OF[w]) for w in watlas])
        CentM = np.stack([cent[c] for c in CAT_WORDS])
        S = Zn @ CentM.T
        colmean = S.mean(0)
        colvar = float(colmean.var())
        dommean = {}
        for c in CAT_WORDS:
            dommean.setdefault(DOMAIN_OF[c], []).append(
                colmean[CAT_WORDS.index(c)])
        dmean = {k: float(np.mean(v)) for k, v in dommean.items()}
        gm = float(colmean.mean())
        ss_dom = float(sum(len(v) * (dmean[k] - gm) ** 2
                           for k, v in dommean.items()))
        ss_cls = float(sum((colmean[CAT_WORDS.index(c)]
                            - dmean[DOMAIN_OF[c]]) ** 2
                           for c in CAT_WORDS))
        share_dom = ss_dom / max(ss_dom + ss_cls, 1e-30)
        p_g3 = bool(share_dom >= 0.30 and (1 - share_dom) >= 0.30)
        print('P2808 [%s] Arm C dom=%.3f cls=%.3f P-G3=%s'
              % (key, share_dom, 1 - share_dom, p_g3), flush=True)

        # Arm D: efficiency
        def nca(Xf, tmpl):
            pred = []
            for i in range(len(Xf)):
                dd = [np.linalg.norm(Xf[i] - cc_) for cc_ in tmpl]
                pred.append(int(np.argmin(dd)))
            return float(np.mean(np.array(pred) == ytrue))

        unitcent = np.stack([unit(cent[c]) for c in CAT_WORDS])
        feats_full = Zn @ unitcent.T
        tmpl_full = {c: np.stack([feats_full[watlas.index(w)]
                                  for w in CATS[c]]).mean(0)
                     for c in CAT_WORDS}
        acc_full = nca(feats_full, list(tmpl_full.values()))
        Qlist = [unit(dW_class[i]) for i in range(10)]
        accs = {}
        for k in range(1, 11):
            Qk = np.stack(Qlist[:k])
            Fk = Zn @ Qk.T
            cfk = {c: np.stack([Fk[watlas.index(w)] for w in CATS[c]])
                   .mean(0) for c in CAT_WORDS}
            accs[k] = nca(Fk, list(cfk.values()))
        Qdom = np.stack([unit(dW_nat_art),
                         unit(liv_mean - non_mean)])
        Fd = Zn @ Qdom.T
        cfd = {c: np.stack([Fd[watlas.index(w)] for w in CATS[c]])
               .mean(0) for c in CAT_WORDS}
        acc_dom2 = nca(Fd, list(cfd.values()))
        p_g4 = bool(accs[10] >= 0.85 * acc_full)
        print('P2808 [%s] Arm D acc_full=%.3f acc10=%.3f accs=%s '
              'dom2=%.3f P-G4=%s'
              % (key, acc_full, accs[10],
                 {k: round(v, 2) for k, v in accs.items()}, acc_dom2,
                 p_g4), flush=True)

        # Arm E: pair energy decomposition
        Vd = np.stack([unit(dW_nat_art), unit(liv_mean - non_mean)])
        pair_rows = []
        dom_shares = {'same_class': [], 'cross_domain': []}
        for w1, w2, tag in PAIRS:
            E1 = E[tid(w1)].astype(np.float64)
            E2 = E[tid(w2)].astype(np.float64)
            dE = E1 - E2
            e_dom = float(sum((unit(v) @ dE) ** 2 for v in Vd))
            e_cls = float(sum((q @ dE) ** 2 for q in Qlist))
            e_tot = float(dE @ dE)
            row = {'pair': '%s-%s' % (w1, w2), 'tag': tag,
                   'dom': round(e_dom / max(e_tot, 1e-30), 3),
                   'cls': round(e_cls / max(e_tot, 1e-30), 3),
                   'res': round((e_tot - e_cls) / max(e_tot, 1e-30), 3)}
            pair_rows.append(row)
            if tag in dom_shares:
                dom_shares[tag].append(row['dom'])
            print('P2808 [%s] Arm E %-14s %-24s dom=%.3f cls=%.3f res=%.3f'
                  % (key, row['pair'], tag, row['dom'], row['cls'],
                     row['res']), flush=True)
        m_same = float(np.mean(dom_shares['same_class']))
        m_xdom = float(np.mean(dom_shares['cross_domain']))
        p_g5 = bool(m_xdom >= 3.0 * m_same)
        print('P2808 [%s] P-G5 same=%.4f cross=%.4f ratio=%.1f P-G5=%s'
              % (key, m_same, m_xdom, m_xdom / max(m_same, 1e-30), p_g5),
              flush=True)

        per_model[key] = {
            'arch': arch, 'tie': tie, 'rms_eps': eps,
            'hidden_size': hidden,
            'gateE_tie_check': round(gateE, 6),
            'dW_zero_sum': '%.2e' % gateG0,
            'nesting': {k: round(v, 3) for k, v in nest.items()},
            'P-G1': p_g1,
            'rho_tree': round(rho_tree, 3), 'P-G2': p_g2,
            'col_var_domain_share': round(share_dom, 3), 'P-G3': p_g3,
            'acc_full': round(acc_full, 3),
            'acc_k': {str(k): round(v, 3) for k, v in accs.items()},
            'acc_2dom': round(acc_dom2, 3), 'P-G4': p_g4,
            'dom_share_same_class': round(m_same, 4),
            'dom_share_cross_domain': round(m_xdom, 4),
            'dom_ratio': round(m_xdom / max(m_same, 1e-30), 1),
            'P-G5': p_g5,
            'pair_rows': pair_rows,
            'merges': [[(CAT_WORDS[a] if a < 10 else 'cluster%d' % a),
                        (CAT_WORDS[b] if b < 10 else 'cluster%d' % b)]
                       for a, b in merges],
        }
        del W_U, E, g
        import gc
        gc.collect()

    p_h1 = all(m['P-G1'] for m in per_model.values())
    p_h2 = all(m['P-G4'] for m in per_model.values())
    p_h3 = all(m['P-G5'] for m in per_model.values())
    verdict = {
        'per_model': {k: {kk: vv for kk, vv in m.items()
                          if kk != 'pair_rows'}
                      for k, m in per_model.items()},
        'nesting_general': p_h1,
        'efficiency_general': p_h2,
        'grading_general': p_h3,
        'hierarchy_law_general': bool(p_h1 and p_h2 and p_h3),
    }
    result = {'phase': 2808, 'prereg': PREREG, 'verdict': verdict,
              'per_model_full': per_model}
    fc.save(OUT / 'result.json', result)
    print('P2808 VERDICT %s' % json.dumps(verdict), flush=True)


if __name__ == '__main__':
    main()
