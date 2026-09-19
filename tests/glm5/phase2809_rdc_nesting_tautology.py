"""Phase 2809 (LPF-22): IS THE 1.000 NESTING RETENTION AN ALGEBRAIC
TUTOLOGY?  (self-falsification first)

Phase 2806/2808 headline: both domain directions lie in the 10-class
direction subspace with projection retention 1.000 on five models.
BUT: dW_dom = mean(cent_NAT) - mean(cent_ART) is a ZERO-SUM linear
combination of the centroids, and dW_class spans (at most) the same
zero-sum combination space V0 = {sum a_i cent_i : sum a_i = 0} of
dimension <= 9.  If dW_class has numerical rank 9, then ANY zero-sum
domain direction necessarily nests -> P-G1 = 1.000 carries NO
information.

Arms (zero-forward, qwen4):
  A  rank audit: singular spectrum of dW_class (2806 word list);
     numerical rank, affine rank of centroids                -> P-J1
  B  random null: 1000 shuffles of the 100-word class labels
     (10x10 preserved); rebuild pseudo dW_class and pseudo domain
     directions (5v4 split, 2806 construction); nesting retention
     distribution                                            -> P-J2
  C  alignment test: real dom-vs-class cosine profile vs null
     distribution (max |cos| of the domain row)              -> P-J3

Prereg (frozen before any readout):
  P-J1  rank precondition iff numerical rank of dW_class == 9
        (necessary condition for tautology)
  P-J2  tautology iff null median nesting retention >= 0.999
  P-J3  alignment_substantive iff real max|cos(dom row over 10 class
        dirs)| >= 95th percentile of the null max|cos|
  verdict: nesting_substantive iff NOT (P-J1 AND P-J2);
           if tautology but P-J3 true, the ALIGNMENT structure (not
           the nesting fact) is the non-trivial remnant.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2809' / 'qwen4_nesting_tautology'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SEED = 2809
N_NULL = 1000

PREREG = {
    'P-J1': 'rank precondition iff numerical rank of dW_class == 9',
    'P-J2': 'tautology iff null median nesting retention >= 0.999',
    'P-J3': 'alignment_substantive iff real max|cos(dom row over 10 '
            'class directions)| >= 95th percentile of null max|cos|',
    'verdict': 'nesting_substantive iff NOT (P-J1 AND P-J2); tautology '
               'downgrades 2806/2808 P-G1 headlines; alignment survives '
               'iff P-J3',
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    exec2806 = json.loads(SRC_2806_EXEC.read_text(encoding='utf-8'))
    CATS = exec2806['cats']
    CAT_WORDS = list(CATS.keys())
    NAT_DOM = exec2806['nat_dom']
    ART_DOM = exec2806['art_dom']

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG, 'seed': SEED, 'n_null': N_NULL}
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

    watlas = [w for v in CATS.values() for w in v]
    for w in watlas + CAT_WORDS:
        tid(w)
    Erows = {w: W_U[tid(w)].astype(np.float64) for w in watlas}

    def unit(x):
        return x / max(np.linalg.norm(x), 1e-30)

    def build(labels):
        """labels: list of class index per word -> (dW (10,2560),
        dom_dirs (2,2560)) using 2806 construction."""
        cents = []
        for c in range(10):
            ws = [watlas[i] for i in range(len(watlas))
                  if labels[i] == c]
            cents.append(np.stack([Erows[w] for w in ws]).mean(0))
        Cm = np.stack(cents)
        dW = Cm - (Cm.sum(0, keepdims=True) - Cm) / 9.0
        nat = Cm[NAT_IDX].mean(0)
        art = Cm[ART_IDX].mean(0)
        doms = np.stack([nat - art])
        return dW, doms

    # ---------- Arm A: rank audit on the real labelling ----------
    labels_real = []
    for ci, c in enumerate(CAT_WORDS):
        labels_real += [ci] * len(CATS[c])
    NAT_IDX = [CAT_WORDS.index(c) for c in NAT_DOM]
    ART_IDX = [CAT_WORDS.index(c) for c in ART_DOM]
    dW_real, dom_real = build(labels_real)
    sv = np.linalg.svd(dW_real, compute_uv=False)
    num_rank = int((sv > sv[0] * 1e-10).sum())
    Cm_real = np.stack([
        np.stack([Erows[w] for w in CATS[c]]).mean(0) for c in CAT_WORDS])
    aff_rank = int(np.linalg.matrix_rank(
        Cm_real - Cm_real.mean(0), tol=sv[0] * 1e-10))
    p_j1 = bool(num_rank == 9)
    print('P2809 Arm A singular spectrum top5=%s last=%s '
          'num_rank=%d aff_rank=%d P-J1=%s'
          % ([round(float(x), 1) for x in sv[:5]],
             round(float(sv[-1]), 6), num_rank, aff_rank, p_j1),
          flush=True)

    def nest_ret(dW, doms):
        Qc, _ = np.linalg.qr(dW.T)
        out = []
        for dv in doms:
            proj = Qc @ (Qc.T @ dv)
            out.append(float(np.linalg.norm(proj)
                             / max(np.linalg.norm(dv), 1e-30)))
        return out

    real_nest = nest_ret(dW_real, dom_real)
    print('P2809 real nesting retention %s (cross-check 2806)'
          % {k: round(v, 4) for k, v in
             zip(('nat_art',), real_nest)}, flush=True)

    # ---------- Arm B: random null ----------
    rng = np.random.default_rng(SEED)
    null_max = []
    null_min = []
    null_cos_max = []
    real_cos_max = max(abs(float(unit(dom_real[0]) @ unit(dW_real[i])))
                       for i in range(10))
    for it in range(N_NULL):
        perm = rng.permutation(len(watlas))
        labels_null = [0] * len(watlas)
        for pos, wi in enumerate(perm):
            labels_null[wi] = pos // 10
        dWn, _ = build(labels_null)
        # pseudo domain direction: random 5v4 zero-sum combo of the
        # pseudo centroids (2806-style construction on shuffled labels)
        groups = list(range(10))
        rng.shuffle(groups)
        cents_n = []
        for c in range(10):
            ws = [watlas[i] for i in range(len(watlas))
                  if labels_null[i] == c]
            cents_n.append(np.stack([Erows[w] for w in ws]).mean(0))
        CmN = np.stack(cents_n)
        domn = CmN[groups[:5]].mean(0) - CmN[groups[5:9]].mean(0)
        nr = nest_ret(dWn, np.stack([domn]))
        null_max.append(max(nr))
        null_min.append(min(nr))
        null_cos_max.append(max(abs(float(unit(domn) @ unit(dWn[i])))
                                for i in range(10)))
    null_max = np.array(null_max)
    p_j2 = bool(float(np.median(null_max)) >= 0.999)
    print('P2809 Arm B null nest max: median=%.4f q05=%.4f q95=%.4f '
          'min=%.4f  P-J2(tautology)=%s'
          % (float(np.median(null_max)), float(np.quantile(null_max, 0.05)),
             float(np.quantile(null_max, 0.95)), float(null_max.min()),
             p_j2), flush=True)

    # ---------- Arm C: alignment vs null ----------
    null_cos_max = np.array(null_cos_max)
    q95 = float(np.quantile(null_cos_max, 0.95))
    p_j3 = bool(real_cos_max >= q95)
    print('P2809 Arm C real max|cos(dom,class)|=%.3f null q95=%.3f '
          'null median=%.3f  P-J3=%s'
          % (real_cos_max, q95, float(np.median(null_cos_max)), p_j3),
          flush=True)

    tautology = bool(p_j1 and p_j2)
    verdict = {
        'dW_singular_top5': [round(float(x), 1) for x in sv[:5]],
        'dW_singular_last': round(float(sv[-1]), 6),
        'numerical_rank': num_rank,
        'affine_rank_centroids': aff_rank,
        'real_nesting_nat_art': round(real_nest[0], 4),
        'null_nest_max_median': round(float(np.median(null_max)), 4),
        'null_nest_max_q05': round(float(np.quantile(null_max, 0.05)), 4),
        'null_nest_max_q95': round(float(np.quantile(null_max, 0.95)), 4),
        'null_nest_max_min': round(float(null_max.min()), 4),
        'real_max_abs_cos_dom_row': round(real_cos_max, 3),
        'null_max_abs_cos_q95': round(q95, 3),
        'null_max_abs_cos_median': round(
            float(np.median(null_cos_max)), 3),
        'rank_precondition': p_j1,
        'tautology': tautology,
        'alignment_substantive': p_j3,
        'nesting_substantive': bool(not tautology),
        'final_reading': ('nesting_fact_is_algebraic_identity_but_'
                          'alignment_structure_nontrivial'
                          if tautology and p_j3 else
                          'nesting_fact_nontrivial' if not tautology else
                          'nesting_fact_trivial_and_alignment_trivial'),
    }
    result = {'phase': 2809, 'prereg': PREREG, 'verdict': verdict}
    fc.save(OUT / 'result.json', result)
    print('P2809 VERDICT %s' % json.dumps(verdict), flush=True)


if __name__ == '__main__':
    main()
