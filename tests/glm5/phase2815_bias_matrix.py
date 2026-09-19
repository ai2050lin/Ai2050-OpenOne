"""Phase 2815 (LPF-28): BIAS-MATRIX KNOWLEDGE STRUCTURE — does the
low-dim PC-coordinate space carry the knowledge network?

2814 falsified micro-field NEIGHBOURS in the 2560-dim residual (281/282
singleton clusters) and reinterpreted 2813's PC readouts as weak signed
gradient axes.  GPS model: each word's knowledge address = its readings
along a few gradient axes = coordinates in the TOP-PC bias matrix
B = [u_w,k * s_k] (283 x K).  If knowledge lives in B, same-class words
must cluster IN B-SPACE (while they do not in the full residual).

Zero-forward; gates: dW vs 2807 <1e-6, eval_words exact vs 2811,
Z_all vs 2813 residual.npz <1e-4.  Residual construction identical to
2813.  SVD of residual -> top-60 PCs; B_K for K in {10,20,50}.

Stats (eval words only, classes with n>=5):
  within-class mean pairwise cos (weighted by pair counts)
  k-NN purity (k=10, leave-self-out)
Nulls: label shuffle x1000 preserving class sizes (both stats).

Prereg (frozen before any readout):
  P-B1  bias_gradient_real iff within-class cos (B, K=50) > shuffle
        q95 AND >= 2 x global mean pairwise cos in B
  P-B2  bias_knn_readable iff kNN purity (B, K=50) > shuffle q95 AND
        >= 0.30 (record chance level)
  verdict: bias_matrix_carries_knowledge = P-B1 AND P-B2
  (K=10/20 curves = descriptive robustness, not criteria)
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
OUT = BASE / 'phase2815' / 'bias_matrix'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SRC_2807 = BASE / 'phase2807' / 'qwen4_heldout'
SRC_2811 = BASE / 'phase2811' / 'noun_superposition'
SRC_2813 = BASE / 'phase2813' / 'residual_geometry'
SEED = 2815
N_SHUFFLE = 1000
K_PREREG = 50
K_CURVE = [10, 20, 50]
KNN_K = 10

PREREG = {
    'P-B1': 'bias_gradient_real iff within-class mean pairwise cos '
            '(B, K=50, eval, weighted) > label-shuffle q95 AND >= 2 x '
            'global mean pairwise cos in B',
    'P-B2': 'bias_knn_readable iff kNN purity (k=10, leave-self-out, '
            'B K=50) > shuffle q95 AND >= 0.30',
    'verdict': 'bias_matrix_carries_knowledge = P-B1 AND P-B2; '
               'K=10/20 curves descriptive',
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
                 'n_shuffle': N_SHUFFLE, 'k_prereg': K_PREREG,
                 'k_curve': K_CURVE, 'knn_k': KNN_K,
                 'note': 'GPS-model test: knowledge address = word '
                         'coordinates along top PCs of the class-removed '
                         'residual; clustering must appear in B-space '
                         'even though it is absent in the full residual'}
    fc.save(OUT / 'execution.json', execution)

    # ---------- gates + residual ----------
    h7 = np.load(SRC_2807 / 'heldout.npz')
    b11 = np.load(SRC_2811 / 'battery.npz')
    b13 = np.load(SRC_2813 / 'residual.npz')
    Z_all = b13['Z_all'].astype(np.float64)
    ytrue = b13['ytrue'].astype(int)
    atlas_labels = np.array([ci for ci, c in enumerate(CAT_WORDS)
                             for _ in CATS[c]])
    n_atlas = len(atlas_labels)
    eval_words = res2811['eval_words']
    atlas_words = [w for v in CATS.values() for w in v]
    all_words = atlas_words + eval_words
    assert len(all_words) == Z_all.shape[0] == 283
    # rebuild dW from lm_head rows via 2807 reference for gate
    dW_ref = h7['dW_class'].astype(np.float64)
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
    except KeyError:
        Wu = Etab
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

    Z_eval = np.stack([zrow_local(Etab, g, eps, tid, w)
                       for w in eval_words])
    gate_Z = float(np.abs(Z_eval - b11['Z_eval'].astype(np.float64)).max())
    gate_Zall = float(np.abs(Z_all - b13['Z_all'].astype(np.float64)).max())
    print('P2815 gates: Zeval_vs_2811=%.2e Zall_vs_2813=%.2e'
          % (gate_Z, gate_Zall), flush=True)
    assert gate_Z < 1e-4 and gate_Zall < 1e-4

    unitD = np.stack([unit(dW_ref[i]) for i in range(10)])
    classQ, _ = np.linalg.qr(unitD.T)
    Zc = Z_all - Z_all.mean(0, keepdims=True)
    resid = Zc - (Zc @ classQ) @ classQ.T
    U, s, Vt = np.linalg.svd(resid, full_matrices=False)

    y_eval = ytrue
    classes = [c for c in range(10) if int((y_eval == c).sum()) >= 5]

    def within_stat(Bk, y):
        tot, wsum = 0.0, 0.0
        Uc = Bk / np.maximum(np.linalg.norm(Bk, axis=1, keepdims=True),
                             1e-30)
        pairs_tot = 0
        for c in classes:
            idx = np.where(y == c)[0]
            m = len(idx)
            if m < 2:
                continue
            S = Uc[idx]
            G = S @ S.T
            iu = np.triu_indices(m, 1)
            tot += float(G[iu].mean()) * len(iu[0])
            wsum += len(iu[0])
            pairs_tot += len(iu[0])
        return tot / max(wsum, 1)

    def global_stat(Bk):
        Uc = Bk / np.maximum(np.linalg.norm(Bk, axis=1, keepdims=True),
                             1e-30)
        y_e = y_eval
        G = Uc @ Uc.T
        iu = np.triu_indices(len(y_e), 1)
        return float(G[iu].mean())

    def knn_purity(Bk, y):
        Uc = Bk / np.maximum(np.linalg.norm(Bk, axis=1, keepdims=True),
                             1e-30)
        G = Uc @ Uc.T
        np.fill_diagonal(G, -1e30)
        hits = 0
        for i in range(len(y)):
            nb = np.argsort(-G[i])[:KNN_K]
            hits += int((y[nb] == y[i]).sum())
        return hits / (len(y) * KNN_K)

    results_curve = {}
    stat50 = base50 = pur50 = None
    for K in K_CURVE:
        Bk = U[:len(y_eval), :K] * s[:K]
        stat = within_stat(Bk, y_eval)
        base = global_stat(Bk)
        pur = knn_purity(Bk, y_eval)
        nulls_stat, nulls_pur = [], []
        for _ in range(N_SHUFFLE):
            ys = y_eval.copy()
            ys[rng.permutation(len(ys))] = \
                y_eval[rng.permutation(len(y_eval))]
            nulls_stat.append(within_stat(Bk, ys))
            nulls_pur.append(knn_purity(Bk, ys))
        q95s = float(np.quantile(nulls_stat, 0.95))
        q95p = float(np.quantile(nulls_pur, 0.95))
        chance = float(sum(((y_eval == c).mean()) ** 2
                           for c in range(10)))
        results_curve[str(K)] = {
            'within_cos': round(stat, 4), 'global_cos': round(base, 4),
            'null_within_q95': round(q95s, 4),
            'knn_purity': round(pur, 4), 'knn_chance': round(chance, 4),
            'null_purity_q95': round(q95p, 4)}
        if K == K_PREREG:
            stat50, base50, pur50, q95s50, q95p50, chance50 = \
                stat, base, pur, q95s, q95p, chance
        print('P2815 K=%d within=%.4f base=%.4f q95=%.4f purity=%.4f '
              'chance=%.4f q95p=%.4f'
              % (K, stat, base, q95s, pur, chance, q95p), flush=True)

    p_b1 = bool(stat50 > q95s50 and stat50 >= 2 * base50)
    p_b2 = bool(pur50 > q95p50 and pur50 >= 0.30)

    per_class = {}
    Bk = U[:len(y_eval), :K_PREREG] * s[:K_PREREG]
    Uc = Bk / np.maximum(np.linalg.norm(Bk, axis=1, keepdims=True), 1e-30)
    for c in classes:
        idx = np.where(y_eval == c)[0]
        if len(idx) < 2:
            continue
        G = Uc[idx] @ Uc[idx].T
        iu = np.triu_indices(len(idx), 1)
        per_class[CAT_WORDS[c]] = {'n': int(len(idx)),
                                   'within_cos': round(float(G[iu].mean()),
                                                       4)}
    print('P2815 per-class within-B cos %s' % json.dumps(per_class),
          flush=True)

    verdict = {
        'k_prereg': K_PREREG,
        'within_cos': round(stat50, 4), 'global_cos': round(base50, 4),
        'null_within_q95': round(q95s50, 4),
        'knn_purity': round(pur50, 4), 'knn_chance': round(chance50, 4),
        'null_purity_q95': round(q95p50, 4),
        'bias_gradient_real': p_b1, 'bias_knn_readable': p_b2,
        'curve': results_curve, 'per_class_within_cos': per_class,
        'bias_matrix_carries_knowledge': bool(p_b1 and p_b2),
    }
    result = {'phase': 2815, 'prereg': PREREG, 'verdict': verdict,
              'eval_words': eval_words}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'bias.npz', B50=Bk.astype(np.float32),
           Vt60=Vt[:60].astype(np.float32),
           spectrum=(s ** 2 / (s ** 2).sum()).astype(np.float32))

    elapsed = time.monotonic() - t0
    cc.ledger('phase2815', elapsed)
    print('P2815 VERDICT %s' % json.dumps(verdict), flush=True)
    print('P2815 elapsed %.1fs' % elapsed, flush=True)


def zrow_local(Etab, g, eps, tid, t):
    e = Etab[tid(t)].astype(np.float64)
    return e / np.sqrt((e ** 2).mean() + eps) * g


if __name__ == '__main__':
    main()
