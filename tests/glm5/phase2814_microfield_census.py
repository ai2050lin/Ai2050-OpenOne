"""Phase 2814 (LPF-27): MICRO-FIELD CENSUS — is the 91% residual a
knowledge-network patchwork?

2813: residual spectrum flat (top-20 PC = 17.9% variance) BUT top PCs
read out as semantic micro-fields (countries, chemical elements,
European countries, Asian food, sea geography, hardware) with
cross-lingual unembed tokens.  Now the decisive test: do micro-fields
COVER the residual population?

Arm M  leader clustering (deterministic battery order) on the 283-word
       class-removed centered residual, tau=0.45 merge threshold;
       coverage: word covered iff cos(resid_w, leave-one-out field
       mean) >= 0.50; singleton fields = uncovered.
       N1 random-assignment null x1000 (sizes preserved, sanity ~0)
       N2 random-token control: 250 random vocab residuals, same
       clustering + coverage (knowledge-specificity margin)
Arm D  diagnostic: metal->fruit mediation — within the 14 metal eval
       words, Spearman rho between element-field alignment
       (cos to loo metal-residual mean) and fruit-direction projection
       (unit dW_fruit); furniture words as control class.

Zero-forward, 2808 protocol (tie-aware); battery identical to 2811
(gates: dW vs 2807 <1e-6, eval_words exact vs 2811, Z vs 2811
battery.npz <1e-4, Z_all vs 2813 residual.npz <1e-4).

Prereg (frozen before any readout):
  P-M1  microfield_census_real iff coverage(all 283, tau=0.45,
        theta=0.50, singletons uncovered) >= 0.70
  P-M2  knowledge_specific iff coverage_eval183 - coverage_rand250
        >= 0.15
  P-D1  fruit_metal_mediation (diagnostic only) iff Spearman rho >= 0.5
  verdict: knowledge_network_coordinates = P-M1 AND P-M2
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
OUT = BASE / 'phase2814' / 'microfield_census'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SRC_2807 = BASE / 'phase2807' / 'qwen4_heldout'
SRC_2811 = BASE / 'phase2811' / 'noun_superposition'
SRC_2813 = BASE / 'phase2813' / 'residual_geometry'
SEED = 2814
N_SHUFFLE = 1000
N_RAND = 250
TAU = 0.45
THETA = 0.50
TAU_CURVE = [0.35, 0.45, 0.55]

PREREG = {
    'P-M1': 'microfield_census_real iff coverage(all 283, tau=0.45, '
            'theta=0.50, singleton fields uncovered) >= 0.70',
    'P-M2': 'knowledge_specific iff coverage_eval183 - '
            'coverage_rand250 >= 0.15',
    'P-D1': 'fruit_metal_mediation (diagnostic only) iff Spearman rho '
            '(metal words: element-field alignment vs fruit-direction '
            'projection) >= 0.5',
    'verdict': 'knowledge_network_coordinates = P-M1 AND P-M2; '
               'P-D1 diagnosis only',
}


def unit(x):
    return x / max(np.linalg.norm(x), 1e-30)


def cos(a, b):
    return float(unit(a) @ unit(b))


def leader_cluster(R, tau):
    n = R.shape[0]
    leaders = []
    for i in range(n):
        best, best_c = None, tau
        for li, mem in enumerate(leaders):
            c = cos(R[i], np.mean(R[mem], axis=0))
            if c > best_c:
                best, best_c = li, c
        if best is None:
            leaders.append([i])
        else:
            leaders[best].append(i)
    return leaders


def coverage_mask(R, leaders, theta):
    n = R.shape[0]
    cov = np.zeros(n, dtype=bool)
    for mem in leaders:
        if len(mem) < 2:
            continue
        S = R[mem]
        for j in range(len(mem)):
            mu = (S.sum(0) - S[j]) / (len(mem) - 1)
            if cos(R[mem[j]], mu) >= theta:
                cov[mem[j]] = True
    return cov


def null_coverage(R, sizes, rng, theta):
    perm = rng.permutation(R.shape[0])
    idx = 0
    hits = 0
    for m in sizes:
        grp = perm[idx:idx + m]
        idx += m
        if m < 2:
            continue
        S = R[grp]
        for j in range(m):
            mu = (S.sum(0) - S[j]) / (m - 1)
            if cos(R[grp[j]], mu) >= theta:
                hits += 1
    return hits / R.shape[0]


def spearman(x, y):
    def rank(v):
        order = np.argsort(v)
        r = np.empty(len(v))
        r[order] = np.arange(len(v), dtype=float)
        vals, inv, cnt = np.unique(v, return_inverse=True,
                                   return_counts=True)
        for k, c in enumerate(cnt):
            if c > 1:
                m = inv == k
                r[m] = r[m].mean()
        return r
    rx, ry = rank(np.asarray(x, float)), rank(np.asarray(y, float))
    rx -= rx.mean()
    ry -= ry.mean()
    return float((rx * ry).sum()
                 / max(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()), 1e-30))


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
                 'n_shuffle': N_SHUFFLE, 'n_rand': N_RAND,
                 'tau': TAU, 'theta': THETA, 'tau_curve': TAU_CURVE,
                 'note': 'leader clustering on 283-word class-removed '
                         'centered residual, deterministic battery '
                         'order 0..282; coverage via leave-one-out '
                         'field means; N2 = random-token control'}
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
    atlas_labels = np.array([ci for ci, c in enumerate(CAT_WORDS)
                             for _ in CATS[c]])

    # ---------- gates ----------
    assert eval_words == res2811['eval_words'], 'eval battery drift'
    h7 = np.load(SRC_2807 / 'heldout.npz')
    cent = {c: np.stack([Wu[tid(w)].astype(np.float64)
                         for w in CATS[c]]).mean(0) for c in CAT_WORDS}
    Cm = np.stack([cent[c] for c in CAT_WORDS])
    dW = Cm - (Cm.sum(0, keepdims=True) - Cm) / 9.0
    gate_dW = float(np.abs(dW - h7['dW_class'].astype(np.float64)).max())
    b11 = np.load(SRC_2811 / 'battery.npz')
    b13 = np.load(SRC_2813 / 'residual.npz')

    def zrow(t):
        e = Etab[tid(t)].astype(np.float64)
        return e / np.sqrt((e ** 2).mean() + eps) * g

    def zrow_id(i):
        e = Etab[i].astype(np.float64)
        return e / np.sqrt((e ** 2).mean() + eps) * g

    Z_eval = np.stack([zrow(w) for w in eval_words])
    Z_atlas = np.stack([zrow(w) for w in atlas_words])
    Z_all = np.vstack([Z_atlas, Z_eval])
    all_words = atlas_words + eval_words
    gate_Z = float(np.abs(Z_eval - b11['Z_eval'].astype(np.float64)).max())
    gate_Zall = float(np.abs(Z_all - b13['Z_all'].astype(np.float64))
                      .max())
    print('P2814 gates: dW=%.2e Zeval=%.2e Zall_vs_2813=%.2e tie=%s'
          % (gate_dW, gate_Z, gate_Zall, tie), flush=True)
    assert gate_dW < 1e-6 and gate_Z < 1e-4 and gate_Zall < 1e-4

    unitD = np.stack([unit(dW[i]) for i in range(10)])
    classQ, _ = np.linalg.qr(unitD.T)
    Zc = Z_all - Z_all.mean(0, keepdims=True)
    resid = Zc - (Zc @ classQ) @ classQ.T

    # ---------- Arm M: leader clustering + coverage ----------
    leaders = leader_cluster(resid, TAU)
    sizes = sorted((len(m) for m in leaders), reverse=True)
    cov = coverage_mask(resid, leaders, THETA)
    coverage_all = float(cov.mean())
    cov_eval = float(cov[len(atlas_words):].mean())
    n_fields = len(leaders)
    n_multi = sum(1 for m in leaders if len(m) >= 2)
    print('P2814 fields=%d multi=%d largest=%s coverage_all=%.3f '
          'coverage_eval=%.3f'
          % (n_fields, n_multi, sizes[:8], coverage_all, cov_eval),
          flush=True)

    cov_curve = {}
    for tau2 in TAU_CURVE:
        ld2 = leader_cluster(resid, tau2)
        c2 = coverage_mask(resid, ld2, THETA)
        cov_curve[str(tau2)] = {'n_fields': len(ld2),
                                'coverage_all': round(float(c2.mean()), 4),
                                'coverage_eval': round(
                                    float(c2[len(atlas_words):].mean()),
                                    4)}
    print('P2814 coverage curve %s' % json.dumps(cov_curve), flush=True)

    null_covs = np.array([null_coverage(resid, sizes, rng, THETA)
                          for _ in range(N_SHUFFLE)])
    print('P2814 N1 random-assignment coverage mean=%.4f q95=%.4f'
          % (float(null_covs.mean()), float(np.quantile(null_covs, 0.95))),
          flush=True)

    # ---------- N2: random-token control ----------
    all_tids = set(tc.values())
    rand_ids = []
    while len(rand_ids) < N_RAND:
        r = int(rng.integers(0, Etab.shape[0]))
        if r > 0 and r not in all_tids and r not in rand_ids:
            rand_ids.append(r)
    Z_rand = np.stack([zrow_id(i) for i in rand_ids])
    Zr_c = Z_rand - Z_all.mean(0, keepdims=True)
    resid_rand = Zr_c - (Zr_c @ classQ) @ classQ.T
    leaders_rand = leader_cluster(resid_rand, TAU)
    cov_rand = coverage_mask(resid_rand, leaders_rand, THETA)
    coverage_rand = float(cov_rand.mean())
    margin = cov_eval - coverage_rand
    p_m2 = bool(margin >= 0.15)
    print('P2814 N2 random-token coverage=%.3f margin_eval=%.3f P-M2=%s'
          % (coverage_rand, margin, p_m2), flush=True)

    # ---------- field readout table ----------
    fields_out = []
    for mem in sorted((m for m in leaders if len(m) >= 2),
                      key=len, reverse=True):
        S = resid[mem]
        mu = S.mean(0)
        pw = [cos(S[i], S[j]) for i in range(len(mem))
              for j in range(i + 1, len(mem))]
        top_ids = np.argsort(-(Wu @ mu))[:8].tolist()
        toks = [tok.decode([t]).strip() for t in top_ids]
        members = [all_words[i] for i in mem]
        fields_out.append({
            'size': len(mem), 'members': members,
            'mean_pw_cos': round(float(np.mean(pw)), 3),
            'top_unembed': toks,
            'class_mix': sorted({CAT_WORDS[int(ytrue[i - len(atlas_words)])]
                                 for i in mem
                                 if i >= len(atlas_words)} |
                                {CAT_WORDS[int(atlas_labels[i])]
                                 for i in mem if i < len(atlas_words)}),
        })
    print('P2814 top fields %s'
          % json.dumps(fields_out[:12]), flush=True)

    # ---------- Arm D: metal->fruit mediation (diagnostic) ----------
    fi = CAT_WORDS.index('fruit')
    dfu = unit(dW[fi])
    d_eval = resid[len(atlas_words):]

    def mediation(cls_idx, n_cls):
        idxs = np.where(ytrue == cls_idx)[0]
        if len(idxs) < 5:
            return None
        S = d_eval[idxs]
        align = [cos(S[j], (S.sum(0) - S[j]) / (len(idxs) - 1))
                 for j in range(len(idxs))]
        proj = [float(d_eval[i] @ dfu) for i in idxs]
        return round(spearman(align, proj), 4)

    rho_metal = mediation(CAT_WORDS.index('metal'), 14)
    rho_furn = mediation(CAT_WORDS.index('furniture'), 13)
    p_d1 = bool(rho_metal is not None and rho_metal >= 0.5)
    print('P2814 Arm D rho_metal=%s rho_furniture=%s P-D1=%s'
          % (rho_metal, rho_furn, p_d1), flush=True)

    # ---------- verdict ----------
    p_m1 = bool(coverage_all >= 0.70)
    verdict = {
        'n_words': len(all_words), 'n_eval': len(eval_words),
        'n_rand': N_RAND,
        'n_fields': n_fields, 'n_multi_fields': n_multi,
        'field_size_head': sizes[:12],
        'coverage_all': round(coverage_all, 4),
        'coverage_eval': round(cov_eval, 4),
        'coverage_curve': cov_curve,
        'n1_null_mean': round(float(null_covs.mean()), 4),
        'n1_null_q95': round(float(np.quantile(null_covs, 0.95)), 4),
        'coverage_rand': round(coverage_rand, 4),
        'margin_eval_minus_rand': round(margin, 4),
        'microfield_census_real': p_m1, 'knowledge_specific': p_m2,
        'fields': fields_out,
        'rand_ids': rand_ids,
        'rho_metal_fruit': rho_metal, 'rho_furniture_fruit': rho_furn,
        'fruit_metal_mediation': p_d1,
        'knowledge_network_coordinates': bool(p_m1 and p_m2),
    }
    result = {'phase': 2814, 'prereg': PREREG, 'verdict': verdict,
              'eval_words': eval_words}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'fields.npz', resid=resid.astype(np.float32),
           cov=cov.astype(np.float32), ytrue=ytrue,
           atlas_labels=atlas_labels)

    elapsed = time.monotonic() - t0
    cc.ledger('phase2814', elapsed)
    print('P2814 VERDICT %s' % json.dumps(verdict), flush=True)
    print('P2814 elapsed %.1fs' % elapsed, flush=True)


if __name__ == '__main__':
    main()
