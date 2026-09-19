"""Phase 2797 (LPF-10): dimension-level noun atlas -- what does EACH
embedding parameter do?

User directive upgrade: map the atlas structure of ALL nouns in one
system, down to the role of every embedding parameter.

Setting: the 100-noun x 2560-dim embedding slice (E30, stored in 2795
atlas.npz) and the same dims of the word-position state (H30).  Key
structural fact: the lens readout margin is EXACTLY additive over
embedding dims, because for a fixed word the RMSNorm denominator is a
constant:

  margin(w,c) = (1/rms_w) * sum_d E[w,d] * g[d] * dW[c,d]
  dW[c,d] = W_U[c,d] - mean_{c'!=c} W_U[c',d]

so each dimension's "carrier contribution" C(d; w, c) is exact, and
causal ablation (zero a dim -> recompute lens) needs NO forward pass.

Prereg (frozen before any readout):
  P-A  category_dims_exist iff >= 25 dims have eta^2(d) above the
       per-dim 99.5th percentile of a 500-permutation label-shuffle
       null (seed 2797) AND the significant dims' argmax category
       assignments cover >= 5 of the 10 categories.
  P-B  single_dim_causal_specificity iff for >= 8/10 of the top-10
       eta^2 dims: zeroing that dim in E changes the own-category
       margin of the dim's assigned category members MORE (|delta|)
       than of non-members (group means).
  P-C  top_dims_carry_prior_spectrum iff zeroing the top-30 eta^2
       dims shrinks the 10-class span of the E-side own-margin class
       means to <= 0.5 of the original span, AND zeroing 30 random
       non-top-30 dims (seed 2797) keeps span ratio >= 0.8.
  P-D  additive_decomposition_exact iff max |margin_decomp -
       margin_lens| over the full 100x10 table < 1e-3 (mathematical
       identity check of the carrier formula).
  D    descriptive: full eta^2 spectra for E and H + Spearman rho;
       significant dim list with assigned category and per-category
       carrier sums; PCA 2D projections of E30 and H30; top-30 E/H
       dim overlap; per-category carrier distribution over dims.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2797' / 'qwen4_dim_attribution'
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
WORDS = [w for v in CATS.values() for w in v]
CAT_OF = {w: c for c, v in CATS.items() for w in v}
CAT_WORDS = list(CATS.keys())
L_READ = 30

N_PERM = 500
SEED = 2797
TOP_BETA = 10
TOP_GAMMA = 30
N_RAND = 30

PREREG = {
    'P-A': 'category_dims_exist iff >= 25 dims with eta^2 above '
           'per-dim 99.5 pct of 500-perm shuffle null AND sig dims '
           'argmax categories cover >= 5/10',
    'P-B': 'single_dim_causal_specificity iff >= 8/10 top-10 eta^2 '
           'dims: |delta own margin| of assigned-cat members > '
           'non-members (group means) when dim zeroed in E',
    'P-C': 'top_dims_carry_prior_spectrum iff top-30 eta^2 dims '
           'zeroed -> E-side own-margin class-mean span ratio <= 0.5 '
           'AND 30 random non-top dims -> span ratio >= 0.8',
    'P-D': 'additive_decomposition_exact iff max |decomp - lens| '
           '< 1e-3 over 100x10',
    'verdict': 'parameter_map_established iff P-A AND P-B AND P-C AND '
               'P-D; distributed_encoding iff NOT P-A',
}


def unit(x):
    return x / max(np.linalg.norm(x), 1e-9)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def margins_from_E(Ew, W_U, g, eps, idx_cat):
    """Ew: (N,2560) -> own/cross margin table (N,10) via exact lens."""
    rms = np.sqrt(np.mean(Ew.astype(np.float64) ** 2, axis=1) + eps)
    Hn = (Ew.astype(np.float64) / rms[:, None]) * g[None, :]
    Z = Hn @ W_U[idx_cat].T.astype(np.float64)
    return Z - (Z.sum(1, keepdims=True) - Z) / 9.0, rms


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    data = np.load(SRC_NPZ, allow_pickle=True)
    words = [str(w) for w in data['words']]
    H30 = data['H30'].astype(np.float64)
    E30 = data['E30'].astype(np.float64)
    assert words == WORDS
    labels = np.array([CAT_WORDS.index(CAT_OF[w]) for w in WORDS])

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG, 'cats': CATS, 'words': WORDS,
                 'l_read': L_READ, 'n_perm': N_PERM, 'seed': SEED,
                 'top_beta': TOP_BETA, 'top_gamma': TOP_GAMMA,
                 'src_npz': str(SRC_NPZ)}
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

    tc = {}

    def tid(t):
        if t not in tc:
            ids = tok(' ' + t, add_special_tokens=False)['input_ids']
            if len(ids) != 1:
                ids = tok(t, add_special_tokens=False)['input_ids']
            assert len(ids) == 1, t
            tc[t] = int(ids[0])
        return tc[t]
    for t in WORDS + CAT_WORDS:
        tid(t)
    idx_cat = [tid(c) for c in CAT_WORDS]
    assert len(set(idx_cat)) == 10

    # torch lens reference for the sanity gate (same as 2795/2796 path)
    device = next(model.parameters()).device
    with torch.inference_mode():
        hn = final_norm(torch.tensor(E30, device=device).float())
        Zt = hn @ torch.tensor(W_U, device=device).T
    Zt = Zt.float().cpu().numpy()[:, idx_cat].astype(np.float64)
    M_torch = Zt - (Zt.sum(1, keepdims=True) - Zt) / 9.0
    src = json.loads(Path(SRC_RESULT).read_text(encoding='utf-8'))
    mce = src['margin_cat_emb']
    own = labels
    s_gate = max(abs(float(M_torch[i, own[i]]) - float(mce[w]))
                 for i, w in enumerate(WORDS))
    print('P2797 gate: torch lens own margin vs 2795 max|d|=%.6f'
          % s_gate, flush=True)
    assert s_gate < 0.05, 'lens path does not reproduce 2795'

    # ---------- P-D: exact additive decomposition ----------
    Wcat = W_U[idx_cat].astype(np.float64)
    dW = Wcat - (Wcat.sum(0, keepdims=True) - Wcat) / 9.0
    rmsE = np.sqrt(np.mean(E30 ** 2, axis=1) + eps)
    contrib = (E30 * g[None, :]) @ dW.T
    M_decomp = contrib / rmsE[:, None]
    err_pd = float(np.max(np.abs(M_decomp - M_torch)))
    p_d = bool(err_pd < 1e-3)
    print('P2797 P-D additive decomposition max err=%.2e -> %s'
          % (err_pd, p_d), flush=True)

    # ---------- Arm A: eta^2 census with permutation null ----------
    mu = E30.mean(0)
    sst = ((E30 - mu[None, :]) ** 2).sum(0)
    sst_H = None

    def ss_between(E, lab):
        onehot = np.zeros((len(lab), 10))
        onehot[np.arange(len(lab)), lab] = 1.0
        muc = (onehot.T @ E) / 10.0
        return ((muc - mu[None, :]) ** 2).sum(0) * 10.0

    eta2_E = ss_between(E30, labels) / sst
    eta2_H_mu = H30.mean(0)
    sst_H = ((H30 - eta2_H_mu[None, :]) ** 2).sum(0)
    eta2_H = ss_between(H30, labels) / sst_H
    rng = np.random.default_rng(SEED)
    null = np.zeros((N_PERM, 2560))
    for p in range(N_PERM):
        pl = labels[rng.permutation(100)]
        null[p] = ss_between(E30, pl) / sst
    q995 = np.quantile(null, 0.995, axis=0)
    sig = eta2_E > q995
    n_sig = int(sig.sum())
    # assigned category = argmax standardized class contrast
    zc = np.zeros((2560, 10))
    for c in range(10):
        inm = E30[labels == c].mean(0)
        outm = E30[labels != c].mean(0)
        sd = E30.std(0) + 1e-9
        zc[:, c] = (inm - outm) / sd
    dim_cat = np.where(sig, zc.argmax(1), -1)
    cov = len(set(int(v) for v in dim_cat if v >= 0))
    p_a = bool(n_sig >= 25 and cov >= 5)
    top = np.argsort(-eta2_E)[:12]
    print('P2797 eta2 top dims: %s' % json.dumps(
        [{'d': int(d), 'eta2': round(float(eta2_E[d]), 4),
          'cat': CAT_WORDS[int(zc[d].argmax())],
          'sig': bool(sig[d])} for d in top]), flush=True)
    print('P2797 n_sig=%d cov=%d/10 P-A=%s' % (n_sig, cov, p_a),
          flush=True)
    carrier = np.zeros((10, 2560))
    for c in range(10):
        ii = np.where(labels == c)[0]
        Cwc = E30[ii] * g[None, :] * dW[c][None, :]
        carrier[c] = np.abs(Cwc).mean(0)
    rho_EH = spearman(eta2_E, eta2_H)
    top30E = set(np.argsort(-eta2_E)[:TOP_GAMMA].tolist())
    top30H = set(np.argsort(-eta2_H)[:TOP_GAMMA].tolist())
    ov = len(top30E & top30H)
    print('P2797 eta2 E-H spearman=%.4f top30 overlap=%d'
          % (rho_EH, ov), flush=True)

    # ---------- Arm B: single-dim causal ablation ----------
    M0, _ = margins_from_E(E30, W_U, g, eps, idx_cat)
    max_err_np = float(np.max(np.abs(M0 - M_torch)))
    print('P2797 numpy lens vs torch max err=%.2e' % max_err_np,
          flush=True)
    order = np.argsort(-eta2_E)[:TOP_BETA]
    b_rows = []
    n_b_ok = 0
    for d in order:
        Ez = E30.copy()
        Ez[:, int(d)] = 0.0
        Mz, _ = margins_from_E(Ez, W_U, g, eps, idx_cat)
        dM = np.abs(M0[np.arange(100), own]
                    - Mz[np.arange(100), own])
        cd = int(zc[int(d)].argmax())
        din = float(dM[labels == cd].mean())
        dout = float(dM[labels != cd].mean())
        ok = bool(din > dout)
        n_b_ok += int(ok)
        b_rows.append({'dim': int(d), 'eta2': float(eta2_E[d]),
                       'cat': CAT_WORDS[cd], 'delta_in': din,
                       'delta_out': dout, 'specific': ok})
        print('P2797 dim %d (%s): |dMargin| in=%.4f out=%.4f %s'
              % (int(d), CAT_WORDS[cd], din, dout, ok), flush=True)
    p_b = bool(n_b_ok >= 8)
    print('P2797 P-B=%s (%d/10)' % (p_b, n_b_ok), flush=True)

    # ---------- Arm C: top-30 carry the prior spectrum ----------
    span_E0 = float(np.max([M0[labels == c, c].mean()
                            for c in range(10)])
                    - np.min([M0[labels == c, c].mean()
                              for c in range(10)]))
    print('P2797 E-side own-margin class-mean spectrum: %s'
          % json.dumps({CAT_WORDS[c]: round(
              float(M0[labels == c, c].mean()), 3)
              for c in range(10)}), flush=True)
    topg = np.argsort(-eta2_E)[:TOP_GAMMA]
    Ez = E30.copy()
    Ez[:, topg] = 0.0
    Mz, _ = margins_from_E(Ez, W_U, g, eps, idx_cat)
    span_Ez = float(np.max([Mz[labels == c, c].mean()
                            for c in range(10)])
                    - np.min([Mz[labels == c, c].mean()
                              for c in range(10)]))
    pool = np.setdiff1d(np.arange(2560), topg)
    rdim = rng.choice(pool, N_RAND, replace=False)
    Er = E30.copy()
    Er[:, rdim] = 0.0
    Mr, _ = margins_from_E(Er, W_U, g, eps, idx_cat)
    span_Er = float(np.max([Mr[labels == c, c].mean()
                            for c in range(10)])
                    - np.min([Mr[labels == c, c].mean()
                              for c in range(10)]))
    r_top = span_Ez / span_E0
    r_rand = span_Er / span_E0
    p_c = bool(r_top <= 0.5 and r_rand >= 0.8)
    print('P2797 span_E0=%.3f top30=%.3f (ratio %.3f) rand30=%.3f '
          '(ratio %.3f) P-C=%s'
          % (span_E0, span_Ez, r_top, span_Er, r_rand, p_c),
          flush=True)

    # ---------- Arm D: PCA projections ----------
    for name, X in (('E', E30), ('H', H30)):
        Xc = X - X.mean(0)
        s, v, vt = np.linalg.svd(Xc, full_matrices=False)
        proj = Xc @ vt[:2].T
        evr = (v[:2] ** 2) / (v ** 2).sum()
        if name == 'E':
            proj_E, evr_E = proj, evr
        else:
            proj_H, evr_H = proj, evr
        print('P2797 PCA %s evr=%.4f,%.4f' % (name, evr[0], evr[1]),
              flush=True)

    verdict = {
        'category_dims_exist': p_a, 'n_sig_dims': n_sig,
        'cat_coverage': cov,
        'single_dim_causal_specificity': p_b, 'n_b_ok': n_b_ok,
        'b_rows': b_rows,
        'top_dims_carry_prior_spectrum': p_c,
        'span_E0': span_E0, 'span_ratio_top30': r_top,
        'span_ratio_rand30': r_rand,
        'additive_decomposition_exact': p_d, 'err_pd': err_pd,
        'eta2_spearman_E_H': rho_EH, 'top30_overlap_E_H': ov,
        'parameter_map_established': bool(p_a and p_b and p_c and p_d),
        'distributed_encoding': bool(not p_a),
    }
    result = {'phase': 2797, 'prereg': PREREG, 'verdict': verdict,
              'sig_dims': [{'dim': int(d), 'eta2': float(eta2_E[d]),
                            'cat': CAT_WORDS[int(dim_cat[d])],
                            'carrier_own': float(carrier[
                                int(dim_cat[d]), d])}
                           for d in np.where(sig)[0]]}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'dim_attribution.npz',
           eta2_E=eta2_E, eta2_H=eta2_H, q995=q995,
           dim_cat=dim_cat, carrier=carrier,
           proj_E2=proj_E, proj_H2=proj_H,
           evr_E=evr_E, evr_H=evr_H,
           labels=labels)
    print('P2797 VERDICT %s' % json.dumps(
        {k: v for k, v in verdict.items() if k != 'b_rows'}),
        flush=True)


if __name__ == '__main__':
    main()
