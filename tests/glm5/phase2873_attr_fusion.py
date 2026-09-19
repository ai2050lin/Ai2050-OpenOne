"""Phase 2873 (LPF MA2 cont): attr-axis fusion third point -- the
growth curve's point 3 on the density-gated protocol (2869).

Context: 2869 established density-matched fusion F(a)=unit([B3, a*B2s])
with a*=0.25, acc 0.8875 (growth v3 = 0.10, boundary).  2870 proved the
attribute axis is geometrically independent of the class axis in unembed
space (P1 max cos 0.0755 < random baseline; P3 orth energy 0.995).
2872 proved mechanism-side separation.  The open question: does the
attribute axis add CLASS-incremental information at the fusion level
(growth curve point 3), i.e. is the two-axis atlas closed at readout?

Blocks (80 words, SEED=2855 vocab, zero forward for rep block):
  B_cf       density-gated class fusion = unit([B3u, 0.25*B2s]), 10+43-d
             [2867 npz + 2868 sig mask; a* frozen from 2869]
  B_attr_rep 15-d attribute profile: proj[w,a] = unit(E_w) . D[a]
             [D frozen from 2870 npz; E rows recomputed from
             safetensors embed_tokens, deterministic tids]
  B_attr_h   15-d hidden-side profile: proj[w,a] = unit(h_w) . D[a]
             [h_w = last hidden state, qwen3-4b single-token forward,
             no context; one model load on CUDA]

Prereg (frozen before any readout):
  A0  anchor check: acc(B_cf) == 0.8875 (2869 reproduction, exact LOO
      procedure); mismatch => protocol_broken, abort interpretation.
  A1  (main) F_rep(a) = unit([B_cf, a*B_attr_rep_u]), a grid
      {0.25,0.5,1,2,4}; a* = argmax acc; fusion_gain iff
      acc(F_rep(a*)) > acc(B_cf) AND acc(F_rep(a*)) > max-alpha null
      p95 (200 label permutations, SEED=2873, each perm takes its own
      max over the grid).  else attr_no_class_gain.
  A2  growth v4 = (acc(F_rep(a*)) - acc(B_cf)) / max(1-acc(B_cf),1e-9);
      <0.1 sublinear_reuse (two-axis closure at readout),
      >=0.1 additive_gain (only meaningful if A1 true).
  A3  acc(B_attr_rep_u) vs null p95 (same perm engine):
      attr_axis_class_leak / attr_axis_class_clean.
  A4  class margin of cos(B_attr_rep) vs null p95:
      attr_profile_class_correlated / attr_profile_class_uncorrelated.
  A5  hidden side: F_h(a) = unit([B_cf, a*B_attr_h_u]) same protocol
      (SEED=28731); fusion_gain_h iff same rule; geometry
      replicability: Spearman(utri cos(B_attr_rep), utri cos(B_attr_h))
      > 0.5 => attr_geometry_hidden_preserved.
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
OUT = BASE / 'phase2873' / 'attr_fusion'
SRC_2867 = BASE / 'phase2867' / 'word_coords_v1' / 'word_coords_v1.npz'
SRC_2868 = BASE / 'phase2868' / 'growth_v2' / 'growth_v2.npz'
SRC_2870 = BASE / 'phase2870' / 'attr_axis_pilot' / 'attr_axis_pilot.npz'
MODEL_DIR = Path(r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b')
SEED = 2873
SEED_H = 28731
N_PERM = 200
ALPHAS = (0.25, 0.5, 1.0, 2.0, 4.0)
ALPHA_CF = 0.25
ANCHOR_ACC = 0.8875

PREREG = {
    'A0': 'acc(B_cf) == 0.8875 anchor (2869 reproduction) else '
          'protocol_broken',
    'A1': 'F_rep(a)=unit([B_cf, a*B_attr_rep_u]); a*=argmax; '
          'fusion_gain iff acc(F_rep(a*)) > acc(B_cf) and > max-alpha '
          'null p95 (200 perms, SEED=2873) else attr_no_class_gain',
    'A2': 'growth v4 = (acc*-acc(B_cf))/(1-acc(B_cf)); <0.1 '
          'sublinear_reuse else additive_gain',
    'A3': 'acc(B_attr_rep_u) vs null p95 => attr_axis_class_leak / '
          'attr_axis_class_clean',
    'A4': 'class margin of cos(B_attr_rep) vs null p95 => '
          'attr_profile_class_correlated / uncorrelated',
    'A5': 'F_h same protocol (SEED=28731); fusion_gain_h; geometry '
          'replicability Spearman(utri cos rep, utri cos h) > 0.5 => '
          'attr_geometry_hidden_preserved',
}


def unit(x):
    return x / max(float(np.linalg.norm(x)), 1e-30)


def loo_nn_acc(C, labels):
    n = len(labels)
    C = C.copy()
    np.fill_diagonal(C, -2.0)
    a = 0
    for i in range(n):
        j = int(np.argmax(C[i]))
        a += int(labels[j] == labels[i])
    return a / n


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    d = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def fusion_scan(base, attr, labels, seed):
    """Density-gated fusion scan + max-alpha null. Returns verdict dict."""
    n = len(labels)
    base_u = np.stack([unit(base[i]) for i in range(n)])
    attr_u = np.stack([unit(attr[i]) for i in range(n)])

    def fused(a):
        raw = np.hstack([base_u, a * attr_u])
        return np.stack([unit(raw[i]) for i in range(n)])

    accs = {}
    Cmats = {}
    for a in ALPHAS:
        C = fused(a) @ fused(a).T
        Cmats[a] = C
        accs[a] = loo_nn_acc(C, labels)
    C0 = base_u @ base_u.T
    acc0 = loo_nn_acc(C0, labels)

    a_star = max(ALPHAS, key=lambda a: accs[a])
    acc_star = accs[a_star]

    rng = np.random.default_rng(seed)
    null_max = []
    for _ in range(N_PERM):
        pl = rng.permutation(labels)
        best = -1.0
        for a in ALPHAS:
            best = max(best, loo_nn_acc(Cmats[a], pl))
        null_max.append(best)
    nm_p95 = float(np.percentile(null_max, 95))

    gain = bool(acc_star > acc0 and acc_star > nm_p95)
    headroom = max(1.0 - acc0, 1e-9)
    growth = (acc_star - acc0) / headroom
    return {
        'acc_base': round(acc0, 4),
        'acc_by_alpha': {str(a): round(accs[a], 4) for a in ALPHAS},
        'alpha_star': a_star,
        'acc_star': round(acc_star, 4),
        'null_max_p95': round(nm_p95, 4),
        'gain': gain,
        'growth': round(growth, 4),
        'growth_label': 'sublinear_reuse' if growth < 0.1
                        else 'additive_gain',
    }, C0, base_u, attr_u


def main():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'result.json').exists():
        raise RuntimeError('result.json exists; delete execution.json and '
                           'result.json before re-run (immutability rule)')

    execution_path = OUT / 'execution.json'
    if not execution_path.exists():
        execution = {'timestamp': fc.stamp(),
                     'source': cc.snapshot(__file__),
                     'prereg': PREREG, 'seed': SEED,
                     'design': 'attr-axis fusion third point: B_attr_rep '
                               '(15-d, frozen 2870 directions) and '
                               'B_attr_h (15-d, qwen4 last hidden state) '
                               'fused into 2869 density-gated class '
                               'fusion B_cf; max-alpha null'}
        fc.save(execution_path, execution)

    # ---------- frozen inputs ----------
    z67 = np.load(SRC_2867, allow_pickle=True)
    B2 = z67['B2'].astype(np.float64)
    B3 = z67['B3'].astype(np.float64)
    labels = z67['labels'].astype(np.int64)
    n_words = len(labels)
    z68 = np.load(SRC_2868, allow_pickle=True)
    sig = z68['sig_mask'].astype(bool)
    assert sig.sum() == 43
    z70 = np.load(SRC_2870, allow_pickle=True)
    D = z70['D'].astype(np.float64)              # (15, 2560) frozen
    labels70 = z70['labels'].astype(np.int64)
    assert np.array_equal(labels, labels70), 'word order mismatch 2867/2870'
    n_attr = D.shape[0]

    B2s = np.stack([unit(B2[i][sig]) for i in range(n_words)])
    B3u = np.stack([unit(B3[i]) for i in range(n_words)])
    raw_cf = np.hstack([B3u, ALPHA_CF * B2s])
    B_cf = np.stack([unit(raw_cf[i]) for i in range(n_words)])

    # ---------- B_attr_rep (zero forward) ----------
    from transformers import AutoTokenizer
    from safetensors import safe_open
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    def single_form(word):
        for form in (' %s' % word, word):
            ids = tok.encode(form, add_special_tokens=False)
            if len(ids) == 1:
                return form, int(ids[0])
        return None, None

    words = json.loads((BASE / 'phase2867' / 'word_coords_v1'
                        / 'result.json').read_text(encoding='utf-8'))['words']
    assert len(words) == n_words
    tids = []
    for w in words:
        form, tid = single_form(w)
        assert tid is not None, 'multi-token: %s' % w
        tids.append(tid)

    idx = json.loads((MODEL_DIR / 'model.safetensors.index.json')
                     .read_text(encoding='utf-8'))
    shard = idx['weight_map']['model.embed_tokens.weight']
    with safe_open(str(MODEL_DIR / shard), framework='pt',
                   device='cpu') as f:
        emb_sel = f.get_tensor('model.embed_tokens.weight')[tids]
    E_rows = emb_sel.float().numpy().astype(np.float64)
    E_unit = np.stack([unit(E_rows[i]) for i in range(n_words)])
    B_attr_rep = E_unit @ D.T                    # (80, 15)

    # ---------- B_attr_h (one CUDA model load, 80 single-token forwards)
    import torch
    from phase2662_symmetric_mapping_contract import load_native
    model, _ = load_native('qwen4')
    model.eval()
    dev = model.get_input_embeddings().weight.device
    hs = []
    with torch.inference_mode():
        for s in range(0, n_words, 16):
            chunk = torch.tensor([tids[s:s + 16]], device=dev).T
            out = model(chunk, output_hidden_states=True)
            hs.append(out.hidden_states[-1][:, 0, :].float().cpu().numpy())
    H = np.concatenate(hs, axis=0).astype(np.float64)
    H_unit = np.stack([unit(H[i]) for i in range(n_words)])
    B_attr_h = H_unit @ D.T
    del model
    torch.cuda.empty_cache()

    # ---------- A0 anchor ----------
    C_cf = B_cf @ B_cf.T
    acc_cf = loo_nn_acc(C_cf, labels)
    a0_ok = abs(acc_cf - ANCHOR_ACC) < 1e-9
    if not a0_ok:
        v0 = {'A0': False, 'acc_B_cf': round(acc_cf, 4),
              'final_verdict': 'protocol_broken'}
        fc.save(OUT / 'result.json',
                {'phase': 2873, 'prereg': PREREG, 'verdict': v0})
        cc.ledger('phase2873_aborted', time.monotonic() - t0)
        print('P2873 VERDICT %s' % json.dumps(v0), flush=True)
        return

    # ---------- A1-A4 (rep block) ----------
    rep_scan, C0, B_cf_u, B_rep_u = fusion_scan(
        B_cf, B_attr_rep, labels, SEED)

    rng = np.random.default_rng(SEED + 1)
    C_rep = B_rep_u @ B_rep_u.T
    acc_rep = loo_nn_acc(C_rep, labels)
    null_rep = [loo_nn_acc(C_rep, rng.permutation(labels))
                for _ in range(N_PERM)]
    acc_rep_p95 = float(np.percentile(null_rep, 95))
    a3 = bool(acc_rep > acc_rep_p95)
    a3_label = 'attr_axis_class_leak' if a3 else 'attr_axis_class_clean'

    same = (labels[:, None] == labels[None, :])
    offm = ~np.eye(n_words, dtype=bool)

    def margin(C, lm):
        return float(np.mean([
            C[i][same[i] & offm[i]].mean() - C[i][(~same[i]) & offm[i]].mean()
            for i in range(n_words)]))

    def null_margins(C):
        vals = []
        for _ in range(N_PERM):
            plm = rng.permutation(labels)
            m = (plm[:, None] == plm[None, :])
            sm = m & offm
            df = (~m) & offm
            vals.append(float(np.mean([
                C[i][sm[i]].mean() - C[i][df[i]].mean()
                for i in range(n_words)])))
        return vals

    m_rep = margin(C_rep, labels)
    nm_rep_p95 = float(np.percentile(null_margins(C_rep), 95))
    a4 = bool(m_rep > nm_rep_p95)
    a4_label = 'attr_profile_class_correlated' if a4 \
        else 'attr_profile_class_uncorrelated'

    # ---------- A5 (hidden block) ----------
    h_scan, _, _, B_h_u = fusion_scan(B_cf, B_attr_h, labels, SEED_H)
    C_h = B_h_u @ B_h_u.T
    iu = np.triu_indices(n_words, 1)
    rho_rep_h = spearman(C_rep[iu], C_h[iu])
    a5_geom = bool(rho_rep_h > 0.5)

    v = {
        'A0': True,
        'acc_B_cf': round(acc_cf, 4),
        'A1': rep_scan,
        'A1_label': 'fusion_gain' if rep_scan['gain']
                    else 'attr_no_class_gain',
        'A2_label': rep_scan['growth_label'],
        'A3': a3, 'A3_label': a3_label,
        'acc_attr_rep_alone': round(acc_rep, 4),
        'acc_attr_rep_null_p95': round(acc_rep_p95, 4),
        'A4': a4, 'A4_label': a4_label,
        'margin_rep': round(m_rep, 5),
        'margin_rep_null_p95': round(nm_rep_p95, 5),
        'A5': h_scan,
        'A5_label': 'fusion_gain_h' if h_scan['gain']
                    else 'attr_no_class_gain_h',
        'rho_rep_h': round(rho_rep_h, 4),
        'A5_geometry': 'attr_geometry_hidden_preserved' if a5_geom
                       else 'attr_geometry_hidden_shifted',
        'final_verdict': 'A1=%s(%s)/growth=%s/A3=%s/A4=%s/A5=%s/geom=%s'
                         % (rep_scan['gain'], rep_scan['growth_label'],
                            rep_scan['growth'], a3_label, a4_label,
                            h_scan['gain'],
                            'preserved' if a5_geom else 'shifted'),
    }

    fc.save(OUT / 'result.json',
            {'phase': 2873, 'prereg': PREREG, 'verdict': v,
             'seed_null': SEED, 'seed_h': SEED_H, 'n_perm': N_PERM,
             'alphas': ALPHAS, 'alpha_cf': ALPHA_CF, 'words': words})
    fc.npz(OUT / 'attr_fusion.npz',
           B_cf=B_cf.astype(np.float32),
           B_attr_rep=B_attr_rep.astype(np.float32),
           B_attr_h=B_attr_h.astype(np.float32),
           labels=labels.astype(np.int64),
           C_rep=C_rep.astype(np.float32),
           C_h=C_h.astype(np.float32))

    elapsed = time.monotonic() - t0
    cc.ledger('phase2873', elapsed)
    print('P2873 VERDICT %s' % json.dumps(v), flush=True)
    print('P2873 elapsed %.1fs' % elapsed, flush=True)


if __name__ == '__main__':
    main()
