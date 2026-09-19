# -*- coding: utf-8 -*-
"""Phase 2906: margin amplitude law + neuron-level readout
attribution (zero-forward; weights read directly from
safetensors, no forward pass).

2905 located the margin carrier: first-order class-mean shift in
the B row space; Fisher F / Hotelling T2 match margin positivity
4/4.  Open: (a) does the ISOTROPIC summary (class means mu_c +
per-class scalar variance) quantitatively RECONSTRUCT the margin
amplitudes (amplitude law)?  (b) at neuron level, which output
neurons' readout directions carry the response-space class-mean
shift Delta_r?

Module A (amplitude law, B row space):
  anchors: a1 margin/acc recomputed from npz == stored (abs
      2e-5); a2 Delta_B recomputed from B npz == 2905 result
      delta_per_layer (abs 1e-4; 2905 stores round(...,4)).
  audit (coverage self-test, rng [2906,0]): 40 reps - draw true
      params (mu0, mu1, sigma), draw 'real' matrix, estimate
      (mu_hat_c, sigma_hat_c) from it, build M1 95% interval from
      400 synth draws, check coverage; pass iff covered in
      [32,40]/40 (Binomial(40,0.95) lower tail ~2%).
  primary (frozen): per group M1 synthesis x_i = mu_hat_c(i) +
      sigma_hat_c * z (isotropic, class sizes fixed, 400 draws,
      rng 2906 shared stream, group order glm4-mlp glm4-attn
      qwen-mlp qwen-attn) -> 95% interval of margin; verdict:
      4/4 real margin_full inside => amplitude_law_confirmed_
      isotropic; 3/4 => amplitude_law_partial; else
      amplitude_law_not_established.
  diagnostics per group: SNR = 0.5||delta||^2 / (0.5(||mu0||^2+
      ||mu1||^2) + mean_c tr(Sigma_c)); dilution ratio
      margin_full / margin_from_mean; first-order analytic
      0.5||delta||^2/E||x||^2.

Module B (neuron readout attribution, response space):
  Delta_r[q] = mean(r_same | lab=1, q) - mean(r_same | lab=0, q)
      from npz r_*_same (same-context responses, 2902/2903).
  dictionary: mlp -> down_proj columns (d_model x inter; column
      k = output-neuron k readout direction); attn -> o_proj
      columns (head-aggregated readout).  Weights via
      safetensors safe_open (bf16 -> float32), window layers
      only; model NOT loaded, no forward.
  coordinates: c = pinv(W) @ Delta_r (W row-full-rank ->
      exact reconstruction); concentration = energy share of
      top-64 coordinates.
  nulls (frozen): 200 label-permutation Delta_r (rng [2906,1],
      preserves real response covariance) -> per-layer p95 of
      top-64 share; Gaussian null (rng [2906,2]) diagnostic.
  reading (descriptive, not in primary verdict): per (model, ch)
      median-over-layers top-64 share vs perm null p95 ->
      readout_concentrated / readout_distributed.

Verdicts (frozen):
  anchors fail            => anchor_fail_all_void
  coverage audit fail     => audit_coverage_fail_all_void
  primary 4/4 | 3/4 | else => amplitude_law_confirmed_isotropic |
      amplitude_law_partial | amplitude_law_not_established

SEED=2906.  Output: phase2906/amplitude_law_neuron_attribution/.
"""
import hashlib
import json
import os
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC = {
    'glm4': (os.path.join(BASE, 'phase2902',
                          'glm4_channel_jacobian_asymmetry',
                          'glm4_channel_jacobian_asymmetry.npz'),
             os.path.join(BASE, 'phase2902',
                          'glm4_channel_jacobian_asymmetry',
                          'result.json'),
             r'D:\AI2050\Ai2050-OpenOne\models\hf'
             r'\glm4-9b-chat-hf', 28, 40),
    'qwen': (os.path.join(BASE, 'phase2903',
                          'qwen_channel_jacobian_decomposition',
                          'qwen_channel_jacobian_decomposition.npz'),
             os.path.join(BASE, 'phase2903',
                          'qwen_channel_jacobian_decomposition',
                          'result.json'),
             r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b',
             26, 36),
}
R2905 = os.path.join(BASE, 'phase2905',
                     'margin_structure_skew_audited',
                     'result.json')
OUT = os.path.join(BASE, 'phase2906',
                   'amplitude_law_neuron_attribution')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2906_run_report.txt')
SEED = 2906
N_SYNTH = 400
N_COV = 40
N_NULL_PERM = 200
TOPK = 64
TOL_A1 = 2e-5
TOL_A2 = 1e-4

PREREG = {
    'mode': 'zero_forward; weights read via safetensors '
            'safe_open (no model load, no forward pass)',
    'sources': 'B matrices + r_*_same from 2902/2903 npz; '
               'stored margin/acc (2902/2903) and '
               'delta_per_layer (2905, round4)',
    'anchor_a1': 'margin/acc recomputed from npz float32 B == '
                 'stored within abs 2e-5, else '
                 'anchor_fail_all_void',
    'anchor_a2': 'Delta_B recomputed from B npz (per-layer class '
                 'mean diff) == 2905 delta_per_layer within abs '
                 '1e-4 (round4 storage), else anchor_fail_all_void',
    'audit_coverage': 'rng [2906,0]: 40 reps; true params (mu0, '
                      'mu1=mu0+delta, sigma0,sigma1; d=10, n=57, '
                      'split 22/35); estimate (mu_hat_c, '
                      'sigma_hat_c) from the drawn matrix; M1 '
                      '95% interval from 400 synth; pass iff '
                      'coverage in [32,40]/40, else '
                      'audit_coverage_fail_all_void',
    'amplitude_law': 'per group M1 isotropic synthesis (mu_hat_c '
                     'measured class means, sigma_hat_c = sqrt(tr'
                     '(Sigma_c)/d) per class, class sizes fixed, '
                     '400 draws, shared rng stream, group order '
                     'glm4-mlp glm4-attn qwen-mlp qwen-attn) -> '
                     '95% interval; real margin_full inside for '
                     '4/4 => amplitude_law_confirmed_isotropic; '
                     '3/4 => amplitude_law_partial; else '
                     'amplitude_law_not_established',
    'neuron_attribution': 'descriptive: Delta_r[q] = class mean '
                          'diff of r_same at layer q; dictionary '
                          'mlp down_proj columns / attn o_proj '
                          'columns; c = pinv(W) @ Delta_r (exact); '
                          'concentration = top-64 energy share; '
                          'null = 200 label-permutation Delta_r '
                          '(rng [2906,1]) per-layer p95; Gaussian '
                          'null (rng [2906,2]) diagnostic; median '
                          'share > perm p95 => readout_concentrated',
    'verdict': 'anchor fail => anchor_fail_all_void; coverage '
               'audit fail => audit_coverage_fail_all_void; 4/4 '
               'groups inside M1 interval => '
               'amplitude_law_confirmed_isotropic; 3/4 => '
               'amplitude_law_partial; else '
               'amplitude_law_not_established',
}


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


def log(msg, lines):
    print(msg, flush=True)
    lines.append(msg)


def unit_rows(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.maximum(n, 1e-30)


def margin_of_B(B, lab):
    U = unit_rows(B)
    Sm = U @ U.T
    n = len(lab)
    eye = np.eye(n, dtype=bool)
    same = (lab[:, None] == lab[None, :]) & (~eye)
    diff = (~eye) & (~same)
    return float(Sm[same].mean() - Sm[diff].mean())


def acc_of_B(B, lab):
    mu = B.mean(axis=0, keepdims=True)
    sd = B.std(axis=0, keepdims=True)
    C = (B - mu) / np.maximum(sd, 1e-30)
    C = unit_rows(C)
    S = C @ C.T
    np.fill_diagonal(S, -2.0)
    nn = S.argmax(axis=1)
    return float(np.mean(lab[nn] == lab))


def margin_from_mean_of(B, lab):
    Mm = np.empty_like(B)
    for c in (0, 1):
        m = lab == c
        if m.any():
            Mm[m] = B[m].mean(axis=0, keepdims=True)
    return margin_of_B(Mm, lab)


def load_window_weight(mt_dir, layer, names):
    """Read one layer's tensors by name from safetensors shards."""
    from safetensors import safe_open
    idx_p = os.path.join(mt_dir, 'model.safetensors.index.json')
    weight_map = {}
    if os.path.exists(idx_p):
        idx = json.load(open(idx_p, encoding='utf-8'))
        weight_map = idx.get('weight_map', {})
    shards = [f for f in os.listdir(mt_dir)
              if f.endswith('.safetensors')]
    out = {}
    for full in names:
        key = full.format(layer=layer)
        shard = weight_map.get(key)
        got = False
        cand = ([shard] if shard else []) + shards
        for sf in cand:
            p = os.path.join(mt_dir, sf)
            if not os.path.exists(p):
                continue
            with safe_open(p, framework='pt') as f:
                if key in f.keys():
                    out[full] = f.get_tensor(key) \
                        .float().numpy()
                    got = True
                    break
        if not got:
            raise KeyError('weight not found: %s' % key)
    return out


def topk_share(c, k=TOPK):
    cc = np.sort(np.abs(c) ** 2)[::-1]
    tot = float(cc.sum())
    return float(cc[:k].sum() / max(tot, 1e-30))


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2906,
                   'name': 'amplitude_law_neuron_attribution',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8': sha8(os.path.abspath(__file__)),
                   'sources': {'s2902': sha8(SRC['glm4'][0]),
                               'r2902': sha8(SRC['glm4'][1]),
                               's2903': sha8(SRC['qwen'][0]),
                               'r2903': sha8(SRC['qwen'][1]),
                               'r2905': sha8(R2905)},
                   'mode': 'zero_forward_matrix_analysis',
                   'seed': SEED, 'n_synth': N_SYNTH,
                   'n_cov_reps': N_COV,
                   'n_null_perm': N_NULL_PERM, 'topk': TOPK,
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    # ---------- load groups ----------
    groups = {}
    for mdl, (npz, rj, mt_dir, lo, hi) in SRC.items():
        z = np.load(npz, allow_pickle=True)
        r = json.load(open(rj, encoding='utf-8'))
        for ch in ('mlp', 'attn'):
            lab = np.asarray(z['labels_lang']).astype(int)
            groups['%s_%s' % (mdl, ch)] = {
                'B': z['B_%s' % ch].astype(np.float64),
                'r': z['r_%s_same' % ch].astype(np.float64),
                'lab': lab, 'mt_dir': mt_dir, 'lo': lo, 'hi': hi,
                'stored_margin': float(r['margins'][ch]['margin']),
                'stored_p95': float(
                    r['margins'][ch]['margin_p95']),
                'stored_acc': float(
                    r['accs'][ch] if 'accs' in r
                    else r['margins'][ch]['acc']),
            }
    r05 = json.load(open(R2905, encoding='utf-8'))

    # ---------- anchors ----------
    anchor = {}
    anchor_ok = True
    for g, it in groups.items():
        mf = margin_of_B(it['B'], it['lab'])
        ac = acc_of_B(it['B'], it['lab'])
        mdl, ch = g.split('_')
        d05 = np.asarray(
            r05['groups']['%s_%s' % (mdl, ch)]
            ['delta_per_layer'], dtype=float)
        d06 = it['B'][it['lab'] == 1].mean(0) \
            - it['B'][it['lab'] == 0].mean(0)
        e2 = float(np.abs(d06 - d05).max())
        ok = (abs(mf - it['stored_margin']) < TOL_A1
              and abs(ac - it['stored_acc']) < TOL_A1
              and e2 < TOL_A2)
        anchor[g] = {'margin_recomp': round(mf, 6),
                     'acc_recomp': round(ac, 6),
                     'deltaB_vs_2905_maxabs': round(e2, 7),
                     'ok': bool(ok)}
        anchor_ok = anchor_ok and ok
        log('anchor %s margin %.5f acc %.5f dLB %.2e ok=%s'
            % (g, mf, ac, e2, ok), lines)

    # ---------- coverage audit ----------
    rng_cov = np.random.default_rng([SEED, 0])
    d, n, n0 = 10, 57, 22
    lab_c = np.array([0] * n0 + [1] * (n - n0))
    covered = 0
    for rep in range(N_COV):
        mu0t = rng_cov.normal(size=d) * 0.3
        dvt = rng_cov.normal(size=d)
        mu1t = mu0t + 0.5 * dvt / np.linalg.norm(dvt)
        s0t = float(rng_cov.uniform(0.3, 1.0))
        s1t = float(rng_cov.uniform(0.3, 1.0))
        Bt = np.empty((n, d))
        m0 = lab_c == 0
        m1 = lab_c == 1
        Bt[m0] = mu0t + s0t * rng_cov.normal(size=(int(m0.sum()), d))
        Bt[m1] = mu1t + s1t * rng_cov.normal(size=(int(m1.sum()), d))
        m_true = margin_of_B(Bt, lab_c)
        u0 = Bt[m0].mean(0)
        u1 = Bt[m1].mean(0)
        s0h = float(Bt[m0].std(0).mean())
        s1h = float(Bt[m1].std(0).mean())
        ms = []
        for _ in range(N_SYNTH):
            Bs = np.empty((n, d))
            Bs[m0] = u0 + s0h * rng_cov.normal(
                size=(int(m0.sum()), d))
            Bs[m1] = u1 + s1h * rng_cov.normal(
                size=(int(m1.sum()), d))
            ms.append(margin_of_B(Bs, lab_c))
        lo_, hi_ = np.percentile(ms, [2.5, 97.5])
        if lo_ <= m_true <= hi_:
            covered += 1
    cov_ok = bool(32 <= covered <= N_COV)
    log('coverage audit: %d/%d pass=%s'
        % (covered, N_COV, cov_ok), lines)

    verdict = None
    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    elif not cov_ok:
        verdict = 'audit_coverage_fail_all_void'

    # ---------- module A: amplitude law ----------
    res_g = {}
    rng_main = np.random.default_rng(SEED)
    if verdict is None:
        for g in ('glm4_mlp', 'glm4_attn', 'qwen_mlp',
                  'qwen_attn'):
            it = groups[g]
            B, lab = it['B'], it['lab']
            m_full = margin_of_B(B, lab)
            m_fmean = margin_from_mean_of(B, lab)
            m0 = lab == 0
            m1 = lab == 1
            mu0 = B[m0].mean(0)
            mu1 = B[m1].mean(0)
            s0 = float(B[m0].std(0).mean())
            s1 = float(B[m1].std(0).mean())
            ms = []
            for _ in range(N_SYNTH):
                Bs = np.empty_like(B)
                Bs[m0] = mu0 + s0 * rng_main.normal(
                    size=(int(m0.sum()), B.shape[1]))
                Bs[m1] = mu1 + s1 * rng_main.normal(
                    size=(int(m1.sum()), B.shape[1]))
                ms.append(margin_of_B(Bs, lab))
            lo_, hi_ = np.percentile(ms, [2.5, 97.5])
            inside = bool(lo_ <= m_full <= hi_)
            dlt = mu1 - mu0
            trS = 0.5 * float((B[m0].var(0).sum()
                               + B[m1].var(0).sum()))
            mu_bar = 0.5 * (mu0 + mu1)
            exx = 0.5 * (float(mu0 @ mu0) + float(mu1 @ mu1)) \
                + trS
            snr = 0.5 * float(dlt @ dlt) / max(exx, 1e-30)
            fo = 0.5 * float(dlt @ dlt) / max(exx, 1e-30)
            res_g[g] = {
                'margin_full': round(m_full, 6),
                'margin_from_mean': round(m_fmean, 6),
                'm1_lo': round(float(lo_), 6),
                'm1_hi': round(float(hi_), 6),
                'inside_m1': inside,
                'synth_median': round(float(np.median(ms)), 6),
                'snr': round(snr, 5),
                'first_order_analytic': round(fo, 5),
                'dilution_ratio': round(m_full
                                        / max(m_fmean, 1e-30), 5),
                'delta_norm': round(float(np.linalg.norm(dlt)),
                                    4),
                'sigma0_iso': round(s0, 5),
                'sigma1_iso': round(s1, 5),
            }
            log('%s: full=%.4f from_mean=%.4f M1=[%.4f,%.4f] '
                'inside=%s SNR=%.4f dilution=%.4f'
                % (g, m_full, m_fmean, lo_, hi_, inside, snr,
                   res_g[g]['dilution_ratio']), lines)
        n_in = sum(1 for e in res_g.values()
                   if e['inside_m1'])
        if n_in == 4:
            verdict = 'amplitude_law_confirmed_isotropic'
        elif n_in == 3:
            verdict = 'amplitude_law_partial'
        else:
            verdict = 'amplitude_law_not_established'
        log('inside count %d/4' % n_in, lines)

    log('==== VERDICT: %s ====' % verdict, lines)

    # ---------- module B: neuron readout attribution ----------
    rng_perm = np.random.default_rng([SEED, 1])
    rng_gauss = np.random.default_rng([SEED, 2])
    neuron = {}
    if verdict is not None and verdict.endswith('all_void'):
        log('skip module B (void gate)', lines)
    else:
        for g in ('glm4_mlp', 'glm4_attn', 'qwen_mlp',
                  'qwen_attn'):
            it = groups[g]
            r, lab = it['r'], it['lab']
            mdl, ch = g.split('_')
            lo, hi = it['lo'], it['hi']
            m0, m1 = lab == 0, lab == 1
            shares, p95s, gauss_med = [], [], []
            for q in range(lo, hi):
                dv = r[:, q - lo, :].mean(0, keepdims=True) * 0
                dv = (r[m1, q - lo, :].mean(0)
                      - r[m0, q - lo, :].mean(0))
                Wn = ('model.layers.{layer}.mlp.down_proj.weight'
                      if ch == 'mlp' else
                      'model.layers.{layer}.self_attn.o_proj'
                      '.weight')
                Wt = load_window_weight(it['mt_dir'], q, [Wn])
                W = Wt[Wn].astype(np.float32)
                if W.shape[0] != dv.shape[0]:
                    W = W.T
                # pinv (row-full-rank W: [d_model, N_cols])
                pinv = np.linalg.pinv(W)
                c_obs = pinv @ dv.astype(np.float32)
                sh = topk_share(c_obs)
                sh_null = []
                for _ in range(N_NULL_PERM):
                    pl = rng_perm.permutation(lab)
                    dv_p = (r[pl == 1, q - lo, :].mean(0)
                            - r[pl == 0, q - lo, :].mean(0))
                    sh_null.append(topk_share(pinv @ dv_p
                                              .astype(np.float32)))
                sh_g = [topk_share(pinv @ rng_gauss.normal(
                    size=dv.shape).astype(np.float32))
                    for _ in range(20)]
                shares.append(sh)
                p95s.append(float(np.percentile(sh_null, 95)))
                gauss_med.append(float(np.median(sh_g)))
                del pinv, W, Wt
                log('  %s L%d: top64 share %.4f (perm p95 %.4f, '
                    'gauss med %.4f)'
                    % (g, q, sh, p95s[-1], gauss_med[-1]), lines)
            med_share = float(np.median(shares))
            med_p95 = float(np.median(p95s))
            neuron[g] = {
                'top64_share_median': round(med_share, 5),
                'perm_p95_median': round(med_p95, 5),
                'concentrated': bool(med_share > med_p95),
                'per_layer_share': [round(x, 5) for x in shares],
                'per_layer_p95': [round(x, 5) for x in p95s],
                'gauss_null_median_median': round(
                    float(np.median(gauss_med)), 5),
            }
            log('%s: median share %.4f vs perm p95 %.4f -> %s'
                % (g, med_share, med_p95,
                   'concentrated'
                   if neuron[g]['concentrated']
                   else 'distributed'), lines)

    res = {
        'phase': 2906, 'model': 'glm4+qwen (matrix analysis)',
        'prereg': PREREG, 'seed': SEED,
        'anchor': anchor, 'anchor_ok': bool(anchor_ok),
        'coverage_audit': {'covered': covered, 'n': N_COV,
                           'pass': cov_ok},
        'groups': res_g, 'neuron_attribution': neuron,
        'final_verdict': verdict,
        'runtime_s': round(time.monotonic() - t0, 1),
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    npz_data = {}
    for g, e in res_g.items():
        for k in ('margin_full', 'margin_from_mean', 'm1_lo',
                  'm1_hi', 'snr', 'delta_norm'):
            npz_data['%s_%s' % (g, k)] = np.asarray(e[k])
    for g, e in neuron.items():
        npz_data['%s_per_layer_share' % g] = np.asarray(
            e['per_layer_share'])
        npz_data['%s_per_layer_p95' % g] = np.asarray(
            e['per_layer_p95'])
    if npz_data:
        np.savez_compressed(os.path.join(
            OUT, 'amplitude_law_neuron_attribution.npz'),
            **npz_data)

    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
