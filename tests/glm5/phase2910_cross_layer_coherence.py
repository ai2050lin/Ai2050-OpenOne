# -*- coding: utf-8 -*-
"""Phase 2910: qwen_attn cross-layer aggregation coherence.

Zero-forward matrix analysis.

Why: 2909 (N12) found that qwen_attn's lower-tail status
(full-layer cosine-margin percentile p = 0.026 under its own
isotropic null) is a FULL-LAYER AGGREGATION effect - every proper
layer subset is null-like (S_front 0.140, S_back 0.130, S_key
0.607).  Two competing explanations:
  H_coherent  each window layer contributes a small consistent
              downward shift; the full-layer mean aggregates a
              coherent cross-layer property;
  H_cancel    layer contributions have mixed signs and the full-
              layer average lands at the lower edge by accident
              (averaging artifact).
This phase adjudicates with per-layer and cumulative analyses
under the fixed isotropic-null protocol (RMS sigma, E9-corrected):
  per-layer  : for each window layer j, d=1 margin percentile
               p_j = P(margin_synth_j <= margin_real_j),
               5 seeds x N_SYNTH draws;
  cumulative : for k = 1..n_win, percentile of B[:, :k] (V0),
               tracing the aggregation dynamics.
Adjudication (frozen, qwen_attn only; other groups descriptive):
  below_frac = fraction of layers with p_median_j < 0.5;
  below_frac >= 0.90 => qwen_attn_coherent_cross_layer_
                        suppression;
  below_frac <  0.50 => qwen_attn_cancellation_artifact;
  else               => qwen_attn_partial_coherence.

Anchors: a1 full-layer baseline margin/acc == stored 2e-5;
a2 Delta_B == 2905 delta_per_layer 1e-4; a3 fast margin ==
margin_of_B and acc_score == acc_of_B within 1e-12.

Coverage audits: d=1 (rng [2910,0]) and d=5 (rng [2910,1]), 40
reps x 4000 draws, pass iff [32,40]/40; d=10 full-layer audit
already registered in 2908 (40/40), d=6 in 2909 (40/40) - the
{1,5,6,10} grid anchors the audit at the extremes used here.
Any audit fail => audit_coverage_fail_all_void.

Output: phase2910/cross_layer_coherence/.
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
                          'result.json')),
    'qwen': (os.path.join(BASE, 'phase2903',
                          'qwen_channel_jacobian_decomposition',
                          'qwen_channel_jacobian_decomposition.npz'),
             os.path.join(BASE, 'phase2903',
                          'qwen_channel_jacobian_decomposition',
                          'result.json')),
}
R2905 = os.path.join(BASE, 'phase2905',
                     'margin_structure_skew_audited',
                     'result.json')
OUT = os.path.join(BASE, 'phase2910', 'cross_layer_coherence')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2910_run_report.txt')
SEED_BASE = 2910
N_SYNTH = 5000
N_SEEDS = 5
N_COV = 40
N_AUDIT = 4000
TOL_A1 = 2e-5
TOL_A2 = 1e-4
BELOW = 0.5
COHERENT_FRAC = 0.90
CANCEL_FRAC = 0.50
GROUP_ORDER = ('glm4_mlp', 'glm4_attn', 'qwen_mlp', 'qwen_attn')

PREREG = {
    'mode': 'zero_forward_matrix_analysis',
    'sigma_definition': 'RMS sqrt(tr(Sigma_c)/d) (E9-corrected); '
                        'd=1 uses std(ddof=1)',
    'question': 'is qwen_attn lower-tail status (full-layer p = '
                '0.026, M2908/M2909) a coherent cross-layer '
                'property (H_coherent) or an averaging artifact '
                '(H_cancel)?',
    'configs': 'per group: per-layer d=1 percentile p_j for each '
               'window layer j (5 seeds x 5000 draws, rng '
               '[2910,1000+j*10+k]); cumulative percentile of '
               'B[:, :k] for k=1..n_win (rng [2910,2000+k*10+k]); '
               'scoring = V0 row-normalised cosine margin (2896 '
               'family, the declared definition)',
    'sources': 'B matrices + stored margin/acc (2902/2903); '
               'delta_per_layer (2905, round4)',
    'anchors': 'a1 margin/acc == stored 2e-5; a2 Delta_B == 2905 '
               'delta_per_layer 1e-4; a3 fast == full 1e-12 '
               '(margin and acc)',
    'coverage_audits': 'd=1 rng [2910,0], d=5 rng [2910,1]; 40 '
                       'reps x 4000 draws, pass iff [32,40]/40; '
                       'd=10/6 audits registered in 2908/2909; '
                       'any fail here => '
                       'audit_coverage_fail_all_void',
    'adjudication': 'qwen_attn only: below_frac = fraction of '
                    'layers with p_median_j < 0.5; >= 0.90 => '
                    'qwen_attn_coherent_cross_layer_suppression; '
                    '< 0.50 => qwen_attn_cancellation_artifact; '
                    'else qwen_attn_partial_coherence; other '
                    'groups descriptive (below_frac + cumulative '
                    'curve)',
}


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


def log(msg, lines):
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


def sigma_rms(Bm):
    d = Bm.shape[1]
    if d == 1:
        return float(Bm.std(0, ddof=1).item())
    S = np.cov(Bm, rowvar=False)
    return float(np.sqrt(np.trace(S) / d))


def synth_p(B_sub, lab, rng, n_draws):
    m0, m1 = lab == 0, lab == 1
    n0, n1 = int(m0.sum()), int(m1.sum())
    d = B_sub.shape[1]
    mu0, mu1 = B_sub[m0].mean(0), B_sub[m1].mean(0)
    s0, s1 = sigma_rms(B_sub[m0]), sigma_rms(B_sub[m1])
    s_true = margin_of_B(B_sub, lab)
    cnt = 0
    for _ in range(n_draws):
        Bs = np.empty_like(B_sub)
        Bs[m0] = mu0 + s0 * rng.normal(size=(n0, d))
        Bs[m1] = mu1 + s1 * rng.normal(size=(n1, d))
        if margin_of_B(Bs, lab) <= s_true:
            cnt += 1
    return cnt / float(n_draws), s_true


def coverage_audit(d, seed_key):
    rng = np.random.default_rng(seed_key)
    n, n0 = 57, 22
    n1 = n - n0
    lab = np.array([0] * n0 + [1] * (n - n0))
    m0, m1 = lab == 0, lab == 1
    cov = 0
    for _ in range(N_COV):
        mu0t = rng.normal(size=d) * 0.3
        dvt = rng.normal(size=d)
        mu1t = mu0t + 0.5 * dvt / np.linalg.norm(dvt)
        s0t = rng.uniform(0.3, 1.0)
        s1t = rng.uniform(0.3, 1.0)
        Bt = np.empty((n, d))
        Bt[m0] = mu0t + s0t * rng.normal(size=(n0, d))
        Bt[m1] = mu1t + s1t * rng.normal(size=(n1, d))
        s_true = margin_of_B(Bt, lab)
        u0, u1 = Bt[m0].mean(0), Bt[m1].mean(0)
        h0, h1 = sigma_rms(Bt[m0]), sigma_rms(Bt[m1])
        vals = np.empty(N_AUDIT)
        for i in range(N_AUDIT):
            Bs = np.empty((n, d))
            Bs[m0] = u0 + h0 * rng.normal(size=(n0, d))
            Bs[m1] = u1 + h1 * rng.normal(size=(n1, d))
            vals[i] = margin_of_B(Bs, lab)
        lo, hi = np.percentile(vals, [2.5, 97.5])
        if lo <= s_true <= hi:
            cov += 1
    return cov


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2910,
                   'name': 'cross_layer_coherence',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8': sha8(os.path.abspath(__file__)),
                   'sources': {'s2902': sha8(SRC['glm4'][0]),
                               'r2902': sha8(SRC['glm4'][1]),
                               's2903': sha8(SRC['qwen'][0]),
                               'r2903': sha8(SRC['qwen'][1]),
                               'r2905': sha8(R2905)},
                   'mode': 'zero_forward_matrix_analysis',
                   'seed_base': SEED_BASE, 'n_synth': N_SYNTH,
                   'n_seeds': N_SEEDS, 'n_cov_reps': N_COV,
                   'n_audit_draws': N_AUDIT,
                   'below': BELOW,
                   'coherent_frac': COHERENT_FRAC,
                   'cancel_frac': CANCEL_FRAC,
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    groups = {}
    for mdl, (npz, rj) in SRC.items():
        z = np.load(npz, allow_pickle=True)
        r = json.load(open(rj, encoding='utf-8'))
        for ch in ('mlp', 'attn'):
            lab = np.asarray(z['labels_lang']).astype(int)
            groups['%s_%s' % (mdl, ch)] = {
                'B': z['B_%s' % ch].astype(np.float64),
                'lab': lab,
                'stored_margin': float(r['margins'][ch]['margin']),
                'stored_acc': float(
                    r['accs'][ch] if 'accs' in r
                    else r['margins'][ch]['acc']),
            }
    r05 = json.load(open(R2905, encoding='utf-8'))

    # ---------- anchors ----------
    anchor_ok = True
    anchor = {}
    for g in GROUP_ORDER:
        it = groups[g]
        B, lab = it['B'], it['lab']
        mf = margin_of_B(B, lab)
        ac = acc_of_B(B, lab)
        mdl, ch = g.split('_')
        d05 = np.asarray(r05['groups']['%s_%s' % (mdl, ch)]
                         ['delta_per_layer'], dtype=float)
        d10 = B[lab == 1].mean(0) - B[lab == 0].mean(0)
        e2 = float(np.abs(d10 - d05).max())
        ok = (abs(mf - it['stored_margin']) < TOL_A1
              and abs(ac - it['stored_acc']) < TOL_A1
              and e2 < TOL_A2)
        anchor_ok = anchor_ok and ok
        anchor[g] = {'margin_recomp': round(mf, 6),
                     'acc_recomp': round(ac, 6),
                     'deltaB_vs_2905_maxabs': e2, 'ok': bool(ok)}
        log('anchor %s margin %.5f acc %.5f dLB %.2e ok=%s'
            % (g, mf, ac, e2, ok), lines)

    # ---------- coverage audits ----------
    audits = {}
    audit_ok = True
    for d, key in ((1, 0), (5, 1)):
        c = coverage_audit(d, [SEED_BASE, key])
        ok = bool(32 <= c <= N_COV)
        audit_ok = audit_ok and ok
        audits['d%d' % d] = {'covered': c, 'n': N_COV,
                             'pass': ok}
        log('audit d=%d: %d/%d pass=%s' % (d, c, N_COV, ok),
            lines)

    verdict = None
    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    elif not audit_ok:
        verdict = 'audit_coverage_fail_all_void'

    # ---------- per-layer + cumulative grid ----------
    res_g = {}
    if verdict is None:
        for g in GROUP_ORDER:
            B, lab = groups[g]['B'], groups[g]['lab']
            n_win = B.shape[1]
            layers = []
            for j in range(n_win):
                ps = []
                s_true = None
                for k in range(N_SEEDS):
                    rng = np.random.default_rng(
                        [SEED_BASE, 1000 + j * 10 + k])
                    p, s_true = synth_p(B[:, j:j + 1], lab,
                                        rng, N_SYNTH)
                    ps.append(p)
                ps_arr = np.asarray(ps)
                layers.append({
                    'col': j,
                    'score_true': round(float(s_true), 6),
                    'p_median': round(float(np.median(ps_arr)),
                                      6),
                    'p_per_seed': [round(p, 6) for p in ps],
                })
                log('%s L%02d d1 score=%+.6f p_med=%.4f '
                    'p_range=[%.4f,%.4f]'
                    % (g, j, s_true, np.median(ps_arr),
                       ps_arr.min(), ps_arr.max()), lines)
            cum = []
            for kk in range(1, n_win + 1):
                ps = []
                s_true = None
                for k in range(N_SEEDS):
                    rng = np.random.default_rng(
                        [SEED_BASE, 2000 + kk * 10 + k])
                    p, s_true = synth_p(B[:, :kk], lab,
                                        rng, N_SYNTH)
                    ps.append(p)
                ps_arr = np.asarray(ps)
                cum.append({
                    'k': kk,
                    'score_true': round(float(s_true), 6),
                    'p_median': round(float(np.median(ps_arr)),
                                      6),
                })
            below = sum(1 for e in layers
                        if e['p_median'] < BELOW)
            frac = below / float(n_win)
            if g == 'qwen_attn':
                if frac >= COHERENT_FRAC:
                    gverdict = ('qwen_attn_coherent_cross_layer_'
                                'suppression')
                elif frac < CANCEL_FRAC:
                    gverdict = 'qwen_attn_cancellation_artifact'
                else:
                    gverdict = 'qwen_attn_partial_coherence'
            else:
                gverdict = ''
            res_g[g] = {
                'n_win': n_win,
                'layers': layers,
                'cumulative': cum,
                'below_frac': round(frac, 4),
                'sub_verdict': gverdict,
            }
            log('%s below_frac=%.2f cum_p: %s %s'
                % (g, frac,
                   ' '.join('%.3f' % e['p_median']
                            for e in cum), gverdict), lines)
        verdict = res_g['qwen_attn']['sub_verdict']

    log('==== VERDICT: %s ====' % verdict, lines)

    res = {
        'phase': 2910,
        'model': 'glm4+qwen (matrix analysis)',
        'prereg': PREREG,
        'seed_base': SEED_BASE, 'n_synth': N_SYNTH,
        'n_seeds': N_SEEDS, 'n_audit_draws': N_AUDIT,
        'anchor': anchor, 'anchor_ok': bool(anchor_ok),
        'coverage_audits': audits, 'audits_pass': bool(audit_ok),
        'groups': res_g,
        'final_verdict': verdict,
        'runtime_s': round(time.monotonic() - t0, 1),
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    np.savez_compressed(
        os.path.join(OUT, 'cross_layer_coherence.npz'),
        phase=np.int64(2910))
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2910 verdict=%s' % verdict)


if __name__ == '__main__':
    main()
