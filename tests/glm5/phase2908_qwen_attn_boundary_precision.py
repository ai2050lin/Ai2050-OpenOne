# -*- coding: utf-8 -*-
"""Phase 2908: qwen_attn M1 boundary precision (high-power MC).

Zero-forward matrix analysis.

Why: 2907 found all four groups at level M1 under the preregistered
RMS sigma, but qwen_attn (margin_full = -0.019459) sits at the M1
lower edge with slack 0.0004, and diag_2907b showed its verdict is
MC-noise borderline (inside 2/4, below 2/4 across the 2x2x2 grid).
At N_SYNTH=400 the interval endpoints carry MC error ~0.002-0.004,
the same order as the slack.  This phase re-adjudicates the qwen_attn
M1 position at N_SYNTH=20000 x 5 independent seeds (800x the draw
mass of one 2907 arm, 100x the total) plus a matched-protocol
coverage audit, and converts the endpoint test into a direct
percentile test: p = P(margin_synth <= margin_full).

Preregistered adjudication (frozen in execution.json):
  p in percentile scale; SE_P = sqrt(0.025*0.975/N_SYNTH) ~ 0.00110;
  inside iff p_median >= 0.025 + 3*SE_P;  below iff p_median
  <= 0.025 - 3*SE_P;  else the true position IS the M1 edge
  (qwen_attn_at_m1_edge) - i.e. the margin sits at the 2.5th
  percentile of its own isotropic null within protocol precision.
Other three groups: descriptive p-values only.

Sigma definition fixed to RMS sqrt(tr(Sigma_c)/d) per the 2906
prereg TEXT / 2907 implementation (E9-corrected); the coverage audit
uses the same definition by construction.

Sources: B matrices + stored margin/acc from 2902/2903 npz;
delta_per_layer (round4) from 2905 result.json (anchor a2).
Output: phase2908/qwen_attn_boundary_precision/.
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
OUT = os.path.join(BASE, 'phase2908', 'qwen_attn_boundary_precision')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2908_run_report.txt')
SEED_BASE = 2908
N_SYNTH = 20000
N_SEEDS = 5
N_COV = 40
TOL_A1 = 2e-5
TOL_A2 = 1e-4
SE_P = float(np.sqrt(0.025 * 0.975 / N_SYNTH))
P_LO = 0.025 - 3 * SE_P
P_HI = 0.025 + 3 * SE_P

PREREG = {
    'mode': 'zero_forward_matrix_analysis',
    'sigma_definition': 'RMS sqrt(tr(Sigma_c)/d) per class, fixed '
                        'by 2906 prereg text / 2907 implementation '
                        '(E9-corrected); audit uses same definition',
    'sources': 'B matrices + stored margin/acc (2902/2903); '
               'delta_per_layer (2905, round4)',
    'anchors': 'a1 margin/acc recomputed == stored within abs 2e-5; '
               'a2 Delta_B recomputed == 2905 delta_per_layer '
               'within abs 1e-4 (round4); a3 fast_margin == '
               'margin_of_B within abs 1e-12 per group',
    'coverage_audit': 'rng [2908,0]: 40 reps, true params (d=10, '
                      'n=57, split 22/35, mu0 ~ 0.3*N(0,I), '
                      'delta = 0.5*unit, sigma0,sigma1 ~ U(0.3,1.0));'
                      ' estimate (mu_hat_c, sigma_hat_c RMS); '
                      'N_SYNTH-draw 95% interval covers true margin;'
                      ' pass iff coverage in [32,40]/40 else '
                      'audit_coverage_fail_all_void',
    'main_synth': 'per group, 5 independent seeds rng [2908,10+k] '
                  'k=0..4, N_SYNTH=20000 draws each, class sizes '
                  'fixed, order glm4-mlp glm4-attn qwen-mlp '
                  'qwen-attn; p = P(margin_synth <= margin_full) '
                  'per seed',
    'adjudication': 'qwen_attn only; p_median over seeds; '
                    'p_median >= 0.025+3*SE_P (%.5f) => '
                    'qwen_attn_inside_m1_confirmed; p_median <= '
                    '0.025-3*SE_P (%.5f) => qwen_attn_below_m1_'
                    'confirmed; else => qwen_attn_at_m1_edge; '
                    'other groups descriptive only'
                    % (P_HI, P_LO),
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


def fast_margin(U, same, diff):
    Sm = U @ U.T
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
    S = np.cov(Bm, rowvar=False)
    return float(np.sqrt(np.trace(S) / S.shape[0]))


def coverage_audit():
    rng = np.random.default_rng([SEED_BASE, 0])
    d, n, n0 = 10, 57, 22
    lab = np.array([0] * n0 + [1] * (n - n0))
    m0, m1 = lab == 0, lab == 1
    eye = np.eye(n, dtype=bool)
    same = (lab[:, None] == lab[None, :]) & (~eye)
    diff = (~eye) & (~same)
    cov = 0
    for rep in range(N_COV):
        mu0t = rng.normal(size=d) * 0.3
        dvt = rng.normal(size=d)
        mu1t = mu0t + 0.5 * dvt / np.linalg.norm(dvt)
        s0t = rng.uniform(0.3, 1.0)
        s1t = rng.uniform(0.3, 1.0)
        Bt = np.empty((n, d))
        Bt[m0] = mu0t + s0t * rng.normal(size=(int(m0.sum()), d))
        Bt[m1] = mu1t + s1t * rng.normal(size=(int(m1.sum()), d))
        m_true = fast_margin(unit_rows(Bt), same, diff)
        u0 = Bt[m0].mean(0)
        u1 = Bt[m1].mean(0)
        h0 = sigma_rms(Bt[m0])
        h1 = sigma_rms(Bt[m1])
        ms = np.empty(N_SYNTH)
        for i in range(N_SYNTH):
            Bs = np.empty((n, d))
            Bs[m0] = u0 + h0 * rng.normal(size=(int(m0.sum()), d))
            Bs[m1] = u1 + h1 * rng.normal(size=(int(m1.sum()), d))
            ms[i] = fast_margin(unit_rows(Bs), same, diff)
        lo, hi = np.percentile(ms, [2.5, 97.5])
        if lo <= m_true <= hi:
            cov += 1
    return cov


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2908,
                   'name': 'qwen_attn_boundary_precision',
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
                   'se_p': SE_P, 'p_lo': P_LO, 'p_hi': P_HI,
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
    for g, it in groups.items():
        B, lab = it['B'], it['lab']
        mf = margin_of_B(B, lab)
        ac = acc_of_B(B, lab)
        U = unit_rows(B)
        n = len(lab)
        eye = np.eye(n, dtype=bool)
        same = (lab[:, None] == lab[None, :]) & (~eye)
        diff = (~eye) & (~same)
        mfast = fast_margin(U, same, diff)
        mdl, ch = g.split('_')
        d05 = np.asarray(r05['groups']['%s_%s' % (mdl, ch)]
                         ['delta_per_layer'], dtype=float)
        d08 = B[lab == 1].mean(0) - B[lab == 0].mean(0)
        e2 = float(np.abs(d08 - d05).max())
        e3 = abs(mf - mfast)
        ok = (abs(mf - it['stored_margin']) < TOL_A1
              and abs(ac - it['stored_acc']) < TOL_A1
              and e2 < TOL_A2 and e3 < 1e-12)
        anchor_ok = anchor_ok and ok
        anchor[g] = {'margin_recomp': round(mf, 6),
                     'acc_recomp': round(ac, 6),
                     'deltaB_vs_2905_maxabs': e2,
                     'fast_vs_full': e3, 'ok': bool(ok)}
        log('anchor %s margin %.5f acc %.5f dLB %.2e '
            'fast %.1e ok=%s'
            % (g, mf, ac, e2, e3, ok), lines)

    cov = coverage_audit()
    cov_ok = bool(32 <= cov <= N_COV)
    log('coverage audit (N_SYNTH=%d): %d/%d pass=%s'
        % (N_SYNTH, cov, N_COV, cov_ok), lines)

    verdict = None
    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    elif not cov_ok:
        verdict = 'audit_coverage_fail_all_void'

    # ---------- high-power percentile adjudication ----------
    res_g = {}
    if verdict is None:
        for g in ('glm4_mlp', 'glm4_attn', 'qwen_mlp',
                  'qwen_attn'):
            B, lab = groups[g]['B'], groups[g]['lab']
            m_full = margin_of_B(B, lab)
            m0, m1 = lab == 0, lab == 1
            n = len(lab)
            eye = np.eye(n, dtype=bool)
            same = (lab[:, None] == lab[None, :]) & (~eye)
            diff = (~eye) & (~same)
            mu0, mu1 = B[m0].mean(0), B[m1].mean(0)
            s0, s1 = sigma_rms(B[m0]), sigma_rms(B[m1])
            n0, n1 = int(m0.sum()), int(m1.sum())
            d = B.shape[1]
            ps, qlos, qhis = [], [], []
            for k in range(N_SEEDS):
                rng = np.random.default_rng([SEED_BASE, 10 + k])
                ms = np.empty(N_SYNTH)
                for i in range(N_SYNTH):
                    Bs = np.empty((n, d))
                    Bs[m0] = mu0 + s0 * rng.normal(size=(n0, d))
                    Bs[m1] = mu1 + s1 * rng.normal(size=(n1, d))
                    ms[i] = fast_margin(unit_rows(Bs),
                                        same, diff)
                ps.append(float(np.mean(ms <= m_full)))
                qlos.append(float(np.percentile(ms, 2.5)))
                qhis.append(float(np.percentile(ms, 97.5)))
            ps_arr = np.asarray(ps)
            p_med = float(np.median(ps_arr))
            if g == 'qwen_attn':
                if p_med >= P_HI:
                    gverdict = 'qwen_attn_inside_m1_confirmed'
                elif p_med <= P_LO:
                    gverdict = 'qwen_attn_below_m1_confirmed'
                else:
                    gverdict = 'qwen_attn_at_m1_edge'
            else:
                gverdict = ''
            res_g[g] = {
                'margin_full': round(m_full, 6),
                'p_per_seed': [round(p, 6) for p in ps],
                'p_median': round(p_med, 6),
                'p_min': round(float(ps_arr.min()), 6),
                'p_max': round(float(ps_arr.max()), 6),
                'q2.5_median': round(float(np.median(qlos)), 6),
                'q97.5_median': round(float(np.median(qhis)), 6),
                'sub_verdict': gverdict,
            }
            log('%s: full=%+.6f p_med=%.5f p_range=[%.5f,%.5f] '
                'q2.5=%.6f q97.5=%.6f %s'
                % (g, m_full, p_med, ps_arr.min(), ps_arr.max(),
                   np.median(qlos), np.median(qhis), gverdict),
               lines)
        verdict = res_g['qwen_attn']['sub_verdict']

    log('==== VERDICT: %s ====' % verdict, lines)

    res = {
        'phase': 2908,
        'model': 'glm4+qwen (matrix analysis)',
        'prereg': PREREG,
        'seed_base': SEED_BASE, 'n_synth': N_SYNTH,
        'n_seeds': N_SEEDS, 'se_p': round(SE_P, 6),
        'thresholds': {'p_lo': round(P_LO, 6),
                       'p_hi': round(P_HI, 6)},
        'anchor': anchor, 'anchor_ok': bool(anchor_ok),
        'coverage_audit': {'covered': cov, 'n': N_COV,
                           'pass': bool(cov_ok)},
        'groups': res_g,
        'final_verdict': verdict,
        'runtime_s': round(time.monotonic() - t0, 1),
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    np.savez_compressed(
        os.path.join(OUT, 'qwen_attn_boundary_precision.npz'),
        phase=np.int64(2908))
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2908 verdict=%s' % verdict)


if __name__ == '__main__':
    main()
