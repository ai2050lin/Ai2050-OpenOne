# -*- coding: utf-8 -*-
"""Phase 2907: attn margin shape-correction law (zero-forward).

2906: M1 isotropic summary (class means + per-class SCALAR
variance) reconstructs both mlp margins but BOTH attn margins
fall BELOW their M1 intervals (glm4 attn 0.1213 < lo 0.1649;
qwen attn -0.0195 < lo -0.0116) - attn within-class shape
beyond isotropy suppresses the margin.  Here the summary is
coarsened stepwise to locate the suppressing component:

  M1  : x_i = mu_c(i) + sigma_c * z          (scalar, 2906)
  M2a : x_i = mu_c(i) + diag(Sigma_c)^{1/2} z (per-layer variance
        profile, no cross-layer correlation)
  M2b : x_i = mu_c(i) + L z, L=chol(Sigma_c)  (full covariance)

per group 400 synth draws -> 95% interval; level(g) = deepest
summary whose interval contains the real margin_full.

Coverage audits (2809 institutionalized; rng [2907,0] for M2a,
[2907,3] for M2b): 40 reps each with non-trivial true Sigma
(M2a: random per-layer variances; M2b: random full PSD), pass
iff covered in [32,40]/40.

Anchors: a1 margin/acc == stored (abs 2e-5); a2 Delta_B ==
2905 delta_per_layer (abs 1e-4, round4 storage).

Verdict (frozen):
  anchors fail                  => anchor_fail_all_void
  either coverage audit fails   => audit_coverage_fail_all_void
  attn group levels:
    both M2a                    => shape_correction_diagonal
    both M2b (not M2a)          => shape_correction_full_covariance
    both none                   => shape_correction_failed
    split                       => shape_correction_mixed
  mlp groups reported as sanity levels only.

Descriptive per group: Sigma_c eigen-profile, layer variance
profile, cross-layer correlation summary, per-level intervals.

SEED=2907.  Output: phase2907/shape_correction_law/.
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
OUT = os.path.join(BASE, 'phase2907', 'shape_correction_law')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2907_run_report.txt')
SEED = 2907
N_SYNTH = 400
N_COV = 40
TOL_A1 = 2e-5
TOL_A2 = 1e-4

PREREG = {
    'mode': 'zero_forward_matrix_analysis',
    'sources': 'B matrices from 2902/2903 npz; stored '
               'margin/acc (2902/2903); delta_per_layer (2905, '
               'round4)',
    'anchors': 'a1 margin/acc recomputed == stored within abs '
               '2e-5; a2 Delta_B recomputed == 2905 '
               'delta_per_layer within abs 1e-4 (round4)',
    'summary_ladder': 'M1: mu_c + sigma_c I (scalar per class); '
                      'M2a: mu_c + diag(Sigma_c)^{1/2} (per-layer '
                      'variance profile); M2b: mu_c + chol'
                      '(Sigma_c) (full covariance, Sigma_c = '
                      'sample covariance of the same class rows, '
                      'n>=d+1 both classes)',
    'coverage_audits': 'rng [2907,0] M2a (true diag Sigma with '
                       'random per-layer variances 0.09-1.0) and '
                       '[2907,3] M2b (true full PSD Sigma = A A^T '
                       '+0.1 I, A random 10x10): 40 reps each, '
                       'estimate from drawn matrix, 400-synth 95% '
                       'interval, pass iff coverage in [32,40]/40; '
                       'failure => audit_coverage_fail_all_void',
    'synthesis': 'per group, per level, 400 draws, shared rng '
                 'stream SEED=2907, order glm4-mlp glm4-attn '
                 'qwen-mlp qwen-attn then M1 M2a M2b; class '
                 'sizes fixed',
    'verdict': 'anchors fail => anchor_fail_all_void; coverage '
               'audit fail => audit_coverage_fail_all_void; attn '
               'groups: both level M2a => '
               'shape_correction_diagonal; both M2b (not M2a) => '
               'shape_correction_full_covariance; both none => '
               'shape_correction_failed; split => '
               'shape_correction_mixed; mlp groups sanity only',
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


def synth_margin(mu0, mu1, L0, L1, lab, rng, n_synth):
    n = len(lab)
    m0 = lab == 0
    m1 = lab == 1
    d = mu0.shape[0]
    ms = []
    for _ in range(n_synth):
        Bs = np.empty((n, d))
        z0 = rng.normal(size=(int(m0.sum()), d))
        z1 = rng.normal(size=(int(m1.sum()), d))
        if L0 is None:
            Bs[m0] = mu0 + z0
        else:
            Bs[m0] = mu0 + z0 @ L0.T
        if L1 is None:
            Bs[m1] = mu1 + z1
        else:
            Bs[m1] = mu1 + z1 @ L1.T
        ms.append(margin_of_B(Bs, lab))
    return np.asarray(ms)


def interval(ms):
    return float(np.percentile(ms, 2.5)), \
        float(np.percentile(ms, 97.5))


def cholesky_psd(S):
    w = np.linalg.eigvalsh(S)
    ridge = 0.0
    for k in range(6):
        try:
            return np.linalg.cholesky(
                S + ridge * np.eye(S.shape[0]))
        except np.linalg.LinAlgError:
            ridge = max(ridge * 10, 1e-10)
    raise np.linalg.LinAlgError('chol failed min eig %g' % w.min())


def coverage_audit(kind):
    rng = np.random.default_rng(
        [SEED, 0 if kind == 'M2a' else 3])
    d, n, n0 = 10, 57, 22
    lab = np.array([0] * n0 + [1] * (n - n0))
    cov = 0
    for rep in range(N_COV):
        mu0t = rng.normal(size=d) * 0.3
        dvt = rng.normal(size=d)
        mu1t = mu0t + 0.5 * dvt / np.linalg.norm(dvt)
        if kind == 'M2a':
            v0 = rng.uniform(0.09, 1.0, size=d)
            v1 = rng.uniform(0.09, 1.0, size=d)
            S0t, S1t = np.diag(v0), np.diag(v1)
            L0t, L1t = np.diag(np.sqrt(v0)), np.diag(np.sqrt(v1))
        else:
            A0 = rng.normal(size=(d, d))
            A1 = rng.normal(size=(d, d))
            S0t = A0 @ A0.T / d + 0.1 * np.eye(d)
            S1t = A1 @ A1.T / d + 0.1 * np.eye(d)
            L0t = cholesky_psd(S0t)
            L1t = cholesky_psd(S1t)
        m0 = lab == 0
        m1 = lab == 1
        Bt = np.empty((n, d))
        Bt[m0] = mu0t + rng.normal(size=(int(m0.sum()), d)) @ L0t.T
        Bt[m1] = mu1t + rng.normal(size=(int(m1.sum()), d)) @ L1t.T
        m_true = margin_of_B(Bt, lab)
        u0 = Bt[m0].mean(0)
        u1 = Bt[m1].mean(0)
        S0h = np.cov(Bt[m0], rowvar=False)
        S1h = np.cov(Bt[m1], rowvar=False)
        if kind == 'M2a':
            L0h = np.diag(np.sqrt(np.diag(S0h)))
            L1h = np.diag(np.sqrt(np.diag(S1h)))
        else:
            L0h = cholesky_psd(S0h)
            L1h = cholesky_psd(S1h)
        ms = synth_margin(u0, u1, L0h, L1h, lab, rng, N_SYNTH)
        lo_, hi_ = interval(ms)
        if lo_ <= m_true <= hi_:
            cov += 1
    return cov


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2907,
                   'name': 'shape_correction_law',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8': sha8(os.path.abspath(__file__)),
                   'sources': {'s2902': sha8(SRC['glm4'][0]),
                               'r2902': sha8(SRC['glm4'][1]),
                               's2903': sha8(SRC['qwen'][0]),
                               'r2903': sha8(SRC['qwen'][1]),
                               'r2905': sha8(R2905)},
                   'mode': 'zero_forward_matrix_analysis',
                   'seed': SEED, 'n_synth': N_SYNTH,
                   'n_cov_reps': N_COV, 'prereg': PREREG},
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
    for g, it in groups.items():
        mf = margin_of_B(it['B'], it['lab'])
        ac = acc_of_B(it['B'], it['lab'])
        mdl, ch = g.split('_')
        d05 = np.asarray(r05['groups']['%s_%s' % (mdl, ch)]
                         ['delta_per_layer'], dtype=float)
        d06 = it['B'][it['lab'] == 1].mean(0) \
            - it['B'][it['lab'] == 0].mean(0)
        e2 = float(np.abs(d06 - d05).max())
        ok = (abs(mf - it['stored_margin']) < TOL_A1
              and abs(ac - it['stored_acc']) < TOL_A1
              and e2 < TOL_A2)
        anchor_ok = anchor_ok and ok
        log('anchor %s margin %.5f acc %.5f dLB %.2e ok=%s'
            % (g, mf, ac, e2, ok), lines)

    # ---------- coverage audits ----------
    cov_a = coverage_audit('M2a')
    cov_a_ok = bool(32 <= cov_a <= N_COV)
    log('coverage M2a: %d/%d pass=%s'
        % (cov_a, N_COV, cov_a_ok), lines)
    cov_b = coverage_audit('M2b')
    cov_b_ok = bool(32 <= cov_b <= N_COV)
    log('coverage M2b: %d/%d pass=%s'
        % (cov_b, N_COV, cov_b_ok), lines)

    verdict = None
    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    elif not (cov_a_ok and cov_b_ok):
        verdict = 'audit_coverage_fail_all_void'

    # ---------- ladder ----------
    res_g = {}
    rng_main = np.random.default_rng(SEED)
    if verdict is None:
        for g in ('glm4_mlp', 'glm4_attn', 'qwen_mlp',
                  'qwen_attn'):
            it = groups[g]
            B, lab = it['B'], it['lab']
            m_full = margin_of_B(B, lab)
            m0 = lab == 0
            m1 = lab == 1
            mu0 = B[m0].mean(0)
            mu1 = B[m1].mean(0)
            S0 = np.cov(B[m0], rowvar=False)
            S1 = np.cov(B[m1], rowvar=False)
            d = B.shape[1]
            s0 = float(np.sqrt(np.trace(S0) / d))
            s1 = float(np.sqrt(np.trace(S1) / d))
            lv = {}
            # M1
            ms = synth_margin(mu0, mu1, s0 * np.eye(d),
                              s1 * np.eye(d), lab, rng_main,
                              N_SYNTH)
            lo_, hi_ = interval(ms)
            lv['M1'] = (lo_ <= m_full <= hi_)
            lv['M1_interval'] = (round(lo_, 6), round(hi_, 6))
            # M2a
            L0a = np.diag(np.sqrt(np.diag(S0)))
            L1a = np.diag(np.sqrt(np.diag(S1)))
            ms = synth_margin(mu0, mu1, L0a, L1a, lab, rng_main,
                              N_SYNTH)
            lo_, hi_ = interval(ms)
            lv['M2a'] = (lo_ <= m_full <= hi_)
            lv['M2a_interval'] = (round(lo_, 6), round(hi_, 6))
            # M2b
            L0b = cholesky_psd(S0)
            L1b = cholesky_psd(S1)
            ms = synth_margin(mu0, mu1, L0b, L1b, lab, rng_main,
                              N_SYNTH)
            lo_, hi_ = interval(ms)
            lv['M2b'] = (lo_ <= m_full <= hi_)
            lv['M2b_interval'] = (round(lo_, 6), round(hi_, 6))
            level = 'M1' if lv['M1'] else (
                'M2a' if lv['M2a'] else (
                    'M2b' if lv['M2b'] else 'none'))
            ev0 = np.linalg.eigvalsh(S0)
            ev1 = np.linalg.eigvalsh(S1)
            Crl = np.corrcoef(B.T)
            off = Crl[~np.eye(d, dtype=bool)]
            entry = {
                'margin_full': round(m_full, 6),
                'level': level,
                'M1_interval': list(lv['M1_interval']),
                'M2a_interval': list(lv['M2a_interval']),
                'M2b_interval': list(lv['M2b_interval']),
                'inside_M1': bool(lv['M1']),
                'inside_M2a': bool(lv['M2a']),
                'inside_M2b': bool(lv['M2b']),
                'sigma0_iso': round(s0, 5),
                'sigma1_iso': round(s1, 5),
                'diag_share_S0': round(
                    float(np.trace(S0) / max(
                        float(np.abs(ev0).sum()), 1e-30)), 4),
                'diag_share_S1': round(
                    float(np.trace(S1) / max(
                        float(np.abs(ev1).sum()), 1e-30)), 4),
                'layer_var_ratio_S0': round(
                    float(np.diag(S0).max()
                          / max(np.diag(S0).min(), 1e-30)), 2),
                'layer_var_ratio_S1': round(
                    float(np.diag(S1).max()
                          / max(np.diag(S1).min(), 1e-30)), 2),
                'cross_layer_corr_mean': round(
                    float(np.mean(np.abs(off))), 4),
            }
            res_g[g] = entry
            log('%s: full=%.4f level=%s M1=[%.4f,%.4f] '
                'M2a=[%.4f,%.4f] M2b=[%.4f,%.4f] '
                'layVarRatio=%.1f corr=%.3f'
                % (g, m_full, level,
                   *lv['M1_interval'], *lv['M2a_interval'],
                   *lv['M2b_interval'],
                   entry['layer_var_ratio_S0'],
                   entry['cross_layer_corr_mean']), lines)
        a_lev = (res_g['glm4_attn']['level'],
                 res_g['qwen_attn']['level'])
        log('attn levels: glm4=%s qwen=%s' % a_lev, lines)
        if a_lev[0] == a_lev[1] == 'M2a':
            verdict = 'shape_correction_diagonal'
        elif a_lev[0] == a_lev[1] == 'M2b':
            verdict = 'shape_correction_full_covariance'
        elif a_lev[0] == a_lev[1] == 'none':
            verdict = 'shape_correction_failed'
        else:
            verdict = 'shape_correction_mixed'

    log('==== VERDICT: %s ====' % verdict, lines)

    res = {
        'phase': 2907, 'model': 'glm4+qwen (matrix analysis)',
        'prereg': PREREG, 'seed': SEED, 'n_synth': N_SYNTH,
        'coverage_audits': {'M2a': cov_a, 'M2b': cov_b,
                            'n': N_COV,
                            'pass': bool(cov_a_ok and cov_b_ok)},
        'anchor_ok': bool(anchor_ok),
        'groups': res_g,
        'final_verdict': verdict,
        'runtime_s': round(time.monotonic() - t0, 1),
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    npz_data = {}
    for g, e in res_g.items():
        npz_data['%s_margin_full' % g] = np.asarray(
            e['margin_full'])
        for k in ('M1_interval', 'M2a_interval', 'M2b_interval'):
            npz_data['%s_%s' % (g, k)] = np.asarray(e[k])
        npz_data['%s_layer_var_S0' % g] = np.asarray([e[
            'layer_var_ratio_S0']])
    if npz_data:
        np.savez_compressed(
            os.path.join(OUT, 'shape_correction_law.npz'),
            **npz_data)

    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
