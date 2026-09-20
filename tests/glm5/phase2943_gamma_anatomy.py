# -*- coding: utf-8 -*-
"""Phase 2943: gamma negative-offset anatomy (zero-forward).

Why: 2937 P2 found OLS s_c = beta*s_func + gamma with null
gamma in [-16.1, -11.6] (same +3.94) and verdict
scale_collapse_rewrite. 2942 found the joint U8 injection
reproduces the raw shift shape (R1 0.74) but overshoots sep
and its per-word median displacement is ~0 while actual is
-29. Unresolved: is the gamma offset an independent mechanism
component, and how much of the readout collapse does the
linear-shrinkage geometry alone explain?

Mode: ZERO FORWARD. Post-hoc anatomy on frozen artifacts
(2937 proj / 2939 coords / 2942 injection npz). All inputs
(beta, gamma, R1) were displayed in 2937/2942 => the combined
statistics below are NEW but quasi-post-hoc in inputs
(discipline 9): verdicts are registered as mechanism-chain
integration, NOT as discovery-grade flips.

Anchors (frozen):
  a1 proj 2937 vs 2939 vs 2942 max abs < 1e-9
  a2 OLS beta/gamma rebuild vs 2937 result.json < 1e-3
  a3 dcks vs c8_null0 - c8_func0 < 1e-6
  a4 sep_func rebuild 185.6975 < 1e-3
  a5 words/labels consistency 2937 vs 2942

Main tests (frozen):
  T1 residual class structure of the collapse: given the
     OLS decomposition y = beta*f + gamma + resid, the
     identity sep(y) = beta*sep_f + sep(resid) holds by
     construction (run4 correction: the earlier rel_err
     form was tautological). The EMPIRICAL test is whether
     the residual carries class structure beyond
     shrinkage: share_c = |sep_resid_c| / (beta_c*|sep_f|)
     < 0.10 for all 4 null sets.
  T2 gamma independence: |gamma_c - gamma_pred_c| > 3.0 for
     all 4 null sets AND median gamma < -5, where gamma_pred_c
     = P_c - (beta_c-1)*mean(f), P_c = mean U8-rebuilt dproj.
  T3 injection shape independence: partial Spearman
     (d_inj, d_null0 | f) for s in {1,2,3,4}, pass iff median
     < 0.3. NOTE: s=2 injection readout is cross-session
     unstable (2942 finding); robustness taken over s.

Verdict (frozen):
  anchor fail                     => anchor_fail_all_void
  T1 pass AND T2 pass AND T3 pass => regime_signature_confirmed
  T1 pass else                    => partial_signature
  T1 fail                         => collapse_not_linear_shrinkage

Descriptive:
  D1 per-condition 3-term decomposition table
  D2 class-conditional gamma split (gamma_L0 - gamma_L1)
  D3 intercept-gap closure: mean(d_inj)+gamma vs mean(d_null)

Output: phase2943/gamma_anatomy/.
"""
import hashlib
import json
import os
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC_2937 = os.path.join(BASE, 'phase2937', 'scale_collapse',
                        'scale_collapse.npz')
SRC_2939 = os.path.join(BASE, 'phase2939', 'rotation_target',
                        'rotation_target.npz')
SRC_2942 = os.path.join(BASE, 'phase2942', 'u8_joint_injection',
                        'u8_joint_injection.npz')
RES_2937 = os.path.join(BASE, 'phase2937', 'scale_collapse',
                        'result.json')
OUT = os.path.join(BASE, 'phase2943', 'gamma_anatomy')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2943_run_report.txt')

RES_SHARE_MAX = 0.10
GAP_GAMMA_MIN = 3.0
GAMMA_MED_MAX = -5.0
PARTIAL_S_MAX = 0.3
S_ROBUST = (1.0, 2.0, 3.0, 4.0)

PREREG = {
    'mode': 'ZERO FORWARD post-hoc anatomy on 2937/2939/2942 '
            'npz; inputs quasi-post-hoc (discipline 9) => '
            'verdict = mechanism-chain integration',
    'question': 'is the gamma negative offset an independent '
                'mechanism component and how much of the '
                'readout collapse does linear-shrinkage '
                'geometry alone explain?',
    'anchors': {
        'a1': 'proj 2937 vs 2939 vs 2942 < 1e-9',
        'a2': 'OLS rebuild vs 2937 result < 1e-3',
        'a3': 'dcks vs c8_null0-c8_func0 < 1e-6',
        'a4': 'sep_func rebuild < 1e-3',
        'a5': 'words/labels consistency',
    },
    'T1': 'residual class-structure share of the sep '
          'collapse: |sep_resid|/(beta*|sep_f|) < 0.10 for '
          'all 4 null sets; note sep(y)=beta*sep_f+sep(resid) '
          'is an identity, the empirical content is the '
          'residual share only',
    'T2': 'gamma independence: |gamma - gamma_pred| > 3.0 '
          'all 4 null sets AND median gamma < -5; '
          'gamma_pred = P_u8 - (beta-1)*mean(f)',
    'T3': 'partial Spearman(d_inj, d_null0 | f), s in '
          '{1,2,3,4}, median < 0.3; s=2 cross-session '
          'instability noted (2942)',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'T1 pass AND T2 pass AND T3 pass => '
               'regime_signature_confirmed; T1 pass else => '
               'partial_signature; T1 fail => '
               'collapse_not_linear_shrinkage',
    'correction_note': 'run4: T1 was written as rel_err of '
                       'sep prediction - tautological by '
                       'construction (resid := y - beta*f - '
                       'gamma makes sep(y) = beta*sep_f + '
                       'sep(resid) an identity; rel_err '
                       '0.0000 carries no evidence). T1 '
                       'redefined to the genuine empirical '
                       'quantity: residual class-structure '
                       'share |sep_resid|/(beta*|sep_f|) < '
                       '0.10 per null set.',
}


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


def log(msg, lines):
    lines.append(msg)
    print(msg, flush=True)


def spear(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def partial_spear(a, b, c):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    rc = np.argsort(np.argsort(c)).astype(float)

    def res(y, z):
        zm = z.mean()
        beta = float(((y - y.mean()) * (z - zm)).sum()
                     / max(((z - zm) ** 2).sum(), 1e-30))
        return y - y.mean() - beta * (z - zm)

    return float(np.corrcoef(res(ra, rc), res(rb, rc))[0, 1])


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2943,
                   'name': 'gamma_anatomy',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2937': sha8(SRC_2937),
                               's2939': sha8(SRC_2939),
                               's2942': sha8(SRC_2942),
                               'r2937': sha8(RES_2937)},
                   'model': 'qwen3-4b', 'zero_forward': True,
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    # ---------- sources ----------
    z37 = np.load(SRC_2937, allow_pickle=True)
    z39 = np.load(SRC_2939, allow_pickle=True)
    z42 = np.load(SRC_2942, allow_pickle=True)
    r37 = json.load(open(RES_2937, encoding='utf-8'))

    conds37 = [str(s) for s in z37['cond_names']]
    proj37 = z37['proj'].astype(np.float64)
    w37_arr = np.asarray(z37['words']).reshape(
        len(z37['words']), -1)
    words37 = [':'.join(str(p) for p in row)
               for row in w37_arr]
    conds39 = [str(s) for s in z39['cond_names']]
    coords = z39['coords'].astype(np.float64)
    proj39 = z39['proj_dir35'].astype(np.float64)
    Vt8 = z42['Vt8'].astype(np.float64)
    u35 = z42['dirs_word'].astype(np.float64)[35]
    lab = z42['labels_lang'].astype(int)
    scales = z42['scales'].astype(float)
    proj_inj = z42['proj_inj'].astype(np.float64)
    proj_f0 = z42['proj_func0'].astype(np.float64)
    proj_n0 = z42['proj_null0'].astype(np.float64)
    dcks = z42['dcks'].astype(np.float64)
    c8_n0 = z42['c8_null0'].astype(np.float64)
    c8_f0 = z42['c8_func0'].astype(np.float64)
    words42 = [str(w) for w in z42['words']]

    i_f = conds37.index('func')

    # ---------- anchors ----------
    a1_diff = float(np.abs(proj37 - proj39).max())
    a1_ok = bool(a1_diff < 1e-9)
    a1b_diff = float(max(np.abs(proj37[i_f] - proj_f0).max(),
                         np.abs(proj37[conds37.index('null0')]
                                - proj_n0).max()))
    a1_ok = bool(a1_ok and a1b_diff < 1e-9)
    log('a1 proj consistency 37v39 %.2e 37v42 %.2e ok=%s'
        % (a1_diff, a1b_diff, a1_ok), lines)

    fits = {}
    a2_diff = 0.0
    x = proj37[i_f]
    for cn in conds37:
        if cn == 'func':
            continue
        y = proj37[conds37.index(cn)]
        xm, ym = x.mean(), y.mean()
        b = float(((x - xm) * (y - ym)).sum()
                  / ((x - xm) ** 2).sum())
        g = float(ym - b * xm)
        fits[cn] = (b, g)
        ref = r37['P2']['fits'][cn]
        a2_diff = max(a2_diff, abs(b - ref['beta']),
                      abs(g - ref['gamma']))
    a2_ok = bool(a2_diff < 1e-3)
    log('a2 OLS rebuild max dev vs 2937 %.2e ok=%s'
        % (a2_diff, a2_ok), lines)

    a3_diff = float(np.abs(dcks - (c8_n0 - c8_f0)).max())
    a3_ok = bool(a3_diff < 1e-6)
    log('a3 dcks vs c8 diff %.2e ok=%s'
        % (a3_diff, a3_ok), lines)

    sep_f = float(x[lab == 0].mean() - x[lab == 1].mean())
    a4_diff = abs(sep_f - 185.6975)
    a4_ok = bool(a4_diff < 1e-3)
    log('a4 sep_func %.4f dev %.2e ok=%s'
        % (sep_f, a4_diff, a4_ok), lines)

    w37_set = set(words37)
    a5_ok = bool(w37_set == set(words42) and len(lab) == 57
                 and len(words37) == 57)
    log('a5 words/labels consistent ok=%s' % a5_ok, lines)

    anchor_ok = bool(a1_ok and a2_ok and a3_ok and a4_ok
                     and a5_ok)
    verdict = None
    t1 = t2 = t3 = d1 = d2 = d3 = None
    save = {}

    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        mf = float(x.mean())
        n_fin37 = z37['fin_norm'].astype(np.float64)

        # ---------- D1/T1: 3-term decomposition ----------
        d1 = {}
        t1_rels = {}
        log('D1/T1 per-condition decomposition '
            '(sep_f %.2f, mean_f %.2f):' % (sep_f, mf), lines)
        for cn in conds37:
            if cn == 'func':
                continue
            b, g = fits[cn]
            y = proj37[conds37.index(cn)]
            d = y - x
            slope_t = (b - 1.0) * x
            const_t = np.full_like(x, g)
            resid = d - slope_t - const_t

            def sep(v):
                return float(v[lab == 0].mean()
                             - v[lab == 1].mean())

            obs_sep = sep(y)
            # identity check (descriptive only, tautological)
            pred_sep = b * sep_f + sep(resid)
            ident_dev = abs(pred_sep - obs_sep) / abs(sep_f)
            # empirical quantity: residual class share
            share = abs(sep(resid)) / max(b * abs(sep_f),
                                          1e-30)
            if cn.startswith('null'):
                t1_rels[cn] = share
            d1[cn] = {
                'beta': round(b, 4), 'gamma': round(g, 4),
                'sep_readout': round(obs_sep, 2),
                'sep_d': round(sep(d), 2),
                'sep_slope_term': round(sep(slope_t), 2),
                'sep_const_term': round(sep(const_t), 2),
                'sep_resid': round(sep(resid), 2),
                'resid_share': round(share, 4),
                'identity_dev': round(ident_dev, 6),
                'mean_dproj': round(float(d.mean()), 2),
                'resid_mean': round(float(resid.mean()), 2),
            }
            log('  %s beta %.3f gamma %.2f sep(readout) %.1f '
                '(d %.1f) = slope %.1f + const %.1f + '
                'resid %.1f | resid_share %.4f '
                '(identity_dev %.1e)'
                % (cn, b, g, obs_sep, sep(d), sep(slope_t),
                   sep(const_t), sep(resid), share,
                   ident_dev), lines)
        t1_pass = bool(len(t1_rels) == 4
                       and all(v < RES_SHARE_MAX
                               for v in t1_rels.values()))
        t1 = {'resid_shares': {k: round(v, 4)
                               for k, v in
                               t1_rels.items()},
              'threshold': RES_SHARE_MAX, 'pass': t1_pass}
        log('T1 residual class share pass=%s' % t1_pass,
            lines)

        # ---------- T2: gamma independence from U8 ----------
        t2_rows = {}
        for ci, cn in enumerate(conds39):
            if cn == 'func':
                continue
            dc = coords[ci] - coords[conds39.index('func')]
            P = float((dc @ Vt8 @ u35).mean())
            b, g = fits[cn]
            gp = P - (b - 1.0) * mf
            t2_rows[cn] = {'gamma': g, 'gamma_pred_u8': gp,
                           'abs_gap': abs(g - gp)}
        gammas = [v['gamma'] for v in t2_rows.values()]
        gaps = [v['abs_gap'] for v in t2_rows.values()]
        t2_pass = bool(all(v > GAP_GAMMA_MIN for v in gaps)
                       and float(np.median(gammas))
                       < GAMMA_MED_MAX)
        t2 = {'rows': {k: {kk: round(vv, 3) for kk, vv
                           in v.items()}
                       for k, v in t2_rows.items()},
              'median_gamma': round(float(np.median(gammas)),
                                    3),
              'min_gap': round(float(min(gaps)), 3),
              'pass': t2_pass}
        log('T2 gamma independence: gaps %s median_gamma %.2f '
            'pass=%s' % ({k: round(v['abs_gap'], 2)
                          for k, v in t2_rows.items()},
                         float(np.median(gammas)), t2_pass),
            lines)

        # ---------- T3: injection partial correlation ----
        d_null = proj_n0 - proj_f0
        i_if = conds39.index('func')
        ps = {}
        for s in S_ROBUST:
            isi = int(np.argmin(np.abs(scales - s)))
            d_inj = proj_inj[isi] - proj_f0
            ps['%.0f' % s] = round(partial_spear(
                d_inj, d_null, proj_f0), 4)
        pmed = float(np.median(list(ps.values())))
        t3_pass = bool(pmed < PARTIAL_S_MAX)
        raw = round(spear(proj_inj[int(np.argmin(
            np.abs(scales - 2.0)))] - proj_f0, d_null), 4)
        t3 = {'partial_per_s': ps, 'median_partial': pmed,
              'raw_spearman_s2': raw,
              'threshold': PARTIAL_S_MAX, 'pass': t3_pass}
        log('T3 partial spearman %s median %.4f '
            '(raw s2 %.4f) pass=%s'
            % (ps, pmed, raw, t3_pass), lines)

        # ---------- D2: class-conditional gamma split ---
        d2 = {}
        for cn in conds37:
            if cn == 'func':
                continue
            y = proj37[conds37.index(cn)]
            gl = {}
            for lb, tag in ((0, 'L0'), (1, 'L1')):
                xs, ys = x[lab == lb], y[lab == lb]
                xm, ym = xs.mean(), ys.mean()
                bb = float(((xs - xm) * (ys - ym)).sum()
                           / max(((xs - xm) ** 2).sum(),
                                 1e-30))
                gl[tag] = round(float(ym - bb * xm), 3)
            d2[cn] = {'gamma_L0': gl['L0'],
                      'gamma_L1': gl['L1'],
                      'diff': round(gl['L0'] - gl['L1'], 3)}
            log('D2 %s gamma_L0 %.2f gamma_L1 %.2f diff %.2f'
                % (cn, gl['L0'], gl['L1'],
                   gl['L0'] - gl['L1']), lines)

        # ---------- D3: intercept-gap closure ----------
        isi2 = int(np.argmin(np.abs(scales - 2.0)))
        d_inj = proj_inj[isi2] - proj_f0
        g0 = fits['null0'][1]
        d3 = {'mean_d_inj': round(float(d_inj.mean()), 2),
              'mean_d_null': round(float(d_null.mean()), 2),
              'gap_raw': round(float(d_inj.mean()
                                     - d_null.mean()), 2),
              'gap_after_gamma': round(float(
                  (d_inj.mean() + g0) - d_null.mean()), 2),
              'resid_mean_null0':
                  d1['null0']['resid_mean'],
              'note': 'median d_inj +0.874 (2942 R2) vs '
                      'mean -23.97: class-asymmetric '
                      'displacement (2942)'}
        log('D3 gap raw %.2f after +gamma %.2f (resid mean '
            '%.2f)' % (d3['gap_raw'], d3['gap_after_gamma'],
                       d1['null0']['resid_mean']), lines)

        # ---------- verdict ----------
        if t1_pass and t2_pass and t3_pass:
            verdict = 'regime_signature_confirmed'
        elif t1_pass:
            verdict = 'partial_signature'
        else:
            verdict = 'collapse_not_linear_shrinkage'

        save = {
            'cond_names': np.array(conds37),
            'beta_gamma': np.array([fits[cn]
                                    for cn in conds37
                                    if cn != 'func']),
            't1_resid_shares': np.array(
                [t1_rels[cn] for cn in sorted(t1_rels)]),
            't2_gaps': np.array(gaps),
            't3_partial': np.array(
                [ps['%.0f' % s] for s in S_ROBUST]),
            'labels_lang': lab,
        }

    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2943, 'model': 'qwen3-4b',
           'prereg': PREREG,
           'anchors': {'a1_diff': float('%.3e' % a1_diff),
                       'a1b_diff': float('%.3e' % a1b_diff),
                       'a1_ok': a1_ok,
                       'a2_diff': float('%.3e' % a2_diff),
                       'a2_ok': a2_ok,
                       'a3_diff': float('%.3e' % a3_diff),
                       'a3_ok': a3_ok,
                       'a4_diff': float('%.3e' % a4_diff),
                       'a4_ok': a4_ok,
                       'a5_ok': a5_ok,
                       'ok': anchor_ok},
           'T1': t1, 'T2': t2, 'T3': t3,
           'D1': d1, 'D2': d2, 'D3': d3,
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if save:
        np.savez_compressed(os.path.join(
            OUT, 'gamma_anatomy.npz'), **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2943 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
