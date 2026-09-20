# -*- coding: utf-8 -*-
"""Phase 2936: semantic-context suppression law formalization.

Why: 2935 showed null-context CI amplification is driven by
context statistics (H2, verdict null_amp_mixed) and the seal
found an amplification-gradient INVERSION: the cells amplified
most are precisely the load-band / early-layer cells with the
largest func CI (L6 x3.15 vs L11 x1.59). This suggests an
ANCHORING model: semantic context subtracts a nearly constant
additive amount from ablation sensitivity, so
    supp(c) = ci_raw_null(c) - ci_raw_func(c) = a + b*ci_func(c)
with a > 0 (additive anchor), rather than a multiplicative
model supp = (k-1)*ci_func (constant ratio). The two models
are distinguishable: additive anchor predicts amp = 1 + a/ci
+ b (large-amp at small-ci cells = gradient inversion as a
COROLLARY).

Mode: ZERO forward. Pure re-analysis of 2935 npz (func +
4 resampled null sets, raw CI recoverable via ci_rel * scale)
and 2934 npz (same/func/null x pos0/pos1/pos01) plus 2930
lin_r, 2931 masks, 2933 func CI.

Anchors (frozen):
  a1 2935 func ci_rel vs 2933 npz ci_rel max abs diff < 1e-12
     (both relative to func scale)
  a2 cell grids identical across 2933/2934/2935
  a3 source SHA chain equals 2935 execution.json sources
  a4 2934 pos1/func raw CI vs 2935 func raw CI max abs
     diff < 1e-4 (cross-phase, same batch57 composition)
  a5 mask counts 469/382/295

Main tests (frozen):
  P1 additive-anchor law: for each null set r (4 sets),
     supp_r(c) = raw_null_r(c) - raw_func(c) vs raw_func(c)
     over 1120 cells: (i) Spearman; (ii) OLS supp = a + b*ci:
     R2_lin; (iii) permutation null for intercept a (rng 2917,
     10000 cell-pair shuffles): p_a = P(a_null >= a_obs);
     (iv) log-log power-fit R2 on supp>0 cells (count of
     supp<=0 registered). Set-level pass: R2_lin >= 0.8 AND
     a_obs > 0 AND p_a <= 0.01 AND R2_lin >= R2_pow.
  P2 mechanism face: k_r(c) = amp_r(c) - 1 vs ci_rel_func(c)
     Spearman over 1120 cells (prediction negative); median
     over 4 sets.
  P3 cross-condition generality: supp_same(c) = raw_same(c) -
     raw_func(c) from 2934 pos1 rows; same P1 test battery
     (one set).
  P4 descriptive: per-layer median supp_r vs median raw_func
     and vs lin_r (layer-level Spearman, 35 layers); within-
     layer Spearman(supp, raw_func) profile (cross-scale
     lesson: layer-stratified report mandatory).

Verdict (frozen):
  anchor fail                              => anchor_fail_all_void
  P1 median over 4 sets passes AND P3 passes (R2>=0.7, a>0,
     p_a<=0.05)                            => anchoring_law_linear_general
  P1 median passes AND P3 R2>=0.5 but battery not met
                                           => anchoring_law_linear_nullonly
  P1 median Spearman >= 0.8 but linear battery not met
                                           => anchoring_law_monotone
  P1 median Spearman >= 0.5                => anchoring_law_partial
  else                                     => anchoring_not_established

Output: phase2936/anchoring_law/.
"""
import hashlib
import json
import os
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC_2927 = os.path.join(BASE, 'phase2927', 'probe_relativity',
                        'probe_relativity.npz')
SRC_2929 = os.path.join(BASE, 'phase2929',
                        'response_structure_atlas',
                        'response_structure_atlas.npz')
SRC_2930 = os.path.join(BASE, 'phase2930', 'direction_flip_control',
                        'direction_flip_control.npz')
SRC_2931 = os.path.join(BASE, 'phase2931', 'skeleton_overlap_null',
                        'skeleton_overlap_null.npz')
SRC_2933 = os.path.join(BASE, 'phase2933', 'full_atlas_ci',
                        'full_atlas_ci.npz')
SRC_2934 = os.path.join(BASE, 'phase2934', 'loadband_anatomy',
                        'loadband_anatomy.npz')
SRC_2935 = os.path.join(BASE, 'phase2935', 'null_amp_anatomy',
                        'null_amp_anatomy.npz')
EXEC_2935 = os.path.join(BASE, 'phase2935', 'null_amp_anatomy',
                         'execution.json')
OUT = os.path.join(BASE, 'phase2936', 'anchoring_law')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2936_run_report.txt')
NH, HD = 32, 128
NL = 36
N_GRP = 10000
RNG_INT = 2917
EXP_N86, EXP_NWD, EXP_SHARED = 469, 382, 295
A1_TOL = 1e-12
A4_TOL = 1e-4

PREREG = {
    'mode': 'ZERO forward. Re-analysis of 2935 npz (func + 4 '
            'null sets, raw CI = ci_rel * scale) + 2934 npz '
            '(same/func/null x 3 cfgs) + 2930 lin_r + 2931 '
            'masks + 2933 func CI. 1120 gated cells.',
    'question': 'what is the functional form of semantic-'
                'context suppression of ablation sensitivity: '
                'additive anchor (supp = a + b*ci, a>0) or '
                'multiplicative (supp proportional to ci)?',
    'anchors': {
        'a1': '2935 func ci_rel vs 2933 npz ci_rel max abs '
              '< 1e-12',
        'a2': 'cell grids identical 2933/2934/2935',
        'a3': 'source SHA chain equals 2935 execution.json '
              'sources',
        'a4': '2934 pos1/func raw CI vs 2935 func raw CI max '
              'abs diff < 1e-4',
        'a5': 'mask counts 469/382/295',
    },
    'P1': 'per null set r: supp_r = raw_null_r - raw_func vs '
          'raw_func over 1120 cells: Spearman + OLS R2_lin + '
          'intercept permutation (rng 2917, 10000) p_a + '
          'log-log power R2 (supp>0 subset); set pass = '
          'R2_lin>=0.8 AND a>0 AND p_a<=0.01 AND '
          'R2_lin>=R2_pow',
    'P2': 'k = amp - 1 vs ci_rel_func Spearman (prediction '
          'negative), median over 4 sets',
    'P3': 'supp_same from 2934 pos1 rows, same battery '
          '(one set)',
    'P4': 'layer-median supp vs raw_func and vs lin_r; '
          'within-layer Spearman profile',
    'verdict': 'anchor fail => anchor_fail_all_void; P1 '
               'median passes AND P3 (R2>=0.7, a>0, p_a<=0.05)'
               ' => anchoring_law_linear_general; P1 median '
               'passes AND P3 R2>=0.5 else => '
               'anchoring_law_linear_nullonly; P1 median '
               'Spearman>=0.8 else => anchoring_law_monotone; '
               'Spearman>=0.5 => anchoring_law_partial; else '
               'anchoring_not_established',
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


def avg_ranks(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x))
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(x, y):
    rx = avg_ranks(x)
    ry = avg_ranks(y)
    if rx.std() < 1e-30 or ry.std() < 1e-30:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def ols_r2(x, y):
    """Return (slope, intercept, r2) of y = a + b x."""
    xm, ym = x.mean(), y.mean()
    sxx = float(((x - xm) ** 2).sum())
    sxy = float(((x - xm) * (y - ym)).sum())
    b = sxy / max(sxx, 1e-30)
    a = float(ym - b * xm)
    pred = a + b * x
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - ym) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
    return b, a, float(r2)


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2936,
                   'name': 'anchoring_law',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2927': sha8(SRC_2927),
                               's2929': sha8(SRC_2929),
                               's2930': sha8(SRC_2930),
                               's2931': sha8(SRC_2931),
                               's2933': sha8(SRC_2933),
                               's2934': sha8(SRC_2934),
                               's2935': sha8(SRC_2935)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'rng_int': RNG_INT, 'n_grp': N_GRP,
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    # ---------- sources ----------
    z29 = np.load(SRC_2929, allow_pickle=True)
    _ = z29['rho_grid'].astype(np.float64)  # source-chain pin
    z30 = np.load(SRC_2930, allow_pickle=True)
    lin_r = z30['lin_r_profile'].astype(np.float64)
    z31 = np.load(SRC_2931, allow_pickle=True)
    S29c = z31['S29_corrected'].astype(bool)
    Smirc = z31['Smir_corrected'].astype(bool)
    shared = S29c & Smirc
    n86, nwd, nsh = int(S29c.sum()), int(Smirc.sum()), \
        int(shared.sum())
    a5_ok = bool(n86 == EXP_N86 and nwd == EXP_NWD
                 and nsh == EXP_SHARED)
    log('a5 mask counts n86=%d nwd=%d shared=%d ok=%s'
        % (n86, nwd, nsh, a5_ok), lines)

    z33 = np.load(SRC_2933, allow_pickle=True)
    cells_33 = [tuple(int(v) for v in c)
                for c in z33['cells']]
    ci33 = z33['ci_rel'].astype(np.float64)
    z34 = np.load(SRC_2934, allow_pickle=True)
    cells_34 = [tuple(int(v) for v in c)
                for c in z34['cells']]
    cfgs_34 = [str(s) for s in z34['cfgs']]
    conds_34 = [str(s) for s in z34['conds']]
    z35 = np.load(SRC_2935, allow_pickle=True)
    cells_35 = [tuple(int(v) for v in c)
                for c in z35['cells']]
    a2_ok = bool(cells_33 == cells_34 == cells_35)
    log('a2 cell grids identical ok=%s (n=%d/%d/%d)'
        % (a2_ok, len(cells_33), len(cells_34),
           len(cells_35)), lines)
    cells = cells_33
    pos_of = {c: i for i, c in enumerate(cells)}

    # a3: source SHA chain vs 2935 execution.json
    exec35 = json.load(open(EXEC_2935, encoding='utf-8'))
    src35 = exec35['sources']
    a3_pairs = {'s2927': sha8(SRC_2927), 's2929': sha8(SRC_2929),
                's2930': sha8(SRC_2930), 's2931': sha8(SRC_2931),
                's2933': sha8(SRC_2933)}
    a3_ok = bool(all(src35[k] == v for k, v in
                     a3_pairs.items()))
    log('a3 source SHA chain vs 2935 ok=%s' % a3_ok, lines)

    # ---------- a1: 2935 func ci_rel vs 2933 ----------
    conds35 = [str(s) for s in z35['cond_names']]
    if_func35 = conds35.index('func')
    ci35_rel = z35['ci_rel'].astype(np.float64)
    scale35 = z35['scale'].astype(np.float64)
    f35_rel = np.array([ci35_rel[if_func35, pos_of[c]]
                        for c in cells])
    a1_diff = float(np.abs(f35_rel - ci33).max())
    a1_ok = bool(a1_diff < A1_TOL)
    log('a1 2935 func ci_rel vs 2933 max abs diff %.2e ok=%s'
        % (a1_diff, a1_ok), lines)

    # ---------- a4: 2934 pos1/func vs 2935 (raw) ----------
    ci34_rel = z34['ci_rel'].astype(np.float64)
    scale34 = z34['scale'].astype(np.float64)
    ic_f = conds_34.index('func')
    ic_s = conds_34.index('same')
    ig_p1 = cfgs_34.index('pos1')
    row_pf = ig_p1 * len(conds_34) + ic_f
    raw_func35 = f35_rel * scale35[if_func35]
    raw_pf34 = ci34_rel[row_pf] * scale34[ic_f]
    a4_diff = float(np.abs(
        [raw_pf34[pos_of[c]] - raw_func35[pos_of[c]]
         for c in cells]).max())
    a4_ok = bool(a4_diff < A4_TOL)
    log('a4 2934 pos1/func raw CI vs 2935 func raw max abs '
        'diff %.2e ok=%s' % (a4_diff, a4_ok), lines)

    anchor_ok = bool(a1_ok and a2_ok and a3_ok and a4_ok
                     and a5_ok)
    verdict = None
    p1 = p2 = p3 = p4 = None
    save = {}

    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        inulls = [i for i, s in enumerate(conds35)
                  if s.startswith('null')]
        null_names = [conds35[i] for i in inulls]
        raw_null35 = {conds35[i]:
                      ci35_rel[i] * scale35[i]
                      for i in inulls}
        pos1_same_rel = ci34_rel[ig_p1 * len(conds_34) + ic_s]
        raw_same = pos1_same_rel * scale34[ic_s]

        def battery(supp, ci_base, rng_int, tag):
            """Full P1/P3 battery on one suppression set."""
            rho = spearman(supp, ci_base)
            b, a_obs, r2_lin = ols_r2(ci_base, supp)
            # intercept permutation null
            rng = np.random.default_rng(rng_int)
            cnt = 0
            for _ in range(N_GRP):
                pm = rng.permutation(len(supp))
                _, a_n, _ = ols_r2(ci_base[pm], supp)
                if a_n >= a_obs:
                    cnt += 1
            p_a = float((cnt + 1) / (N_GRP + 1))
            # log-log power fit on supp>0 subset
            m = (supp > 0) & (ci_base > 0)
            n_nonpos = int((supp <= 0).sum())
            if m.sum() >= 10:
                lx = np.log(ci_base[m])
                ly = np.log(supp[m])
                _, _, r2_pow = ols_r2(lx, ly)
            else:
                r2_pow = float('nan')
            ok = bool(r2_lin >= 0.8 and a_obs > 0
                      and p_a <= 0.01 and r2_lin >= r2_pow)
            res = {'tag': tag,
                   'spearman': round(rho, 4),
                   'slope': round(float(b), 4),
                   'intercept': round(a_obs, 6),
                   'p_intercept': float('%.3e' % p_a),
                   'r2_linear': round(r2_lin, 4),
                   'r2_power_loglog': round(r2_pow, 4),
                   'n_supp_le0': n_nonpos,
                   'pass': ok}
            log('battery %s: rho %.4f a %.6f p_a %.3e '
                'R2lin %.4f R2pow %.4f supp<=0 %d pass=%s'
                % (tag, rho, a_obs, p_a, r2_lin, r2_pow,
                   n_nonpos, ok), lines)
            return res

        # ---------- P1: per null set ----------
        p1 = {'sets': []}
        for k, nm in enumerate(null_names):
            supp = (raw_null35[nm] - raw_func35)
            r = battery(supp, raw_func35,
                        RNG_INT + k, nm)
            p1['sets'].append(r)
        med_rho = float(np.median(
            [r['spearman'] for r in p1['sets']]))
        med_r2 = float(np.median(
            [r['r2_linear'] for r in p1['sets']]))
        all_pass = all(r['pass'] for r in p1['sets'])
        med_pass = bool(np.median(
            [float(r['pass']) for r in p1['sets']]) > 0.5)
        p1['median_spearman'] = round(med_rho, 4)
        p1['median_r2_linear'] = round(med_r2, 4)
        p1['all_sets_pass'] = all_pass
        log('P1 median rho %.4f median R2lin %.4f '
            'all_pass=%s' % (med_rho, med_r2, all_pass),
            lines)

        # ---------- P2: k = amp - 1 vs ci_func ----------
        amp = z35['amp'].astype(np.float64)
        amp_names = [str(s) for s in z35['amp_names']]
        k_rhos = []
        for i, nm in enumerate(amp_names):
            kk = amp[i] - 1.0
            k_rhos.append(spearman(kk, f35_rel))
        p2 = {'rhos': [round(v, 4) for v in k_rhos],
              'median': round(float(np.median(k_rhos)), 4)}
        log('P2 k-vs-ci_func rhos %s median %.4f '
            '(prediction negative)'
            % (p2['rhos'], p2['median']), lines)

        # ---------- P3: same-condition battery ----------
        supp_same = raw_same - raw_func35
        p3 = battery(supp_same, raw_func35,
                     RNG_INT + 100, 'same(pos1)')

        # ---------- P4: layer profiles ----------
        med_supp_l = np.zeros(NL)
        med_ci_l = np.zeros(NL)
        med_supp_same_l = np.zeros(NL)
        wl_rhos = np.zeros(NL)
        for li in range(1, NL):
            ix = [pos_of[(h, li)] for h in range(NH)]
            med_supp_l[li] = float(np.median(
                raw_null35[null_names[0]][ix]
                - raw_func35[ix]))
            med_ci_l[li] = float(np.median(
                raw_func35[ix]))
            med_supp_same_l[li] = float(np.median(
                supp_same[ix]))
            wl_rhos[li] = spearman(
                raw_null35[null_names[0]][ix]
                - raw_func35[ix], raw_func35[ix])
        r_law_supp_ci = spearman(med_supp_l[1:],
                                 med_ci_l[1:])
        r_law_supp_linr = spearman(med_supp_l[1:],
                                   lin_r[1:])
        r_law_same_ci = spearman(med_supp_same_l[1:],
                                 med_ci_l[1:])
        p4 = {'layer_median_spearman_supp_ci':
              round(r_law_supp_ci, 4),
              'layer_median_spearman_supp_linr':
              round(r_law_supp_linr, 4),
              'layer_median_spearman_same_ci':
              round(r_law_same_ci, 4),
              'within_layer_rho_median':
              round(float(np.median(wl_rhos[1:])), 4),
              'within_layer_rho_max':
              round(float(wl_rhos[1:].max()), 4),
              'med_supp_by_layer':
              [round(float(v), 6)
               for v in med_supp_l[1:]],
              'med_ci_by_layer':
              [round(float(v), 6)
               for v in med_ci_l[1:]]}
        log('P4 layer-level: rho(supp_l, ci_l) %.4f '
            'rho(supp_l, lin_r) %.4f | within-layer rho '
            'median %.4f max %.4f'
            % (r_law_supp_ci, r_law_supp_linr,
               p4['within_layer_rho_median'],
               p4['within_layer_rho_max']), lines)

        # ---------- verdict ----------
        p3_pass_strict = bool(p3['r2_linear'] >= 0.7
                              and p3['intercept'] > 0
                              and p3['p_intercept'] <= 0.05)
        p3_pass_loose = bool(p3['r2_linear'] >= 0.5)
        if med_pass and p3_pass_strict:
            verdict = 'anchoring_law_linear_general'
        elif med_pass and p3_pass_loose:
            verdict = 'anchoring_law_linear_nullonly'
        elif med_rho >= 0.8:
            verdict = 'anchoring_law_monotone'
        elif med_rho >= 0.5:
            verdict = 'anchoring_law_partial'
        else:
            verdict = 'anchoring_not_established'

        save = {
            'cells': np.array(cells, dtype=np.int64),
            'supp_null0': (raw_null35[null_names[0]]
                           - raw_func35),
            'supp_null1': (raw_null35[null_names[1]]
                           - raw_func35),
            'supp_same': supp_same,
            'raw_func': raw_func35,
            'wl_rhos': wl_rhos}

    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2936, 'model': 'qwen3-4b',
           'prereg': PREREG,
           'anchors': {'a1_diff': float('%.3e' % a1_diff),
                       'a1_ok': a1_ok,
                       'a2_ok': a2_ok,
                       'a3_ok': a3_ok,
                       'a4_diff': float('%.3e' % a4_diff),
                       'a4_ok': a4_ok,
                       'a5_ok': a5_ok,
                       'ok': anchor_ok},
           'P1': p1, 'P2': p2, 'P3': p3, 'P4': p4,
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if save:
        np.savez_compressed(os.path.join(OUT,
                                         'anchoring_law.npz'),
                            **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2936 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
