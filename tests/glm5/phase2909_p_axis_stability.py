# -*- coding: utf-8 -*-
"""Phase 2909: p-axis spectrum stability (preregistered).

Zero-forward matrix analysis.

Why: 2908 established a new spectrum axis - the percentile of the
real margin within its own isotropic null, p = P(margin_synth <=
margin_full) - which orders the four channels (qwen mlp 0.474 >
glm4 mlp 0.375 > glm4 attn 0.168 > qwen attn 0.028) independently
of the margin-amplitude order.  Before promoting p to a second
primary spectrum axis, its robustness must be tested against
(a) margin-family scoring variants, (b) window-layer subsets,
under a fixed synthesis protocol and matched coverage audits.

Configurations (frozen):
  margin variants, full layer set:
    V0_full   baseline 2896-family margin (row-normalised B,
              same-off-diag minus diff-off-diag cosine mean)
    V1_diagin same term includes the diagonal (Sm incl diag)
    V2_colz   column z-score of B before row normalisation, V0
    V3_acc    LOO nearest-neighbour accuracy (acc family)
  layer subsets, scored by V0:
    S_front   first half of the window-layer columns
    S_back    second half
    S_key     single key layer = argmax |delta_per_layer| (2905)

Adjudication (frozen):
  p per configuration = median over 5 seeds of
  P(score_synth <= score_true), N_SYNTH=10000 draws/seed.
  order  : p_med ordering of the four channels under V0_full
           (baseline) must be reproduced exactly under V1_diagin,
           V2_colz, V3_acc and under S_front, S_back, S_key.
  tail   : qwen_attn p_med < 0.05 in ALL 7 non-baseline configs
           (6 variants/subsets + baseline itself must also be
           < 0.05) => qwen_attn_lower_tail_robust.
  interior: glm4_attn p_med in [0.05, 0.95] in all 7 configs
           => glm4_attn_interior_robust.
  verdict: not order_stable => p_axis_order_unstable;
           stable & tail & interior => p_axis_second_dimension_
           confirmed; else => p_axis_order_stable_edge_cases.

Anchors: a1 margin/acc (full-layer baseline) == stored 2e-5;
a2 Delta_B == 2905 delta_per_layer 1e-4; a3 fast margin ==
margin_of_B and acc_score == acc_of_B within 1e-12.

Coverage audits (per scoring family): rng [2909,100+v] for the
four margin variants (d=10) and [2909,200+s] for the three layer
subsets (d = 6/6/1); 40 reps, N_AUDIT=4000 draws, pass iff
coverage in [32,40]/40; any audit fail => audit_coverage_fail_
all_void.

Sigma definition: RMS sqrt(tr(Sigma_c)/d) (E9-corrected);
d=1 subset uses std(ddof=1).
Output: phase2909/p_axis_stability/.
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
OUT = os.path.join(BASE, 'phase2909', 'p_axis_stability')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2909_run_report.txt')
SEED_BASE = 2909
N_SYNTH = 10000
N_SEEDS = 5
N_COV = 40
N_AUDIT = 4000
TOL_A1 = 2e-5
TOL_A2 = 1e-4
TAIL_LO = 0.05
INTERIOR_LO, INTERIOR_HI = 0.05, 0.95
GROUP_ORDER = ('glm4_mlp', 'glm4_attn', 'qwen_mlp', 'qwen_attn')
VAR_KINDS = ('V0_full', 'V1_diagin', 'V2_colz', 'V3_acc')
SUB_KINDS = ('S_front', 'S_back', 'S_key')

PREREG = {
    'mode': 'zero_forward_matrix_analysis',
    'sigma_definition': 'RMS sqrt(tr(Sigma_c)/d) (E9-corrected); '
                        'd=1 subset uses std(ddof=1)',
    'configs': 'margin variants full-layer: V0_full baseline, '
               'V1_diagin (same term incl diagonal), V2_colz '
               '(column z-score then V0), V3_acc (LOO NN acc); '
               'layer subsets scored by V0: S_front first half '
               'columns, S_back second half, S_key single column '
               'argmax|delta_per_layer| (2905)',
    'sources': 'B matrices + stored margin/acc (2902/2903); '
               'delta_per_layer (2905, round4)',
    'anchors': 'a1 full-layer baseline margin/acc == stored '
               'within 2e-5; a2 Delta_B == 2905 delta_per_layer '
               'within 1e-4; a3 fast margin == margin_of_B and '
               'acc_score == acc_of_B within 1e-12',
    'coverage_audits': 'per scoring family: rng [2909,100+v] '
                       'variants (d=10), [2909,200+s] subsets '
                       '(d=6/6/1); 40 reps, N_AUDIT=4000 draws, '
                       'pass iff coverage in [32,40]/40; any '
                       'fail => audit_coverage_fail_all_void',
    'main': 'per config, 5 seeds rng [2909,10+cfg_index*10+k], '
            'N_SYNTH=10000 draws/seed, p = P(score_synth <= '
            'score_true), reported as seed-median',
    'adjudication': 'order: four-channel p_med ordering under '
                    'V0_full reproduced exactly under V1/V2/V3 '
                    'and S_front/S_back/S_key; tail: qwen_attn '
                    'p_med < 0.05 in all 7 configs; interior: '
                    'glm4_attn p_med in [0.05,0.95] in all 7; '
                    'verdict: order unstable => '
                    'p_axis_order_unstable; stable & tail & '
                    'interior => p_axis_second_dimension_'
                    'confirmed; else p_axis_order_stable_edge_'
                    'cases',
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


def acc_score(B, lab):
    return acc_of_B(B, lab)


def margin_v0(B, lab):
    return margin_of_B(B, lab)


def margin_v1_diagin(B, lab):
    U = unit_rows(B)
    Sm = U @ U.T
    n = len(lab)
    same_in = (lab[:, None] == lab[None, :])
    eye = np.eye(n, dtype=bool)
    diff = (~eye) & (~same_in)
    return float(Sm[same_in].mean() - Sm[diff].mean())


def margin_v2_colz(B, lab):
    mu = B.mean(axis=0, keepdims=True)
    sd = B.std(axis=0, keepdims=True)
    return margin_of_B((B - mu) / np.maximum(sd, 1e-30), lab)


SCORES = {
    'V0_full': margin_v0,
    'V1_diagin': margin_v1_diagin,
    'V2_colz': margin_v2_colz,
    'V3_acc': acc_score,
}


def sigma_rms(Bm):
    d = Bm.shape[1]
    if d == 1:
        return float(Bm.std(0, ddof=1).item())
    S = np.cov(Bm, rowvar=False)
    return float(np.sqrt(np.trace(S) / d))


def synth_p(B_sub, lab, score_fn, rng, n_draws):
    m0, m1 = lab == 0, lab == 1
    n0, n1 = int(m0.sum()), int(m1.sum())
    d = B_sub.shape[1]
    mu0, mu1 = B_sub[m0].mean(0), B_sub[m1].mean(0)
    s0, s1 = sigma_rms(B_sub[m0]), sigma_rms(B_sub[m1])
    s_true = score_fn(B_sub, lab)
    cnt = 0
    for _ in range(n_draws):
        Bs = np.empty_like(B_sub)
        Bs[m0] = mu0 + s0 * rng.normal(size=(n0, d))
        Bs[m1] = mu1 + s1 * rng.normal(size=(n1, d))
        if score_fn(Bs, lab) <= s_true:
            cnt += 1
    return cnt / float(n_draws), s_true


def coverage_audit(score_kind, d, seed_key):
    score_fn = SCORES[score_kind]
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
        s_true = score_fn(Bt, lab)
        u0, u1 = Bt[m0].mean(0), Bt[m1].mean(0)
        h0, h1 = sigma_rms(Bt[m0]), sigma_rms(Bt[m1])
        vals = np.empty(N_AUDIT)
        for i in range(N_AUDIT):
            Bs = np.empty((n, d))
            Bs[m0] = u0 + h0 * rng.normal(size=(n0, d))
            Bs[m1] = u1 + h1 * rng.normal(size=(n1, d))
            vals[i] = score_fn(Bs, lab)
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
        json.dump({'phase': 2909,
                   'name': 'p_axis_stability',
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
                   'tail_lo': TAIL_LO,
                   'interior': [INTERIOR_LO, INTERIOR_HI],
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
        mfast = margin_v0(B, lab)
        afast = acc_score(B, lab)
        mdl, ch = g.split('_')
        d05 = np.asarray(r05['groups']['%s_%s' % (mdl, ch)]
                         ['delta_per_layer'], dtype=float)
        d09 = B[lab == 1].mean(0) - B[lab == 0].mean(0)
        e2 = float(np.abs(d09 - d05).max())
        e3 = max(abs(mf - mfast), abs(ac - afast))
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

    # ---------- coverage audits ----------
    audits = {}
    audit_ok = True
    for vi, kind in enumerate(VAR_KINDS):
        c = coverage_audit(kind, 10,
                           [SEED_BASE, 100 + vi])
        ok = bool(32 <= c <= N_COV)
        audit_ok = audit_ok and ok
        audits[kind] = {'covered': c, 'n': N_COV, 'pass': ok}
        log('audit %s (d=10): %d/%d pass=%s'
            % (kind, c, N_COV, ok), lines)
    for si, kind in enumerate(SUB_KINDS):
        d_sub = 6 if kind in ('S_front', 'S_back') else 1
        c = coverage_audit('V0_full', d_sub,
                           [SEED_BASE, 200 + si])
        ok = bool(32 <= c <= N_COV)
        audit_ok = audit_ok and ok
        audits[kind] = {'covered': c, 'n': N_COV,
                        'd': d_sub, 'pass': ok}
        log('audit %s (d=%d): %d/%d pass=%s'
            % (kind, d_sub, c, N_COV, ok), lines)

    verdict = None
    order0 = None
    order_detail = None
    tail_ok = None
    interior_ok = None
    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    elif not audit_ok:
        verdict = 'audit_coverage_fail_all_void'

    # ---------- main grid ----------
    res_cfg = {}
    if verdict is None:
        for gi, g in enumerate(GROUP_ORDER):
            B, lab = groups[g]['B'], groups[g]['lab']
            mdl, ch = g.split('_')
            d05 = np.asarray(r05['groups']['%s_%s' % (mdl, ch)]
                             ['delta_per_layer'], dtype=float)
            key_col = int(np.abs(d05).argmax())
            n_win = B.shape[1]
            half = n_win // 2
            cfgs = [('V0_full', B)]
            for kind in VAR_KINDS[1:]:
                cfgs.append((kind, B))
            for kind in SUB_KINDS:
                if kind == 'S_front':
                    cfgs.append((kind, B[:, :half]))
                elif kind == 'S_back':
                    cfgs.append((kind, B[:, half:]))
                else:
                    cfgs.append((kind, B[:, key_col:key_col + 1]))
            res_g = {}
            for ci, (kind, B_sub) in enumerate(cfgs):
                score_fn = (SCORES['V0_full']
                            if kind in SUB_KINDS
                            else SCORES[kind])
                ps = []
                for k in range(N_SEEDS):
                    rng = np.random.default_rng(
                        [SEED_BASE, 10 + ci * 10 + k])
                    p, s_true = synth_p(B_sub, lab,
                                        score_fn, rng,
                                        N_SYNTH)
                    ps.append(p)
                ps_arr = np.asarray(ps)
                res_g[kind] = {
                    'score_true': round(float(s_true), 6),
                    'p_per_seed': [round(p, 6) for p in ps],
                    'p_median': round(float(np.median(ps_arr)),
                                      6),
                }
                log('%s %s: score=%.6f p_med=%.5f '
                    'p_range=[%.5f,%.5f]'
                    % (g, kind, s_true, np.median(ps_arr),
                       ps_arr.min(), ps_arr.max()), lines)
            res_cfg[g] = res_g

        # ---------- adjudication ----------
        order0 = tuple(sorted(GROUP_ORDER,
                              key=lambda g: -res_cfg[g]
                              ['V0_full']['p_median']))
        order_ok, order_detail = True, {}
        for kind in VAR_KINDS[1:] + SUB_KINDS:
            od = tuple(sorted(GROUP_ORDER,
                              key=lambda g: -res_cfg[g]
                              [kind]['p_median']))
            same = od == order0
            order_detail[kind] = {'order': list(od),
                                  'matches_baseline': bool(same)}
            order_ok = order_ok and same
            log('order %s: %s match=%s'
                % (kind, '>'.join(od), same), lines)
        log('baseline order: %s' % '>'.join(order0), lines)

        tail_ok = True
        for kind in VAR_KINDS + SUB_KINDS:
            pm = res_cfg['qwen_attn'][kind]['p_median']
            if not (pm < TAIL_LO):
                tail_ok = False
        interior_ok = True
        for kind in VAR_KINDS + SUB_KINDS:
            pm = res_cfg['glm4_attn'][kind]['p_median']
            if not (INTERIOR_LO <= pm <= INTERIOR_HI):
                interior_ok = False
        log('tail_ok(qwen_attn<0.05 all 7)=%s '
            'interior_ok(glm4_attn in [0.05,0.95] all 7)=%s'
            % (tail_ok, interior_ok), lines)
        if not order_ok:
            verdict = 'p_axis_order_unstable'
        elif tail_ok and interior_ok:
            verdict = 'p_axis_second_dimension_confirmed'
        else:
            verdict = 'p_axis_order_stable_edge_cases'

    log('==== VERDICT: %s ====' % verdict, lines)

    res = {
        'phase': 2909,
        'model': 'glm4+qwen (matrix analysis)',
        'prereg': PREREG,
        'seed_base': SEED_BASE, 'n_synth': N_SYNTH,
        'n_seeds': N_SEEDS, 'n_audit_draws': N_AUDIT,
        'anchor': anchor, 'anchor_ok': bool(anchor_ok),
        'coverage_audits': audits, 'audits_pass': bool(audit_ok),
        'configs': res_cfg,
        'order_baseline': list(order0) if order0 else None,
        'order_detail': order_detail,
        'tail_ok': None if tail_ok is None else bool(tail_ok),
        'interior_ok': (None if interior_ok is None
                        else bool(interior_ok)),
        'final_verdict': verdict,
        'runtime_s': round(time.monotonic() - t0, 1),
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    np.savez_compressed(
        os.path.join(OUT, 'p_axis_stability.npz'),
        phase=np.int64(2909))
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2909 verdict=%s' % verdict)


if __name__ == '__main__':
    main()
