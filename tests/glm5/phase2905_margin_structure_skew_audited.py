# -*- coding: utf-8 -*-
"""Phase 2905: B row-space margin structure - skew-audited
detector, FIRST main reading of margin_within on P.

2904 established (N11): the 2896-family margin metrics detect
class-mean shifts (1st moment) and within-class skewness /
asymmetric-subcluster structure (3rd moment), but are blind to
covariance anisotropy by first-moment identity.  The audit-2nd
construction of 2904 (covariance difference) was therefore
mathematically incapable of passing - construction error, not
detector failure.  Here audit_2nd_order is replaced by the
diagnostic-validated SKEW construction (90% mass at +2w, 10% at
-18w, mean exactly 0; w0 orthogonal w1; 20 independent instances,
pass iff detection rate >= 12/20; point estimate 74/100 in the
2904b diagnostic).  Everything else verbatim 2904.

Anchor a1: margin/acc recomputed from npz float32 B match stored
2902/2903 values within abs 2e-5, else anchor_fail_all_void.
audit_1st_order / audit_negative / from_mean identity unchanged.
Positive set P = groups with stored margin > stored margin_p95
(2896-seed perms, frozen) = expected {glm4_attn, qwen_mlp}.
Null: SEED=2905, 1000 label permutations per group (single shared
stream, group order glm4-mlp, glm4-attn, qwen-mlp, qwen-attn),
p95_within recomputed under permutation.

Verdict (frozen):
  anchor fail                  => anchor_fail_all_void
  audit_negative/1st fail      => algebraic_audit_fail_all_void
  audit_2nd_order_skew < 12/20 => detector_insensitive_all_void
  P empty                      => structure_not_established
  all P: margin_within <= p95_within
                               => margin_carried_by_class_mean_shift
  all P: margin_within > p95_within
                               => margin_carried_by_higher_order_structure
  mixed                        => structure_mixed_across_carriers

Descriptive per group: Fisher F, Hotelling T2 / rho1, spectra of
B and B_within, per-layer delta.

SEED=2905.  Output: phase2905/margin_structure_skew_audited/.
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
OUT = os.path.join(BASE, 'phase2905',
                   'margin_structure_skew_audited')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2905_run_report.txt')
SEED = 2905
N_PERM = 1000
TOL = 2e-5
N_AUDIT2 = 20
AUDIT2_MIN = 12

PREREG = {
    'sources': 'B matrices from 2902 npz (glm4 78x12) and 2903 '
               'npz (qwen 57x10), lang labels verbatim; stored '
               'margin/margin_p95 from 2902/2903 result.json',
    'anchor_a1': 'margin_full and acc recomputed from npz float32 '
                 'B must match stored values within abs 2e-5, '
                 'else anchor_fail_all_void',
    'algebraic_audit': 'audit_1st_order (class-mean broadcast + '
                       'N(0,1), delta=2.0*e1, split 22/35) '
                       'expects margin_full > p95_full and '
                       'margin_within <= p95_within; '
                       'audit_2nd_order_skew (mean-zero rows, '
                       'class-dependent skew: 90% mass +2w, 10% '
                       'mass -18w, w0_|_w1 random orthonormal, '
                       'n=57 d=10 split 22/35; 20 independent '
                       'instances, rng substreams [2905,2,k]) '
                       'pass iff detection rate (margin_within > '
                       'perm p95) >= 12/20; failure => '
                       'detector_insensitive_all_void; '
                       'audit_negative (N(0,I)) expects both '
                       'margins below p95; failure of negative or '
                       '1st => algebraic_audit_fail_all_void; '
                       'audit_from_mean_identity per group: '
                       'margin(broadcast) == 1 - cos(unit mu0, '
                       'unit mu1) within 1e-9',
    'audit_basis': 'N11 (2904): margin metrics detect 1st-order '
                   'class-mean shifts and 3rd-moment skew; '
                   'covariance anisotropy is first-moment-blind; '
                   'skew construction validated in 2904b '
                   'diagnostic (detection 74/100 vs iid 3/100)',
    'decomposition': 'per (model, ch): margin_from_mean '
                     '(first-order carrier), margin_within '
                     '(higher-order carrier), Fisher F, Hotelling '
                     'T2 / CCA rho1, spectra of B and B_within, '
                     'per-layer delta',
    'null': 'SEED=2905, 1000 label permutations per group, single '
            'shared stream, group order glm4-mlp glm4-attn '
            'qwen-mlp qwen-attn; p95_within from margin_within '
            'under permuted labels (within-center recomputed '
            'under permutation); margin_full null descriptive '
            'only (stored 2896 p95 stays the frozen positive '
            'criterion)',
    'positive_set': 'P = groups with stored margin > stored '
                    'margin_p95 (2896-seed perms, frozen)',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'audit_negative or audit_1st_order fail => '
               'algebraic_audit_fail_all_void; '
               'audit_2nd_order_skew rate < 12/20 => '
               'detector_insensitive_all_void; P empty => '
               'structure_not_established; all P margin_within <= '
               'p95_within => margin_carried_by_class_mean_shift; '
               'all P margin_within > p95_within => '
               'margin_carried_by_higher_order_structure; mixed '
               '=> structure_mixed_across_carriers',
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


def within_center(B, lab):
    Bw = B.copy()
    for c in (0, 1):
        m = lab == c
        if m.any():
            Bw[m] -= Bw[m].mean(axis=0, keepdims=True)
    return Bw


def margin_within_of(B, lab):
    return margin_of_B(within_center(B, lab), lab)


def margin_from_mean_of(B, lab):
    Mm = np.empty_like(B)
    for c in (0, 1):
        m = lab == c
        if m.any():
            Mm[m] = B[m].mean(axis=0, keepdims=True)
    return margin_of_B(Mm, lab)


def fisher_t2(B, lab):
    n, d = B.shape
    l0 = lab == 0
    l1 = lab == 1
    n0 = int(l0.sum())
    n1 = int(l1.sum())
    m0 = B[l0].mean(0)
    m1 = B[l1].mean(0)
    delta = m1 - m0
    S0 = np.cov(B[l0], rowvar=False)
    S1 = np.cov(B[l1], rowvar=False)
    SW = ((n0 - 1) * S0 + (n1 - 1) * S1) / (n0 + n1 - 2)
    t2 = float((n0 * n1 / float(n0 + n1))
               * (delta @ np.linalg.solve(SW, delta)))
    rho1 = float(np.sqrt(t2 / (t2 + n - 2)))
    mu = B.mean(0)
    SB = n0 * np.outer(m0 - mu, m0 - mu) \
        + n1 * np.outer(m1 - mu, m1 - mu)
    f_ratio = float(np.trace(SB) / max(np.trace(SW), 1e-30))
    return delta, t2, rho1, f_ratio


def spectra(M):
    sv = np.linalg.svd(M - M.mean(0, keepdims=True),
                       compute_uv=False)
    return {'top1_over_top2': float(sv[0] / max(sv[1], 1e-30)),
            'participation_ratio': float(
                sv.sum() ** 2 / max(float((sv ** 2).sum()),
                                    1e-30))}


def skew_rows(rng, n, w, big=2.0, tail=18.0, pf=0.9):
    Z = rng.normal(size=(n, len(w)))
    side = rng.random(n) < pf
    mu = np.where(side, big, -tail)[:, None] * w[None, :]
    return Z + mu


def audit_2nd_skew_rate():
    det = 0
    mws = []
    for k in range(N_AUDIT2):
        rng = np.random.default_rng([SEED, 2, k])
        n, d, n0 = 57, 10, 22
        lab = np.array([0] * n0 + [1] * (n - n0))
        w0 = rng.normal(size=d)
        w0 = w0 / np.linalg.norm(w0)
        w1 = rng.normal(size=d)
        w1 = w1 - (w1 @ w0) * w0
        w1 = w1 / np.linalg.norm(w1)
        B = np.empty((n, d))
        m0 = lab == 0
        m1 = lab == 1
        B[m0] = skew_rows(rng, int(m0.sum()), w0)
        B[m1] = skew_rows(rng, int(m1.sum()), w1)
        prng = np.random.default_rng([SEED, 2, k, 77])
        nw = [margin_within_of(B, prng.permutation(lab))
              for _ in range(N_PERM)]
        p95w = float(np.percentile(nw, 95))
        mw = margin_within_of(B, lab)
        mws.append(mw)
        if mw > p95w:
            det += 1
    return det, mws


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2905,
                   'name': 'margin_structure_skew_audited',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8': sha8(os.path.abspath(__file__)),
                   'sources': {'s2902': sha8(SRC['glm4'][0]),
                               'r2902': sha8(SRC['glm4'][1]),
                               's2903': sha8(SRC['qwen'][0]),
                               'r2903': sha8(SRC['qwen'][1])},
                   'mode': 'zero_forward_matrix_analysis',
                   'seed': SEED, 'n_perm': N_PERM,
                   'n_audit2_instances': N_AUDIT2,
                   'audit2_min_detect': AUDIT2_MIN,
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    # ---------- load groups ----------
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
                'stored_p95': float(
                    r['margins'][ch]['margin_p95']),
                'stored_acc': float(
                    r['accs'][ch] if 'accs' in r
                    else r['margins'][ch]['acc']),
            }
    log('groups loaded: %s' % {k: v['B'].shape
                               for k, v in groups.items()}, lines)

    # ---------- a1 anchor ----------
    anchor = {}
    anchor_ok = True
    for g, it in groups.items():
        mf = margin_of_B(it['B'], it['lab'])
        ac = acc_of_B(it['B'], it['lab'])
        ok = (abs(mf - it['stored_margin']) < TOL
              and abs(ac - it['stored_acc']) < TOL)
        anchor[g] = {'margin_recomp': round(mf, 6),
                     'acc_recomp': round(ac, 6),
                     'ok': bool(ok)}
        anchor_ok = anchor_ok and ok
        log('anchor %s margin %.5f (stored %.5f) acc %.5f '
            '(stored %.5f) ok=%s'
            % (g, mf, it['stored_margin'], ac, it['stored_acc'],
               ok), lines)

    # ---------- audit 1st ----------
    rng = np.random.default_rng([SEED, 1])
    n, d, n0 = 57, 10, 22
    lab_a = np.array([0] * n0 + [1] * (n - n0))
    mu = np.where(lab_a == 0, -1.0, 1.0)[:, None] \
        * np.eye(1, d)[0][None, :]
    B_a = mu + rng.normal(size=(n, d))
    prng = np.random.default_rng([SEED, 11])
    nf = [margin_of_B(B_a, prng.permutation(lab_a))
          for _ in range(N_PERM)]
    nw = [margin_within_of(B_a, prng.permutation(lab_a))
          for _ in range(N_PERM)]
    a1_full = margin_of_B(B_a, lab_a)
    a1_within = margin_within_of(B_a, lab_a)
    a1_p95f = float(np.percentile(nf, 95))
    a1_p95w = float(np.percentile(nw, 95))
    audit1_ok = bool(a1_full > a1_p95f and a1_within <= a1_p95w)
    log('audit 1st: full %.4f (p95 %.4f) within %.4f (p95 %.4f) '
        'ok=%s' % (a1_full, a1_p95f, a1_within, a1_p95w,
                   audit1_ok), lines)

    # ---------- audit 2nd (skew, 20 instances) ----------
    det, mws = audit_2nd_skew_rate()
    audit2_ok = bool(det >= AUDIT2_MIN)
    log('audit 2nd skew: detection %d/%d (pass >= %d) ok=%s '
        'median margin_within=%+.4f'
        % (det, N_AUDIT2, AUDIT2_MIN, audit2_ok,
           float(np.median(mws))), lines)

    # ---------- audit negative ----------
    rng = np.random.default_rng([SEED, 3])
    B_n = rng.normal(size=(n, d))
    prng = np.random.default_rng([SEED, 13])
    nf = [margin_of_B(B_n, prng.permutation(lab_a))
          for _ in range(N_PERM)]
    nw = [margin_within_of(B_n, prng.permutation(lab_a))
          for _ in range(N_PERM)]
    an_full = margin_of_B(B_n, lab_a)
    an_within = margin_within_of(B_n, lab_a)
    an_p95f = float(np.percentile(nf, 95))
    an_p95w = float(np.percentile(nw, 95))
    auditn_ok = bool(an_full < an_p95f and an_within < an_p95w)
    log('audit neg: full %.4f (p95 %.4f) within %.4f (p95 %.4f) '
        'ok=%s' % (an_full, an_p95f, an_within, an_p95w,
                   auditn_ok), lines)

    # ---------- verdict gate ----------
    verdict = None
    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    elif not (auditn_ok and audit1_ok):
        verdict = 'algebraic_audit_fail_all_void'
    elif not audit2_ok:
        verdict = 'detector_insensitive_all_void'

    # ---------- main analysis ----------
    res_g = {}
    rng_main = np.random.default_rng(SEED)
    from_mean_id_all = True
    if verdict is None:
        P = [g for g, it in groups.items()
             if it['stored_margin'] > it['stored_p95']]
        log('positive set P=%s' % P, lines)
        for g in ('glm4_mlp', 'glm4_attn', 'qwen_mlp',
                  'qwen_attn'):
            it = groups[g]
            B, lab = it['B'], it['lab']
            mf = margin_of_B(B, lab)
            mw = margin_within_of(B, lab)
            mm = margin_from_mean_of(B, lab)
            mu0 = B[lab == 0].mean(0)
            mu1 = B[lab == 1].mean(0)
            ident = 1.0 - float(unit_rows(mu0[None, :])[0]
                                @ unit_rows(mu1[None, :])[0])
            ident_ok = abs(mm - ident) < 1e-9
            from_mean_id_all = from_mean_id_all and ident_ok
            delta, t2, rho1, fr = fisher_t2(B, lab)
            Bw = within_center(B, lab)
            _, t2w, rho1w, frw = fisher_t2(Bw, lab)
            nf, nw, nF, nT = [], [], [], []
            for _ in range(N_PERM):
                pl = rng_main.permutation(lab)
                nf.append(margin_of_B(B, pl))
                nw.append(margin_within_of(B, pl))
                dl, t2p, _, fp = fisher_t2(B, pl)
                nF.append(fp)
                nT.append(t2p)
            p95w = float(np.percentile(nw, 95))
            entry = {
                'n_words': int(len(lab)),
                'n_win': int(B.shape[1]),
                'lang_counts': [int((lab == 0).sum()),
                                int((lab == 1).sum())],
                'stored_margin': it['stored_margin'],
                'stored_p95': it['stored_p95'],
                'positive': bool(it['stored_margin']
                                 > it['stored_p95']),
                'margin_full': round(mf, 6),
                'margin_from_mean': round(mm, 6),
                'from_mean_identity_ok': bool(ident_ok),
                'margin_within': round(mw, 6),
                'null_p95_full_2905': round(
                    float(np.percentile(nf, 95)), 6),
                'null_p95_within': round(p95w, 6),
                'within_exceeds_p95': bool(mw > p95w),
                'F': round(fr, 4),
                'F_null_p95': round(
                    float(np.percentile(nF, 95)), 4),
                'T2': round(t2, 2),
                'T2_null_p95': round(
                    float(np.percentile(nT, 95)), 2),
                'rho1': round(rho1, 4),
                'F_within': round(frw, 4),
                'rho1_within': round(rho1w, 4),
                'delta_norm': round(
                    float(np.linalg.norm(delta)), 4),
                'delta_per_layer': [round(float(x), 4)
                                    for x in delta],
                'spectra_B': {k: round(v, 3)
                              for k, v in spectra(B).items()},
                'spectra_B_within': {
                    k: round(v, 3)
                    for k, v in spectra(Bw).items()},
            }
            res_g[g] = entry
            log('%s: full=%.4f from_mean=%.4f within=%.4f '
                'p95w=%.4f within_over=%s F=%.3f(Fp95=%.3f) '
                'T2=%.1f(Tp95=%.1f) ident_ok=%s'
                % (g, mf, mm, mw, p95w,
                   entry['within_exceeds_p95'], fr,
                   entry['F_null_p95'], t2,
                   entry['T2_null_p95'], ident_ok), lines)
        pos = [g for g in res_g if res_g[g]['positive']]
        if not pos:
            verdict = 'structure_not_established'
        else:
            flags = [res_g[g]['within_exceeds_p95'] for g in pos]
            if all(flags):
                verdict = ('margin_carried_by_'
                           'higher_order_structure')
            elif not any(flags):
                verdict = ('margin_carried_by_'
                           'class_mean_shift')
            else:
                verdict = 'structure_mixed_across_carriers'
        log('from_mean identity all ok=%s' % from_mean_id_all,
            lines)

    log('==== VERDICT: %s ====' % verdict, lines)

    res = {
        'phase': 2905, 'model': 'glm4+qwen (matrix analysis)',
        'prereg': PREREG, 'seed': SEED, 'n_perm': N_PERM,
        'anchor': anchor, 'anchor_ok': bool(anchor_ok),
        'audits': {
            'audit_1st_order': {
                'margin_full': round(a1_full, 5),
                'margin_within': round(a1_within, 5),
                'p95_full': round(a1_p95f, 5),
                'p95_within': round(a1_p95w, 5),
                'ok': audit1_ok},
            'audit_2nd_order_skew': {
                'detection': det, 'n_instances': N_AUDIT2,
                'min_required': AUDIT2_MIN,
                'margin_within_median': round(
                    float(np.median(mws)), 5),
                'ok': audit2_ok},
            'audit_negative': {
                'margin_full': round(an_full, 5),
                'margin_within': round(an_within, 5),
                'p95_full': round(an_p95f, 5),
                'p95_within': round(an_p95w, 5),
                'ok': auditn_ok},
            'from_mean_identity_all_ok':
                bool(from_mean_id_all) if res_g else None,
        },
        'groups': res_g,
        'final_verdict': verdict,
        'runtime_s': round(time.monotonic() - t0, 1),
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    npz_data = {}
    for g, e in res_g.items():
        pre = g
        for k in ('margin_full', 'margin_from_mean',
                  'margin_within', 'null_p95_within', 'F',
                  'T2', 'rho1', 'delta_per_layer'):
            npz_data['%s_%s' % (pre, k)] = np.asarray(e[k])
    if npz_data:
        np.savez_compressed(
            os.path.join(OUT, 'margin_structure_skew_audited.npz'),
            **npz_data)

    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
