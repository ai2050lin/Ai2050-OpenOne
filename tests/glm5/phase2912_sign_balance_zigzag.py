# -*- coding: utf-8 -*-
"""Phase 2912: sign-balance zigzag formalisation + word attribution.

Zero-forward matrix analysis.

Why: 2911 located the qwen_attn margin alternation carrier in the
per-layer class-wise sign balance gap_j = |pos_frac0_j -
pos_frac1_j| (zigzag 7/8, descriptive), driven mainly by class-0
positive-rate fluctuation.  This phase formalises the zigzag under
a word-level permutation null and attributes the fluctuation to
individual words.

Probes (all zero-forward, per group):
  P1 zigzag test : k = #{inner j: (gap_{j+1}-gap_j)*(gap_j-gap_
     j-1) < 0} over the n_win-layer gap sequence; null = per-layer
     independent permutation of the class labels (class sizes
     fixed 22/35), N_PERM=20000; one-sided p = P(perm_k >= obs_k).
  P2 lag-1 autocorrelation of diff(gap): r = corr(d[:-1], d[1:]);
     zigzag => negative r; permutation p = P(perm_r <= obs_r).
  P3 word attribution (descriptive): per-word sign-flip count
     f_i across adjacent layers (sign(B[:,j]), zeros -> +1);
     top-5 flip share within each class; exact binomial tail for
     the max-flip word P(X >= f_max | n_pairs, p_hat) with
     p_hat = pooled flip rate.

Adjudication (frozen):
  guards: anchors fail => anchor_fail_all_void; calibration
  (rng [2912,0]: 200 random 57x10 sign matrices with 22/35 split,
  1000 perms each; v2 tie-corrected: frac of zigzag p in
  [0.05,0.95] in [0.80,0.97] AND p_mean within 3 SE of the
  analytic expectation 0.5+tau/2, tau = sum pk^2 of the pooled
  null zigzag distribution) else audit_calib_fail_all_void.
  qwen_attn: p_zig <= 0.05 AND p_rho <= 0.05 required for
  'confirmed'; specificity by S = groups significant on both:
    S == {qwen_attn}                  => gap_zigzag_confirmed_
                                         qwen_attn_specific
    qwen_attn in S and S subset of    => gap_zigzag_confirmed_
    {qwen_attn, glm4_attn}               attn_channel_shared
    |S| >= 3                          => gap_zigzag_generic
    else                              => gap_zigzag_confirmed_
                                         partial_specificity
  exactly one of p_zig/p_rho significant => gap_zigzag_weak_
  partial; none => gap_zigzag_absent.  P3 descriptive.

Anchors: a1 full-layer margin/acc == stored 2e-5; a2 Delta_B ==
2905 1e-4; a3 recomputed sign margins == 2910 score_true 1e-6.

Output: phase2912/sign_balance_zigzag/.
"""
import hashlib
import json
import os
import time
from math import comb

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
R2910 = os.path.join(BASE, 'phase2910',
                     'cross_layer_coherence', 'result.json')
OUT = os.path.join(BASE, 'phase2912', 'sign_balance_zigzag')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2912_run_report.txt')
SEED_BASE = 2912
N_PERM = 20000
N_CALIB = 200
N_CALIB_PERM = 1000
TOL_A1 = 2e-5
TOL_A2 = 1e-4
ALPHA = 0.05
GROUP_ORDER = ('glm4_mlp', 'glm4_attn', 'qwen_mlp', 'qwen_attn')

PREREG = {
    'mode': 'zero_forward_matrix_analysis',
    'probes': 'P1 zigzag of per-layer gap sequence (k inner '
              'direction reversals), null = per-layer independent '
              'class-label permutation (sizes fixed), one-sided '
              'p = P(perm >= obs); P2 lag-1 autocorr of diff(gap),'
              ' one-sided negative p = P(perm <= obs); P3 '
              'descriptive word-level flip counts, top-5 share, '
              'exact binomial tail for max-flip word',
    'sources': 'B + stored margin/acc (2902/2903); delta (2905); '
               '2910 score_true (anchor a3)',
    'anchors': 'a1 margin/acc == stored 2e-5; a2 Delta_B == 2905 '
               '1e-4; a3 sign margins == 2910 score_true 1e-6',
    'calibration_v1_superseded': 'rng [2912,0]: 200 random 57x10 '
                 'sign matrices (22/35 split), 1000 perms each; '
                 'frac of zigzag p in [0.05,0.95] in [0.80,0.97] '
                 'AND p_median in [0.40,0.60]',
    'calibration': 'v2 rng [2912,0]: 200 random 57x10 sign '
                   'matrices (22/35 split), 1000 perms each; '
                   'guard1 frac of zigzag p in [0.05,0.95] must '
                   'be in [0.80,0.97]; guard2 p_mean within 3 SE '
                   '(SE = 0.5/sqrt(200)) of the analytic '
                   'tie-corrected expectation 0.5 + tau/2, tau = '
                   'sum pk^2 from the pooled null zigzag '
                   'distribution; else audit_calib_fail_all_void',
    'calibration_v2_note': 'run1 (v1) failed: p_median 0.687 > '
                 '0.60 -> all_void.  diag_2912a/b proved the null '
                 'construction unbiased (iid obs vs null zigzag '
                 'mean 5.18 vs 5.21, null k matches binomial(8, '
                 '2/3), E[k] = 5.33 by the 2/3 up-down reversal '
                 'law) and the failure to be a v1 band defect: '
                 'with p = P(perm >= obs) ties included, E[p] = '
                 '0.5 + tau/2 ~ 0.61 for the discrete null, so '
                 'the p median centers near 0.70 (diag_2912b, '
                 '1000 matrices: p_mean 0.618 vs theory 0.610).  '
                 'v2 replaces the median band by the analytic '
                 'p_mean band; p definition and all other frozen '
                 'elements unchanged; run1 products deleted per '
                 'rerun discipline',
    'adjudication': 'qwen_attn confirmed iff p_zig<=0.05 and '
                    'p_rho<=0.05; specificity by S = groups '
                    'significant on both (S=={qwen_attn} => '
                    'gap_zigzag_confirmed_qwen_attn_specific; '
                    'qwen_attn in S subset {qwen_attn,glm4_attn} '
                    '=> attn_channel_shared; |S|>=3 => generic; '
                    'else partial_specificity); one significant '
                    '=> gap_zigzag_weak_partial; none => '
                    'gap_zigzag_absent; P3 descriptive',
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


def sign_mat(B):
    S = np.sign(B)
    S[S == 0] = 1.0
    return S


def gap_seq(Sm, lab, perm=None):
    """gap_j = |pos_frac0 - pos_frac1| per layer from signs."""
    if perm is not None:
        labp = lab[perm]
    else:
        labp = lab
    m0, m1 = labp == 0, labp == 1
    n0 = max(int(m0.sum()), 1)
    n1 = max(int(m1.sum()), 1)
    pf0 = (Sm[m0] > 0).mean(axis=0)
    pf1 = (Sm[m1] > 0).mean(axis=0)
    return np.abs(pf0 - pf1), (n0, n1)


def zigzag_k(g):
    d = np.diff(g)
    return int(np.sum(d[:-1] * d[1:] < 0))


def rho1(g):
    d = np.diff(g)
    if len(d) < 3 or np.std(d[:-1]) < 1e-12 \
            or np.std(d[1:]) < 1e-12:
        return 0.0
    return float(np.corrcoef(d[:-1], d[1:])[0, 1])


def perm_p_obs(Sm, lab, rng, n_perm):
    """Per-layer INDEPENDENT class-label permutation null.

    lab is ordered [0]*n0 + [1]*n1; per layer j we independently
    permute the 57 words, so the first 22 rows of the permuted
    column are a random class-0 draw.  This destroys the class-
    sign association in each layer independently, keeping the
    per-layer sign margins fixed."""
    g0, _ = gap_seq(Sm, lab)
    k_obs = zigzag_k(g0)
    r_obs = rho1(g0)
    n, w = Sm.shape
    n0 = int((lab == 0).sum())
    ck = cr = 0
    for _ in range(n_perm):
        idx = np.argsort(rng.random((n, w)), axis=0)
        Sp = np.take_along_axis(Sm, idx, axis=0)
        pf0 = (Sp[:n0] > 0).mean(axis=0)
        pf1 = (Sp[n0:] > 0).mean(axis=0)
        gp = np.abs(pf0 - pf1)
        if zigzag_k(gp) >= k_obs:
            ck += 1
        if rho1(gp) <= r_obs:
            cr += 1
    p_zig = (ck + 1) / float(n_perm + 1)
    p_rho = (cr + 1) / float(n_perm + 1)
    return k_obs, p_zig, r_obs, p_rho


def calibration():
    """v2: null-correctness check with discrete-tie analytic band.

    v1 required the p median in [0.40, 0.60]; that band is wrong
    for a discrete null: with p = P(perm >= obs) (ties included),
    E[p] = 1/2 + tau/2 where tau = sum pk^2 over the null zigzag
    distribution (~0.22 here), so E[p] ~ 0.61 and the v1 median
    band spuriously fails (diag_2912a/b: obs and null zigzag
    distributions identical, mean 5.18 vs 5.21 on iid, null k
    matches binomial(8, 2/3)).  v2 keeps the frac-in-band guard
    and checks the p MEAN against the analytic tie-corrected
    expectation with a 3-SE band."""
    rng = np.random.default_rng([SEED_BASE, 0])
    n = 57
    lab = np.array([0] * 22 + [1] * 35)
    ps = []
    hist = np.zeros(9, dtype=int)
    for _ in range(N_CALIB):
        B = rng.normal(size=(n, 10))
        Sm = sign_mat(B)
        g0, _ = gap_seq(Sm, lab)
        k_obs = zigzag_k(g0)
        ck = 0
        for _ in range(N_CALIB_PERM):
            idx = np.argsort(rng.random((n, 10)), axis=0)
            Sp = np.take_along_axis(Sm, idx, axis=0)
            pf0 = (Sp[:22] > 0).mean(axis=0)
            pf1 = (Sp[22:] > 0).mean(axis=0)
            gp = np.abs(pf0 - pf1)
            kp = zigzag_k(gp)
            hist[kp] += 1
            if kp >= k_obs:
                ck += 1
        ps.append((ck + 1) / float(N_CALIB_PERM + 1))
    ps = np.asarray(ps)
    pk = hist / float(hist.sum())
    tau = float((pk ** 2).sum())
    e_p = 0.5 + 0.5 * tau
    se_m = 0.5 / float(np.sqrt(N_CALIB))
    frac = float(np.mean((ps >= 0.05) & (ps <= 0.95)))
    pmean = float(ps.mean())
    pmed = float(np.median(ps))
    ok = bool(0.80 <= frac <= 0.97
              and abs(pmean - e_p) <= 3.0 * se_m)
    return {'version': 'v2_tie_corrected',
            'frac_in_band': round(frac, 4),
            'p_mean': round(pmean, 4),
            'p_median': round(pmed, 4),
            'tau_pool': round(tau, 4),
            'e_p_theory': round(e_p, 4),
            'band': [round(e_p - 3 * se_m, 4),
                     round(e_p + 3 * se_m, 4)],
            'pass': ok}


def word_attribution(Sm, lab):
    out = {}
    for cls in (0, 1):
        idx = np.where(lab == cls)[0]
        Sc = Sm[idx]
        flips = (Sc[:, 1:] != Sc[:, :-1]).sum(axis=1)
        total = int(flips.sum())
        order = np.argsort(-flips)
        top5 = [(int(idx[i]), int(flips[i]))
                for i in order[:5]]
        p_hat = total / float(flips.size) \
            if flips.size else 0.0
        f_max = int(flips.max()) if len(flips) else 0
        n_pairs = Sc.shape[1] - 1
        tail = (sum(comb(n_pairs, k)
                    for k in range(f_max, n_pairs + 1))
                / 2.0 ** n_pairs) if n_pairs > 0 else 1.0
        # exact tail under independent-per-word binomial with
        # pooled rate is not literally binomial(9, p_hat); the
        # comb(., .)/2^n term is the fair-coin tail, reported as
        # such
        out['class%d' % cls] = {
            'total_flips': total,
            'pooled_rate': round(p_hat, 4),
            'top5_word_idx_flip': top5,
            'top5_share': round(
                sum(f for _, f in top5) / max(total, 1), 4),
            'max_flips': f_max,
            'fair_coin_tail_max': round(tail, 6),
        }
    return out


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2912,
                   'name': 'sign_balance_zigzag',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8': sha8(os.path.abspath(__file__)),
                   'sources': {'s2902': sha8(SRC['glm4'][0]),
                               'r2902': sha8(SRC['glm4'][1]),
                               's2903': sha8(SRC['qwen'][0]),
                               'r2903': sha8(SRC['qwen'][1]),
                               'r2905': sha8(R2905),
                               'r2910': sha8(R2910)},
                   'mode': 'zero_forward_matrix_analysis',
                   'seed_base': SEED_BASE, 'n_perm': N_PERM,
                   'n_calib': N_CALIB,
                   'n_calib_perm': N_CALIB_PERM,
                   'alpha': ALPHA, 'prereg': PREREG},
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
    r10 = json.load(open(R2910, encoding='utf-8'))

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
        d12 = B[lab == 1].mean(0) - B[lab == 0].mean(0)
        e2 = float(np.abs(d12 - d05).max())
        s10 = [e['score_true']
               for e in r10['groups'][g]['layers']]
        s12 = [margin_of_B(B[:, j:j + 1], lab)
               for j in range(B.shape[1])]
        e3 = float(np.abs(np.asarray(s12)
                          - np.asarray(s10)).max())
        ok = (abs(mf - it['stored_margin']) < TOL_A1
              and abs(ac - it['stored_acc']) < TOL_A1
              and e2 < TOL_A2 and e3 < 1e-6)
        anchor_ok = anchor_ok and ok
        anchor[g] = {'margin_recomp': round(mf, 6),
                     'acc_recomp': round(ac, 6),
                     'deltaB_vs_2905_maxabs': e2,
                     'score_vs_2910_maxabs': e3, 'ok': bool(ok)}
        log('anchor %s margin %.5f acc %.5f dLB %.2e '
            'sLB %.2e ok=%s' % (g, mf, ac, e2, e3, ok), lines)

    calib = calibration()
    log('calibration v2: frac_in_band=%.4f p_mean=%.4f '
        'p_median=%.4f tau=%.4f e_p=%.4f band=%s pass=%s'
        % (calib['frac_in_band'], calib['p_mean'],
           calib['p_median'], calib['tau_pool'],
           calib['e_p_theory'], calib['band'], calib['pass']),
        lines)

    verdict = None
    s_set = None
    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    elif not calib['pass']:
        verdict = 'audit_calib_fail_all_void'

    # ---------- probes ----------
    res_g = {}
    if verdict is None:
        for gi, g in enumerate(GROUP_ORDER):
            it = groups[g]
            B, lab = it['B'], it['lab']
            Sm = sign_mat(B)
            rng = np.random.default_rng([SEED_BASE, 10 + gi])
            k_obs, p_zig, r_obs, p_rho = perm_p_obs(
                Sm, lab, rng, N_PERM)
            words = word_attribution(Sm, lab)
            res_g[g] = {
                'gap_sequence': [round(v, 4) for v in
                                 gap_seq(Sm, lab)[0]],
                'zigzag_k': k_obs, 'p_zigzag': round(p_zig, 6),
                'rho1': round(r_obs, 4),
                'p_rho1': round(p_rho, 6),
                'words': words,
            }
            log('%s: gap=%s zigzag k=%d p=%.5f | rho1=%+.3f '
                'p=%.5f | c0 top5_share=%.2f max_flips=%d '
                'tail=%.5f'
                % (g, res_g[g]['gap_sequence'], k_obs, p_zig,
                   r_obs, p_rho,
                   words['class0']['top5_share'],
                   words['class0']['max_flips'],
                   words['class0']['fair_coin_tail_max']),
                lines)

        s_set = [g for g in GROUP_ORDER
                 if res_g[g]['p_zigzag'] <= ALPHA
                 and res_g[g]['p_rho1'] <= ALPHA]
        qa = res_g['qwen_attn']
        log('both-significant set S = %s' % s_set, lines)
        if qa['p_zigzag'] <= ALPHA and qa['p_rho1'] <= ALPHA:
            if s_set == ['qwen_attn']:
                verdict = ('gap_zigzag_confirmed_'
                           'qwen_attn_specific')
            elif ('qwen_attn' in s_set
                    and set(s_set) <= {'qwen_attn',
                                       'glm4_attn'}):
                verdict = ('gap_zigzag_confirmed_'
                           'attn_channel_shared')
            elif len(s_set) >= 3:
                verdict = 'gap_zigzag_generic'
            else:
                verdict = ('gap_zigzag_confirmed_'
                           'partial_specificity')
        elif qa['p_zigzag'] <= ALPHA or qa['p_rho1'] <= ALPHA:
            verdict = 'gap_zigzag_weak_partial'
        else:
            verdict = 'gap_zigzag_absent'

    log('==== VERDICT: %s ====' % verdict, lines)

    res = {
        'phase': 2912,
        'model': 'glm4+qwen (matrix analysis)',
        'prereg': PREREG,
        'seed_base': SEED_BASE, 'n_perm': N_PERM,
        'alpha': ALPHA,
        'anchor': anchor, 'anchor_ok': bool(anchor_ok),
        'calibration': calib,
        'groups': res_g,
        'both_sig_set': s_set,
        'final_verdict': verdict,
        'runtime_s': round(time.monotonic() - t0, 1),
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    np.savez_compressed(
        os.path.join(OUT, 'sign_balance_zigzag.npz'),
        phase=np.int64(2912))
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2912 verdict=%s' % verdict)


if __name__ == '__main__':
    main()
