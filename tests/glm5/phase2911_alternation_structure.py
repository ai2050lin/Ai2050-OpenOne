# -*- coding: utf-8 -*-
"""Phase 2911: alternation structure - formal tests + specificity.

Zero-forward matrix analysis.

Why: 2910 found qwen_attn's per-layer d=1 margin contributions
alternate sign across adjacent window layers (L00-L05 strictly:
0.327(-), 0.778(+), 0.104(-), 0.697(+), 0.253(-), 0.849(+)), and
its lower tail is carried by second-half aggregation.  Two open
questions: (1) is the alternation statistically formal (not a
reading of noise), and (2) is it qwen_attn-specific or shared?

Three independent probes per group, all zero-forward:
  P1 margin-score flips : adjacent-sign flip rate of the per-layer
     d=1 margin scores (recomputed here, anchored against the
     2910 registered score_true values); exact binomial tail
     P(X >= flips | n_pairs, 0.5).
  P2 delta flips : sign-flip rate of the 2905 delta_per_layer
     sequence (zero-adjacent pairs excluded, n_eff recorded);
     same exact binomial tail.
  P3 column oscillation : osc = mean_corr(adjacent cols) -
     mean_corr(cols two apart) of B; permutation null (column
     order shuffles, 10000 perms) -> one-sided p = P(perm <= obs)
     (negative osc = oscillation).

Adjudication (frozen):
  guards: anchors a1/a2/a3 fail => anchor_fail_all_void;
  calibration audit (rng [2911,0]: 200 iid 57x10 matrices, 1000
  perms each) - frac of permutation p in [0.05,0.95] must be in
  [0.80,0.97], else audit_calib_fail_all_void.
  qwen_attn alternation: P1 p<=0.05 AND P3 p<=0.05 required.
  specificity by the set S of groups with significant negative
  osc (P3 p<=0.05):
    S == {qwen_attn}                  => alternation_confirmed_
                                         qwen_attn_specific
    qwen_attn in S and S subset of    => alternation_confirmed_
    {qwen_attn, glm4_attn}               attn_channel_shared
    |S| >= 3                          => alternation_generic
    else                              => alternation_confirmed_
                                         partial_specificity
  P1 or P3 not significant for qwen_attn =>
    alternation_not_confirmed_margin_only / osc_only / neither.
  P2 (delta flips) descriptive: mechanism-level evidence if
  qwen_attn delta flip tail p <= 0.05.

Anchors: a1 full-layer margin/acc == stored 2e-5; a2 Delta_B ==
2905 delta_per_layer 1e-4; a3 recomputed per-layer d=1 scores ==
2910 registered score_true within 1e-6.

Output: phase2911/alternation_structure/.
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
OUT = os.path.join(BASE, 'phase2911', 'alternation_structure')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2911_run_report.txt')
SEED_BASE = 2911
N_PERM = 10000
N_CALIB = 200
N_CALIB_PERM = 1000
TOL_A1 = 2e-5
TOL_A2 = 1e-4
ALPHA = 0.05
GROUP_ORDER = ('glm4_mlp', 'glm4_attn', 'qwen_mlp', 'qwen_attn')

PREREG = {
    'mode': 'zero_forward_matrix_analysis',
    'probes': 'P1 adjacent-sign flip rate of per-layer d=1 margin '
              'scores + exact binomial tail; P2 sign-flip rate of '
              '2905 delta_per_layer (zero pairs excluded) + same '
              'tail; P3 osc = mean_corr(adj) - mean_corr(skip1) '
              'of B columns, 10000 column-permutation null, '
              'one-sided p = P(perm <= obs)',
    'sources': 'B + stored margin/acc (2902/2903); '
               'delta_per_layer (2905); 2910 score_true (anchor '
               'a3)',
    'anchors': 'a1 margin/acc == stored 2e-5; a2 Delta_B == 2905 '
               '1e-4; a3 per-layer d=1 scores == 2910 score_true '
               '1e-6',
    'calibration': 'rng [2911,0]: 200 iid 57x10 normal matrices, '
                   '1000 perms each; frac of perm-p in [0.05,0.95]'
                   ' must be in [0.80,0.97] and median p in '
                   '[0.40,0.60], else audit_calib_fail_all_void',
    'adjudication': 'qwen_attn needs P1 p<=0.05 and P3 p<=0.05; '
                    'specificity by S = groups with P3 p<=0.05: '
                    'S=={qwen_attn} => '
                    'alternation_confirmed_qwen_attn_specific; '
                    'qwen_attn in S subset {qwen_attn,glm4_attn} '
                    '=> alternation_confirmed_attn_channel_shared;'
                    ' |S|>=3 => alternation_generic; else '
                    'alternation_confirmed_partial_specificity; '
                    'P1 or P3 fails => alternation_not_confirmed_'
                    'margin_only/osc_only/neither; P2 descriptive',
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


def osc_index(B):
    C = np.corrcoef(B.T)
    w = C.shape[0]
    adj = [C[j, j + 1] for j in range(w - 1)]
    skip = [C[j, j + 2] for j in range(w - 2)]
    return float(np.mean(adj) - np.mean(skip))


def osc_perm_p(B, rng, n_perm):
    obs = osc_index(B)
    w = B.shape[1]
    cnt = 0
    for _ in range(n_perm):
        if osc_index(B[:, rng.permutation(w)]) <= obs:
            cnt += 1
    return obs, (cnt + 1) / float(n_perm + 1)


def flip_stats(vals):
    signs = np.sign(np.asarray(vals, dtype=float))
    flips = 0
    n_eff = 0
    for j in range(len(signs) - 1):
        a, b = signs[j], signs[j + 1]
        if a == 0 or b == 0:
            continue
        n_eff += 1
        if a != b:
            flips += 1
    tail = (sum(comb(n_eff, k)
                for k in range(flips, n_eff + 1)) / 2.0 ** n_eff
            if n_eff > 0 else 1.0)
    return flips, n_eff, tail


def calibration():
    rng = np.random.default_rng([SEED_BASE, 0])
    ps = []
    for _ in range(N_CALIB):
        B = rng.normal(size=(57, 10))
        _, p = osc_perm_p(B, rng, N_CALIB_PERM)
        ps.append(p)
    ps = np.asarray(ps)
    frac = float(np.mean((ps >= 0.05) & (ps <= 0.95)))
    med = float(np.median(ps))
    ok = bool(0.80 <= frac <= 0.97 and 0.40 <= med <= 0.60)
    return {'frac_in_band': round(frac, 4),
            'p_median': round(med, 4), 'pass': ok}


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2911,
                   'name': 'alternation_structure',
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
        d11 = B[lab == 1].mean(0) - B[lab == 0].mean(0)
        e2 = float(np.abs(d11 - d05).max())
        s10 = [e['score_true']
               for e in r10['groups'][g]['layers']]
        s11 = [margin_of_B(B[:, j:j + 1], lab)
               for j in range(B.shape[1])]
        e3 = float(np.abs(np.asarray(s11)
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
    log('calibration: frac_in_band=%.4f p_median=%.4f pass=%s'
        % (calib['frac_in_band'], calib['p_median'],
           calib['pass']), lines)

    verdict = None
    s_set = None
    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    elif not calib['pass']:
        verdict = 'audit_calib_fail_all_void'

    # ---------- probes ----------
    res_g = {}
    if verdict is None:
        for g in GROUP_ORDER:
            it = groups[g]
            B, lab = it['B'], it['lab']
            n_win = B.shape[1]
            scores = [margin_of_B(B[:, j:j + 1], lab)
                      for j in range(n_win)]
            f1, n1, p1 = flip_stats(scores)
            mdl, ch = g.split('_')
            delta = list(r05['groups']['%s_%s' % (mdl, ch)]
                         ['delta_per_layer'])
            f2, n2, p2 = flip_stats(delta)
            rng = np.random.default_rng(
                [SEED_BASE, 10 + GROUP_ORDER.index(g)])
            obs, p3 = osc_perm_p(B, rng, N_PERM)
            res_g[g] = {
                'margin_scores': [round(s, 6) for s in scores],
                'margin_flips': f1, 'margin_pairs': n1,
                'margin_flip_p': round(p1, 6),
                'delta_flips': f2, 'delta_pairs': n2,
                'delta_flip_p': round(p2, 6),
                'delta_signs': [int(np.sign(d)) for d in delta],
                'osc_index': round(obs, 6),
                'osc_perm_p': round(p3, 6),
            }
            log('%s: margin flips %d/%d p=%.5f | delta flips '
                '%d/%d p=%.5f | osc=%.5f perm_p=%.5f'
                % (g, f1, n1, p1, f2, n2, p2, obs, p3), lines)

        s_set = [g for g in GROUP_ORDER
                 if res_g[g]['osc_perm_p'] <= ALPHA
                 and res_g[g]['osc_index'] < 0]
        qa = res_g['qwen_attn']
        p1_ok = qa['margin_flip_p'] <= ALPHA
        p3_ok = qa['osc_perm_p'] <= ALPHA
        log('significant-osc set S = %s' % s_set, lines)
        if not (p1_ok and p3_ok):
            if p1_ok and not p3_ok:
                verdict = 'alternation_not_confirmed_margin_only'
            elif p3_ok and not p1_ok:
                verdict = 'alternation_not_confirmed_osc_only'
            else:
                verdict = 'alternation_not_confirmed_neither'
        elif s_set == ['qwen_attn']:
            verdict = 'alternation_confirmed_qwen_attn_specific'
        elif ('qwen_attn' in s_set
                and set(s_set) <= {'qwen_attn', 'glm4_attn'}):
            verdict = 'alternation_confirmed_attn_channel_shared'
        elif len(s_set) >= 3:
            verdict = 'alternation_generic'
        else:
            verdict = 'alternation_confirmed_partial_specificity'

    log('==== VERDICT: %s ====' % verdict, lines)

    res = {
        'phase': 2911,
        'model': 'glm4+qwen (matrix analysis)',
        'prereg': PREREG,
        'seed_base': SEED_BASE, 'n_perm': N_PERM,
        'alpha': ALPHA,
        'anchor': anchor, 'anchor_ok': bool(anchor_ok),
        'calibration': calib,
        'groups': res_g,
        'sig_osc_set': s_set,
        'final_verdict': verdict,
        'runtime_s': round(time.monotonic() - t0, 1),
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    np.savez_compressed(
        os.path.join(OUT, 'alternation_structure.npz'),
        phase=np.int64(2911))
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2911 verdict=%s' % verdict)


if __name__ == '__main__':
    main()
