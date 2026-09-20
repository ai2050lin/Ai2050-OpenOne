# -*- coding: utf-8 -*-
"""Phase 2923: polarity sign anatomy of attribute events -- zero-forward.

Why: 2922 found all 13 attribute events are polarity-aligned
(p_pole = 0.0005 floor) but the driving SIGN splits 8 LOW vs 5
HIGH (d_pole = (mean r[hi] - mean r[lo]) / pooled_std). Open
question (2922 candidate A): is the driving sign predictable from
the event's top-|r| word polarity composition, from its layer
position, or is it event-specific?

Mode: zero forward. Artifact domain on the 2921 npz (B_speed/
B_size/B_moist, pole labels, words) + 2922 result/npz as the
registered reference (d_pole values, linkage edges).

Main test P1 (frozen BEFORE running): for each event, f_k = mean
of s_i over the top-k |r| words (s_i = +1 HIGH, -1 LOW), k = 5
(primary) and k = 10 (secondary). T_obs = #{events:
sign(f_k) == sign(d_pole)} (f_k = 0 counts as mismatch).
Null: 2000 global pole-label permutations (size-preserving,
independent per event, fresh default_rng(2898)); under a
permutation both d_pole and f_k are recomputed from the permuted
labels (top-|r| word set unchanged). p_perm =
(#{T_perm >= T_obs} + 1)/2001. VERDICT CRITERION (frozen):
p_perm <= 0.05 AND T_obs >= 10 at k = 5.

P2 layer position (quasi-post-hoc: 2922 output already showed
d_pole and peak layers side by side, direction expectation
known): sign x peak-median-split Fisher exact (one-sided,
LOW-concentrated-in-shallow) + point-biserial correlation.
Descriptive weight only.

P3 d_pole decomposition (descriptive): raw means m_hi, m_lo and
their sign combination - contrast type (opposite signs) vs
magnitude type (same sign, |m_lo| > |m_hi|).

P4 same-head cross-category sign pairs (quasi-post-hoc,
descriptive): the 8 heads from 2922 P4.

P5 linkage-edge sign homogeneity (quasi-post-hoc, descriptive +
exact binomial): the 4 size edges from 2922 - are endpoints
sign-matched?

Adjudication (frozen):
  anchor fail => anchor_fail_all_void;
  P1 criterion met => polarity_sign_topword_predicted;
  else P2 Fisher p <= 0.05 => polarity_sign_layer_structured;
  else => polarity_sign_event_specific.

Output: phase2923/polarity_sign_anatomy/.
"""
import hashlib
import json
import os
import time
from math import comb

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC21 = os.path.join(BASE, 'phase2921', 'attr_vocab_expansion',
                     'attr_vocab_expansion.npz')
SRC22 = os.path.join(BASE, 'phase2922', 'attr_event_anatomy')
OUT = os.path.join(BASE, 'phase2923', 'polarity_sign_anatomy')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2923_run_report.txt')
N_PERM = 2000
FDR_Q = 0.05
NH, NL = 32, 36

PREREG = {
    'mode': 'zero-forward polarity sign anatomy on the 2921 npz '
            'with 2922 registered d_pole/edges as reference '
            '(no model load)',
    'question': '2922 candidate A: is the driving sign (d_pole '
                '< 0 LOW-driven vs > 0 HIGH-driven) of the 13 '
                'attribute events predictable from the top-|r| '
                'word polarity composition, from layer position, '
                'or event-specific?',
    'events': {'speed': [[18, 16], [14, 12]],
               'size': [[21, 7], [24, 19], [12, 18], [26, 21],
                        [25, 15], [11, 23], [11, 27], [22, 2],
                        [18, 7]],
               'moist': [[8, 15], [10, 9]]},
    'main_test': 'f_k = mean sign(+1 HIGH/-1 LOW) over top-k |r| '
                 'words, k=5 primary / k=10 secondary; T_obs = '
                 '#{sign(f_k)==sign(d_pole)}, f_k=0 counts as '
                 'mismatch; null = 2000 global size-preserving '
                 'pole permutations (fresh default_rng(2898), '
                 'independent per event; both d_pole and f_k '
                 'recomputed under permuted labels, top-|r| set '
                 'unchanged); p_perm = (#{T_perm>=T_obs}+1)/2001; '
                 'CRITERION: p_perm <= 0.05 AND T_obs >= 10 at '
                 'k=5',
    'anchors': {
        'a1': '2921 sign_M recomputed from npz B per axis '
              '(2922 a1 verbatim): median absdiff < 1e-6, max '
              '< 0.15, n(>0.02) <= 5, sig cells < 0.05',
        'a2': 'd_pole recomputed (2922 formula) vs 2922 result '
              'P1 values: |diff| < 1e-3 for all 13 events',
        'a3': '2922 npz spot-check: rho_obs38 edge (21,7)-(18,7) '
              '= 0.6764 +/- 1e-4, its p_pair 0.029485 +/- 1e-6, '
              'sign_M_ref == 2921 npz sign_M',
    },
    'probes': {
        'P1': 'main test as above (both k reported)',
        'P2': 'sign x peak-median Fisher exact (one-sided) + '
              'point-biserial (quasi-post-hoc, descriptive '
              'weight)',
        'P3': 'm_hi/m_lo raw means + sign combination 2x2: '
              'contrast vs magnitude type (descriptive)',
        'P4': 'same-head cross-category sign pairs (descriptive)',
        'P5': 'size linkage-edge sign homogeneity 4/4 exact '
              'binomial (descriptive)',
    },
    'verdict': 'anchor fail => anchor_fail_all_void; P1 met => '
               'polarity_sign_topword_predicted; else P2 Fisher '
               'p <= 0.05 => polarity_sign_layer_structured; '
               'else => polarity_sign_event_specific',
}

EVENTS = {
    'speed': [(18, 16), (14, 12)],
    'size': [(21, 7), (24, 19), (12, 18), (26, 21), (25, 15),
             (11, 23), (11, 27), (22, 2), (18, 7)],
    'moist': [(8, 15), (10, 9)],
}
AX_AI = {'speed': 1, 'size': 2, 'moist': 3}


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


def log(msg, lines):
    lines.append(msg)
    print(msg, flush=True)


def masks(lab):
    n = len(lab)
    eye = np.eye(n, dtype=bool)
    same = (lab[:, None] == lab[None, :]) & (~eye)
    diff = (~eye) & (~same)
    return same, diff


def sign_recompute(B, lab):
    same_m, diff_m = masks(lab)
    out = np.zeros((B.shape[0], B.shape[2]))
    for h in range(B.shape[0]):
        for li in range(B.shape[2]):
            s = np.sign(B[h, :, li])
            s[s == 0] = 1.0
            Gm = np.outer(s, s)
            out[h, li] = Gm[same_m].mean() - Gm[diff_m].mean()
    return out


def d_pole_of(r, hi, lo):
    v_hi = r[hi].astype(np.float64)
    v_lo = r[lo].astype(np.float64)
    pooled = np.sqrt(max(v_hi.var() + v_lo.var(), 1e-30))
    return (float(v_hi.mean() - v_lo.mean()) / float(pooled),
            float(v_hi.mean()), float(v_lo.mean()))


def fisher_one_sided(a, b, c, d):
    """P(X >= a) under hypergeom(N, r1=a+b, c1=a+c); table
    [[a, b], [c, d]] with rows LOW/HIGH, cols shallow/deep."""
    n = a + b + c + d
    r1 = a + b
    c1 = a + c
    num = sum(comb(r1, x) * comb(n - r1, c1 - x)
              for x in range(a, min(r1, c1) + 1))
    return num / comb(n, c1)


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2923,
                   'name': 'polarity_sign_anatomy',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2921_npz': sha8(SRC21),
                               's2922_result':
                                   sha8(os.path.join(
                                       SRC22, 'result.json')),
                               's2922_npz': sha8(os.path.join(
                                   SRC22,
                                   'attr_event_anatomy.npz'))},
                   'n_perm': N_PERM, 'fdr_q': FDR_Q,
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    z = np.load(SRC21, allow_pickle=True)
    B = {ax: z['B_%s' % ax].astype(np.float64)
         for ax in ('speed', 'size', 'moist')}
    sign_M = z['sign_M'].astype(np.float64)
    p_maxT = z['p_maxT'].astype(np.float64)
    pole = {ax: np.asarray(z['labels_%s' % ax]).astype(int)
            for ax in ('speed', 'size', 'moist')}
    with open(os.path.join(SRC22, 'result.json'),
              encoding='utf-8') as f:
        res22 = json.load(f)
    z22 = np.load(os.path.join(SRC22, 'attr_event_anatomy.npz'),
                  allow_pickle=True)

    # ---------- anchors ----------
    a1 = {}
    a1_ok = True
    for ax in ('speed', 'size', 'moist'):
        sre = sign_recompute(B[ax], pole[ax])
        D = np.abs(sre - sign_M[AX_AI[ax]])
        sc = [(h, li) for h in range(NH) for li in range(NL)
              if p_maxT[AX_AI[ax]][h, li] <= FDR_Q]
        sc_max = max(float(D[h, li]) for (h, li) in sc)
        ok = bool(np.median(D) < 1e-6 and D.max() < 0.15
                  and np.sum(D > 0.02) <= 5 and sc_max < 0.05)
        a1[ax] = {'median': float(np.median(D)),
                  'max': float(D.max()), 'ok': ok}
        a1_ok = a1_ok and ok
    d22 = {}
    for e in res22['P1']:
        d22['%s%s' % (e['axis'], e['event'])] = e['d_pole']
    a2 = {}
    a2_ok = True
    for ax in ('speed', 'size', 'moist'):
        lab = pole[ax]
        hi = lab == 1
        lo = lab == 0
        for (h, li) in EVENTS[ax]:
            d, _, _ = d_pole_of(B[ax][h, :, li], hi, lo)
            ref = d22['%s[%d, %d]' % (ax, h, li)]
            ok = abs(d - ref) < 1e-3
            a2['%s(%d,%d)' % (ax, h, li)] = {
                'recalc': round(d, 6), 'ref': ref, 'ok': ok}
            a2_ok = a2_ok and ok
    # a3: locate the (21,7)-(18,7) edge in the 38-pair layout
    pairs_all = []
    for ax in ('speed', 'size', 'moist'):
        ev = EVENTS[ax]
        for x in range(len(ev)):
            for y in range(x + 1, len(ev)):
                pairs_all.append((ax, ev[x], ev[y]))
    edge_i = [s_i for s_i, (ax, e1, e2) in enumerate(pairs_all)
              if ax == 'size'
              and {tuple(e1), tuple(e2)} == {(21, 7), (18, 7)}][0]
    rho_ref = float(z22['rho_obs38'][edge_i])
    pp_ref = float(z22['p_pair38'][edge_i])
    sm_same = np.array_equal(
        z22['sign_M_ref'].astype(np.float64), sign_M)
    a3 = {'edge_idx': edge_i, 'rho': round(rho_ref, 6),
          'p_pair': round(pp_ref, 6),
          'sign_M_ref_eq': bool(sm_same)}
    a3_ok = bool(abs(rho_ref - 0.6764) < 1e-4
                 and abs(pp_ref - 0.029485) < 1e-6 and sm_same)
    anchor_ok = bool(a1_ok and a2_ok and a3_ok)
    log('a1 ok=%s %s' % (a1_ok,
                         {ax: (round(a1[ax]['median'], 12),
                               round(a1[ax]['max'], 5))
                          for ax in a1}), lines)
    log('a2 ok=%s (13/13 |diff|<1e-3)' % a2_ok, lines)
    log('a3 %s ok=%s' % (a3, a3_ok), lines)

    verdict = None
    p1 = p2 = p3 = p4 = p5 = None
    save = {}
    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        # ---------- frozen observation set ----------
        obs = []
        for ax in ('speed', 'size', 'moist'):
            lab = pole[ax]
            hi = lab == 1
            lo = lab == 0
            for (h, li) in EVENTS[ax]:
                r = B[ax][h, :, li]
                d, m_hi, m_lo = d_pole_of(r, hi, lo)
                sgn = np.where(lab == 1, 1.0, -1.0)
                order = np.argsort(-np.abs(r))
                f5 = float(sgn[order[:5]].mean())
                f10 = float(sgn[order[:10]].mean())
                obs.append({'axis': ax, 'event': [h, li],
                            'd_pole': d, 'm_hi': m_hi,
                            'm_lo': m_lo, 'f5': f5, 'f10': f10,
                            'top5': [str(w) for w in
                                     np.asarray(z['words_%s' % ax])
                                     [order[:5]]],
                            'hi_labels_top5':
                                [int(x) for x in lab[order[:5]]]})

        def agree(fk, d):
            if abs(fk) < 1e-12:
                return False
            return (fk > 0) == (d > 0)

        t5_obs = sum(1 for o in obs if agree(o['f5'], o['d_pole']))
        t10_obs = sum(1 for o in obs
                      if agree(o['f10'], o['d_pole']))
        # permutation null (rng 2898, independent per event)
        rng = np.random.default_rng(2898)
        perms = {}
        for ax in ('speed', 'size', 'moist'):
            n = len(pole[ax])
            pm = np.empty((N_PERM, n), dtype=np.int64)
            for t in range(N_PERM):
                pm[t] = rng.permutation(n)
            perms[ax] = pm
        t5_perm = np.zeros(N_PERM, dtype=np.int64)
        t10_perm = np.zeros(N_PERM, dtype=np.int64)
        for t in range(N_PERM):
            c5 = c10 = 0
            for oi, o in enumerate(obs):
                ax = o['axis']
                h, li = o['event']
                r = B[ax][h, :, li]
                lab_p = pole[ax][perms[ax][t]]
                hi_p = lab_p == 1
                lo_p = lab_p == 0
                d_p, _, _ = d_pole_of(r, hi_p, lo_p)
                sgn_p = np.where(lab_p == 1, 1.0, -1.0)
                order = np.argsort(-np.abs(r))
                f5p = float(sgn_p[order[:5]].mean())
                f10p = float(sgn_p[order[:10]].mean())
                c5 += int(agree(f5p, d_p))
                c10 += int(agree(f10p, d_p))
            t5_perm[t] = c5
            t10_perm[t] = c10
        p5_perm = (int(np.sum(t5_perm >= t5_obs)) + 1.0) \
            / (N_PERM + 1.0)
        p10_perm = (int(np.sum(t10_perm >= t10_obs)) + 1.0) \
            / (N_PERM + 1.0)
        p1 = {'t5_obs': t5_obs, 't10_obs': t10_obs,
              'p5_perm': round(p5_perm, 6),
              'p10_perm': round(p10_perm, 6),
              't5_perm_p50': float(np.percentile(t5_perm, 50)),
              't5_perm_max': int(t5_perm.max()),
              't10_perm_p50': float(np.percentile(t10_perm, 50)),
              't10_perm_max': int(t10_perm.max()),
              'events': [{'axis': o['axis'],
                          'event': o['event'],
                          'd_pole': round(o['d_pole'], 4),
                          'f5': round(o['f5'], 3),
                          'f10': round(o['f10'], 3),
                          'agree5': bool(agree(o['f5'],
                                               o['d_pole'])),
                          'agree10': bool(agree(o['f10'],
                                                o['d_pole'])),
                          'top5': o['top5'],
                          'hi_labels_top5':
                              o['hi_labels_top5']}
                         for o in obs]}
        log('P1: T5 %d/13 (p %.4f, perm p50 %.1f max %d) | '
            'T10 %d/13 (p %.4f)'
            % (t5_obs, p5_perm, p1['t5_perm_p50'],
               p1['t5_perm_max'], t10_obs, p10_perm), lines)
        log('P1 detail: %s'
            % [(o['axis'], tuple(o['event']),
                round(o['d_pole'], 3), round(o['f5'], 3),
                round(o['f10'], 3))
               for o in obs], lines)

        # ---------- P2 layer position ----------
        pks = []
        sgn_d = []
        for o in obs:
            pks.append(int(np.argmax(sign_M[AX_AI[o['axis']]][
                o['event'][0], :])))
            sgn_d.append(1 if o['d_pole'] > 0 else -1)
        pks = np.array(pks)
        sgn_d = np.array(sgn_d)
        med = float(np.median(pks))
        low = sgn_d < 0
        a = int(np.sum(low & (pks <= med)))
        b = int(np.sum(low & (pks > med)))
        c = int(np.sum((~low) & (pks <= med)))
        dd = int(np.sum((~low) & (pks > med)))
        fisher_p = fisher_one_sided(a, b, c, dd)
        s_centered = sgn_d - sgn_d.mean()
        p_centered = pks - pks.mean()
        r_pb = float(s_centered @ p_centered
                     / np.sqrt((s_centered @ s_centered)
                               * (p_centered @ p_centered)))
        p2 = {'peak_layers': pks.tolist(),
              'signs': sgn_d.tolist(),
              'median_peak': med,
              'fisher_table': {'low_shallow': a, 'low_deep': b,
                               'high_shallow': c,
                               'high_deep': dd},
              'fisher_p': round(fisher_p, 6),
              'point_biserial': round(r_pb, 4),
              'note': 'quasi-post-hoc (2922 output showed d_pole '
                      'and peaks together)'}
        log('P2: table %s fisher p %.4f r_pb %.3f'
            % (p2['fisher_table'], fisher_p, r_pb), lines)

        # ---------- P3 decomposition ----------
        p3 = []
        for o in obs:
            st = 'contrast' if (o['m_hi'] > 0) != (o['m_lo'] > 0) \
                else 'magnitude'
            p3.append({'axis': o['axis'],
                       'event': o['event'],
                       'm_hi': round(o['m_hi'], 4),
                       'm_lo': round(o['m_lo'], 4),
                       'type': st})
        n_contrast = sum(1 for x in p3 if x['type'] == 'contrast')
        log('P3: %s | contrast %d magnitude %d'
            % (p3, n_contrast, 13 - n_contrast), lines)

        # ---------- P4 same-head sign pairs ----------
        by_head = {}
        for o in obs:
            by_head.setdefault(o['event'][0], []).append(
                (o['axis'], tuple(o['event']),
                 round(o['d_pole'], 3)))
        p4 = {'heads': {str(h): v for h, v
                        in sorted(by_head.items())
                        if len(v) >= 2},
              'note': 'descriptive; h18 = (+1.36 speed, -1.44 '
                      'size) opposite-sign pair'}
        log('P4: %s' % p4['heads'], lines)

        # ---------- P5 edge sign homogeneity (2922's 4 sig edges) ----------
        de = {tuple(o['event']): (1 if o['d_pole'] > 0 else -1)
              for o in obs if o['axis'] == 'size'}
        edge_signs = []
        for e in res22['P2']['edges']:
            ea, eb = e['pair']
            s1 = de[tuple(ea)]
            s2 = de[tuple(eb)]
            edge_signs.append((list(ea), list(eb), s1 == s2))
        assert len(edge_signs) == 4
        k_match = sum(1 for (_, _, m) in edge_signs if m)
        binom_p = sum(comb(4, x) for x in range(k_match, 5)) \
            / 16.0
        p5 = {'edges': [[e[0], e[1], bool(e[2])]
                        for e in edge_signs],
              'k_match': k_match,
              'exact_binomial_p': round(binom_p, 6),
              'note': 'quasi-post-hoc descriptive'}
        log('P5: %d/4 sign-matched edges, binom p %.4f'
            % (k_match, binom_p), lines)

        # ---------- verdict ----------
        p1_met = bool(p5_perm <= FDR_Q and t5_obs >= 10)
        if p1_met:
            verdict = 'polarity_sign_topword_predicted'
        elif fisher_p <= FDR_Q:
            verdict = 'polarity_sign_layer_structured'
        else:
            verdict = 'polarity_sign_event_specific'

        save = {
            'd_pole13': np.array([o['d_pole'] for o in obs],
                                 dtype=np.float64),
            'f5_13': np.array([o['f5'] for o in obs],
                              dtype=np.float64),
            'f10_13': np.array([o['f10'] for o in obs],
                               dtype=np.float64),
            't5_perm': t5_perm.astype(np.int64),
            't10_perm': t10_perm.astype(np.int64),
            'm_hi13': np.array([o['m_hi'] for o in obs],
                               dtype=np.float64),
            'm_lo13': np.array([o['m_lo'] for o in obs],
                               dtype=np.float64),
            'event_ids_speed': np.array(EVENTS['speed'], dtype=int),
            'event_ids_size': np.array(EVENTS['size'], dtype=int),
            'event_ids_moist': np.array(EVENTS['moist'], dtype=int),
            'peak_layers': pks.astype(np.int64),
        }

    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2923,
           'model': 'qwen3-4b (zero forward)',
           'prereg': PREREG,
           'anchors': {'a1': a1, 'a1_ok': a1_ok,
                       'a2': a2, 'a2_ok': a2_ok,
                       'a3': a3, 'a3_ok': a3_ok,
                       'ok': anchor_ok},
           'P1': p1, 'P2': p2, 'P3': p3, 'P4': p4, 'P5': p5,
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if save:
        np.savez_compressed(
            os.path.join(OUT, 'polarity_sign_anatomy.npz'), **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2923 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
