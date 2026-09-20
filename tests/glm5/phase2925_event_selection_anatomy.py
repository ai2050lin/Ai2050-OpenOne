# -*- coding: utf-8 -*-
"""Phase 2925: event selection anatomy -- zero-forward.

Why: 2924 located the 2923 layer structure as an EVENT-SELECTION
property: the 13 maxT-sig attribute event cells are bipolar
contrast detectors with depth-organized signs, while the grid
background is not. Open question (2924 candidate A): WHAT
distinguishes the 13 sig cells from same-layer non-sig cells
beyond the margin criterion itself? Which grid features select
events?

Mode: zero forward. Artifact domain on the 2921 npz (B matrices,
pole labels) + 2923/2924 npz references.

Features per cell (h, l, ax), all computable from B_ax:
  f1 margin          (the selection criterion itself; reference
                      only, EXCLUDED from the verdict family)
  f2 phi             pole-sign association of the response
                      vector: 2x2 table (hi/lo x r>=0/r<0), phi
                      = (a*d - b*c)/sqrt(...); r == 0 counted on
                      the positive side (2917 sign convention)
  f3 d_pole_abs      |(m_hi - m_lo) / pooled_std|
  f4 contrast_raw    |m_hi - m_lo|
  f5 pr_word         participation ratio of r
  f6 top3_mass       top-3 |r| mass fraction
  f7 mean_abs_r      mean |r| (response magnitude)
  f8 neighbor_phi    max phi over same-head adjacent layers
                      (l-1, l+1, in-range) - spatial channel
                      context

Design (frozen): for each of the 13 sig cells, the comparison
set is the same-axis same-layer non-sig heads. Feature percentile
pct = mean over the pooled set (sig cell + non-sig cells) of
(value <= sig value); 1.0 = highest. Per feature, 13 pcts; test =
sign test vs 0.5 (one-sided, exact binomial; p floor 1/8192 <<
0.05, granularity pre-check, discipline 7 - family = the 7
non-trivial features, BH q = 0.05, exploratory).

Verdict criteria (frozen):
  anchor fail => anchor_fail_all_void;
  n_BH_sig >= 3 AND median pct(phi) >= 0.9
      => events_polar_separation_selected;
  n_BH_sig >= 3 => events_multifeature_selected;
  else => events_margin_only.

Anchors (frozen):
  a1 2921 sign_M recomputed from npz B per axis (2922 a1
     verbatim);
  a2 13-event d_pole recompute vs 2923 npz d_pole13 bit-exact
     (< 1e-9);
  a3 2924 npz rho_axis[size] == -0.0607 +/- 1e-4.

Output: phase2925/event_selection_anatomy/.
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
SRC23 = os.path.join(BASE, 'phase2923', 'polarity_sign_anatomy')
SRC24 = os.path.join(BASE, 'phase2924', 'depth_polarity_gradient')
OUT = os.path.join(BASE, 'phase2925', 'event_selection_anatomy')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2925_run_report.txt')
FDR_Q = 0.05
NH, NL = 32, 36

PREREG = {
    'mode': 'zero-forward event selection anatomy on the 2921 '
            'npz (no model load) + 2923/2924 npz references',
    'question': '2924 candidate A: what distinguishes the 13 '
                'maxT-sig attribute event cells from same-layer '
                'non-sig cells beyond the margin criterion '
                'itself - which grid features select events?',
    'features': 'f1 margin (reference, excluded from verdict '
                'family); f2 phi pole-sign association (r==0 -> '
                'positive side); f3 |d_pole|; f4 |m_hi-m_lo|; '
                'f5 PR_word; f6 top3_mass; f7 mean|r|; f8 '
                'neighbor_phi (same-head l+/-1 max)',
    'design': 'per sig cell: percentile of its feature value '
              'within the pooled set (itself + same-axis '
              'same-layer non-sig heads), pct = mean(value <= '
              'sig value); per feature 13 pcts; sign test vs '
              '0.5 one-sided exact binomial (p floor 1/8192, '
              'granularity pre-check ok); family = 7 non-'
              'trivial features, BH q=0.05, exploratory',
    'verdict': 'anchor fail => anchor_fail_all_void; n_BH_sig '
               '>= 3 AND median pct(phi) >= 0.9 => '
               'events_polar_separation_selected; n_BH_sig >= 3 '
               '=> events_multifeature_selected; else => '
               'events_margin_only',
    'anchors': {
        'a1': '2921 sign_M recompute per axis (2922 a1 '
              'verbatim)',
        'a2': '13-event d_pole recompute vs 2923 npz d_pole13 '
              'bit-exact < 1e-9',
        'a3': '2924 npz rho_axis[size] == -0.0607 +/- 1e-4',
    },
}

EVENTS = {
    'speed': [(18, 16), (14, 12)],
    'size': [(21, 7), (24, 19), (12, 18), (26, 21), (25, 15),
             (11, 23), (11, 27), (22, 2), (18, 7)],
    'moist': [(8, 15), (10, 9)],
}
AX_AI = {'speed': 1, 'size': 2, 'moist': 3}
FEATS = ['phi', 'd_pole_abs', 'contrast_raw', 'pr_word',
         'top3_mass', 'mean_abs_r', 'neighbor_phi']


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
    return float(v_hi.mean() - v_lo.mean()) / float(pooled)


def part_ratio(x):
    a = np.abs(x)
    s1 = a.sum()
    s2 = float((a * a).sum())
    return float(s1 * s1 / max(s2, 1e-30))


def feats_of(B, h, l, hi, lo):
    r = B[h, :, l]
    m_hi = float(r[hi].mean())
    m_lo = float(r[lo].mean())
    a = int(np.sum(hi & (r >= 0)))
    b = int(np.sum(hi & (r < 0)))
    c = int(np.sum(lo & (r >= 0)))
    d = int(np.sum(lo & (r < 0)))
    den = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    phi = (a * d - b * c) / max(den, 1e-30)
    nb = []
    for l2 in (l - 1, l + 1):
        if 0 <= l2 < NL:
            r2 = B[h, :, l2]
            a2 = int(np.sum(hi & (r2 >= 0)))
            b2 = int(np.sum(hi & (r2 < 0)))
            c2 = int(np.sum(lo & (r2 >= 0)))
            d2 = int(np.sum(lo & (r2 < 0)))
            den2 = np.sqrt((a2 + b2) * (c2 + d2) * (a2 + c2)
                           * (b2 + d2))
            nb.append((a2 * d2 - b2 * c2) / max(den2, 1e-30))
    aa = np.abs(r)
    return {'phi': float(phi),
            'd_pole_abs': abs(d_pole_of(r, hi, lo)),
            'contrast_raw': abs(m_hi - m_lo),
            'pr_word': part_ratio(r),
            'top3_mass': float(np.sort(aa)[::-1][:3].sum()
                               / max(aa.sum(), 1e-30)),
            'mean_abs_r': float(aa.mean()),
            'neighbor_phi': float(max(nb)) if nb else 0.0}


def sign_test_p(k, n):
    return sum(comb(n, x) for x in range(k, n + 1)) / 2.0 ** n


def bh_q(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    qs = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        val = pvals[i] * m / (rank + 1)
        prev = min(prev, val)
        qs[i] = prev
    return qs


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2925,
                   'name': 'event_selection_anatomy',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2921_npz': sha8(SRC21),
                               's2923_npz': sha8(os.path.join(
                                   SRC23,
                                   'polarity_sign_anatomy.npz')),
                               's2924_npz': sha8(os.path.join(
                                   SRC24,
                                   'depth_polarity_gradient.npz'))},
                   'fdr_q': FDR_Q,
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
    z23 = np.load(os.path.join(SRC23,
                               'polarity_sign_anatomy.npz'),
                  allow_pickle=True)
    z24 = np.load(os.path.join(SRC24,
                               'depth_polarity_gradient.npz'),
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
    a2_ok = True
    a2_max = 0.0
    oi = 0
    for ax in ('speed', 'size', 'moist'):
        lab = pole[ax]
        hi = lab == 1
        lo = lab == 0
        for (h, li) in EVENTS[ax]:
            diff = abs(d_pole_of(B[ax][h, :, li], hi, lo)
                       - float(z23['d_pole13'][oi]))
            a2_max = max(a2_max, diff)
            a2_ok = a2_ok and diff < 1e-9
            oi += 1
    rho_size = float(z24['rho_axis'][1])
    a3_ok = bool(abs(rho_size - (-0.0607)) < 1e-4)
    a3 = {'rho_axis_size': round(rho_size, 6)}
    anchor_ok = bool(a1_ok and a2_ok and a3_ok)
    log('a1 ok=%s | a2 ok=%s max_diff %.1e | a3 %s ok=%s'
        % (a1_ok, a2_ok, a2_max, a3, a3_ok), lines)

    verdict = None
    p1 = None
    save = {}
    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        # ---------- feature percentiles ----------
        rows = []
        for ax in ('speed', 'size', 'moist'):
            lab = pole[ax]
            hi = lab == 1
            lo = lab == 0
            sig_cells = [(h, li) for h in range(NH)
                         for li in range(NL)
                         if p_maxT[AX_AI[ax]][h, li] <= FDR_Q]
            sig_set = set(sig_cells)
            for (h, li) in EVENTS[ax]:
                fv = feats_of(B[ax], h, li, hi, lo)
                fv['margin'] = float(sign_M[AX_AI[ax]][h, li])
                nonsig = [hh for hh in range(NH)
                          if (hh, li) not in sig_set]
                pct = {}
                for k, v in fv.items():
                    pool = [v]
                    for hh in nonsig:
                        pool.append(feats_of(
                            B[ax], hh, li, hi, lo)[k]
                            if k != 'margin'
                            else float(sign_M[AX_AI[ax]][hh, li]))
                    pool = np.array(pool)
                    pct[k] = float(np.mean(pool <= v))
                rows.append({'axis': ax, 'event': [h, li],
                             'feats': fv, 'pct': pct})
        # per-feature stats
        stats = {}
        for k in FEATS:
            pcts = np.array([r['pct'][k] for r in rows])
            n_gt = int(np.sum(pcts > 0.5))
            p = sign_test_p(n_gt, 13)
            stats[k] = {'pcts': [round(float(x), 4)
                                 for x in pcts],
                        'median': round(float(np.median(pcts)),
                                        4),
                        'n_gt_half': n_gt,
                        'sign_p': round(p, 6)}
        ps = [stats[k]['sign_p'] for k in FEATS]
        qs = bh_q(ps)
        for k, q in zip(FEATS, qs):
            stats[k]['bh_q'] = round(float(q), 6)
            stats[k]['bh_sig'] = bool(q <= FDR_Q)
        n_bh = sum(1 for k in FEATS if stats[k]['bh_sig'])
        med_phi = float(np.median(
            [r['pct']['phi'] for r in rows]))
        log('P1 per-feature: %s'
            % [(k, stats[k]['median'], stats[k]['n_gt_half'],
                stats[k]['sign_p'], stats[k]['bh_q'],
                stats[k]['bh_sig']) for k in FEATS], lines)
        log('P1 margin reference: median pct %.4f (trivially '
            'high - criterion itself)'
            % float(np.median([r['pct']['margin']
                               for r in rows])), lines)
        log('P1 n_BH_sig %d/7 | median pct(phi) %.4f'
            % (n_bh, med_phi), lines)
        detail = [(r['axis'], tuple(r['event']),
                   round(r['pct']['phi'], 3),
                   round(r['pct']['d_pole_abs'], 3),
                   round(r['pct']['neighbor_phi'], 3))
                  for r in rows]
        log('P1 detail (axis, event, phi, |d_pole|, nbr_phi): %s'
            % detail, lines)

        p1 = {'features': stats, 'n_bh_sig': n_bh,
              'median_pct_phi': round(med_phi, 4),
              'rows_pct': [{r['axis']: None,
                            'event': r['event'],
                            'pct': {k: round(v, 4) for k, v
                                    in r['pct'].items()}}
                           for r in rows]}

        # ---------- verdict ----------
        if n_bh >= 3 and med_phi >= 0.9:
            verdict = 'events_polar_separation_selected'
        elif n_bh >= 3:
            verdict = 'events_multifeature_selected'
        else:
            verdict = 'events_margin_only'

        save = {
            'pct_matrix': np.array(
                [[r['pct'][k] for k in
                  ['margin'] + FEATS] for r in rows],
                dtype=np.float64),
            'feat_keys': np.array(['margin'] + FEATS,
                                  dtype=object),
            'event_ids_speed': np.array(EVENTS['speed'],
                                        dtype=int),
            'event_ids_size': np.array(EVENTS['size'],
                                       dtype=int),
            'event_ids_moist': np.array(EVENTS['moist'],
                                        dtype=int),
        }

    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2925,
           'model': 'qwen3-4b (zero forward)',
           'prereg': PREREG,
           'anchors': {'a1': a1, 'a1_ok': a1_ok,
                       'a2_ok': a2_ok, 'a2_max_diff': a2_max,
                       'a3': a3, 'a3_ok': a3_ok,
                       'ok': anchor_ok},
           'P1': p1,
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if save:
        np.savez_compressed(
            os.path.join(OUT, 'event_selection_anatomy.npz'),
            **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2925 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
