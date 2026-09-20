# -*- coding: utf-8 -*-
"""Phase 2922: attribute event anatomy -- zero-forward artifact domain.

Why: 2921 flipped the 2920 verdict (attribute_events_found: speed 2,
size 9, moist 2 events under the same frozen sign-Gram maxT as the
24-event lang atlas). Open question (2921 candidate A): do the 13
new attribute events have the same anatomy as the lang events
(word-level decomposition, polarity alignment, sparsity, temporal
curves), and do they link within their axis (Spearman rho between
word-response vectors)?

Mode: zero forward. Artifact domain on the 2921 npz only
(B_speed (32,43,36), B_size (32,48,36), B_moist (32,35,36),
sign_M, p_maxT, pole labels, words) + 2917 npz for the P5
lang-vs-attr comparison.
Unit of analysis: word-response vector r_e = B_ax[h, :, l]
(N_ax-dim, one scalar per word; pole label per word).

Linkage statistic: Spearman rho between word-response vectors of
two same-axis events. Null: 2000 draws of layer-distance-matched
non-sig cell pairs (per-axis non-sig sets), maxT (Westfall-Young
single-step) family-wise correction over the merged 38-pair family
(speed 1 + size 36 + moist 1). Granularity pre-check BEFORE
observation: maxT p floor = 1/2001 = 0.0005 << q = 0.05, no BH
anywhere (2917/2918 lesson applied).

Polarity statistic (new, per event): sign agreement with the pole
partition. p_pole = (#{perm: max(|m_hi|,|m_lo|) >= obs} + 1)/2001
under 2000 size-preserving pole-label permutations (fresh
default_rng(2897), independent of the linkage null stream).
pole-aligned event: p_pole <= 0.05.

Events (frozen, from 2921 result P1 by margin desc):
  speed: (18,16),(14,12)
  size:  (21,7),(24,19),(12,18),(26,21),(25,15),(11,23),(11,27),
         (22,2),(18,7)
  moist: (8,15),(10,9)
lang 24 events recomputed from the 2917 npz (registry anchor).

Anchors (frozen):
  a1 per attr axis: sign_M recomputed from npz B_ax via the 2917
     formula (sign outer Gram, same/diff pole masks): median
     absdiff < 1e-6, max < 0.15, n_cells(>0.02) <= 5, all sig
     cells absdiff < 0.05.
  a2 registry: per-axis sig set size == 2/9/2 (lang 24 from the
     2917 npz); per-axis top1 cell == 2921 result P1 top15[0] with
     margin within 1e-4.
  a3 lang anchor (2918 a2 verbatim): 2917 sign_M argmax == (7,19),
     margin 1.34634 +/- 5e-3, p_maxT[7,19] <= 0.0051.

Probes (frozen):
  P1 word anatomy per attr event: n_pos/n_neg, pole alignment
     |mean sign| hi/lo + p_pole, d_pole (standardized hi-lo mean
     difference), top-5 pos/neg words, PR_word + top-3 mass vs
     same-layer non-sig null percentile, temporal (peak layer,
     adjacent contrast, temporal PR, n positive layers).
  P2 within-axis linkage: rho matrix per axis, p_pair (maxT over
     the merged 38-pair family), edges (p <= 0.05), size
     components (union-find), n_linked_size (verdict axis),
     n_linked_all.
  P3 layer distribution: attr event peak layers vs lang events.
  P4 head-channel notes: within-axis h11 pair via P2 p; same-head
     cross-category event pairs (lang vs attr heads 21/24/26/18)
     registered descriptively (rho not computable across vocab).
  P5 lang vs attr anatomy comparison (descriptive): same P1
     metrics on the 24 lang events from the 2917 npz.

Adjudication (frozen):
  anchor fail => anchor_fail_all_void;
  n_linked_size >= 1 AND pole_majority (>= 7/13 p_pole <= 0.05)
      => attr_events_linked_polar;
  n_linked_size >= 1 => attr_events_linked_polarity_mixed;
  n_linked_size == 0 AND pole_majority
      => attr_events_scattered_polar;
  else => attr_events_scattered_polarity_mixed.

Output: phase2922/attr_event_anatomy/.
"""
import hashlib
import json
import os
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC21 = os.path.join(BASE, 'phase2921', 'attr_vocab_expansion',
                     'attr_vocab_expansion.npz')
SRC21_RES = os.path.join(BASE, 'phase2921', 'attr_vocab_expansion',
                         'result.json')
SRC17 = os.path.join(BASE, 'phase2917', 'event_atlas',
                     'event_atlas.npz')
OUT = os.path.join(BASE, 'phase2922', 'attr_event_anatomy')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2922_run_report.txt')
N_NULL = 2000
FDR_Q = 0.05
NH, NL = 32, 36

PREREG = {
    'mode': 'zero-forward artifact-domain attribute event anatomy '
            'on the 2921 npz (no model load) + 2917 npz for the '
            'P5 lang comparison',
    'question': '2921 candidate A: do the 13 attribute events '
                '(speed 2 / size 9 / moist 2) have the same '
                'anatomy as the 24 lang events (word decomposition, '
                'pole alignment, sparsity, temporal curves), and do '
                'they link within their axis (Spearman rho between '
                'word-response vectors)?',
    'events': {'speed': [[18, 16], [14, 12]],
               'size': [[21, 7], [24, 19], [12, 18], [26, 21],
                        [25, 15], [11, 23], [11, 27], [22, 2],
                        [18, 7]],
               'moist': [[8, 15], [10, 9]]},
    'protocol': 'linkage stat = Spearman rho between N_ax-dim '
                'word-response vectors B_ax[h,:,l]; null = 2000 '
                'layer-distance-matched non-sig cell-pair draws '
                '(per-axis non-sig sets); maxT family-wise over '
                'the merged 38-pair family (speed 1 + size 36 + '
                'moist 1); polarity stat per event: size-'
                'preserving pole-label permutations x2000 (fresh '
                'default_rng(2897), independent stream), p_pole = '
                '(#{max(|m_hi|,|m_lo|) >= obs} + 1)/2001; '
                'granularity pre-check: maxT p floor 1/2001 = '
                '0.0005 << 0.05, no BH anywhere',
    'anchors': {
        'a1': 'per attr axis sign_M recomputed from npz B_ax '
              '(2917 formula, pole masks): median absdiff < 1e-6, '
              'max < 0.15, n_cells(>0.02) <= 5, all sig cells '
              '< 0.05',
        'a2': 'registry: per-axis sig sizes 2/9/2 (lang 24 from '
              '2917 npz); per-axis top1 cell == 2921 P1 top15[0], '
              'margin within 1e-4',
        'a3': '2917 sign_M argmax == (7,19), margin 1.34634 +/- '
              '5e-3, p_maxT[7,19] <= 0.0051',
    },
    'probes': {
        'P1': 'word anatomy per attr event: n_pos/n_neg, pole '
              'align hi/lo + p_pole, d_pole, top-5 words both '
              'sides, PR_word + top-3 mass vs same-layer non-sig '
              'null percentile, temporal metrics',
        'P2': 'within-axis linkage: rho + p_pair (38-pair merged '
              'maxT), edges, size components, n_linked_size '
              '(verdict axis), n_linked_all',
        'P3': 'layer distribution attr vs lang',
        'P4': 'head-channel notes: h11 within-axis pair; same-head '
              'cross-category pairs (descriptive)',
        'P5': 'lang vs attr anatomy comparison (descriptive)',
    },
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'n_linked_size>=1 AND pole_majority(>=7/13) => '
               'attr_events_linked_polar; n_linked_size>=1 => '
               'attr_events_linked_polarity_mixed; n_linked_size==0 '
               'AND pole_majority => attr_events_scattered_polar; '
               'else => attr_events_scattered_polarity_mixed',
}

EVENTS = {
    'speed': [(18, 16), (14, 12)],
    'size': [(21, 7), (24, 19), (12, 18), (26, 21), (25, 15),
             (11, 23), (11, 27), (22, 2), (18, 7)],
    'moist': [(8, 15), (10, 9)],
}
TOP1_EXPECT = {'speed': (18, 16, 1.01608),
               'size': (21, 7, 1.23834),
               'moist': (8, 15, 0.99258)}
AX_KEYS = {'speed': ('B_speed', 'sign_M', 1),
           'size': ('B_size', 'sign_M', 2),
           'moist': ('B_moist', 'sign_M', 3)}


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


def ranks_ord(M):
    order = np.argsort(M, axis=1)
    r = np.empty(M.shape, dtype=np.float64)
    rows = np.arange(M.shape[0])[:, None]
    r[rows, order] = np.arange(M.shape[1],
                               dtype=np.float64)[None, :] + 1.0
    return r


def rowcorr(Rc):
    num = Rc @ Rc.T
    den = np.sqrt(np.sum(Rc * Rc, axis=1))
    den = np.maximum(den[None, :] * den[:, None], 1e-30)
    return num / den


def spearman_pair(x, y):
    Ra = ranks_ord(x[None, :])[0]
    Rb = ranks_ord(y[None, :])[0]
    Ra = Ra - Ra.mean()
    Rb = Rb - Rb.mean()
    den = np.sqrt(Ra @ Ra * Rb @ Rb)
    return float(Ra @ Rb / max(den, 1e-30))


def part_ratio(x):
    a = np.abs(x)
    s1 = a.sum()
    s2 = float((a * a).sum())
    return float(s1 * s1 / max(s2, 1e-30))


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


def anatomy_p1(r, words, pole, lab_is_pole, sig_set, li, sign_row,
               rng_pole):
    """P1 metrics for one event (r: word-response vector)."""
    s = np.sign(r)
    s[s == 0] = 1.0
    order = np.argsort(-r)
    top_pos = [(words[n], round(float(r[n]), 4))
               for n in order[:5]]
    top_neg = [(words[n], round(float(r[n]), 4))
               for n in order[::-1][:5]]
    out = {'n_pos': int(np.sum(s > 0)),
           'n_neg': int(np.sum(s < 0)),
           'top_pos': top_pos, 'top_neg': top_neg}
    if lab_is_pole:
        hi = pole == 1
        lo = pole == 0
        m_hi = float(np.abs(s[hi].mean()))
        m_lo = float(np.abs(s[lo].mean()))
        obs = max(m_hi, m_lo)
        nh, nl = int(hi.sum()), int(lo.sum())
        n_perm = rng_pole.shape[0]
        cnt = 0
        for t in range(n_perm):
            p = rng_pole[t]
            sp = s[p]
            m1 = abs(sp[:nh].mean())
            m2 = abs(sp[nh:].mean())
            if max(m1, m2) >= obs:
                cnt += 1
        p_pole = (cnt + 1.0) / (n_perm + 1.0)
        v_hi = r[hi].astype(np.float64)
        v_lo = r[lo].astype(np.float64)
        pooled = np.sqrt(max(v_hi.var() + v_lo.var(), 1e-30))
        out['pole_align'] = {'hi': round(m_hi, 3),
                             'lo': round(m_lo, 3)}
        out['p_pole'] = round(p_pole, 6)
        out['d_pole'] = round(
            float((v_hi.mean() - v_lo.mean()) / pooled), 4)
    else:
        langs = sorted(set(pole.tolist()))
        align = {str(g): round(float(np.abs(
            s[pole == g].mean())), 3) for g in langs}
        out['lang_align'] = align
    a = np.abs(r)
    out['pr_word'] = round(part_ratio(r), 3)
    out['top3_mass'] = round(
        float(np.sort(a)[::-1][:3].sum() / max(a.sum(), 1e-30)), 4)
    out['temporal'] = temporal(sign_row)
    return out


def pr_pct(r, B, sig_set, li):
    nonsig = [h for h in range(NH) if (h, li) not in sig_set]
    prs = np.array([part_ratio(B[h, :, li]) for h in nonsig])
    return round(float(np.mean(prs <= part_ratio(r))), 4)


def temporal(c):
    pk = int(np.argmax(c))
    nb = [c[q] for q in (pk - 1, pk + 1) if 0 <= q < NL]
    contrast = float(c[pk] - np.mean(nb)) if nb else 0.0
    return {'peak_layer': pk,
            'contrast_adj': round(contrast, 5),
            'pr_time': round(part_ratio(c), 3),
            'n_pos_layers': int(np.sum(c > 0))}


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2922,
                   'name': 'attr_event_anatomy',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2921_npz': sha8(SRC21),
                               's2921_result': sha8(SRC21_RES),
                               's2917_npz': sha8(SRC17)},
                   'n_null_iter': N_NULL, 'fdr_q': FDR_Q,
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
    words = {ax: [str(w) for w in z['words_%s' % ax]]
             for ax in ('speed', 'size', 'moist')}
    with open(SRC21_RES, encoding='utf-8') as f:
        res21 = json.load(f)

    z17 = np.load(SRC17, allow_pickle=True)
    B17 = z17['B_heads'].astype(np.float64)
    sign17 = z17['sign_M'].astype(np.float64)
    p17 = z17['p_maxT'].astype(np.float64)
    lab17 = np.asarray(z17['labels_lang']).astype(int)
    words17_raw = [str(w) for w in z17['words']]

    # ---------- anchors ----------
    a1 = {}
    a1_ok = True
    sig_cells = {}
    for ax in ('speed', 'size', 'moist'):
        lab = pole[ax]
        sre = sign_recompute(B[ax], lab)
        ai = AX_KEYS[ax][2]
        D = np.abs(sre - sign_M[ai])
        sc = sorted([(h, li) for h in range(NH)
                     for li in range(NL)
                     if p_maxT[ai][h, li] <= FDR_Q])
        sig_cells[ax] = set(sc)
        sc_max = max(float(D[h, li]) for (h, li) in sc) if sc else 0
        ok = bool(np.median(D) < 1e-6 and D.max() < 0.15
                  and np.sum(D > 0.02) <= 5 and sc_max < 0.05)
        a1[ax] = {'median': float(np.median(D)),
                  'max': float(D.max()),
                  'n_gt_0p02': int(np.sum(D > 0.02)),
                  'sig_max': sc_max, 'ok': ok}
        a1_ok = a1_ok and ok
    sig17 = sorted([(h, li) for h in range(NH)
                    for li in range(NL) if p17[h, li] <= FDR_Q])
    reg_sizes = {'speed': 2, 'size': 9, 'moist': 2}
    a2 = {'sig_sizes': {ax: len(sig_cells[ax])
                        for ax in ('speed', 'size', 'moist')},
          'lang_sig': len(sig17)}
    a2_ok = bool(all(len(sig_cells[ax]) == reg_sizes[ax]
                     for ax in reg_sizes) and len(sig17) == 24)
    top1_detail = {}
    for ax in ('speed', 'size', 'moist'):
        h, li, m_exp = TOP1_EXPECT[ax]
        m_obs = float(sign_M[AX_KEYS[ax][2]][h, li])
        ok = abs(m_obs - m_exp) < 1e-4
        top1_detail[ax] = {'cell': [h, li],
                           'margin_obs': round(m_obs, 6),
                           'margin_exp': m_exp, 'ok': ok}
        a2_ok = a2_ok and ok
    hm, lm = np.unravel_index(np.argmax(sign17), sign17.shape)
    a3_ok = bool(int(hm) == 7 and int(lm) == 19
                 and abs(float(sign17[7, 19]) - 1.34634) < 5e-3
                 and float(p17[7, 19]) <= 0.0051)
    a3 = {'argmax': [int(hm), int(lm)],
          'margin_7_19': round(float(sign17[7, 19]), 6),
          'p_7_19': round(float(p17[7, 19]), 6), 'ok': a3_ok}
    anchor_ok = bool(a1_ok and a2_ok and a3_ok)
    log('a1 ok=%s %s' % (a1_ok,
                         {ax: (round(a1[ax]['median'], 12),
                               round(a1[ax]['max'], 5),
                               a1[ax]['n_gt_0p02'])
                          for ax in a1}), lines)
    log('a2 sig sizes %s lang %d top1 %s ok=%s'
        % (a2['sig_sizes'], len(sig17), top1_detail, a2_ok), lines)
    log('a3 %s ok=%s' % (a3, a3_ok), lines)

    verdict = None
    p1 = p2 = p3 = p4 = p5 = None
    save = {}
    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        # ---------- pole permutation matrix (rng2 stream) ----------
        rng2 = np.random.default_rng(2897)
        perm_mat = {}
        for ax in ('speed', 'size', 'moist'):
            n = len(pole[ax])
            nh = int((pole[ax] == 1).sum())
            pm = np.empty((N_NULL, n), dtype=np.int64)
            for t in range(N_NULL):
                pm[t] = rng2.permutation(n)
            perm_mat[ax] = pm

        # ---------- P1 attr anatomy ----------
        p1 = []
        for ax in ('speed', 'size', 'moist'):
            ai = AX_KEYS[ax][2]
            for (h, li) in EVENTS[ax]:
                r = B[ax][h, :, li]
                e = {'axis': ax, 'event': [h, li],
                     'margin': round(
                         float(sign_M[ai][h, li]), 5)}
                e.update(anatomy_p1(
                    r, words[ax], pole[ax], True, sig_cells[ax],
                    li, sign_M[ai][h, :], perm_mat[ax]))
                e['pr_pct_vs_layer_null'] = pr_pct(
                    r, B[ax], sig_cells[ax], li)
                p1.append(e)
        pole_major = [e for e in p1 if e['p_pole'] <= FDR_Q]
        log('P1 attr: %s'
            % [(e['axis'], tuple(e['event']), e['p_pole'],
                e['d_pole'], e['pr_word'], e['top3_mass'],
                e['temporal']['peak_layer'])
               for e in p1], lines)
        log('P1 pole-aligned (p<=0.05): %d/13 %s'
            % (len(pole_major),
               [(e['axis'], tuple(e['event'])) for e in pole_major]),
               lines)

        # ---------- P2 within-axis linkage ----------
        pairs_all = []
        for ax in ('speed', 'size', 'moist'):
            ev = EVENTS[ax]
            for x in range(len(ev)):
                for y in range(x + 1, len(ev)):
                    pairs_all.append((ax, ev[x], ev[y]))
        assert len(pairs_all) == 38
        rho_obs = []
        d_s = []
        for (ax, e1, e2) in pairs_all:
            rho_obs.append(spearman_pair(
                B[ax][e1[0], :, e1[1]], B[ax][e2[0], :, e2[1]]))
            d_s.append(abs(e1[1] - e2[1]))
        rho_obs = np.array(rho_obs)
        d_s = np.array(d_s)
        nonsig = {}
        for ax in ('speed', 'size', 'moist'):
            nonsig[ax] = {li: np.array(
                [h for h in range(NH) if (h, li)
                 not in sig_cells[ax]])
                for li in range(NL)}
        rng1 = np.random.default_rng(2896)
        valid_lp = {}
        for d in range(NL):
            if d == 0:
                lp = [(la, la) for la in range(NL)]
            else:
                lp = [(la, lb) for la in range(NL)
                      for lb in (la - d, la + d)
                      if 0 <= lb < NL]
            valid_lp[d] = np.array(lp)
        null_all = np.zeros((N_NULL, 38), dtype=np.float32)
        for t in range(N_NULL):
            for s_i, (ax, e1, e2) in enumerate(pairs_all):
                d = int(d_s[s_i])
                lp = valid_lp[d]
                la, lb = lp[rng1.integers(len(lp))]
                la, lb = int(la), int(lb)
                ha = nonsig[ax][la][
                    rng1.integers(len(nonsig[ax][la]))]
                cand = nonsig[ax][lb]
                if d == 0:
                    cand = cand[cand != ha]
                hb = cand[rng1.integers(len(cand))]
                null_all[t, s_i] = spearman_pair(
                    B[ax][ha, :, la], B[ax][hb, :, lb])
        max_null = null_all.max(axis=1)

        def p_maxT_(rho_arr):
            cnt = np.sum(max_null[None, :] >= rho_arr[:, None],
                         axis=1)
            return (cnt + 1.0) / (N_NULL + 1.0)

        p_pair = p_maxT_(rho_obs)
        idx_map = {ax: {e: k for k, e
                        in enumerate(EVENTS[ax])}
                   for ax in ('speed', 'size', 'moist')}
        p_table = []
        for s_i, (ax, e1, e2) in enumerate(pairs_all):
            p_table.append({'axis': ax, 'pair': [list(e1),
                                                 list(e2)],
                            'rho': round(float(rho_obs[s_i]), 5),
                            'p_pair': round(float(p_pair[s_i]),
                                            6)})
        n_linked_all = int(np.sum(p_pair <= FDR_Q))
        size_mask = np.array([ax == 'size'
                              for (ax, _, _) in pairs_all])
        n_linked_size = int(np.sum(p_pair[size_mask] <= FDR_Q))
        edges = [p_table[s_i] for s_i in range(38)
                 if p_pair[s_i] <= FDR_Q]
        # size components (union-find)
        par = list(range(9))

        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x

        for s_i, (ax, e1, e2) in enumerate(pairs_all):
            if ax == 'size' and p_pair[s_i] <= FDR_Q:
                ra, rb = find(idx_map['size'][e1]), \
                    find(idx_map['size'][e2])
                if ra != rb:
                    par[ra] = rb
        comps = {}
        for k_i, e in enumerate(EVENTS['size']):
            comps.setdefault(find(k_i), []).append(e)
        comp_list = sorted([sorted(c) for c in comps.values()],
                           key=lambda c: (-len(c), c[0]))
        p2 = {'pairs': p_table, 'n_linked_all': n_linked_all,
              'n_linked_size': n_linked_size,
              'edges': edges,
              'size_components': [[list(e) for e in c]
                                  for c in comp_list]}
        log('P2: n_linked_all %d/38, n_linked_size %d/36 | '
            'edges %s | size comps %s'
            % (n_linked_all, n_linked_size,
               [(tuple(e['pair'][0]), tuple(e['pair'][1]),
                 e['axis'], e['rho'], e['p_pair'])
                for e in edges], comp_list), lines)

        # ---------- P3 layer distribution ----------
        attr_peaks = [e['temporal']['peak_layer'] for e in p1]
        lang_p1 = []
        for (h, li) in sig17:
            r = B17[h, :, li]
            e = {'axis': 'lang', 'event': [h, li],
                 'margin': round(float(sign17[h, li]), 5)}
            e.update(anatomy_p1(r, words17_raw, lab17, False,
                                set(sig17), li, sign17[h, :],
                                None))
            e['pr_pct_vs_layer_null'] = pr_pct(
                r, B17, set(sig17), li)
            lang_p1.append(e)
        lang_peaks = [e['temporal']['peak_layer']
                      for e in lang_p1]
        p3 = {'attr_peak_layers': attr_peaks,
              'lang_peak_layers': lang_peaks,
              'attr_median_peak': float(np.median(attr_peaks)),
              'lang_median_peak': float(np.median(lang_peaks))}
        log('P3: attr median peak %.1f %s | lang median peak %.1f'
            % (p3['attr_median_peak'], attr_peaks,
               p3['lang_median_peak']), lines)

        # ---------- P4 head-channel notes ----------
        same_head = {}
        for e in p1 + lang_p1:
            same_head.setdefault(e['event'][0], []).append(
                (e['axis'], tuple(e['event']), e['margin']))
        cross_cat = [
            {'head': h, 'members': v}
            for h, v in sorted(same_head.items())
            if len(set(ax for (ax, _, _) in v)) >= 2]
        p4 = {'same_head_cross_category': cross_cat,
              'note': 'rho not computable across vocabularies; '
                      'h11 (11,23)-(11,27) within-axis pair '
                      'adjudicated by P2'}
        log('P4 cross-cat heads: %s'
            % [(c['head'],
                [(m[0], m[1]) for m in c['members']])
               for c in cross_cat], lines)

        # ---------- P5 comparison ----------
        def med(vals):
            return round(float(np.median(vals)), 3)

        p5 = {
            'lang': {'n': len(lang_p1),
                     'pr_word_med': med([e['pr_word']
                                         for e in lang_p1]),
                     'top3_mass_med': med([e['top3_mass']
                                           for e in lang_p1]),
                     'pr_pct_med': med([e['pr_pct_vs_layer_null']
                                        for e in lang_p1]),
                     'peak_med': med(lang_peaks),
                     'pr_time_med': med([e['temporal']['pr_time']
                                         for e in lang_p1]),
                     'n_pos_layers_med': med(
                         [e['temporal']['n_pos_layers']
                          for e in lang_p1]),
                     'align_med': med([max(e['lang_align']
                                           .values())
                                       for e in lang_p1])},
            'attr': {'n': len(p1),
                     'pr_word_med': med([e['pr_word']
                                         for e in p1]),
                     'top3_mass_med': med([e['top3_mass']
                                           for e in p1]),
                     'pr_pct_med': med([e['pr_pct_vs_layer_null']
                                        for e in p1]),
                     'peak_med': med(attr_peaks),
                     'pr_time_med': med([e['temporal']['pr_time']
                                         for e in p1]),
                     'n_pos_layers_med': med(
                         [e['temporal']['n_pos_layers']
                          for e in p1]),
                     'align_med': med([max(e['pole_align']
                                           .values())
                                       for e in p1]),
                     'pole_aligned': len(pole_major),
                     'd_pole_med': med([abs(e['d_pole'])
                                        for e in p1])}}
        log('P5 lang vs attr: %s' % p5, lines)

        # ---------- verdict ----------
        pole_majority = len(pole_major) >= 7
        if n_linked_size >= 1 and pole_majority:
            verdict = 'attr_events_linked_polar'
        elif n_linked_size >= 1:
            verdict = 'attr_events_linked_polarity_mixed'
        elif pole_majority:
            verdict = 'attr_events_scattered_polar'
        else:
            verdict = 'attr_events_scattered_polarity_mixed'

        save = {
            'word_r_speed': np.stack(
                [B['speed'][h, :, li]
                 for (h, li) in EVENTS['speed']]).astype(np.float32),
            'word_r_size': np.stack(
                [B['size'][h, :, li]
                 for (h, li) in EVENTS['size']]).astype(np.float32),
            'word_r_moist': np.stack(
                [B['moist'][h, :, li]
                 for (h, li) in EVENTS['moist']]).astype(np.float32),
            'event_ids_speed': np.array(EVENTS['speed'], dtype=int),
            'event_ids_size': np.array(EVENTS['size'], dtype=int),
            'event_ids_moist': np.array(EVENTS['moist'], dtype=int),
            'rho_obs38': rho_obs.astype(np.float32),
            'p_pair38': p_pair.astype(np.float32),
            'null_all38': null_all,
            'd_s38': d_s.astype(np.int64),
            'pole_p13': np.array([e['p_pole'] for e in p1],
                                 dtype=np.float64),
            'lang_event_ids': np.array(sig17, dtype=int),
            'sign_M_ref': sign_M.astype(np.float32),
            'sign17_ref': sign17.astype(np.float32),
        }

    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2922,
           'model': 'qwen3-4b (zero forward)',
           'prereg': PREREG,
           'anchors': {'a1': a1, 'a1_ok': a1_ok,
                       'a2': {**a2, 'top1': top1_detail},
                       'a2_ok': a2_ok, 'a3': a3, 'a3_ok': a3_ok,
                       'ok': anchor_ok},
           'P1': p1, 'P2': p2, 'P3': p3, 'P4': p4, 'P5': p5,
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if save:
        np.savez_compressed(
            os.path.join(OUT, 'attr_event_anatomy.npz'), **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2922 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
