# -*- coding: utf-8 -*-
"""Phase 2918: (head,layer) event anatomy -- zero-forward artifact domain.

Why: 2917 established the full 32x36 sign-margin atlas under maxT
family-wise correction: 24 significant events, known 2/6 survive
((7,19) top-of-atlas, (27,24)), the other 4 known are window-family-
relative only, and 22 NOVEL events concentrate at early layers
(L1-L16 hold 16/24). Open question (2917 candidate A): are the
novel early-layer events the EARLY-LAYER FORM of the same mechanism
as the late known events, or a distinct mechanism family?

Mode: zero forward. Artifact domain on the 2917 npz only
(B_heads (32,57,36) fp32, sign_M, p_maxT, words, labels_lang).
Unit of analysis: word-response vector r_e = B_heads[h, :, l]
(57-dim, one scalar per word).

Linkage statistic: Spearman rho between word-response vectors of
two events. Null: 2000 draws of layer-distance-matched non-sig
cell pairs (p_maxT > 0.05), maxT (Westfall-Young single-step)
family-wise correction over TWO frozen families:
  (i) the 276-pair family of all 24 sig events (for the linkage
      graph),
  (ii) the 10-slot family of focus x known pairs (for the verdict
      p_link; slot statistic is the pair rho, verdict uses the
      max over each focus event's 2 known refs).
Granularity check BEFORE observation (2917 lesson applied):
maxT p floor = 1/2001 = 0.0005 << q = 0.05 -> no BH anywhere.

Focus events (frozen): first 5 novel events of the 2917 sig list
sorted by margin desc = (26,6), (8,2), (25,3), (22,12), (24,23).
Known refs (frozen): (7,19), (27,24) -- the two full-family
survivors. Early novel def (frozen): novel sig events, layer <= 16.

Anchors (frozen):
  a1 sign_M recomputed from npz B_heads via the 2917 formula
     (sign outer Gram, same/diff lang masks): median absdiff
     < 1e-6, max absdiff < 0.15 (fp32-storage sign flips of
     near-zero responses may move isolated cells by <= ~0.11),
     cells with absdiff > 0.02 at most 5, and all 24 sig cells
     absdiff < 0.05.
  a2 (7,19) is the argmax cell of the npz sign_M; recomputed
     margin within 5e-3 of 1.34634; npz p_maxT[7,19] <= 0.0051.
  registry: npz-derived sig set == 24 == 2917 result P1 n_sig.

Probes (frozen):
  P1 word anatomy per focus/known event: n_pos/n_neg, top-5
     positive/negative words, per-language sign alignment
     |mean sign|, word-level participation ratio PR_word and
     top-3 mass fraction vs same-layer non-sig null percentile
     (input-side gating density).
  P2 linkage: focus x known rho + p_link (maxT over 10-slot
     family), rho_early (max rho to early-novel others),
     n_linked (verdict axis), internal coherence, 276-pair
     linkage graph edges (p_pair <= 0.05) with early/late
     composition.
  P3 temporal curves: for focus/known/multi-event heads: peak
     layer, adjacent contrast, temporal PR over |sign_M row|,
     n positive layers (layer-side gating density).
  P4 head-channel reuse: within-head sig-event pairs (h26, h24,
     h21, h4, h1) + descriptive h7 (19,34): rho vs layer-distance
     matched null95.
  P5 descriptive.

Adjudication (frozen):
  anchor/registry fail => anchor_fail_all_void;
  n_linked >= 3 => early_events_same_mechanism;
  n_linked == 0 AND mean(rho_early) > mean(rho_known)
      => early_events_own_family;
  n_linked == 0 => early_events_unlinked_scattered;
  else => early_events_mixed_structure.

Output: phase2918/event_anatomy/.
"""
import hashlib
import json
import os
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC = os.path.join(BASE, 'phase2917', 'event_atlas',
                   'event_atlas.npz')
SRC_RES = os.path.join(BASE, 'phase2917', 'event_atlas',
                       'result.json')
OUT = os.path.join(BASE, 'phase2918', 'event_anatomy')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2918_run_report.txt')
N_NULL = 2000
FDR_Q = 0.05
EARLY_MAX = 16
N_FOCUS = 5
KNOWN_REF = [(7, 19), (27, 24)]
NH, NW, NL = 32, 57, 36

PREREG = {
    'mode': 'zero-forward artifact-domain event anatomy on the '
            '2917 atlas npz (no model load)',
    'question': '2917 candidate A: are the novel early-layer '
                'events the early-layer form of the same mechanism '
                'as the late known events, or a distinct mechanism '
                'family; plus event anatomy (word decomposition, '
                'sparsity, temporal curves, head reuse) as the '
                'empirical basis for density-gating '
                'operationalization',
    'focus': 'first 5 novel events of the 2917 sig list by margin '
             'desc: (26,6),(8,2),(25,3),(22,12),(24,23)',
    'known_refs': KNOWN_REF,
    'early_novel_def': 'novel sig events with layer <= 16',
    'protocol': 'linkage stat = Spearman rho between 57-dim '
                'word-response vectors B_heads[h,:,l]; null = 2000 '
                'layer-distance-matched non-sig cell-pair draws; '
                'maxT family-wise over the 276-pair family '
                '(graph) and the 10-slot focus x known family '
                '(verdict p_link); same-layer (d=0) event pairs '
                'exist (e.g. (26,6)/(21,6) at L6) and are included '
                'in family+null with ha != hb; granularity '
                'pre-check: maxT p floor 1/2001 = 0.0005 << 0.05, '
                'no BH anywhere (2917 lesson applied before '
                'observation); ties: ordinal ranks, tie audit '
                'reported',
    'anchors': {
        'a1': 'sign_M recomputed from npz B_heads (2917 formula): '
              'median absdiff < 1e-6, max < 0.15, n_cells(>0.02) '
              '<= 5, all 24 sig cells < 0.05',
        'a2': '(7,19) argmax of npz sign_M; recomputed margin '
              'within 5e-3 of 1.34634; p_maxT[7,19] <= 0.0051',
        'registry': 'npz sig set == 24 == 2917 P1 n_sig',
    },
    'probes': {
        'P1': 'word anatomy: n_pos/n_neg, top-5 words both sides, '
              'per-language sign alignment, PR_word + top-3 mass '
              'vs same-layer non-sig null percentile',
        'P2': 'linkage table + n_linked + 276-pair graph '
              '(maxT q=0.05) with early/late composition',
        'P3': 'temporal curves: peak, adjacent contrast, temporal '
              'PR, n positive layers',
        'P4': 'head-channel reuse: within-head sig pairs + h7 '
              '(19,34) vs layer-distance-matched null95',
        'P5': 'descriptive',
    },
    'verdict': 'anchor/registry fail => anchor_fail_all_void; '
               'n_linked>=3 => early_events_same_mechanism; '
               'n_linked==0 AND mean(rho_early)>mean(rho_known) => '
               'early_events_own_family; n_linked==0 => '
               'early_events_unlinked_scattered; else => '
               'early_events_mixed_structure',
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


def spearman_rows(A, Bm_):
    Ra = ranks_ord(A)
    Rb = ranks_ord(Bm_)
    Ra = Ra - Ra.mean(axis=1, keepdims=True)
    Rb = Rb - Rb.mean(axis=1, keepdims=True)
    num = np.sum(Ra * Rb, axis=1)
    den = np.sqrt(np.sum(Ra * Ra, axis=1)
                  * np.sum(Rb * Rb, axis=1))
    return num / np.maximum(den, 1e-30)


def part_ratio(x):
    a = np.abs(x)
    s1 = a.sum()
    s2 = float((a * a).sum())
    return float(s1 * s1 / max(s2, 1e-30))


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2918,
                   'name': 'event_anatomy',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2917_npz': sha8(SRC),
                               's2917_result': sha8(SRC_RES)},
                   'n_null_iter': N_NULL, 'fdr_q': FDR_Q,
                   'early_max_layer': EARLY_MAX,
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    z = np.load(SRC, allow_pickle=True)
    Bm = z['B_heads'].astype(np.float64)      # (32,57,36)
    sign_M = z['sign_M'].astype(np.float64)   # (32,36)
    p_maxT = z['p_maxT'].astype(np.float64)
    lab_lang = np.asarray(z['labels_lang']).astype(int)
    words_raw = [str(w) for w in z['words']]
    words = [w.split(':') for w in words_raw]  # lang, ck, word
    assert Bm.shape == (NH, NW, NL)
    assert len(words) == NW

    # ---------- anchors ----------
    same_m, diff_m = masks(lab_lang)
    sign_re = np.zeros((NH, NL))
    for h in range(NH):
        for li in range(NL):
            s = np.sign(Bm[h, :, li])
            s[s == 0] = 1.0
            Gm = np.outer(s, s)
            sign_re[h, li] = Gm[same_m].mean() \
                - Gm[diff_m].mean()
    D = np.abs(sign_re - sign_M)
    sig24 = sorted([(h, li) for h in range(NH) for li in range(NL)
                    if p_maxT[h, li] <= FDR_Q],
                   key=lambda e: -sign_M[e[0], e[1]])
    sig_set = set(sig24)
    sig24_max = max(float(D[h, li]) for (h, li) in sig24)
    a1 = {'median_diff': float(np.median(D)),
          'max_diff': float(D.max()),
          'n_cells_gt_0p02': int(np.sum(D > 0.02)),
          'sig24_max_diff': sig24_max}
    a1_ok = bool(a1['median_diff'] < 1e-6 and a1['max_diff'] < 0.15
                 and a1['n_cells_gt_0p02'] <= 5
                 and sig24_max < 0.05)
    hm, lm = np.unravel_index(np.argmax(sign_M), sign_M.shape)
    a2 = {'argmax_cell': [int(hm), int(lm)],
          'margin_re_7_19': round(float(sign_re[7, 19]), 6),
          'p_maxT_7_19': round(float(p_maxT[7, 19]), 6)}
    a2_ok = bool(int(hm) == 7 and int(lm) == 19
                 and abs(float(sign_re[7, 19]) - 1.34634) < 5e-3
                 and float(p_maxT[7, 19]) <= 0.0051)
    reg_ok = bool(len(sig24) == 24)
    anchor_ok = bool(a1_ok and a2_ok and reg_ok)
    log('a1 med %.2e max %.4f n>0.02 %d sig24max %.2e ok=%s'
        % (a1['median_diff'], a1['max_diff'],
           a1['n_cells_gt_0p02'], sig24_max, a1_ok), lines)
    log('a2 argmax %s margin_re %.6f p %.4f ok=%s | registry24 %s'
        % (a2['argmax_cell'], a2['margin_re_7_19'],
           a2['p_maxT_7_19'], a2_ok, reg_ok), lines)

    verdict = None
    p1 = p2 = p3 = p4 = p5 = None
    save = {}
    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        # ---------- focus / refs ----------
        novel = [(h, li) for (h, li) in sig24
                 if (h, li) not in KNOWN_REF
                 and not any(h == hk and abs(li - lk) <= 1
                             for (hk, lk) in KNOWN_REF)]
        focus = novel[:N_FOCUS]
        early_novel = [e for e in novel if e[1] <= EARLY_MAX]
        log('focus %s | early_novel %d/%d'
            % (focus, len(early_novel), len(novel)), lines)

        # tie audit (ordinal ranks validity)
        tie_max = 0
        for (h, li) in sig24:
            v = Bm[h, :, li]
            _, cnt = np.unique(v, return_counts=True)
            tie_max = max(tie_max, int(cnt.max()))
        log('tie audit: max multiplicity %d' % tie_max, lines)

        # ---------- pairwise rho among 24 sig events ----------
        k = len(sig24)
        V = np.stack([Bm[h, :, li] for (h, li) in sig24])
        Rk = ranks_ord(V)
        Rk = Rk - Rk.mean(axis=1, keepdims=True)
        C = rowcorr(Rk)
        pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
        d_s = np.array([abs(sig24[i][1] - sig24[j][1])
                        for (i, j) in pairs])
        rho_pairs = np.array([C[i, j] for (i, j) in pairs])
        np_pairs = len(pairs)          # 276
        fk_slots = [s for s, (i, j) in enumerate(pairs)
                    if sig24[i] in focus and sig24[j] in KNOWN_REF
                    or sig24[j] in focus
                    and sig24[i] in KNOWN_REF]
        fk_slots = [s for s, (i, j) in enumerate(pairs)
                    if (sig24[i] in focus
                        and sig24[j] in KNOWN_REF)
                    or (sig24[j] in focus
                        and sig24[i] in KNOWN_REF)]
        assert len(fk_slots) == N_FOCUS * len(KNOWN_REF)

        # ---------- null: layer-distance-matched non-sig pairs ----------
        nonsig = {li: np.array([h for h in range(NH)
                                if (h, li) not in sig_set])
                  for li in range(NL)}
        valid_lp = {}
        for d in range(0, NL):
            if d == 0:
                lp = [(la, la) for la in range(NL)]
            else:
                lp = [(la, lb) for la in range(NL)
                      for lb in (la - d, la + d)
                      if 0 <= lb < NL]
            if lp:
                valid_lp[d] = np.array(lp)
        assert all(int(d) in valid_lp for d in np.unique(d_s))
        rng = np.random.default_rng(2896)
        null_all = np.zeros((N_NULL, np_pairs), dtype=np.float32)
        for t in range(N_NULL):
            la = np.empty(np_pairs, dtype=int)
            lb = np.empty(np_pairs, dtype=int)
            ha = np.empty(np_pairs, dtype=int)
            hb = np.empty(np_pairs, dtype=int)
            for s in range(np_pairs):
                d = int(d_s[s])
                lp = valid_lp[d]
                i0 = rng.integers(len(lp))
                la[s], lb[s] = lp[i0]
                ha[s] = nonsig[la[s]][
                    rng.integers(len(nonsig[la[s]]))]
                cand = nonsig[lb[s]]
                if d == 0:
                    cand = cand[cand != ha[s]]
                hb[s] = cand[rng.integers(len(cand))]
            B1 = Bm[ha, :, la]
            B2 = Bm[hb, :, lb]
            null_all[t] = spearman_rows(B1, B2)
        max_null276 = null_all.max(axis=1)
        max_null10 = null_all[:, fk_slots].max(axis=1)
        m276 = np.sort(max_null276)
        m10 = np.sort(max_null10)

        def p_maxT_(rho_arr, sorted_max):
            idx = np.searchsorted(sorted_max, rho_arr, side='left')
            cnt_ge = len(sorted_max) - idx
            return (cnt_ge + 1.0) / (len(sorted_max) + 1.0)

        p_pair = np.ones((k, k))
        pp = p_maxT_(rho_pairs, m276)
        for s, (i, j) in enumerate(pairs):
            p_pair[i, j] = p_pair[j, i] = pp[s]

        # ---------- P2 linkage / verdict ----------
        idx_of = {e: i for i, e in enumerate(sig24)}
        focus_table = []
        n_linked = 0
        rho_early_list = []
        rho_known_list = []
        for e in focus:
            i = idx_of[e]
            rk = {}
            for kk in KNOWN_REF:
                j = idx_of[kk]
                rk[str(kk)] = round(float(C[i, j]), 5)
            rho_best = max(float(C[i, idx_of[kk]])
                           for kk in KNOWN_REF)
            sl = [s for s in fk_slots
                  if sig24[pairs[s][0]] == e
                  or sig24[pairs[s][1]] == e]
            rho_best2 = max(rho_pairs[s] for s in sl)
            p_link = float((np.sum(max_null10 >= rho_best2) + 1)
                           / (N_NULL + 1))
            others = [idx_of[x] for x in early_novel if x != e]
            rho_e = max((float(C[i, j]) for j in others),
                        default=0.0)
            n_linked += int(p_link <= FDR_Q)
            rho_known_list.append(rho_best)
            rho_early_list.append(rho_e)
            focus_table.append({
                'event': list(e), 'margin':
                    round(float(sign_M[e[0], e[1]]), 5),
                'rho_known': rk, 'rho_best_known':
                    round(rho_best, 5), 'p_link':
                    round(p_link, 6), 'rho_early':
                    round(rho_e, 5), 'linked':
                    bool(p_link <= FDR_Q)})
        rho_known_mean = float(np.mean(rho_known_list))
        rho_early_mean = float(np.mean(rho_early_list))
        # graph
        edges = [(sig24[i], sig24[j], float(C[i, j]),
                  float(p_pair[i, j]))
                 for i in range(k) for j in range(i + 1, k)
                 if p_pair[i, j] <= FDR_Q]
        # union-find components
        par = list(range(k))

        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x

        for (a, b, _, _) in edges:
            ra, rb = find(idx_of[a]), find(idx_of[b])
            if ra != rb:
                par[ra] = rb
        comps = {}
        for i, e in enumerate(sig24):
            comps.setdefault(find(i), []).append(e)
        comp_list = sorted(
            [sorted(c) for c in comps.values()],
            key=lambda c: (-len(c), c[0]))
        ee = sum(1 for (a, b, _, _) in edges
                 if a[1] <= EARLY_MAX and b[1] <= EARLY_MAX)
        ll = sum(1 for (a, b, _, _) in edges
                 if a[1] > EARLY_MAX and b[1] > EARLY_MAX)
        el = len(edges) - ee - ll
        p2 = {'focus_table': focus_table,
              'n_linked': n_linked,
              'rho_known_mean': round(rho_known_mean, 5),
              'rho_early_mean': round(rho_early_mean, 5),
              'tie_max': tie_max,
              'graph': {'n_edges': len(edges),
                        'n_early_early': ee, 'n_early_late': el,
                        'n_late_late': ll,
                        'components': [[list(e) for e in c]
                                       for c in comp_list],
                        'edges': [[list(a), list(b),
                                   round(r, 5), round(pp, 6)]
                                  for (a, b, r, pp) in edges]}}
        log('P2: n_linked %d/%d | rho_known_mean %.4f '
            'rho_early_mean %.4f | graph edges %d (ee %d el %d '
            'll %d) comps %d'
            % (n_linked, N_FOCUS, rho_known_mean, rho_early_mean,
               len(edges), ee, el, ll, len(comp_list)), lines)
        log('P2 focus: %s'
            % [(t['event'], t['rho_best_known'], t['p_link'],
                t['rho_early'], t['linked'])
               for t in focus_table], lines)

        # ---------- P1 word anatomy ----------
        nonsig_by_layer = {li: [h for h in range(NH)
                                if (h, li) not in sig_set]
                           for li in range(NL)}

        def pr_word_all_cells():
            out = {}
            for li in range(NL):
                vals = [part_ratio(Bm[h, :, li])
                        for h in nonsig_by_layer[li]]
                out[li] = np.array(vals)
            return out

        pr_null = pr_word_all_cells()
        p1 = []
        for e in focus + KNOWN_REF:
            h, li = e
            r = Bm[h, :, li]
            s = np.sign(r)
            s[s == 0] = 1.0
            order = np.argsort(-r)
            top_pos = [(words[n][0], words[n][2],
                        round(float(r[n]), 4))
                       for n in order[:5]]
            top_neg = [(words[n][0], words[n][2],
                        round(float(r[n]), 4))
                       for n in order[::-1][:5]]
            langs = sorted(set(lab_lang.tolist()))
            align = {}
            for g in langs:
                m = lab_lang == g
                align[str(g)] = round(
                    float(np.abs(s[m].mean())), 3)
            prw = part_ratio(r)
            a = np.abs(r)
            top3 = float(np.sort(a)[::-1][:3].sum()
                         / max(a.sum(), 1e-30))
            pct = float(np.mean(pr_null[li] <= prw))
            c = sign_M[h, :]
            pk = int(np.argmax(c))
            nb = [c[q] for q in (pk - 1, pk + 1) if 0 <= q < NL]
            contrast = float(c[pk] - np.mean(nb)) if nb else 0.0
            prt = part_ratio(c)
            p1.append({
                'event': list(e),
                'margin': round(float(sign_M[h, li]), 5),
                'n_pos': int(np.sum(s > 0)),
                'n_neg': int(np.sum(s < 0)),
                'top_pos': top_pos, 'top_neg': top_neg,
                'lang_align': align,
                'pr_word': round(prw, 3),
                'top3_mass': round(top3, 4),
                'pr_pct_vs_layer_null': round(pct, 4),
                'temporal': {'peak_layer': pk,
                             'contrast_adj': round(contrast, 5),
                             'pr_time': round(prt, 3),
                             'n_pos_layers': int(np.sum(c > 0))}})
        log('P1 anatomy: %s'
            % [(e['event'], e['pr_word'], e['top3_mass'],
                e['pr_pct_vs_layer_null'],
                e['temporal']['peak_layer'],
                e['temporal']['pr_time'])
               for e in p1], lines)

        # ---------- P4 head-channel reuse ----------
        by_head = {}
        for e in sig24:
            by_head.setdefault(e[0], []).append(e[1])
        reuse = []
        d_pool = {}
        for s in range(np_pairs):
            d_pool.setdefault(int(d_s[s]), []).append(
                null_all[:, s])
        for h, ls in sorted(by_head.items()):
            if len(ls) < 2:
                continue
            for x in range(len(ls)):
                for y in range(x + 1, len(ls)):
                    l1, l2 = ls[x], ls[y]
                    d = abs(l1 - l2)
                    r = float(spearman_rows(
                        Bm[h, :, l1][None, :],
                        Bm[h, :, l2][None, :])[0])
                    n95 = float(np.percentile(
                        np.concatenate(d_pool[d]), 95)) \
                        if d in d_pool else None
                    reuse.append({
                        'head': h, 'pair': [l1, l2],
                        'rho': round(r, 5), 'd': d,
                        'null95_d': round(n95, 5)
                        if n95 is not None else None,
                        'above_null95':
                            bool(n95 is not None and r > n95)})
        r7 = float(spearman_rows(Bm[7, :, 19][None, :],
                                 Bm[7, :, 34][None, :])[0])
        n95_15 = float(np.percentile(
            np.concatenate(d_pool[15]), 95))
        reuse.append({'head': 7, 'pair': [19, 34],
                      'rho': round(r7, 5), 'd': 15,
                      'null95_d': round(n95_15, 5),
                      'above_null95': bool(r7 > n95_15),
                      'note': 'descriptive, (7,34) not sig'})
        p4 = {'multi_event_heads':
              {str(h): ls for h, ls in sorted(by_head.items())
               if len(ls) >= 2},
              'reuse_table': reuse}
        log('P4 reuse: %s'
            % [(t['head'], t['pair'], t['rho'],
                t['null95_d'], t['above_null95'])
               for t in reuse], lines)

        # ---------- P3 curves (saved npz) ----------
        curve_heads = sorted(set([e[0] for e in focus]
                                 + [e[0] for e in KNOWN_REF]
                                 + [7, 4, 26, 24, 21, 1]))
        p3 = {'curve_heads': curve_heads,
              'curves': {str(h): [round(float(x), 5)
                                  for x in sign_M[h, :]]
                         for h in curve_heads}}
        p5 = {'note': 'curves in result P3; full matrices in npz'}

        # ---------- verdict ----------
        if n_linked >= 3:
            verdict = 'early_events_same_mechanism'
        elif n_linked == 0 and rho_early_mean > rho_known_mean:
            verdict = 'early_events_own_family'
        elif n_linked == 0:
            verdict = 'early_events_unlinked_scattered'
        else:
            verdict = 'early_events_mixed_structure'
        save = {'word_r': V.astype(np.float32),
                'event_ids': np.array(sig24, dtype=int),
                'rho24': C.astype(np.float32),
                'p_pair24': p_pair.astype(np.float32),
                'null_all': null_all,
                'max_null276': max_null276.astype(np.float32),
                'max_null10': max_null10.astype(np.float32),
                'focus_ids': np.array(focus, dtype=int),
                'known_ids': np.array(KNOWN_REF, dtype=int),
                'early_novel_ids':
                    np.array(early_novel, dtype=int),
                'curve_heads': np.array(curve_heads, dtype=int),
                'sign_M_ref': sign_M.astype(np.float32),
                'labels_lang': lab_lang,
                'words': np.array(words_raw, dtype=object)}

    log('==== VERDICT: %s ====' % verdict, lines)

    res = {'phase': 2918, 'model': 'qwen3-4b (zero forward)',
           'prereg': PREREG,
           'anchors': {'a1': a1, 'a1_ok': a1_ok, 'a2': a2,
                       'a2_ok': a2_ok, 'registry_ok': reg_ok,
                       'ok': anchor_ok},
           'P1': p1, 'P2': p2, 'P3': p3, 'P4': p4, 'P5': p5,
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if save:
        np.savez_compressed(
            os.path.join(OUT, 'event_anatomy.npz'), **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2918 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
