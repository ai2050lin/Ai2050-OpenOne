# -*- coding: utf-8 -*-
"""Phase 2928: survivor core anatomy -- why do 7 events survive
the probe change?

Why: 2927 showed the lang event atlas is a probe x circuitry
interaction product (overlap 7/24, jaccard 0.143) with a
7-event probe-invariant hard core: (1,6) (5,6) (7,19) (8,2)
(14,9) (20,8) (21,6), anchored by (7,19) (layer-internal rank
#1 under BOTH probes). Open questions (2927 candidate A):
(a) is the 7/24 overlap above chance?  (b) what feature
distinguishes survivors from lost events - the natural
candidate is DUAL-PROBE STRENGTH (high margin percentile in
BOTH calibers); (c) is the word-response STRUCTURE of survivor
events probe-invariant?

Mode: zero forward, artifact domain on the 2927 npz (B86,
B_word, sign_M86, sign_M_word, p_maxT86, p_maxT_word,
labels_lang) + 2917 npz + 2887 labels.

P1 overlap null calibration (frozen): the maxT machinery on
FIXED G matrices with R=100 independent permutation sets
(rng base 2902, seeds 2902+i, N_PERM=200 each, vectorized) -
each repeat draws E86' and Ewd' from the same G stacks and
counts overlap; p_overlap = mean(overlap_null >= 7). Pass iff
p <= 0.05. This preserves the grid correlation structure
(fixed G) while randomizing label masks - the honest null for
"two maxT sets on correlated grids".

P2 dual-strength discrimination (frozen main test): per event
in E17 (24) compute pct86 = mean(margin86 in its layer <= its
margin86) and pctwd (same under word caliber); statistic
min_pct = min(pct86, pctwd). Groups: survivor (7) vs lost
(17). Mann-Whitney U one-sided + label permutation (rng 2903,
10000). Pass iff p <= 0.05 in the survivor-higher direction
=> core_is_dual_strong.

P3 response-structure invariance (descriptive + test): per
E17 event, rho_e = Spearman(B86[h,:,l], B_word[h,:,l]) over
57 words (average ranks). Same U test survivor vs lost. Pass
iff p <= 0.05 survivor-higher => core_response_structure_
invariant.

Verdict (frozen):
  anchor fail => anchor_fail_all_void;
  P1 pass AND P2 pass => survivor_core_dual_strength_confirmed;
  P1 pass AND P2 fail => survivor_core_overlap_only;
  else => survivor_core_not_established.

Anchors (frozen):
  a1 sign_M86 (2927 npz) vs 2917 npz max abs diff < 1e-4;
  a2 rebuilt sets: |E17|=24, |Ewd|=32, overlap=7;
  a3 margin86(7,19) rounds to 1.01827 (2927 seal value);
  a4 cos_profile median == 0.1651 +/- 1e-3.

Output: phase2928/survivor_core_anatomy/.
"""
import hashlib
import json
import os
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC2927 = os.path.join(BASE, 'phase2927', 'probe_relativity',
                       'probe_relativity.npz')
SRC2917 = os.path.join(BASE, 'phase2917', 'event_atlas',
                       'event_atlas.npz')
OUT = os.path.join(BASE, 'phase2928', 'survivor_core_anatomy')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2928_run_report.txt')
NH, NL, NW = 32, 36, 57
FDR_Q = 0.05
N_PERM = 200
R_REPEAT = 100
RNG_BASE = 2902
RNG_UT = 2903
N_UT = 10000
TOP1 = (7, 19)

PREREG = {
    'mode': 'zero-forward survivor core anatomy on the 2927 npz '
            '(no model load)',
    'question': '2927 candidate A: is the 7/24 probe overlap '
                'above chance, and what distinguishes the 7 '
                'survivor events from the 17 lost ones - dual-'
                'probe margin strength and/or probe-invariant '
                'word-response structure?',
    'P1': 'overlap null: maxT on fixed G stacks, R=%d repeat '
          'permutation sets (rng base %d, N_PERM=%d), '
          'p_overlap = mean(overlap_null >= 7); pass iff p <= '
          '0.05' % (R_REPEAT, RNG_BASE, N_PERM),
    'P2': 'min_pct = min(pct86, pctwd) per E17 event, layer-'
          'internal margin percentile under each caliber; '
          'survivor (7) vs lost (17), Mann-Whitney U one-sided '
          '+ permutation rng %d x %d; pass iff p <= 0.05 '
          'survivor-higher => core_is_dual_strong'
          % (RNG_UT, N_UT),
    'P3': 'rho_e = Spearman(B86[h,:,l], B_word[h,:,l]) per E17 '
          'event; same U test; pass iff p <= 0.05 survivor-'
          'higher => core_response_structure_invariant',
    'verdict': 'anchor fail => anchor_fail_all_void; P1 pass '
               'AND P2 pass => '
               'survivor_core_dual_strength_confirmed; P1 pass '
               'AND P2 fail => survivor_core_overlap_only; '
               'else => survivor_core_not_established',
    'anchors': {
        'a1': 'sign_M86 vs 2917 npz max abs diff < 1e-4',
        'a2': 'rebuilt |E17|=24 |Ewd|=32 overlap=7',
        'a3': 'margin86(7,19) == 1.34634 AND margin_word(7,19) '
              '== 1.01827 (2927 seal values)',
        'a4': 'cos_profile median == 0.1651 +/- 1e-3',
    },
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


def mannwhitney_u(x, y):
    """U = #{(a, b): a > b} + 0.5 #{a == b}, x vs y."""
    u = 0.0
    for a in x:
        u += float(np.sum(a > y)) + 0.5 * float(np.sum(a == y))
    return u


def perm_u_p(x, y, rng, n_perm):
    """One-sided p: P(U_perm >= U_obs) under pooled label
    permutation. x is the 'higher' group."""
    nx, ny = len(x), len(y)
    pooled = np.concatenate([x, y])
    u_obs = mannwhitney_u(x, y)
    cnt = 0
    for _ in range(n_perm):
        rp = rng.permutation(pooled)
        xp = rp[:nx]
        yp = rp[nx:]
        if mannwhitney_u(xp, yp) >= u_obs - 1e-12:
            cnt += 1
    return (cnt + 1) / (n_perm + 1), u_obs


def maxT_events_fast(G_stack, lab, seed, n_perm=N_PERM):
    """Vectorized maxT on fixed G stack.

    G_stack: (NH*NL, NW, NW) Gram matrices (sign outer).
    Returns p_maxT (NH*NL,) and the E set of cell indices with
    p <= FDR_Q (flattened h*NL+li convention)."""
    E = G_stack.shape[0]
    rng = np.random.default_rng(seed)
    same_m, diff_m = masks(lab)
    n = len(lab)
    cs = int(same_m.sum())
    cd = int(diff_m.sum())
    same_flat = np.empty((n_perm, n * n), dtype=np.float32)
    diff_flat = np.empty((n_perm, n * n), dtype=np.float32)
    for p in range(n_perm):
        pl = rng.permutation(lab)
        sm, df = masks(pl)
        same_flat[p] = sm.ravel().astype(np.float32)
        diff_flat[p] = df.ravel().astype(np.float32)
    Gf = G_stack.reshape(E, n * n).astype(np.float32)
    true_same = same_m.ravel().astype(np.float32)
    true_diff = diff_m.ravel().astype(np.float32)
    obs = (Gf @ true_same) / cs - (Gf @ true_diff) / cd
    V = (Gf @ same_flat.T) / cs \
        - (Gf @ diff_flat.T) / cd          # (E, n_perm)
    max_v = V.max(axis=0)                  # (n_perm,)
    p_maxT = (np.sum(max_v[None, :] >= obs[:, None], axis=1)
              + 1) / (n_perm + 1)
    sig = set(int(i) for i in np.nonzero(p_maxT <= FDR_Q)[0])
    return p_maxT, sig


def gram_stack(B):
    """(NH*NL, NW, NW) sign-outer Gram stack from B (NH, NW, NL)."""
    out = np.empty((NH * NL, NW, NW), dtype=np.float32)
    for h in range(NH):
        for li in range(NL):
            s = np.sign(B[h, :, li])
            s[s == 0] = 1.0
            out[h * NL + li] = np.outer(s, s).astype(np.float32)
    return out


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2928,
                   'name': 'survivor_core_anatomy',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8': sha8(os.path.abspath(__file__)),
                   'sources': {'s2927': sha8(SRC2927),
                               's2917': sha8(SRC2917)},
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    z = np.load(SRC2927, allow_pickle=True)
    B86 = z['B86'].astype(np.float64)
    Bwd = z['B_word'].astype(np.float64)
    sm86 = z['sign_M86'].astype(np.float64)
    smwd = z['sign_M_word'].astype(np.float64)
    pm86 = z['p_maxT86'].astype(np.float64)
    pmwd = z['p_maxT_word'].astype(np.float64)
    lab = np.asarray(z['labels_lang']).astype(int)
    cos_prof = z['cos_profile'].astype(np.float64)
    z17 = np.load(SRC2917, allow_pickle=True)
    sm17 = z17['sign_M'].astype(np.float64)

    # ---------- anchors ----------
    a1_diff = float(np.abs(sm86 - sm17).max())
    a1_ok = bool(a1_diff < 1e-4)
    E17 = set((h, li) for h in range(NH) for li in range(NL)
              if pm86[h, li] <= FDR_Q)
    Ewd = set((h, li) for h in range(NH) for li in range(NL)
              if pmwd[h, li] <= FDR_Q)
    a2_ok = bool(len(E17) == 24 and len(Ewd) == 32
                 and len(E17 & Ewd) == 7)
    a3_ok = bool(round(float(sm86[7, 19]), 5) == 1.34634
                 and round(float(smwd[7, 19]), 5) == 1.01827)
    a4_ok = bool(abs(float(np.median(cos_prof)) - 0.1651) < 1e-3)
    anchor_ok = bool(a1_ok and a2_ok and a3_ok and a4_ok)
    log('a1 diff %.2e ok=%s | a2 sets ok=%s (24/32/7) | a3 %s '
        '| a4 %s'
        % (a1_diff, a1_ok, a2_ok,
           (round(float(sm86[7, 19]), 5),
            round(float(smwd[7, 19]), 5)), a4_ok), lines)

    verdict = None
    p1 = p2 = p3 = None
    save = {}
    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        survivor = sorted(E17 & Ewd)
        lost = sorted(E17 - Ewd)
        new = sorted(Ewd - E17)
        assert len(survivor) == 7 and len(lost) == 17 \
            and len(new) == 25

        # ---------- P1: overlap null calibration ---------------
        Gs86 = gram_stack(B86)
        Gswd = gram_stack(Bwd)
        overlaps = np.empty(R_REPEAT)
        for r in range(R_REPEAT):
            seed = RNG_BASE + r
            _, s86 = maxT_events_fast(Gs86, lab, seed)
            _, swd = maxT_events_fast(Gswd, lab, seed)
            overlaps[r] = len(s86 & swd)
        n_ge = int(np.sum(overlaps >= 7))
        p_overlap = float(n_ge + 1) / (R_REPEAT + 1)
        p1_pass = bool(p_overlap <= 0.05)
        p1 = {'R': R_REPEAT, 'seed_base': RNG_BASE,
              'overlap_null_median':
                  float(np.median(overlaps)),
              'overlap_null_max': float(overlaps.max()),
              'n_ge_obs': n_ge, 'p_overlap': round(p_overlap, 4),
              'pass': p1_pass}
        log('P1 overlap null: median %.1f max %.1f | n>=7 %d/%d '
            '| p %.4f pass=%s'
            % (np.median(overlaps), overlaps.max(), n_ge,
               R_REPEAT, p_overlap, p1_pass), lines)

        # ---------- P2: dual-strength discrimination -----------
        def layer_pct(SM, h, li):
            col = SM[:, li]
            return float(np.mean(col <= SM[h, li]))

        rows2 = []
        for (h, li) in sorted(E17):
            p86 = layer_pct(sm86, h, li)
            pwd = layer_pct(smwd, h, li)
            grp = 'survivor' if (h, li) in Ewd else 'lost'
            rows2.append({'event': [h, li], 'grp': grp,
                          'pct86': p86, 'pctwd': pwd,
                          'min_pct': min(p86, pwd)})
        x_sur = np.array([r['min_pct'] for r in rows2
                          if r['grp'] == 'survivor'])
        y_los = np.array([r['min_pct'] for r in rows2
                          if r['grp'] == 'lost'])
        rng_ut = np.random.default_rng(RNG_UT)
        p2_p, u2 = perm_u_p(x_sur, y_los, rng_ut, N_UT)
        p2_pass = bool(p2_p <= 0.05)
        p2 = {'survivor_min_pct_median':
                  round(float(np.median(x_sur)), 4),
              'lost_min_pct_median':
                  round(float(np.median(y_los)), 4),
              'U_obs': round(u2, 1), 'p': round(p2_p, 4),
              'pass': p2_pass,
              'rows': rows2}
        log('P2 min_pct: survivor median %.4f lost median %.4f '
            '| U %.1f p %.4f pass=%s'
            % (np.median(x_sur), np.median(y_los), u2, p2_p,
               p2_pass), lines)
        log('P2 detail: %s'
            % [(tuple(r['event']), r['grp'],
                round(r['min_pct'], 3)) for r in rows2], lines)

        # ---------- P3: response-structure invariance ----------
        rows3 = []
        for (h, li) in sorted(E17):
            rho_e = spearman(B86[h, :, li], Bwd[h, :, li])
            grp = 'survivor' if (h, li) in Ewd else 'lost'
            rows3.append({'event': [h, li], 'grp': grp,
                          'rho': rho_e})
        x3 = np.array([r['rho'] for r in rows3
                       if r['grp'] == 'survivor'])
        y3 = np.array([r['rho'] for r in rows3
                       if r['grp'] == 'lost'])
        rng_ut3 = np.random.default_rng(RNG_UT + 1)
        p3_p, u3 = perm_u_p(x3, y3, rng_ut3, N_UT)
        p3_pass = bool(p3_p <= 0.05)
        p3 = {'survivor_rho_median':
                  round(float(np.median(x3)), 4),
              'lost_rho_median': round(float(np.median(y3)), 4),
              'U_obs': round(u3, 1), 'p': round(p3_p, 4),
              'pass': p3_pass,
              'rows': [{'event': r['event'], 'grp': r['grp'],
                        'rho': round(r['rho'], 4)}
                       for r in rows3]}
        log('P3 rho(B86, Bwd): survivor median %.4f lost median '
            '%.4f | U %.1f p %.4f pass=%s'
            % (np.median(x3), np.median(y3), u3, p3_p, p3_pass),
            lines)
        log('P3 detail: %s'
            % [(tuple(r['event']), r['grp'], r['rho'])
               for r in p3['rows']], lines)

        # ---------- verdict ----------
        if p1_pass and p2_pass:
            verdict = 'survivor_core_dual_strength_confirmed'
        elif p1_pass:
            verdict = 'survivor_core_overlap_only'
        else:
            verdict = 'survivor_core_not_established'

        save = {'overlap_null': overlaps,
                'E17_ids': np.array(sorted(E17), dtype=int),
                'Ewd_ids': np.array(sorted(Ewd), dtype=int),
                'min_pct_rows': np.array(
                    [[r['pct86'], r['pctwd'], r['min_pct']]
                     for r in rows2], dtype=np.float64),
                'rho_rows': np.array(
                    [r['rho'] for r in rows3], dtype=np.float64)}

    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2928,
           'model': 'qwen3-4b (zero forward on 2927 npz)',
           'prereg': PREREG,
           'anchors': {'a1_diff': float('%.3e' % a1_diff),
                       'a1_ok': a1_ok,
                       'a2_ok': a2_ok,
                       'a3_margin86_top1':
                           round(float(sm86[7, 19]), 5),
                       'a3_ok': a3_ok,
                       'a4_cos_median':
                           round(float(np.median(cos_prof)), 4),
                       'a4_ok': a4_ok,
                       'ok': anchor_ok},
           'P1': p1, 'P2': p2, 'P3': p3,
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if save:
        np.savez_compressed(
            os.path.join(OUT, 'survivor_core_anatomy.npz'),
            **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2928 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
