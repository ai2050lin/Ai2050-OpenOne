# -*- coding: utf-8 -*-
"""Phase 2929: grid-wide response-structure atlas.

Why: 2928 established that maxT significant-set overlap is
uninformative (chance-level under fixed Gram structure) and
that the CORRECT probe-invariance statistics are event-level:
dual-caliber layer strength (P2 p 0.0049) and word-response
structure rho (P3 p 0.0025, survivor 0.825 vs lost 0.377).
Open question (2928 candidate A): what does rho(B86, B_word)
look like over ALL 1152 cells - is there a probe-invariant
SKELETON (cells whose 57-word response pattern is probe-
independent), how big is it, and is it coupled to or orthogonal
from the event selection?

Mode: zero forward, artifact domain on the 2927 npz (B86,
B_word, sign_M86, sign_M_word, p_maxT86, p_maxT_word) + 2928
npz (rho_rows for the 24 E17 events) + 2917 npz.

P1 full-grid rho map: rho_grid[h,l] = Spearman(B86[h,:,l],
B_word[h,:,l]) over the 57 words (average ranks). Layer-wise
permutation null (frozen): within layer l, permute the 32 head
rows of B_word (rng base 2904, 2000 permutations per layer,
vectorized on precomputed average ranks); p_perm[h,l] =
mean(rho_perm >= rho_obs). SKELETON (frozen definition): cells
with rho_obs >= the layer's null 95th percentile; skeleton
size vs the 5% expectation (57.6 cells) via exact binomial
(rng-free, comb).

P2 event coupling (frozen main test): event cells = E17 union
Ewd (24 + 32 - 7 = 49); U test of rho(event cells) vs
rho(non-event cells), rank-based Mann-Whitney with 10000
label permutations (rng 2905). Coupled iff p <= 0.05 in the
event-higher direction.

P3 distribution shape (descriptive): rho deciles; gap check
between skeleton and background.

Verdict (frozen):
  anchor fail => anchor_fail_all_void;
  skeleton_present AND events_rho_coupled   => skeleton_event_aligned;
  skeleton_present AND events_rho_uncoupled => skeleton_orthogonal_to_events;
  else => skeleton_not_established.
(skeleton_present iff n_skel >= 2x expectation AND binom p
 <= 0.001; skeleton_sparse otherwise.)

Anchors (frozen):
  a1 E17-event rho recompute vs 2928 npz rho_rows max diff < 1e-9;
  a2 rebuilt groups: |E17|=24 |Ewd|=32 overlap=7;
  a3 sign_M86 vs 2917 npz max abs diff < 1e-4.

Output: phase2929/response_structure_atlas/.
"""
import hashlib
import json
import os
import time
from math import comb

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC2927 = os.path.join(BASE, 'phase2927', 'probe_relativity',
                       'probe_relativity.npz')
SRC2928 = os.path.join(BASE, 'phase2928', 'survivor_core_anatomy',
                       'survivor_core_anatomy.npz')
SRC2917 = os.path.join(BASE, 'phase2917', 'event_atlas',
                       'event_atlas.npz')
OUT = os.path.join(BASE, 'phase2929', 'response_structure_atlas')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2929_run_report.txt')
NH, NL, NW = 32, 36, 57
FDR_Q = 0.05
N_PERM_L = 2000
RNG_PERM = 2904
RNG_U = 2905
N_U = 10000
EXP5 = NH * NL * 0.05

PREREG = {
    'mode': 'zero-forward grid-wide response-structure atlas on '
            'the 2927 npz (no model load)',
    'question': '2928 candidate A: is there a probe-invariant '
                'response-structure skeleton over the 1152-cell '
                'grid, how big is it, and is it coupled to or '
                'orthogonal from the event selection?',
    'P1': 'rho_grid[h,l] = Spearman(B86[h,:,l], B_word[h,:,l]); '
          'layer-wise head-permutation null (rng base %d, %d '
          'perms/layer, vectorized ranks); skeleton = rho_obs '
          '>= layer null p95; size test vs 5%% expectation '
          '(%.1f cells, exact binomial)'
          % (RNG_PERM, N_PERM_L, EXP5),
    'P2': 'event cells = E17 union Ewd (49); U test rho(event) '
          'vs rho(non-event), rank-based, %d permutations rng '
          '%d; coupled iff p <= 0.05 event-higher'
          % (N_U, RNG_U),
    'P3': 'rho deciles + skeleton/background gap, descriptive',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'skeleton_present AND events_rho_coupled => '
               'skeleton_event_aligned; skeleton_present AND '
               'events_rho_uncoupled => '
               'skeleton_orthogonal_to_events; else => '
               'skeleton_not_established (skeleton_present iff '
               'n_skel >= 2x expectation AND binom p <= 0.001)',
    'anchors': {
        'a1': 'E17-event rho recompute vs 2928 npz rho_rows '
              'max diff < 1e-9',
        'a2': 'rebuilt |E17|=24 |Ewd|=32 overlap=7',
        'a3': 'sign_M86 vs 2917 npz max abs diff < 1e-4',
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


def centered(M):
    return M - M.mean(axis=1, keepdims=True)


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2929,
                   'name': 'response_structure_atlas',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8': sha8(os.path.abspath(__file__)),
                   'sources': {'s2927': sha8(SRC2927),
                               's2928': sha8(SRC2928),
                               's2917': sha8(SRC2917)},
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    z = np.load(SRC2927, allow_pickle=True)
    B86 = z['B86'].astype(np.float64)
    Bwd = z['B_word'].astype(np.float64)
    sm86 = z['sign_M86'].astype(np.float64)
    pm86 = z['p_maxT86'].astype(np.float64)
    pmwd = z['p_maxT_word'].astype(np.float64)
    z28 = np.load(SRC2928, allow_pickle=True)
    rho_rows_28 = z28['rho_rows'].astype(np.float64)
    z17 = np.load(SRC2917, allow_pickle=True)
    sm17 = z17['sign_M'].astype(np.float64)

    # ---------- anchors ----------
    E17 = set((h, li) for h in range(NH) for li in range(NL)
              if pm86[h, li] <= FDR_Q)
    Ewd = set((h, li) for h in range(NH) for li in range(NL)
              if pmwd[h, li] <= FDR_Q)
    a2_ok = bool(len(E17) == 24 and len(Ewd) == 32
                 and len(E17 & Ewd) == 7)
    a3_diff = float(np.abs(sm86 - sm17).max())
    a3_ok = bool(a3_diff < 1e-4)
    rho_grid = np.zeros((NH, NL))
    for h in range(NH):
        for li in range(NL):
            rho_grid[h, li] = spearman(B86[h, :, li],
                                       Bwd[h, :, li])
    E17_sorted = sorted(E17)
    diffs_a1 = []
    for idx, (h, li) in enumerate(E17_sorted):
        diffs_a1.append(abs(rho_grid[h, li]
                            - float(rho_rows_28[idx])))
    a1_max = float(max(diffs_a1))
    a1_ok = bool(a1_max < 1e-9)
    anchor_ok = bool(a1_ok and a2_ok and a3_ok)
    log('a1 max diff %.2e ok=%s | a2 ok=%s | a3 %.2e ok=%s'
        % (a1_max, a1_ok, a2_ok, a3_diff, a3_ok), lines)

    verdict = None
    p1 = p2 = p3 = None
    save = {}
    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        # ---------- P1: layer-wise permutation null ------------
        rng = np.random.default_rng(RNG_PERM)
        p_perm = np.zeros((NH, NL))
        null_p95 = np.zeros(NL)
        Rwd_rank = {}
        for li in range(NL):
            Rwd_rank[li] = np.stack(
                [avg_ranks(Bwd[h, :, li]) for h in range(NH)])
        for li in range(NL):
            R86 = np.stack([avg_ranks(B86[h, :, li])
                            for h in range(NH)])
            R86c = centered(R86)
            Rwdc = centered(Rwd_rank[li])
            denom = np.sqrt((R86c ** 2).sum(1)
                            * (Rwdc ** 2).sum(1))
            rho_layer = (R86c * Rwdc).sum(1) / np.maximum(
                denom, 1e-30)
            rho_grid[:, li] = rho_layer
            perms = np.stack(
                [rng.permutation(NH) for _ in range(N_PERM_L)])
            # rho_perm[k, h] = corr(R86[h], Rwd[perms[k][h]])
            Rw_p = Rwdc[perms]                     # (K, NH, 57)
            num = np.einsum('hd,khd->kh', R86c, Rw_p)
            den = np.sqrt(
                (R86c ** 2).sum(1)[None, :]
                * (Rw_p ** 2).sum(2))              # (K, NH)
            rho_perm = num / np.maximum(den, 1e-30)
            p_perm[:, li] = np.mean(
                rho_perm >= rho_layer[None, :], axis=0)
            null_p95[li] = float(np.percentile(rho_perm, 95))
        skel_mask = rho_grid >= null_p95[None, :]
        n_skel = int(skel_mask.sum())
        # exact binomial tail via lgamma (float-safe);
        # log imported as _log to avoid shadowing the module logger
        from math import lgamma, exp, log as _log
        n_tot = NH * NL
        p_binom = 0.0
        for x in range(n_skel, n_tot + 1):
            logc = (lgamma(n_tot + 1) - lgamma(x + 1)
                    - lgamma(n_tot - x + 1))
            term = exp(logc + x * _log(0.05)
                       + (n_tot - x) * _log(0.95))
            p_binom += term
            if term == 0.0 and x > n_skel + 50:
                break
        skeleton_present = bool(n_skel >= 2 * EXP5
                                and p_binom <= 0.001)
        p1 = {'n_skeleton': n_skel,
              'expectation_5pct': EXP5,
              'binom_p': p_binom,
              'skeleton_present': skeleton_present,
              'rho_median': round(float(np.median(rho_grid)), 4),
              'rho_p90': round(float(np.percentile(rho_grid,
                                                   90)), 4),
              'rho_max': round(float(rho_grid.max()), 4),
              'layer_null_p95_median':
                  round(float(np.median(null_p95)), 4),
              'skeleton_layers': sorted(
                  int(li) for li in set(
                      li for h, li in zip(
                          *np.nonzero(skel_mask))))}
        log('P1: n_skel %d (expect %.1f) binom p %.3e present=%s '
            '| rho median %.4f p90 %.4f max %.4f'
            % (n_skel, EXP5, p_binom, skeleton_present,
               np.median(rho_grid),
               np.percentile(rho_grid, 90), rho_grid.max()),
            lines)
        log('P1 skeleton layer profile: %s'
            % [(l, int(skel_mask[:, l].sum()))
               for l in range(NL)], lines)

        # ---------- P2: event coupling -------------------------
        event_cells = sorted(E17 | Ewd)
        event_flag = np.zeros(NH * NL, dtype=bool)
        for (h, li) in event_cells:
            event_flag[h * NL + li] = True
        rho_flat = rho_grid.ravel()
        r_ev = np.sort(rho_flat[event_flag])
        r_bg = np.sort(rho_flat[~event_flag])
        nx, nbg = len(r_ev), len(r_bg)
        pooled = np.concatenate([r_ev, r_bg])
        order = np.argsort(pooled, kind='mergesort')
        ranks_p = np.empty(len(pooled))
        sp = pooled[order]
        i = 0
        while i < len(pooled):
            j = i
            while j + 1 < len(pooled) and sp[j + 1] == sp[i]:
                j += 1
            ranks_p[order[i:j + 1]] = (i + j) / 2.0
            i = j + 1
        R_ev = ranks_p[:nx].sum()
        U_obs = R_ev - nx * (nx + 1) / 2.0
        rng_u = np.random.default_rng(RNG_U)
        cnt = 0
        idx_all = np.arange(nx + nbg)
        half = nx * (nx + 1) / 2.0
        for _ in range(N_U):
            rp = rng_u.permutation(idx_all)[:nx]
            if ranks_p[rp].sum() - half >= U_obs - 1e-9:
                cnt += 1
        p2_p = float(cnt + 1) / (N_U + 1)
        events_coupled = bool(p2_p <= 0.05)
        p2 = {'n_event_cells': nx, 'n_background': nbg,
              'rho_event_median':
                  round(float(np.median(r_ev)), 4),
              'rho_background_median':
                  round(float(np.median(r_bg)), 4),
              'U_obs': float(U_obs), 'p': round(p2_p, 4),
              'events_rho_coupled': events_coupled}
        log('P2: rho event median %.4f (n=%d) vs background '
            'median %.4f (n=%d) | U %.1f p %.4f coupled=%s'
            % (np.median(r_ev), nx, np.median(r_bg), nbg,
               U_obs, p2_p, events_coupled), lines)
        # group detail (survivor/lost/new)
        grp_det = {}
        for name, cells in (('survivor', sorted(E17 & Ewd)),
                            ('lost', sorted(E17 - Ewd)),
                            ('new', sorted(Ewd - E17))):
            vals = [rho_grid[h, li] for (h, li) in cells]
            grp_det[name] = round(float(np.median(vals)), 4)
        p2['group_medians'] = grp_det
        log('P2 group rho medians: %s' % grp_det, lines)

        # ---------- P3: distribution shape ---------------------
        dec = [round(float(np.percentile(rho_flat, q)), 4)
               for q in (10, 20, 30, 40, 50, 60, 70, 80, 90)]
        skel_vals = rho_grid[skel_mask]
        bg_vals = rho_grid[~skel_mask]
        gap_lo = float(bg_vals.max())
        gap_hi = float(skel_vals.min())
        p3 = {'deciles': dec,
              'skeleton_rho_min': round(gap_hi, 4),
              'background_rho_max': round(gap_lo, 4),
              'hard_gap': bool(gap_hi > gap_lo),
              'skeleton_in_event': int(
                  np.sum([skel_mask[h, li]
                          for (h, li) in event_cells]))}
        log('P3 deciles: %s | skel min %.4f bg max %.4f '
            'hard_gap=%s | skeleton cells in event set: %d/49'
            % (dec, gap_hi, gap_lo, gap_hi > gap_lo,
               p3['skeleton_in_event']), lines)

        # ---------- verdict ----------
        if skeleton_present and events_coupled:
            verdict = 'skeleton_event_aligned'
        elif skeleton_present and not events_coupled:
            verdict = 'skeleton_orthogonal_to_events'
        else:
            verdict = 'skeleton_not_established'

        save = {'rho_grid': rho_grid,
                'p_perm': p_perm,
                'null_p95': null_p95,
                'skel_mask': skel_mask,
                'event_flag': event_flag}

    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2929,
           'model': 'qwen3-4b (zero forward on 2927 npz)',
           'prereg': PREREG,
           'anchors': {'a1_max_diff': float('%.3e' % a1_max),
                       'a1_ok': a1_ok,
                       'a2_ok': a2_ok,
                       'a3_diff': float('%.3e' % a3_diff),
                       'a3_ok': a3_ok,
                       'ok': anchor_ok},
           'P1': p1, 'P2': p2, 'P3': p3,
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if save:
        np.savez_compressed(
            os.path.join(OUT, 'response_structure_atlas.npz'),
            **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2929 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
