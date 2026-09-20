# -*- coding: utf-8 -*-
"""Phase 2931: skeleton overlap null calibration (zero forward).

Why: 2930 P3 reported jaccard(skeleton_2929, skeleton_mirror)
= 0.556 (327 shared cells) and interpreted it as "the skeleton
body is convention-invariant". Discipline 11 (2928 lesson)
requires ANY overlap claim to pass a null calibration first:
maxT significant-set overlap at 2928 was EXACTLY chance under
fixed Gram structure. The skeleton overlap must pass the same
test before "convention-invariant body" is claimed.

Mode: zero forward on the 2927 / 2929 / 2930 npz (no model
load). Skeletons are value-threshold selections; permuting the
head order within a layer permutes the skeleton labels with the
values, so the null is built by INDEPENDENT double head
permutation (one for each caliber), preserving each column's
value multiset (layer hotspots + skeleton size) and destroying
the cross-caliber cell pairing.

Anchors (frozen):
  a1 rho_grid recompute from 2927 npz (B86, B_word) vs 2929
     npz rho_grid max abs diff < 1e-9
  a2 rho_mirror recompute from 2930 npz (B86, B_mirror) vs
     2930 npz rho_mirror_grid max abs diff < 1e-9
  a3 set rebuild |skel29|==501 |skelmir|==414 inter==327
  a4 2929 null_p95 replay (rng 2904 verbatim) vs 2929 npz
     null_p95 max abs diff < 1e-9

P1 main test (frozen): corrected caliber (degenerate cells
excluded via rank-std gate on B86/B_word/B_mirror ranks) =>
obs_jacc; null = R=1000 independent double within-layer head
permutations (rng 2906) of both skeleton masks; verdict:
  obs >= null p95 => skeleton_overlap_above_chance
  null p50 < obs < null p95 => skeleton_overlap_borderline
  obs <= null p50 => skeleton_overlap_chance_level
P1b sanity: same-permutation pairing should reproduce obs
jaccard bit-exactly on every replicate (implementation check).

P2 cell-level pairing (descriptive): Spearman(rho29, rhomir)
over surviving cells + per-layer profile + survivor-7 values.

P3 degenerate-gate audit (descriptive): count exact-zero rho
cells vs rank-std degenerate cells; L0 vs other layers.

P4 independence reference: |S29|*|Smir|/N_cells (pure
independence expectation for the intersection).

Output: phase2931/skeleton_overlap_null/.
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
SRC2929 = os.path.join(BASE, 'phase2929',
                       'response_structure_atlas',
                       'response_structure_atlas.npz')
SRC2930 = os.path.join(BASE, 'phase2930',
                       'direction_flip_control',
                       'direction_flip_control.npz')
OUT = os.path.join(BASE, 'phase2931', 'skeleton_overlap_null')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2931_run_report.txt')
NH, NL, NW = 32, 36, 57
R_NULL = 1000
RNG_NULL = 2906
SURV = [(1, 6), (5, 6), (7, 19), (8, 2), (14, 9), (20, 8),
        (21, 6)]

PREREG = {
    'mode': 'zero-forward skeleton overlap null calibration '
            'on the 2927/2929/2930 npz (no model load)',
    'question': '2930 candidate A: is the skeleton overlap '
                'jaccard 0.556 (327 shared cells) above the '
                'chance level under fixed statistical '
                'structure (discipline 11), i.e. is the '
                '"convention-invariant body" claim earned?',
    'null': 'independent double within-layer head permutation '
            '(one rng-drawn permutation per caliber per layer, '
            'rng base %d, R=%d): permute both skeleton masks '
            'along heads, preserving each column value '
            'multiset (layer hotspots, skeleton sizes) and '
            'destroying cross-caliber cell pairing'
            % (RNG_NULL, R_NULL),
    'gate': 'degenerate cells (rank-vector std <= 1e-10 on '
            'B86, B_word, or B_mirror 57-word ranks) excluded '
            'from BOTH calibers before overlap (discipline 12 '
            'non-degeneracy gate; removes the L0 all-32 '
            '0>=0 artifact)',
    'anchors': {
        'a1': 'rho_grid recompute (2927 npz) vs 2929 npz '
              'max abs < 1e-9',
        'a2': 'rho_mirror recompute (2930 npz) vs 2930 npz '
              'max abs < 1e-9',
        'a3': 'set rebuild 501/414/inter 327',
        'a4': '2929 null_p95 replay (rng 2904) max abs < 1e-9',
    },
    'P1': 'obs_jacc (corrected caliber) vs null jaccard '
          'distribution; obs >= p95 => '
          'skeleton_overlap_above_chance; p50 < obs < p95 => '
          'skeleton_overlap_borderline; obs <= p50 => '
          'skeleton_overlap_chance_level',
    'P1b': 'sanity: same-permutation pairing must reproduce '
           'obs jaccard bit-exactly on every replicate',
    'P2': 'Spearman(rho29, rhomir) grid + per-layer + '
          'survivor-7 values, descriptive',
    'P3': 'degenerate-gate audit: exact-zero rho cells vs '
          'rank-std degenerate cells by layer, descriptive',
    'P4': 'independence reference |S29|*|Smir|/N_cells',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'obs >= null p95 => '
               'skeleton_overlap_above_chance; p50 < obs < '
               'p95 => skeleton_overlap_borderline; '
               'obs <= p50 => skeleton_overlap_chance_level',
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
        json.dump({'phase': 2931,
                   'name': 'skeleton_overlap_null',
                   'created': time.strftime(
                       '%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2927': sha8(SRC2927),
                               's2929': sha8(SRC2929),
                               's2930': sha8(SRC2930)},
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    # ---------- sources ----------
    z27 = np.load(SRC2927, allow_pickle=True)
    B86 = z27['B86'].astype(np.float64)
    Bwd = z27['B_word'].astype(np.float64)
    z29 = np.load(SRC2929, allow_pickle=True)
    rho29 = z29['rho_grid'].astype(np.float64)
    skel29 = z29['skel_mask'].astype(bool)
    null_p95_29 = z29['null_p95'].astype(np.float64)
    z30 = np.load(SRC2930, allow_pickle=True)
    Bmir = z30['B_mirror'].astype(np.float64)
    rhomir = z30['rho_mirror_grid'].astype(np.float64)
    skelmir = z30['skel_mask_mirror'].astype(bool)

    # ---------- anchors ----------
    rho29_rc = np.zeros((NH, NL))
    rhomir_rc = np.zeros((NH, NL))
    for h in range(NH):
        for li in range(NL):
            rho29_rc[h, li] = spearman(B86[h, :, li],
                                       Bwd[h, :, li])
            rhomir_rc[h, li] = spearman(B86[h, :, li],
                                        Bmir[h, :, li])
    a1_diff = float(np.abs(rho29_rc - rho29).max())
    a2_diff = float(np.abs(rhomir_rc - rhomir).max())
    a3_ok = bool(int(skel29.sum()) == 501
                 and int(skelmir.sum()) == 414
                 and int((skel29 & skelmir).sum()) == 327)
    # a4: replay 2929 null p95 (rng 2904 verbatim)
    rng4 = np.random.default_rng(2904)
    p95_replay = np.zeros(NL)
    for li in range(NL):
        R86c = centered(np.stack(
            [avg_ranks(B86[h, :, li]) for h in range(NH)]))
        Rwdc = centered(np.stack(
            [avg_ranks(Bwd[h, :, li]) for h in range(NH)]))
        perms_h = np.stack(
            [rng4.permutation(NH) for _ in range(2000)])
        Rw_p = Rwdc[perms_h]
        num = np.einsum('hd,khd->kh', R86c, Rw_p)
        den = np.sqrt((R86c ** 2).sum(1)[None, :]
                      * (Rw_p ** 2).sum(2))
        rho_perm = num / np.maximum(den, 1e-30)
        p95_replay[li] = float(np.percentile(rho_perm, 95))
    a4_diff = float(np.abs(p95_replay - null_p95_29).max())
    a1_ok = bool(a1_diff < 1e-9)
    a2_ok = bool(a2_diff < 1e-9)
    a4_ok = bool(a4_diff < 1e-9)
    anchor_ok = bool(a1_ok and a2_ok and a3_ok and a4_ok)
    log('a1 %.2e ok=%s | a2 %.2e ok=%s | a3 ok=%s | '
        'a4 %.2e ok=%s'
        % (a1_diff, a1_ok, a2_diff, a2_ok, a3_ok,
           a4_diff, a4_ok), lines)

    verdict = None
    p1 = p2 = p3 = p4 = None
    save = {}
    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        # ---------- degenerate gate ----------
        gate = np.zeros((NH, NL), dtype=bool)
        n_zero_rho = 0
        n_deg = 0
        deg_layers = {}
        for h in range(NH):
            for li in range(NL):
                s86 = avg_ranks(B86[h, :, li]).std()
                swd = avg_ranks(Bwd[h, :, li]).std()
                smir = avg_ranks(Bmir[h, :, li]).std()
                if min(s86, swd, smir) <= 1e-10:
                    gate[h, li] = True
                    n_deg += 1
                    deg_layers[li] = \
                        deg_layers.get(li, 0) + 1
                if rho29[h, li] == 0.0:
                    n_zero_rho += 1
        keep = ~gate
        S29 = skel29 & keep
        Smir = skelmir & keep
        n29 = int(S29.sum())
        nmir = int(Smir.sum())
        inter = int((S29 & Smir).sum())
        union = int((S29 | Smir).sum())
        obs_jacc = inter / max(union, 1)
        log('gate: degenerate cells %d (layers %s) | '
            'exact-zero rho cells %d | corrected sizes '
            '%d/%d inter %d obs_jacc %.4f'
            % (n_deg, sorted(deg_layers.items()), n_zero_rho,
               n29, nmir, inter, obs_jacc), lines)

        # ---------- P1 main null ----------
        rng = np.random.default_rng(RNG_NULL)
        j_null = np.zeros(R_NULL)
        for r in range(R_NULL):
            S29p = np.zeros_like(S29)
            Smirp = np.zeros_like(Smir)
            for li in range(NL):
                pi = rng.permutation(NH)
                sg = rng.permutation(NH)
                S29p[:, li] = S29[pi, li]
                Smirp[:, li] = Smir[sg, li]
            i_r = int((S29p & Smirp).sum())
            u_r = int((S29p | Smirp).sum())
            j_null[r] = i_r / max(u_r, 1)
        p50 = float(np.percentile(j_null, 50))
        p95 = float(np.percentile(j_null, 95))
        if obs_jacc >= p95:
            verdict = 'skeleton_overlap_above_chance'
        elif obs_jacc > p50:
            verdict = 'skeleton_overlap_borderline'
        else:
            verdict = 'skeleton_overlap_chance_level'
        p1 = {'obs_jacc': round(obs_jacc, 4),
              'n29_corrected': n29, 'nmir_corrected': nmir,
              'inter_corrected': inter,
              'null_median': round(p50, 4),
              'null_p95': round(p95, 4),
              'null_max': round(float(j_null.max()), 4),
              'null_mean': round(float(j_null.mean()), 4),
              'exceed_frac':
                  round(float(np.mean(j_null >= obs_jacc)),
                        4)}
        log('P1: obs jacc %.4f | null median %.4f mean %.4f '
            'p95 %.4f max %.4f | P(null>=obs) %.4f => %s'
            % (obs_jacc, p50, j_null.mean(), p95,
               j_null.max(), np.mean(j_null >= obs_jacc),
               verdict), lines)

        # ---------- P1b sanity (same permutation) ----------
        rng_s = np.random.default_rng(RNG_NULL)
        max_dev = 0.0
        for r in range(R_NULL):
            S29p = np.zeros_like(S29)
            Smirp = np.zeros_like(Smir)
            for li in range(NL):
                pi = rng_s.permutation(NH)
                rng_s.permutation(NH)
                S29p[:, li] = S29[pi, li]
                Smirp[:, li] = Smir[pi, li]
            i_r = int((S29p & Smirp).sum())
            u_r = int((S29p | Smirp).sum())
            max_dev = max(max_dev,
                          abs(i_r / max(u_r, 1) - obs_jacc))
        p1['sanity_same_perm_max_dev'] = float(
            '%.3e' % max_dev)
        p1['sanity_ok'] = bool(max_dev == 0.0)
        log('P1b sanity: same-perm pairing max dev %.2e '
            '(expect 0)' % max_dev, lines)

        # ---------- P2 cell-level pairing ----------
        flat29 = rho29[keep]
        flatmir = rhomir[keep]
        rho_cells = spearman(flat29, flatmir)
        per_layer = []
        for li in range(NL):
            if keep[:, li].any():
                per_layer.append(
                    (li, round(float(spearman(
                        rho29[keep[:, li], li],
                        rhomir[keep[:, li], li])), 4)))
        surv = []
        for (h, li) in SURV:
            surv.append({'cell': [h, li],
                         'rho29': round(float(
                             rho29[h, li]), 4),
                         'rhomir': round(float(
                             rhomir[h, li]), 4),
                         'in_S29': bool(S29[h, li]),
                         'in_Smir': bool(Smir[h, li])})
        p2 = {'spearman_cells': round(rho_cells, 4),
              'per_layer_worst': min(per_layer,
                                     key=lambda t: t[1]),
              'per_layer_best': max(per_layer,
                                    key=lambda t: t[1]),
              'survivor7': surv}
        log('P2: cell-level Spearman(rho29, rhomir) %.4f | '
            'worst layer %s best %s'
            % (rho_cells, p2['per_layer_worst'],
               p2['per_layer_best']), lines)
        log('P2 survivor7: %s'
            % [(d['cell'], d['rho29'], d['rhomir'],
                d['in_S29'], d['in_Smir']) for d in surv],
            lines)

        # ---------- P3/P4 descriptive ----------
        p3 = {'n_degenerate': n_deg,
              'degenerate_layers':
                  {str(k): int(v) for k, v in
                   sorted(deg_layers.items())},
              'n_exact_zero_rho': n_zero_rho}
        exp_ind = n29 * nmir / float(NH * NL)
        p4 = {'independence_expected_inter':
                  round(exp_ind, 1),
              'observed_inter': inter,
              'ratio_vs_independence':
                  round(inter / max(exp_ind, 1e-30), 3)}
        log('P3: degenerate %d (layers %s) | exact-zero rho '
            'cells %d' % (n_deg, sorted(deg_layers.items()),
                          n_zero_rho), lines)
        log('P4: independence expected inter %.1f vs observed '
            '%d (ratio %.3f)'
            % (exp_ind, inter,
               inter / max(exp_ind, 1e-30)), lines)

        save = {'gate': gate,
                'S29_corrected': S29,
                'Smir_corrected': Smir,
                'null_jacc': j_null}

    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2931,
           'model': 'qwen3-4b (zero forward on 2927/2929/2930 '
                    'npz)',
           'prereg': PREREG,
           'anchors': {'a1_diff': float('%.3e' % a1_diff),
                       'a1_ok': a1_ok,
                       'a2_diff': float('%.3e' % a2_diff),
                       'a2_ok': a2_ok,
                       'a3_ok': a3_ok,
                       'a4_diff': float('%.3e' % a4_diff),
                       'a4_ok': a4_ok,
                       'ok': anchor_ok},
           'P1': p1, 'P2': p2, 'P3': p3, 'P4': p4,
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if save:
        np.savez_compressed(
            os.path.join(OUT, 'skeleton_overlap_null.npz'),
            **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2931 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
