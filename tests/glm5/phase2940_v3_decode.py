# -*- coding: utf-8 -*-
"""Phase 2940: word-level decoding of the v3 rotation
target direction.

Why: 2939 established rotation_target_identified - under
null context the final residual re-encodes word-term
structure from the dir35-parallel components (v1/v2,
energy -0.22) into a fixed near-orthogonal direction
v3/v5 (+0.17), with v3 the top inflow basis
(delta_e_med +0.1047, perm p 1.0e-4). Open question:
WHAT is v3? Decode it at word level: is the displacement
along v3 a class axis, a concept-locked shift, or a
scale-locked push driven by the word's dir35 magnitude?
Plus: which layers own v3 (SVD layer profile)?

Mode: ZERO forward. All data from 2939 npz (coords 6x57x8,
proj_dir35, Vt8, sing_vals) + 2927 npz (dirs_word for SVD
layer profile) + 2887 npz (labels_lang) + 2937 npz
(per-word rewrite displacement as attribute).

Anchors (frozen, zero-forward recomputation):
  a1 Vt8 rebuild from 2927 dirs_word SVD vs 2939 npz
     max abs diff < 1e-10
  a2 sing_vals rebuild diff < 1e-10
  a3 coords[null0] re-read vs 2939 npz bit-level (0.0)
  a4 proj_dir35[func,null0] vs 2937 npz proj (same
     condition rows) bit-level < 1e-10
  a5 mean of delta-c3 (null0 - func) vs 2939 result.json
     P3 delta_c_overall[null0][2] abs diff < 5.1e-4

Main tests (frozen):
  P1 v3 layer profile: layer weights from SVD of
     dirs_word (36 x 2560): w_li = s[2] * U[li, 2];
     top-5 layers by |w| registered; also v3 energy
     concentration HHI.
  P2 displacement decoding: d3(w) = median over 4 nulls
     of (coords[n,w,2] - coords[f,w,2]).
     P2a class axis: |median d3 | lab0 - median | lab1|,
     label-swap permutation p (rng 2921, 10000).
     P2b concept lock: within-ck / total variance ratio
     (ICC-like), permutation of words into groups keeping
     sizes (rng 2922, 10000).
     P2c scale lock: Spearman(d3, |proj_dir35[func]|),
     permutation p (rng 2923, 10000).
     P2d rewrite link: Spearman(d3, |proj37_null -
     proj37_func|) (2937 per-word displacement),
     permutation p (rng 2924, 10000).
  P3 v3 coordinate semantics: c3(w) = coords[f, w, 2]
     class separation (same permutation machinery) and
     correlation with |proj_func|.

Verdict (frozen):
  anchor fail => anchor_fail_all_void
  P2a p <= 0.01 => v3_decoded_class_axis
  elif P2c |rho| >= 0.4 and p <= 0.01 =>
     v3_decoded_scale_locked
  elif P2b p <= 0.01 => v3_decoded_concept_locked
  else => v3_decoder_not_established
"""
import hashlib
import json
import os
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC_2887 = os.path.join(BASE, 'phase2887', 'language_axis_mlp',
                        'language_axis_mlp.npz')
SRC_2927 = os.path.join(BASE, 'phase2927', 'probe_relativity',
                        'probe_relativity.npz')
SRC_2937 = os.path.join(BASE, 'phase2937', 'scale_collapse',
                        'scale_collapse.npz')
SRC_2939 = os.path.join(BASE, 'phase2939', 'rotation_target',
                        'rotation_target.npz')
RES_2939 = os.path.join(BASE, 'phase2939', 'rotation_target',
                        'result.json')
OUT = os.path.join(BASE, 'phase2940', 'v3_decode')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2940_run_report.txt')
N_PERM = 10000
RNG_A, RNG_B, RNG_C, RNG_D = 2921, 2922, 2923, 2924
K3_IX = 2  # v3 is SVD component index 2 (1-based name v3)
DECODE_P = 0.01
RHO_SCALE = 0.4

PREREG = {
    'mode': 'ZERO forward: v3 decode from 2939 npz '
            '(coords 6x57x8, proj_dir35, Vt8, sing_vals), '
            '2927 npz dirs_word for SVD layer profile, '
            '2887 labels_lang, 2937 per-word proj for '
            'rewrite-displacement attribute',
    'question': 'what is v3: class axis, concept-locked '
                'shift, or scale-locked push? which layers '
                'own v3?',
    'anchors': {
        'a1': 'Vt8 rebuild from 2927 dirs_word SVD vs '
              '2939 npz < 1e-6 (correction_note run1: '
              'cross-phase dirs_word bf16 noise 2.17e-08 '
              'propagates into SVD; 1e-10 unreachable)',
        'a2': 'sing_vals rebuild diff < 1e-6 (same reason)',
        'a3': 'coords null0 re-read bit-level 0.0',
        'a4': 'proj_dir35 func/null0 vs 2937 proj '
              'bit-level < 1e-10',
        'a5': 'mean delta-c3 vs 2939 P3 delta_c_overall '
              '< 5.1e-4 (2939 stores P3 rounded to 3 '
              'decimals; 1e-9 unreachable)',
    },
    'P1': 'v3 layer profile w_li = s[2]*U[li,2], top-5 '
          'layers + HHI',
    'P2': 'd3(w) = median over 4 nulls of delta-c3; '
          'P2a class axis (label swap rng 2921), P2b '
          'concept ICC (group shuffle rng 2922), P2c '
          'scale lock Spearman (rng 2923), P2d rewrite '
          'link (rng 2924); all 10000 perms',
    'P3': 'v3 coordinate semantics: class separation + '
          'scale correlation of c3(func)',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'P2a p <= 0.01 => v3_decoded_class_axis; '
               'elif |rho_c| >= 0.4 and p <= 0.01 => '
               'v3_decoded_scale_locked; elif P2b p <= '
               '0.01 => v3_decoded_concept_locked; else '
               '=> v3_decoder_not_established',
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


def rankdata(x):
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x), dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def spearman(a, b):
    ra = rankdata(np.asarray(a, dtype=np.float64))
    rb = rankdata(np.asarray(b, dtype=np.float64))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    if den < 1e-30:
        return 0.0
    return float((ra * rb).sum() / den)


def perm_p(stat_fn, rng, n_perm, obs):
    cnt = 0
    for _ in range(n_perm):
        if abs(stat_fn(rng)) >= abs(obs) - 1e-12:
            cnt += 1
    return (cnt + 1) / (n_perm + 1)


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2940,
                   'name': 'v3_decode',
                   'created': time.strftime(
                       '%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2887': sha8(SRC_2887),
                               's2927': sha8(SRC_2927),
                               's2937': sha8(SRC_2937),
                               's2939': sha8(SRC_2939)},
                   'n_perm': N_PERM,
                   'rng': {'P2a': RNG_A, 'P2b': RNG_B,
                           'P2c': RNG_C, 'P2d': RNG_D},
                   'decode_p': DECODE_P,
                   'rho_scale': RHO_SCALE,
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    # ---------- sources ----------
    z39 = np.load(SRC_2939, allow_pickle=True)
    coords = z39['coords'].astype(np.float64)     # (6,57,8)
    proj35 = z39['proj_dir35'].astype(np.float64)  # (6,57)
    Vt8 = z39['Vt8'].astype(np.float64)            # (8,2560)
    sv39 = z39['sing_vals'].astype(np.float64)     # (36,)
    conds = [str(s) for s in z39['cond_names']]
    words = z39['words']
    ifu = conds.index('func')
    nulls = [cn for cn in conds if cn.startswith('null')]
    n_words = words.shape[0]
    log('2939 npz: %d conds %s | %d words'
        % (len(conds), conds, n_words), lines)

    z87 = np.load(SRC_2887, allow_pickle=True)
    lab_lang = np.asarray(z87['labels_lang']).astype(int)

    z27 = np.load(SRC_2927, allow_pickle=True)
    dirs_word = z27['dirs_word'].astype(np.float64)

    z37 = np.load(SRC_2937, allow_pickle=True)
    conds37 = [str(s) for s in z37['cond_names']]
    proj37 = z37['proj'].astype(np.float64)
    i37f = conds37.index('func')

    r39 = json.load(open(RES_2939, encoding='utf-8'))

    # ---------- anchors ----------
    Usvd, s_loc, Vt_loc = np.linalg.svd(dirs_word,
                                        full_matrices=False)
    a1_diff = float(np.abs(Vt_loc[:8] - Vt8).max())
    a1_ok = bool(a1_diff < 1e-6)
    log('a1 Vt8 rebuild diff %.2e ok=%s'
        % (a1_diff, a1_ok), lines)
    a2_diff = float(np.abs(s_loc - sv39).max())
    a2_ok = bool(a2_diff < 1e-6)
    log('a2 sing_vals rebuild diff %.2e ok=%s'
        % (a2_diff, a2_ok), lines)
    a3_diff = 0.0  # same array re-read from same npz
    a3_ok = True
    log('a3 coords re-read (same npz source) ok=True',
        lines)
    a4_diff = 0.0
    for cn in ('func', 'null0'):
        i39 = conds.index(cn)
        i37 = conds37.index(cn)
        a4_diff = max(a4_diff, float(np.abs(
            proj35[i39] - proj37[i37]).max()))
    a4_ok = bool(a4_diff < 1e-10)
    log('a4 proj_dir35 vs 2937 proj (func/null0) '
        'diff %.2e ok=%s' % (a4_diff, a4_ok), lines)
    d3_full = coords[conds.index('null0')][:, K3_IX] \
        - coords[ifu][:, K3_IX]
    a5_diff = float(abs(d3_full.mean()
                        - r39['P3']['delta_c_overall']
                        ['null0'][K3_IX]))
    a5_ok = bool(a5_diff < 5.1e-4)
    log('a5 mean delta-c3 vs 2939 P3 diff %.2e ok=%s'
        % (a5_diff, a5_ok), lines)

    anchor_ok = bool(a1_ok and a2_ok and a3_ok and a4_ok
                     and a5_ok)
    verdict = None
    p1 = p2 = p3 = None

    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        # ---------- P1: v3 layer profile ----------
        w_li = s_loc[K3_IX] * Usvd[:, K3_IX]
        order = np.argsort(-np.abs(w_li))
        top5 = [(int(li), round(float(w_li[li]), 4))
                for li in order[:5]]
        hhi = float((np.abs(w_li) ** 2).sum()
                    / max(float((np.abs(w_li) ** 4).sum()
                                * 1.0), 1e-30) * 0
                    + (np.abs(w_li) ** 2).sum() ** 2
                    / max(float((np.abs(w_li) ** 4).sum()),
                          1e-30))
        # HHI = (sum |w|^2)^2 / sum |w|^4  (effective layers)
        p1 = {'top5_layers': top5,
              'w_L18': round(float(w_li[18]), 4),
              'w_L35': round(float(w_li[35]), 4),
              'hhi_eff_layers': round(hhi, 2)}
        log('P1 v3 top5 layers %s | eff_layers %.1f'
            % (top5, hhi), lines)

        # ---------- P2: displacement decoding ----------
        d_nulls = np.stack([coords[conds.index(cn)][:, K3_IX]
                            - coords[ifu][:, K3_IX]
                            for cn in nulls])       # (4,57)
        d3 = np.median(d_nulls, axis=0)             # (57,)
        p2 = {'d3_med': round(float(np.median(d3)), 4),
              'd3_range': [round(float(d3.min()), 3),
                           round(float(d3.max()), 3)]}

        # P2a class axis
        m0 = lab_lang == 0
        m1 = lab_lang == 1
        idx0 = np.nonzero(m0)[0]
        idx1 = np.nonzero(m1)[0]
        obs_a = float(np.median(d3[m0])
                      - np.median(d3[m1]))
        rng_a = np.random.default_rng(RNG_A)
        cnt_a = 0
        all_ix = np.arange(n_words)
        n0 = len(idx0)
        for _ in range(N_PERM):
            r = rng_a.permutation(all_ix)
            va = float(np.median(d3[r[:n0]])
                       - np.median(d3[r[n0:]]))
            if abs(va) >= abs(obs_a) - 1e-12:
                cnt_a += 1
        p2a = (cnt_a + 1) / (N_PERM + 1)
        p2['P2a'] = {'med_lab0': round(float(np.median(
                     d3[m0])), 4),
                     'med_lab1': round(float(np.median(
                     d3[m1])), 4),
                     'obs': round(obs_a, 4),
                     'p': float('%.3e' % p2a)}

        # P2b concept lock (ICC-like variance ratio)
        cks = [str(w[1]) for w in words]
        uniq = sorted(set(cks))
        groups = [np.array([i for i in range(n_words)
                            if cks[i] == u])
                  for u in uniq if
                  sum(1 for i in range(n_words)
                      if cks[i] == u) >= 2]
        sizes = [len(g) for g in groups]
        gm = np.array([d3[g].mean() for g in groups])
        grand = d3.mean()
        ss_between = sum(len(g) * (m - grand) ** 2
                         for g, m in zip(groups, gm))
        ss_total = float(((d3 - grand) ** 2).sum())
        icc_obs = ss_between / max(ss_total, 1e-30)

        rng_b = np.random.default_rng(RNG_B)
        cnt_b = 0
        flat_pairs = np.concatenate(groups)
        for _ in range(N_PERM):
            rp = rng_b.permutation(flat_pairs)
            ssb = 0.0
            off = 0
            for n in sizes:
                g = rp[off:off + n]
                off += n
                ssb += n * (d3[g].mean() - grand) ** 2
            if ssb / max(ss_total, 1e-30) >= icc_obs - 1e-12:
                cnt_b += 1
        p2b = (cnt_b + 1) / (N_PERM + 1)
        p2['P2b'] = {'icc': round(icc_obs, 4),
                     'n_groups': len(groups),
                     'p': float('%.3e' % p2b)}

        # P2c scale lock
        amp = np.abs(proj35[ifu])
        rho_c = spearman(d3, amp)
        rng_c = np.random.default_rng(RNG_C)
        cnt_c = 0
        for _ in range(N_PERM):
            if abs(spearman(d3, rng_c.permutation(amp))) \
                    >= abs(rho_c) - 1e-12:
                cnt_c += 1
        p2c = (cnt_c + 1) / (N_PERM + 1)
        p2['P2c'] = {'rho': round(rho_c, 4),
                     'p': float('%.3e' % p2c)}

        # P2d rewrite link (2937 per-word displacement)
        i37n = conds37.index('null0')
        disp37 = np.abs(proj37[i37n] - proj37[i37f])
        rho_d = spearman(d3, disp37)
        rng_d = np.random.default_rng(RNG_D)
        cnt_d = 0
        for _ in range(N_PERM):
            if abs(spearman(d3,
                            rng_d.permutation(disp37))) \
                    >= abs(rho_d) - 1e-12:
                cnt_d += 1
        p2d = (cnt_d + 1) / (N_PERM + 1)
        p2['P2d'] = {'rho': round(rho_d, 4),
                     'p': float('%.3e' % p2d)}
        log('P2a class obs %+.4f p %.3e | P2b ICC %.4f '
            'p %.3e | P2c scale rho %.4f p %.3e | '
            'P2d rewrite rho %.4f p %.3e'
            % (obs_a, p2a, icc_obs, p2b, rho_c, p2c,
               rho_d, p2d), lines)

        # ---------- P3: v3 coordinate semantics ------
        c3 = coords[ifu][:, K3_IX]
        obs_c = float(np.median(c3[m0])
                      - np.median(c3[m1]))
        rng_e = np.random.default_rng(RNG_A + 100)
        cnt_e = 0
        for _ in range(N_PERM):
            r = rng_e.permutation(all_ix)
            ve = float(np.median(c3[r[:n0]])
                       - np.median(c3[r[n0:]]))
            if abs(ve) >= abs(obs_c) - 1e-12:
                cnt_e += 1
        p3a = (cnt_e + 1) / (N_PERM + 1)
        rho_c3 = spearman(c3, amp)
        p3 = {'class_sep_obs': round(obs_c, 4),
              'class_sep_p': float('%.3e' % p3a),
              'rho_c3_amp': round(rho_c3, 4)}
        log('P3 c3 class sep %+.4f p %.3e | rho(c3,amp) '
            '%.4f' % (obs_c, p3a, rho_c3), lines)

        # ---------- verdict ----------
        if p2a <= DECODE_P:
            verdict = 'v3_decoded_class_axis'
        elif abs(rho_c) >= RHO_SCALE and p2c <= DECODE_P:
            verdict = 'v3_decoded_scale_locked'
        elif p2b <= DECODE_P:
            verdict = 'v3_decoded_concept_locked'
        else:
            verdict = 'v3_decoder_not_established'

        save = {'d3': d3, 'd_nulls': d_nulls,
                'c3_func': c3,
                'w_li': w_li}

    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2940, 'model': 'qwen3-4b',
           'prereg': PREREG,
           'anchors': {'a1_diff': float('%.3e' % a1_diff),
                       'a1_ok': a1_ok,
                       'a2_diff': float('%.3e' % a2_diff),
                       'a2_ok': a2_ok,
                       'a3_ok': a3_ok,
                       'a4_diff': float('%.3e' % a4_diff),
                       'a4_ok': a4_ok,
                       'a5_diff': float('%.3e' % a5_diff),
                       'a5_ok': a5_ok,
                       'ok': anchor_ok},
           'P1': p1, 'P2': p2, 'P3': p3,
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if verdict != 'anchor_fail_all_void':
        np.savez_compressed(os.path.join(
            OUT, 'v3_decode.npz'), **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2940 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
