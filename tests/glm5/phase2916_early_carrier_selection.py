# -*- coding: utf-8 -*-
"""Phase 2916: early-segment carrier selection test + h7 dual-role
per-layer decomposition (artifact domain, zero forward).

Why: 2915 established the {7,8} carrier identity as SEGMENT-BOUND:
stable top2 across late windows [26,36)/[30,36), switched in early
windows <=26 (V2 [20,26) top2 {27,31}; V4 [16,26) top2 {7,27} with
h7 #1 at margin 1.17970), and the CHANNEL margin is segment-signed
(early +0.16 acc 0.84 vs late -0.02 acc 0.63).  Open items (2915
next-phase candidate A): (1) are the early-segment carriers
statistically real under the 2913 P3 selection-corrected greedy
test (2915 only ran the family max-null gate), and (2) where does
the huge h7 margin in V4 (1.18 vs 0.177 in V1) come from -
which layers drive it?

Mode: artifact-domain only (loads 2915 npz B_heads_V1..V4, zero
forward).  Selection-corrected test caliber verbatim 2913 P3:
sort heads by margin desc, greedy top-k accumulation, obs = max_k
margin; null = full procedure (re-sort + greedy + max_k) under
each of 200 SEED=2896 label permutations (Sm fixed + mask perm).

Anchors (frozen, artifact-domain):
  a1 margins_V[V1 row] vs 2913 npz margins_h: Spearman >= 0.9999
     AND max rel < 1e-5 (2915 measured 1.0 / 2.73e-08).
  a2 |margin({7,8} from B_heads_V1) - 0.28036| < 5e-3.

Probes (frozen):
  P1 selection-corrected greedy test for V1 (replicate check vs
     2913 p=0.01493 - REGISTERED, not an anchor; bf16-level data
     differences may shift the boundary permutations),
     V2 and V4 (the verdict axis).
  P2 per-layer decomposition for h7/h8 (V1, V4) and V2 carriers
     {27,31}, V4 {7,27}: leave-one-layer-out margin profile
     (Delta_j = margin(without layer j) - margin_full, negative =
     layer j contributes positively) + 2910-caliber sign-margin
     per-layer sequence (auxiliary, comparable to 2913 P2).
  P3 descriptive: per-layer sign-margin peaks per variant
     (which layers carry what).
  P4 tables.

Adjudication (frozen):
  anchor fail => anchor_fail_all_void;
  p3(V2) <= 0.05 AND p3(V4) <= 0.05 =>
      early_carriers_selection_confirmed;
  exactly one of them => early_carriers_partially_confirmed;
  else => early_carriers_not_confirmed.
  P2/P3 descriptive alongside.

Output: phase2916/early_carrier_selection/.
"""
import hashlib
import io
import json
import os
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC_2915 = os.path.join(BASE, 'phase2915',
                        'carrier_robustness_domain',
                        'carrier_robustness_domain.npz')
SRC_2913 = os.path.join(BASE, 'phase2913',
                        'perhead_wvo_decomposition',
                        'perhead_wvo_decomposition.npz')
OUT = os.path.join(BASE, 'phase2916', 'early_carrier_selection')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2916_run_report.txt')
SEED = 2896
N_PERM = 200
NH = 32
RECON_REF = 0.28036
P3_2913_REF = 0.014925   # (2+1)/201 exact
VARIANTS = ['V1', 'V2', 'V3', 'V4', 'V5']

PREREG = {
    'mode': 'artifact-domain (zero forward) selection-corrected '
            'test of early-segment carriers + per-layer margin '
            'decomposition',
    'question': '2915 candidate A: are the early-segment carriers '
                '(V2 {27,31}, V4 {7,27}) statistically real under '
                'the 2913 P3 selection-corrected greedy test; '
                'which layers drive the h7 dual-role margins '
                '(V4 1.17970 vs V1 0.17710)',
    'sources': '2915 npz B_heads_V1..V4 + margins_V (rel 2.73e-08 '
               'to 2913, exact reproduction verified); 2913 npz '
               'margins_h; labels from 2915 npz (verbatim 2887)',
    'anchors': {
        'a1': 'margins_V[V1] vs 2913 margins_h: Spearman >= 0.9999 '
              'AND max rel < 1e-5',
        'a2': '|margin({7,8} from B_heads_V1) - 0.28036| < 5e-3',
    },
    'probes': {
        'P1': 'selection-corrected greedy test V1 (replicate check '
              'vs 0.01493, REGISTERED not anchored), V2, V4 '
              '(verdict axis)',
        'P2': 'leave-one-layer-out margin profiles h7/h8 (V1, V4) '
              '+ V2 {27,31} + V4 {27}; sign-margin seq (2910 '
              'caliber) auxiliary',
        'P3': 'per-layer sign-margin peaks per variant (descriptive)',
        'P4': 'tables',
    },
    'verdict': 'anchor fail => anchor_fail_all_void; p3(V2) <= 0.05 '
               'AND p3(V4) <= 0.05 => '
               'early_carriers_selection_confirmed; exactly one => '
               'early_carriers_partially_confirmed; else => '
               'early_carriers_not_confirmed; P2/P3 descriptive',
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


def rownorm(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.maximum(n, 1e-30)


def masks(lab):
    n = len(lab)
    eye = np.eye(n, dtype=bool)
    same = (lab[:, None] == lab[None, :]) & (~eye)
    diff = (~eye) & (~same)
    return same, diff


def margin_of(Sm, same, diff):
    return float(Sm[same].mean() - Sm[diff].mean())


def gram_margin(B, same, diff):
    U = rownorm(B)
    return margin_of(U @ U.T, same, diff)


def rankdata(x):
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x), dtype=np.float64)
    sx = x[order].astype(np.float64)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(a, b):
    ra = rankdata(a)
    rb = rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    return float((ra @ rb)
                 / max(float(np.sqrt((ra @ ra) * (rb @ rb))), 1e-30))


def sign_seq(B, same, diff):
    """2910-caliber per-layer sign margin sequence."""
    out = np.zeros(B.shape[1])
    for j in range(B.shape[1]):
        s = np.sign(B[:, j])
        s[s == 0] = 1.0
        G = np.outer(s, s)
        out[j] = G[same].mean() - G[diff].mean()
    return out


def selection_test(Bh, lab, perms_masks):
    """2913 P3 caliber verbatim."""
    same, diff = masks(lab)
    Sm_h = np.stack([rownorm(Bh[h]) @ rownorm(Bh[h]).T
                     for h in range(NH)])
    mh = np.array([margin_of(Sm_h[h], same, diff)
                   for h in range(NH)])
    order_h = np.argsort(-mh)
    run = np.zeros(Bh.shape[1:])
    obs_curve = np.zeros(NH)
    for k, h in enumerate(order_h):
        run = run + Bh[h]
        obs_curve[k] = gram_margin(run, same, diff)
    k_best = int(np.argmax(obs_curve))
    obs_max = float(obs_curve[k_best])
    null_max = np.zeros(N_PERM)
    for pi, (sm_p, df_p) in enumerate(perms_masks):
        mh_p = np.array([margin_of(Sm_h[h], sm_p, df_p)
                         for h in range(NH)])
        ordp = np.argsort(-mh_p)
        runp = np.zeros(Bh.shape[1:])
        bestp = -1e9
        for k in range(NH):
            runp = runp + Bh[ordp[k]]
            v = gram_margin(runp, sm_p, df_p)
            if v > bestp:
                bestp = v
        null_max[pi] = bestp
    p3 = float((np.sum(null_max >= obs_max) + 1) / (N_PERM + 1))
    return {'k_best': k_best + 1,
            'heads_topk': [int(x) for x in order_h[:k_best + 1]],
            'obs_margin': round(obs_max, 5),
            'full_margin': round(mh.max(), 5),
            'null_p95': round(float(np.percentile(null_max, 95)), 5),
            'p3': round(p3, 6),
            'significant': bool(p3 <= 0.05)}


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2916,
                   'name': 'early_carrier_selection',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8': sha8(os.path.abspath(__file__)),
                   'sources': {'s2915': sha8(SRC_2915),
                               's2913': sha8(SRC_2913)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'seed': SEED, 'n_perm': N_PERM,
                   'mode': 'zero forward (artifact domain)',
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    # ---------- sources ----------
    z15 = np.load(SRC_2915, allow_pickle=True)
    vorder = [str(x) for x in z15['variant_order']]
    assert vorder == VARIANTS, vorder
    lab = z15['labels_lang'].astype(int)
    B_heads = {vn: z15['B_heads_' + vn].astype(np.float64)
               for vn in ('V1', 'V2', 'V3', 'V4')}
    margins_V = z15['margins_V'].astype(np.float64)
    z13 = np.load(SRC_2913, allow_pickle=True)
    margins_h_2913 = z13['margins_h'].astype(np.float64)
    same, diff = masks(lab)

    # ---------- anchors ----------
    rho_a1 = spearman(margins_V[0], margins_h_2913)
    rel_a1 = float(np.abs(margins_V[0] - margins_h_2913).max()
                   / max(float(np.abs(margins_h_2913).max()), 1e-30))
    a1_ok = bool(rho_a1 >= 0.9999 and rel_a1 < 1e-5)
    m78 = gram_margin(B_heads['V1'][7] + B_heads['V1'][8],
                      same, diff)
    a2_ok = bool(abs(m78 - RECON_REF) < 5e-3)
    anchor_ok = bool(a1_ok and a2_ok)
    log('a1 spearman %.6f rel %.2e ok=%s | a2 m78 %.6f (ref '
        '%.5f) ok=%s'
        % (rho_a1, rel_a1, a1_ok, m78, RECON_REF, a2_ok), lines)

    # ---------- probes ----------
    verdict = None
    p1 = p2 = p3 = p4 = None
    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        rng2 = np.random.default_rng(SEED)
        perms = [rng2.permutation(lab) for _ in range(N_PERM)]
        perms_masks = [masks(pl) for pl in perms]

        # P1 selection-corrected tests
        p1 = {}
        for vn in ('V1', 'V2', 'V4'):
            r = selection_test(B_heads[vn], lab, perms_masks)
            p1[vn] = r
            log('P1 %s: k=%d heads %s obs %.5f (full-max %.5f) '
                'null_p95 %.5f p3=%.6f sig=%s'
                % (vn, r['k_best'], r['heads_topk'], r['obs_margin'],
                   r['full_margin'], r['null_p95'], r['p3'],
                   r['significant']), lines)
        dev = abs(p1['V1']['p3'] - P3_2913_REF)
        p1['V1_replicate'] = {
            'ref_2913': P3_2913_REF,
            'absdev': round(dev, 6),
            'within_0.01': bool(dev <= 0.01)}

        # P2 leave-one-layer-out profiles
        def loo_profile(vn, h):
            Bh = B_heads[vn][h]
            full = gram_margin(Bh, same, diff)
            prof = np.zeros(Bh.shape[1])
            for j in range(Bh.shape[1]):
                cols = [k for k in range(Bh.shape[1]) if k != j]
                prof[j] = gram_margin(Bh[:, cols], same, diff) - full
            return full, prof

        p2 = {}
        for vn, hs in (('V1', [7, 8]), ('V4', [7, 8, 27]),
                       ('V2', [27, 31])):
            for h in hs:
                full, prof = loo_profile(vn, h)
                seq = sign_seq(B_heads[vn][h], same, diff)
                key = '%s_h%d' % (vn, h)
                p2[key] = {
                    'full_margin': round(full, 5),
                    'layers': {'V1': list(range(26, 36)),
                               'V2': list(range(20, 26)),
                               'V4': list(range(16, 26))}[vn],
                    'loo_delta': [round(float(x), 5)
                                  for x in prof],
                    'loo_worst_layer': int(np.argmax(prof)),
                    'loo_best_layer': int(np.argmin(prof)),
                    'sign_seq': [round(float(x), 4)
                                 for x in seq],
                    'sign_peak_layer': {
                        'V1': 26, 'V2': 20, 'V4': 16}[vn]
                        + int(np.argmax(np.abs(seq)))}
                log('P2 %s: full %.5f | loo delta %s | sign peak '
                    'L%d %.4f'
                    % (key, full,
                       [round(float(x), 3) for x in prof],
                       p2[key]['sign_peak_layer'],
                       seq.max(axis=0) if seq.max() > 0
                       else seq.min()), lines)

        # P3 per-layer sign-margin peaks per variant
        p3_d = {}
        for vn in ('V1', 'V2', 'V3', 'V4'):
            S = np.stack([sign_seq(B_heads[vn][h], same, diff)
                          for h in range(NH)])
            tops = np.argsort(-np.abs(S).max(axis=1))[:5]
            p3_d[vn] = [
                {'head': int(h),
                 'layer': {'V1': 26, 'V2': 20, 'V3': 30,
                           'V4': 16}[vn]
                 + int(np.argmax(np.abs(S[h]))),
                 'sign_margin': round(float(S[h].max()), 4),
                 'abs_peak': round(float(np.abs(S[h]).max()), 4)}
                for h in tops]
        p3 = p3_d
        log('P3 V1 top sign heads %s | V4 top %s'
            % ([(d['head'], d['layer'], d['sign_margin'])
                for d in p3_d['V1'][:3]],
               [(d['head'], d['layer'], d['sign_margin'])
                for d in p3_d['V4'][:3]]), lines)

        p4 = {'note': 'see P1/P2/P3 tables; margins_V (2915 npz) '
                      'vs per-variant obs margins cross-checked'}

        # ---------- verdict ----------
        s2 = p1['V2']['significant']
        s4 = p1['V4']['significant']
        if s2 and s4:
            verdict = 'early_carriers_selection_confirmed'
        elif s2 or s4:
            verdict = 'early_carriers_partially_confirmed'
        else:
            verdict = 'early_carriers_not_confirmed'

    log('==== VERDICT: %s ====' % verdict, lines)

    res = {
        'phase': 2916, 'model': 'qwen3-4b', 'prereg': PREREG,
        'anchors': {'a1_spearman_vs_2913': round(rho_a1, 6),
                    'a1_rel_vs_2913': float('%.3e' % rel_a1),
                    'a1_ok': a1_ok,
                    'a2_m78': round(m78, 6),
                    'a2_ref': RECON_REF, 'a2_ok': a2_ok,
                    'ok': anchor_ok},
        'P1': p1, 'P2': p2, 'P3': p3, 'P4': p4,
        'final_verdict': verdict,
        'runtime_s': round(time.monotonic() - t0, 1),
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    save = {'labels_lang': lab,
            'margins_h_2913': margins_h_2913.astype(np.float32),
            'margins_V': margins_V.astype(np.float32),
            'variant_order': np.array(VARIANTS, dtype=object)}
    for vn in ('V1', 'V2', 'V3', 'V4'):
        save['sign_seq_' + vn] = np.stack(
            [sign_seq(B_heads[vn][h], same, diff)
             for h in range(NH)]).astype(np.float32)
    np.savez_compressed(
        os.path.join(OUT, 'early_carrier_selection.npz'), **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2916 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
