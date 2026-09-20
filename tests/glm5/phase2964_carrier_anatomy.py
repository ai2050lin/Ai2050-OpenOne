# -*- coding: utf-8 -*-
"""Phase 2964: carrier anatomy of the load-band class
effect (preregistered). Plan-v2 stage-2.

Why: 2963 established band_class_effect_beyond_frequency
(function-vs-content band gap ~1.46, Freedman-Lane p
2e-04). Open question: WHERE does the gap live - which
layers and which heads carry it? And does the head-level
gap align with the 2947 head-importance order (D_L17) or
the 2953 early-flipper heads (h20/h21)?

Design (frozen before any observation):
  30 FRESH English single-token words: F15 closed-class
  function words (pronouns/prepositions, tid 566-7241)
  + N15 content nouns (tid 3241-26752). Plus 3 ANCHOR
  words re-forwarded from the 2963 list (excluded from
  all tests). Protocol: 2937 pass1 verbatim single
  forwards [the, w]; captures o_proj input pos1 at all
  36 layers; C[li,w,h] per-head contribution to u35
  readout (2947/2962/2963 spec).

Tests (frozen):
  T1 confirmatory replication: Freedman-Lane on F u N
     (30 words), B ~ rank(tid) + group, residual
     permutation (rng 2968, 10000), two-sided; gate
     p <= 0.01 (if fail => class_effect_not_replicated,
     everything else void).
  T2 layer localization: per-layer gap g_li =
     med_F - med_N (36 layers); maxT family 36 (label
     permutation, rng 2969, 10000); report layers with
     maxT p <= 0.01; concentration = top-3-layer share
     of summed |g_li|.
  T3 head localization: per-head gap at the single
     largest-|gap| layer; maxT family 32 (rng 2970,
     10000); report significant heads.
  T4 descriptive (no gate): spearman(gap_L17, 2947
     D_L17); top-gap-5 intersections with 2947 top5
     {22,19,0,7,10}, 2953 early flippers {20,21},
     2953 keep_L17; group medians; content-internal
     rho(B, tid).

Anchors (frozen):
  a1 Vt8 rebuild from 2927 vs 2939 npz < 1e-6
  a2 determinism < 1e-4
  a3 chunk-vs-direct < 1e-9 at L16/L17 (anchor words)
  a4 single-token 33/33 + fresh 30/30
  a5 B of 3 anchor words vs 2963 npz < 1e-4 rel
  a6 non-degeneracy: per-layer per-word std > 0 (36/36)
     and per-head std at L17 > 0 (32/32)

Verdict (frozen):
  anchor fail => anchor_fail_all_void
  T1 p > 0.01 => class_effect_not_replicated
  T1 pass, n_sig_layers >= 1 and n_sig_heads >= 1
      => carrier_localized_layers_heads
  T1 pass, n_sig_layers >= 1 and n_sig_heads == 0
      => carrier_layer_level_only
  T1 pass, n_sig_layers == 0 => carrier_diffuse_registered
"""
import hashlib
import json
import os
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC_2927 = os.path.join(BASE, 'phase2927', 'probe_relativity',
                        'probe_relativity.npz')
SRC_2939 = os.path.join(BASE, 'phase2939', 'rotation_target',
                        'rotation_target.npz')
SRC_2963 = os.path.join(BASE, 'phase2963',
                        'frequency_controlled_band',
                        'freq_band.npz')
SRC_2947 = os.path.join(BASE, 'phase2947', 'head_anatomy',
                        'head_anatomy.npz')
SRC_2953 = os.path.join(BASE, 'phase2953', 'a11_s_response',
                        'a11_s_response.npz')
OUT = os.path.join(BASE, 'phase2964', 'carrier_anatomy')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2964_run_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NH, HD = 32, 128
NL = 36
N_PERM = 10000
RNG_T1, RNG_T2, RNG_T3 = 2968, 2969, 2970
P_TH = 0.01

FUNC = ['he', 'we', 'us', 'his', 'they', 'their', 'our',
        'her', 'them', 'its', 'she', 'him', 'without',
        'among', 'unless']
NOUN = ['window', 'engine', 'train', 'camera', 'ship',
        'radio', 'cloud', 'island', 'storm', 'forest',
        'ocean', 'sword', 'desert', 'crown', 'temple']
ANCHOR_WORDS = ['people', 'for', 'garden']

PREREG = {
    'mode': '33 single forwards (30 fresh test + 3 anchor '
            'words), 2937 pass1 protocol verbatim, NO '
            'ablation; captures o_proj input pos1 all 36 '
            'layers -> C[li,w,h] per-head contribution',
    'question': 'which layers and heads carry the '
                'function-vs-content load-band class gap '
                '(2963), and do they align with the 2947 '
                'head order / 2953 early flippers?',
    'word_list': {'function': FUNC, 'noun': NOUN,
                  'anchors_from_2963': ANCHOR_WORDS},
    'anchors': {
        'a1': 'Vt8 rebuild vs 2939 npz < 1e-6',
        'a2': 'determinism < 1e-4',
        'a3': 'chunk-vs-direct < 1e-9 L16/L17',
        'a4': 'single-token 33/33 + fresh 30/30',
        'a5': 'B of 3 anchor words vs 2963 npz < 1e-4 rel',
        'a6': 'non-degeneracy 36/36 layers, 32/32 heads',
    },
    'T1': 'Freedman-Lane replication F u N: B ~ rank(tid) '
          '+ group; residual perm rng 2968 x10000; gate '
          'p <= 0.01',
    'T2': 'per-layer gap med_F - med_N, maxT family 36 '
          '(rng 2969 x10000); top-3 concentration',
    'T3': 'per-head gap at largest-|gap| layer, maxT '
          'family 32 (rng 2970 x10000)',
    'T4': 'descriptive: overlap with 2947 D_L17/top5, '
          '2953 flippers {20,21} / keep_L17',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'T1 > 0.01 => class_effect_not_replicated; '
               'T1 pass & sig layers >=1 & sig heads >=1 '
               '=> carrier_localized_layers_heads; T1 '
               'pass & layers >=1 & heads 0 => '
               'carrier_layer_level_only; T1 pass & '
               'layers 0 => carrier_diffuse_registered',
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


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2964,
                   'name': 'carrier_anatomy',
                   'created': time.strftime(
                       '%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2927': sha8(SRC_2927),
                               's2939': sha8(SRC_2939),
                               's2963': sha8(SRC_2963),
                               's2947': sha8(SRC_2947),
                               's2953': sha8(SRC_2953)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'n_perm': N_PERM,
                   'rng': {'T1': RNG_T1, 'T2': RNG_T2,
                           'T3': RNG_T3},
                   'p_threshold': P_TH,
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    # ---------- sources ----------
    z27 = np.load(SRC_2927, allow_pickle=True)
    dirs27 = z27['dirs_word'].astype(np.float64)
    z39 = np.load(SRC_2939, allow_pickle=True)
    U, s_loc, Vt_loc = np.linalg.svd(dirs27,
                                     full_matrices=False)
    a1_diff = float(np.abs(Vt_loc[:8]
                           - z39['Vt8']).max())
    a1_ok = bool(a1_diff < 1e-6)
    log('a1 Vt8 rebuild diff %.2e ok=%s'
        % (a1_diff, a1_ok), lines)
    u35 = dirs27[NL - 1]
    z63 = np.load(SRC_2963, allow_pickle=True)
    w63 = [str(w).split(':') for w in z63['words']]
    B63 = z63['B'].astype(np.float64)
    z47 = np.load(SRC_2947, allow_pickle=True)
    D17_47 = z47['D_L17'].astype(np.float64)
    z53 = np.load(SRC_2953, allow_pickle=True)
    keep17_53 = z53['keep_L17'].astype(int)

    # ---------- fresh-list data ----------
    prior_words = set(w for g, w in w63)
    import io as _io
    z87 = np.load(os.path.join(BASE, 'phase2887',
                               'language_axis_mlp',
                               'language_axis_mlp.npz'),
                  allow_pickle=True)
    prior_words |= set(str(w).split(':')[2]
                       for w in z87['words'])

    # ---------- model ----------
    import torch
    import sys
    sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
    from phase2662_symmetric_mapping_contract import \
        load_native
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        MD, local_files_only=True, trust_remote_code=True,
        use_fast=True)
    test_groups = ([('function', w) for w in FUNC]
                   + [('noun', w) for w in NOUN])
    tid_map = {}
    n_single = 0
    n_fresh = 0
    for _, w in test_groups + [
            ('anchor', w) for w in ANCHOR_WORDS]:
        ids = tok(' ' + w, add_special_tokens=False)[
            'input_ids']
        if len(ids) != 1:
            ids = tok(w, add_special_tokens=False)[
                'input_ids']
        assert len(ids) == 1, '%s -> %s' % (w, ids)
        tid_map[w] = int(ids[0])
        n_single += 1
        if w not in prior_words:
            n_fresh += 1
    a4_ok = bool(n_single == 33 and n_fresh == 30)
    log('a4 single-token %d/33, fresh %d/30 ok=%s'
        % (n_single, n_fresh, a4_ok), lines)
    ids_the = tok(' the', add_special_tokens=False)[
        'input_ids']
    assert len(ids_the) == 1
    func_tid = int(ids_the[0])

    model, _ = load_native('qwen4')
    model.eval()
    layers = model.model.layers
    log('model loaded', lines)

    cap_op = {li: [] for li in range(NL)}
    fin_cap = {}
    state_fin = {'on': False}
    handles = []

    def hook_op(li):
        def h(module, args, kwargs):
            x = args[0] if args else kwargs.get('input')
            if x is None or x.dim() < 2:
                return None
            cap_op[li].append(
                x[:, 1, :].detach().float().cpu().numpy())
            return None
        return h

    def pre_norm(module, args, kwargs):
        if state_fin['on']:
            fin_cap['x'] = args[0][:, -1, :].detach() \
                .float().cpu().numpy()

    for li in range(NL):
        handles.append(
            layers[li].self_attn.o_proj
            .register_forward_pre_hook(
                hook_op(li), with_kwargs=True))
    handles.append(model.model.norm
                   .register_forward_pre_hook(
                       pre_norm, with_kwargs=True))

    def clear_cap():
        for li in cap_op:
            del cap_op[li][:]

    def forward1(toks):
        clear_cap()
        fin_cap.pop('x', None)
        state_fin['on'] = True
        with torch.no_grad():
            model(torch.tensor([toks], device='cuda'))
        state_fin['on'] = False
        return (fin_cap['x'].astype(np.float64),
                {li: cap_op[li][0].astype(np.float64)
                 for li in range(NL)})

    M = np.zeros((NL, NH * HD))
    for li in range(NL):
        Wo = layers[li].self_attn.o_proj.weight.detach() \
            .float().cpu().numpy()
        M[li] = u35 @ Wo

    def contributions(op):
        C = np.zeros((NL, NH))
        prof = np.zeros(NL)
        for li in range(NL):
            x = op[li].reshape(-1)
            xm = (x * M[li]).reshape(NH, HD)
            C[li] = xm.sum(axis=1)
            prof[li] = float(C[li].sum())
        return C, prof

    def band_of(prof):
        return (float(prof[6:13].mean())
                - float(prof[28:36].mean()))

    # ---------- a2 determinism ----------
    fin_a, _ = forward1([func_tid, tid_map['people']])
    fin_b, _ = forward1([func_tid, tid_map['people']])
    a2_rel = float(np.abs(fin_a - fin_b).max()
                   / max(float(np.abs(fin_a).max()),
                         1e-30))
    a2_ok = bool(a2_rel < 1e-4)
    log('a2 determinism rel %.2e ok=%s'
        % (a2_rel, a2_ok), lines)

    # ---------- anchor forwards (a3/a5) ----------
    a5_rel = 0.0
    a3_rel = 0.0
    for w in ANCHOR_WORDS:
        fin, op = forward1([func_tid, tid_map[w]])
        _, prof = contributions(op)
        i63 = [i for i, (g, ww) in enumerate(w63)
               if ww == w][0]
        a5_rel = max(a5_rel,
                     abs(band_of(prof) - B63[i63])
                     / max(abs(B63[i63]), 1e-30))
        for li in (16, 17):
            x = op[li].reshape(-1)
            direct = float(np.dot(x, M[li]))
            xm = (x * M[li]).reshape(NH, HD)
            a3_rel = max(a3_rel,
                         abs(direct - float(xm.sum()))
                         / max(abs(direct), 1e-30))
    a5_ok = bool(a5_rel < 1e-4)
    a3_ok = bool(a3_rel < 1e-9)
    log('a3 chunk-vs-direct rel %.2e ok=%s | a5 B vs '
        '2963 rel %.2e ok=%s'
        % (a3_rel, a3_ok, a5_rel, a5_ok), lines)

    # ---------- main sweep: 30 test words ----------
    C_all = np.zeros((NL, 30, NH))
    B_all = np.zeros(30)
    for i, (_, w) in enumerate(test_groups):
        fin, op = forward1([func_tid, tid_map[w]])
        C, prof = contributions(op)
        C_all[:, i, :] = C
        B_all[i] = band_of(prof)
        if (i + 1) % 10 == 0:
            log('sweep [%d/30]' % (i + 1), lines)

    # ---------- a6 non-degeneracy ----------
    lab = np.array([0] * 15 + [1] * 15)
    mF = lab == 0
    mN = lab == 1
    a6_ok = bool(all(C_all[li].std(axis=0).min() > 0
                     for li in range(NL))
                 and C_all[17].std(axis=0).min() > 0)
    log('a6 non-degeneracy ok=%s (C17 min std %.3e)'
        % (a6_ok, C_all[17].std(axis=0).min()), lines)

    anchor_ok = bool(a1_ok and a2_ok and a3_ok and a4_ok
                     and a5_ok and a6_ok)
    verdict = None
    t1 = t2 = t3 = t4 = None

    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        tid_test = np.array([tid_map[w]
                             for _, w in test_groups])

        # ---------- T1 Freedman-Lane replication ----
        x_fc = rankdata(tid_test.astype(np.float64))
        x_fc = (x_fc - x_fc.mean()) / x_fc.std()
        g_fc = mN.astype(np.float64)
        Xr = np.stack([np.ones(30), x_fc], axis=1)
        beta_r, *_ = np.linalg.lstsq(Xr, B_all,
                                     rcond=None)
        resid = B_all - Xr @ beta_r
        Xf = np.stack([np.ones(30), x_fc, g_fc], axis=1)

        def full_coef(bb):
            beta, *_ = np.linalg.lstsq(Xf, bb,
                                       rcond=None)
            return float(beta[2])

        obs1 = full_coef(B_all)
        rng1 = np.random.default_rng(RNG_T1)
        cnt1 = 0
        for _ in range(N_PERM):
            ep = rng1.permutation(resid)
            if abs(full_coef(Xr @ beta_r + ep)) \
                    >= abs(obs1) - 1e-12:
                cnt1 += 1
        p1 = (cnt1 + 1) / (N_PERM + 1)
        t1 = {'coef_group': round(float(obs1), 4),
              'p_freedman_lane': float('%.3e' % p1),
              'med_B_F': round(float(np.median(
                  B_all[mF])), 4),
              'med_B_N': round(float(np.median(
                  B_all[mN])), 4)}
        log('T1 FL coef %+.4f p %.3e | medB F %.4f N %.4f'
            % (obs1, p1, t1['med_B_F'], t1['med_B_N']),
            lines)

        if p1 <= P_TH:
            # ---------- T2 layer localization --------
            gap_l = np.array([float(np.median(
                C_all[li][mF].sum(axis=1))
                - np.median(C_all[li][mN].sum(axis=1)))
                for li in range(NL)])

            def gap_stats(mat30):
                return np.array([float(np.median(
                    mat30[li][mF].sum(axis=1))
                    - np.median(
                    mat30[li][mN].sum(axis=1)))
                    for li in range(NL)])

            rng2 = np.random.default_rng(RNG_T2)
            fam2 = np.zeros(N_PERM)
            for k in range(N_PERM):
                r = rng2.permutation(30)
                gs = gap_stats(C_all[:, r, :])
                fam2[k] = float(np.abs(gs).max())
            thr2 = float(np.quantile(
                fam2, 1 - P_TH))
            sig_layers = [li for li in range(NL)
                          if abs(gap_l[li]) >= thr2]
            order_l = np.argsort(-np.abs(gap_l))
            conc3 = (float(np.abs(
                gap_l[order_l[:3]]).sum())
                / max(float(np.abs(gap_l).sum()),
                      1e-30))
            t2 = {'n_sig_layers': len(sig_layers),
                  'sig_layers': sig_layers,
                  'top5_layers': [
                      (int(li),
                       round(float(gap_l[li]), 4))
                      for li in order_l[:5]],
                  'top3_concentration':
                      round(conc3, 4),
                  'maxT_thr': round(thr2, 4)}
            log('T2 sig layers %s | top5 %s | conc3 %.4f'
                % (sig_layers, t2['top5_layers'],
                   conc3), lines)

            # ---------- T3 head localization --------
            li_top = int(order_l[0])
            gap_h = np.array([float(np.median(
                C_all[li_top][mF][:, h])
                - np.median(C_all[li_top][mN][:, h]))
                for h in range(NH)])

            def head_stats(mat30):
                return np.array([float(np.median(
                    mat30[mF][:, h])
                    - np.median(mat30[mN][:, h]))
                    for h in range(NH)])

            rng3 = np.random.default_rng(RNG_T3)
            fam3 = np.zeros(N_PERM)
            for k in range(N_PERM):
                r = rng3.permutation(30)
                hs = head_stats(C_all[li_top][r, :])
                fam3[k] = float(np.abs(hs).max())
            thr3 = float(np.quantile(fam3, 1 - P_TH))
            sig_heads = [h for h in range(NH)
                         if abs(gap_h[h]) >= thr3]
            order_h = np.argsort(-np.abs(gap_h))
            top5_h = [int(h) for h in order_h[:5]]
            t3 = {'layer': li_top,
                  'n_sig_heads': len(sig_heads),
                  'sig_heads': sig_heads,
                  'top5_heads': [
                      (h, round(float(gap_h[h]), 4))
                      for h in top5_h],
                  'maxT_thr': round(thr3, 4)}
            log('T3 layer %d sig heads %s | top5 %s'
                % (li_top, sig_heads, t3['top5_heads']),
                lines)

            # ---------- T4 descriptive overlaps -----
            gap17 = np.array([float(np.median(
                C_all[17][mF][:, h])
                - np.median(C_all[17][mN][:, h]))
                for h in range(NH)])
            rho_d17 = spearman(gap17, D17_47)
            top5_47 = {22, 19, 0, 7, 10}
            t4 = {'spearman_gap17_D17_2947':
                      round(float(rho_d17), 4),
                  'top5_heads_this': top5_h,
                  'intersect_2947_top5': sorted(
                      set(top5_h) & top5_47),
                  'intersect_2953_flippers':
                      sorted(set(top5_h) & {20, 21}),
                  'intersect_2953_keep17': sorted(
                      set(top5_h)
                      & set(int(k) for k
                            in keep17_53)),
                  'rho_B_tid_content': round(float(
                      spearman(B_all[mN],
                               tid_test[mN])), 4)}
            log('T4 rho(gap17, D2947) %.4f | top5x2947 %s '
                'xflippers %s xkeep %s'
                % (rho_d17, t4['intersect_2947_top5'],
                   t4['intersect_2953_flippers'],
                   t4['intersect_2953_keep17']),
                lines)

            # ---------- verdict ----------
            if len(sig_layers) >= 1 \
                    and len(sig_heads) >= 1:
                verdict = \
                    'carrier_localized_layers_heads'
            elif len(sig_layers) >= 1:
                verdict = 'carrier_layer_level_only'
            else:
                verdict = 'carrier_diffuse_registered'
            save = {'C': C_all.astype(np.float32),
                    'B': B_all, 'tids': tid_test,
                    'labels': lab,
                    'words': np.array(
                        ['%s:%s' % (g, w)
                         for g, w in test_groups]),
                    'gap_layers': gap_l,
                    'gap_heads_L%d' % li_top: gap_h}
        else:
            verdict = 'class_effect_not_replicated'
            t2 = t3 = t4 = None
            save = {'C': C_all.astype(np.float32),
                    'B': B_all, 'tids': tid_test,
                    'labels': lab,
                    'words': np.array(
                        ['%s:%s' % (g, w)
                         for g, w in test_groups])}

    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2964, 'model': 'qwen3-4b',
           'prereg': PREREG,
           'anchors': {'a1_diff': float('%.3e' % a1_diff),
                       'a1_ok': a1_ok,
                       'a2_rel': float('%.3e' % a2_rel),
                       'a2_ok': a2_ok,
                       'a3_rel': float('%.3e' % a3_rel),
                       'a3_ok': a3_ok,
                       'a4_ok': a4_ok,
                       'a5_rel': float('%.3e' % a5_rel),
                       'a5_ok': a5_ok,
                       'a6_ok': a6_ok,
                       'ok': anchor_ok},
           'T1': t1, 'T2': t2, 'T3': t3, 'T4': t4,
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if verdict != 'anchor_fail_all_void':
        np.savez_compressed(os.path.join(
            OUT, 'carrier_anatomy.npz'), **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2964 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
