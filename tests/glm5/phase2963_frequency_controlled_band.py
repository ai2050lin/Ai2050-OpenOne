# -*- coding: utf-8 -*-
"""Phase 2963: frequency-controlled recheck of the
function-vs-content load-band gap (preregistered).
Plan-v2 stage-2, follows 2962.

Why: 2962 found B(band profile) descriptive gap
function +1.23 vs content CONFOUNDED by token id
(rho(tokid, B) = -0.605). This phase tests the gap
with frequency control on a FRESH word list.

Design (frozen before any B observation):
  45 fresh English single-token words, 3 groups x 15:
  F function (tid 369-3425), C common nouns
  (tid 1251-4627, interleaved with F), R rare nouns
  (tid 13551-46118). Plus 5 ANCHOR words re-used from
  the 2962 list (excluded from all tests): protocol
  identity anchor vs 2962 npz B values.
  Protocol: 2937 pass1 verbatim single forwards
  [the, w]; captures o_proj input pos1 (all 36 layers)
  + final-norm input; B(w) = mean prof[L6-12] -
  mean prof[L28-35], prof = sum_h per-head contribution
  (2947/2962 spec).

Tests (frozen):
  T1 primary, Freedman-Lane permutation on F u C
     (30 words): full model B ~ rank(tid) + group,
     reduced B ~ rank(tid); stat = group coefficient;
     permute reduced residuals (rng 2965, 10000),
     two-sided p.
  T2 within-content frequency gradient: Spearman(B,
     tid) over C u R (30 content words), tid
     permutation (rng 2967, 10000). Strong = |rho| >= 0.4
     and p <= 0.01.
  T3 secondary corroboration: tid-stratified block
     permutation over F u C (sorted by tid, blocks of 6,
     labels permuted within blocks, rng 2966); VALID
     gate >= 4/5 mixed blocks (else registered invalid,
     no gate).
  T4 descriptive (no gate): group medians F/C/R;
     content regression predicted B at F tids
     (extrapolation caveat); rho(tokid, B) over all 45.

Anchors (frozen):
  a1 Vt8 rebuild from 2927 dirs_word vs 2939 npz < 1e-6
  a2 determinism (repeat anchor 'apple' forward) < 1e-4
  a3 chunk-vs-direct < 1e-9 at L16/L17 (3 anchor words)
  a4 single-token 45/45 AND fresh-list check (no word
     in 2962 lists or 2887 57-word list)
  a5 cross-phase protocol anchor: B of 5 anchor words
     vs 2962 npz < 1e-4 relative
  a6 non-degeneracy: per-word std C[17] > 0 (45/45),
     std B > 0 within each group

Verdict (frozen):
  anchor fail => anchor_fail_all_void
  T1 p <= 0.01 and T2 weak => band_class_effect_beyond_frequency
  T1 p <= 0.01 and T2 strong => band_gap_frequency_confounded
  T1 p > 0.05 => band_gap_frequency_controlled_away
  else => band_gap_mixed_registered
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
SRC_2962 = os.path.join(BASE, 'phase2962',
                        'word_class_signature_matrix',
                        'signature_matrix.npz')
OUT = os.path.join(BASE, 'phase2963',
                   'frequency_controlled_band')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2963_run_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NH, HD = 32, 128
NL = 36
N_PERM = 10000
RNG_T1, RNG_T3, RNG_T2 = 2965, 2966, 2967
BAND_LOAD = (6, 12)
BAND_DEEP = (28, 35)
STRONG_RHO, P_TH, P_NULL_TH = 0.4, 0.01, 0.05

FUNCTION = ['for', 'this', 'it', 'from', 'by', 'so',
            'which', 'over', 'than', 'under', 'after',
            'before', 'between', 'against', 'whether']
COMMON = ['people', 'game', 'world', 'color', 'power',
          'body', 'school', 'family', 'money', 'story',
          'word', 'food', 'fire', 'friend', 'music']
RARE = ['garden', 'basket', 'bridge', 'guitar', 'anchor',
        'pocket', 'mirror', 'hammer', 'pepper', 'tunnel',
        'drawer', 'ladder', 'candle', 'ribbon', 'pencil']
ANCHOR_WORDS = ['apple', 'the', 'freedom', 'clock', 'idea']

PREREG = {
    'mode': '50 single forwards (45 test + 5 anchor '
            'words), 2937 pass1 protocol verbatim, NO '
            'ablation; B = band profile 2962 spec',
    'question': 'is the function-vs-content load-band '
                'gap a word-CLASS effect beyond token-id '
                'frequency, or a frequency artifact?',
    'word_list': {'function': FUNCTION, 'common': COMMON,
                  'rare': RARE, 'anchors_from_2962':
                      ANCHOR_WORDS},
    'anchors': {
        'a1': 'Vt8 rebuild vs 2939 npz < 1e-6',
        'a2': 'determinism repeat apple < 1e-4',
        'a3': 'chunk-vs-direct < 1e-9 L16/L17',
        'a4': 'single-token 45/45 + fresh-list check',
        'a5': 'B of 5 anchor words vs 2962 npz < 1e-4 '
              'relative',
        'a6': 'non-degeneracy gates',
    },
    'T1': 'Freedman-Lane F u C: B ~ rank(tid) + group '
          'vs B ~ rank(tid); residual permutation rng '
          '2965 x10000 two-sided',
    'T2': 'Spearman(B, tid) over C u R, tid perm rng '
          '2967 x10000; strong = |rho|>=0.4 and p<=0.01',
    'T3': 'tid-stratified block permutation (blocks of '
          '6, >=4/5 mixed gate) rng 2966 x10000; '
          'secondary',
    'T4': 'descriptive: group medians, content-'
          'regression extrapolation to F tids, '
          'rho(tokid,B) all 45',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'T1<=0.01 and T2 weak => '
               'band_class_effect_beyond_frequency; '
               'T1<=0.01 and T2 strong => '
               'band_gap_frequency_confounded; T1>0.05 => '
               'band_gap_frequency_controlled_away; else '
               '=> band_gap_mixed_registered',
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
        json.dump({'phase': 2963,
                   'name': 'frequency_controlled_band',
                   'created': time.strftime(
                       '%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2927': sha8(SRC_2927),
                               's2939': sha8(SRC_2939),
                               's2962': sha8(SRC_2962)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'n_perm': N_PERM,
                   'rng': {'T1': RNG_T1, 'T3': RNG_T3,
                           'T2': RNG_T2},
                   'thresholds': {'strong_rho': STRONG_RHO,
                                  'primary': P_TH,
                                  'null': P_NULL_TH},
                   'bands': {'load': list(BAND_LOAD),
                             'deep': list(BAND_DEEP)},
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    # ---------- sources ----------
    z27 = np.load(SRC_2927, allow_pickle=True)
    dirs27 = z27['dirs_word'].astype(np.float64)
    z39 = np.load(SRC_2939, allow_pickle=True)
    _ = z39['Vt8'].astype(np.float64)   # source-chain pin
    U, s_loc, Vt_loc = np.linalg.svd(dirs27,
                                     full_matrices=False)
    a1_diff = float(np.abs(Vt_loc[:8]
                           - z39['Vt8']).max())
    a1_ok = bool(a1_diff < 1e-6)
    log('a1 Vt8 rebuild diff %.2e ok=%s'
        % (a1_diff, a1_ok), lines)
    u35 = dirs27[NL - 1]

    z62 = np.load(SRC_2962, allow_pickle=True)
    w62 = [str(w).split(':') for w in z62['words']]
    B62 = z62['B'].astype(np.float64)
    tid62 = z62['tids'].astype(int)

    # ---------- fresh-list check data ----------
    prior_words = {}
    for (g, w), t in zip(w62, tid62):
        prior_words[w] = (g, int(t))
    z87 = np.load(os.path.join(BASE, 'phase2887',
                               'language_axis_mlp',
                               'language_axis_mlp.npz'),
                  allow_pickle=True)
    for wtok in z87['words']:
        prior_words[str(wtok).split(':')[2]] = ('old57',
                                                -1)

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
    test_groups = ([('function', w) for w in FUNCTION]
                   + [('common', w) for w in COMMON]
                   + [('rare', w) for w in RARE])
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
    a4_ok = bool(n_single == 50 and n_fresh == 45)
    log('a4 single-token %d/50, fresh %d/45 ok=%s'
        % (n_single, n_fresh, a4_ok), lines)
    func_tid = tid_map['the']

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

    def band_of(op):
        prof = np.zeros(NL)
        for li in range(NL):
            x = op[li].reshape(-1)
            prof[li] = float((x * M[li]).sum())
        return (float(prof[BAND_LOAD[0]:BAND_LOAD[1] + 1]
                      .mean())
                - float(prof[BAND_DEEP[0]:BAND_DEEP[1] + 1]
                        .mean())), prof

    # ---------- a2 determinism ----------
    fin_a, op_a = forward1([func_tid,
                            tid_map['apple']])
    fin_b, _ = forward1([func_tid, tid_map['apple']])
    a2_rel = float(np.abs(fin_a - fin_b).max()
                   / max(float(np.abs(fin_a).max()),
                         1e-30))
    a2_ok = bool(a2_rel < 1e-4)
    log('a2 determinism rel %.2e ok=%s'
        % (a2_rel, a2_ok), lines)

    # ---------- anchor forwards (a3/a5) ----------
    a5_rel = 0.0
    a3_rel = 0.0
    B_anchor = {}
    C_anchor = {}
    for w in ANCHOR_WORDS:
        fin, op = forward1([func_tid, tid_map[w]])
        B_anchor[w], prof = band_of(op)
        C_anchor[w] = op
        i62 = [i for i, (g, ww) in enumerate(w62)
               if ww == w][0]
        a5_rel = max(a5_rel, abs(B_anchor[w] - B62[i62])
                     / max(abs(B62[i62]), 1e-30))
        for li in (16, 17):
            x = op[li].reshape(-1)
            direct = float(np.dot(x, M[li]))
            xm = (x * M[li]).reshape(NH, HD)
            a3_rel = max(a3_rel, abs(direct
                                     - float(xm.sum()))
                         / max(abs(direct), 1e-30))
    a5_ok = bool(a5_rel < 1e-4)
    a3_ok = bool(a3_rel < 1e-9)
    log('a3 chunk-vs-direct rel %.2e ok=%s | a5 B vs '
        '2962 rel %.2e ok=%s'
        % (a3_rel, a3_ok, a5_rel, a5_ok), lines)

    # ---------- main sweep: 45 test words ----------
    B_all = np.zeros(45)
    C17 = np.zeros((45, NH))
    for i, (_, w) in enumerate(test_groups):
        fin, op = forward1([func_tid, tid_map[w]])
        B_all[i], _ = band_of(op)
        x = op[17].reshape(-1)
        C17[i] = (x * M[17]).reshape(NH, HD).sum(axis=1)
        if (i + 1) % 15 == 0:
            log('sweep [%d/45]' % (i + 1), lines)

    # ---------- a6 non-degeneracy ----------
    g_ix = {g: np.array([i for i, (gg, _) in
                         enumerate(test_groups)
                         if gg == g])
            for g in ('function', 'common', 'rare')}
    a6_ok = bool(C17.std(axis=1).min() > 0
                 and B_all[g_ix['function']].std() > 0
                 and B_all[g_ix['common']].std() > 0
                 and B_all[g_ix['rare']].std() > 0)
    log('a6 non-degeneracy ok=%s (C17 min std %.3e)'
        % (a6_ok, C17.std(axis=1).min()), lines)

    anchor_ok = bool(a1_ok and a2_ok and a3_ok and a4_ok
                     and a5_ok and a6_ok)
    verdict = None
    t1 = t2 = t3 = t4 = None

    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        tid_test = np.array([tid_map[w]
                             for _, w in test_groups])
        lab = np.array([0] * 15 + [1] * 15 + [2] * 15)
        mF = lab == 0
        mC = lab == 1
        mR = lab == 2
        mFC = mF | mC

        # ---------- T1 Freedman-Lane ----------
        x_fc = rankdata(tid_test[mFC].astype(np.float64))
        x_fc = (x_fc - x_fc.mean()) / x_fc.std()
        g_fc = (mC[mFC]).astype(np.float64)
        b_fc = B_all[mFC]
        Xr = np.stack([np.ones(len(x_fc)), x_fc], axis=1)
        beta_r, *_ = np.linalg.lstsq(Xr, b_fc, rcond=None)
        resid = b_fc - Xr @ beta_r
        Xf = np.stack([np.ones(len(x_fc)), x_fc, g_fc],
                      axis=1)

        def full_coef(bb):
            beta, *_ = np.linalg.lstsq(Xf, bb,
                                       rcond=None)
            return float(beta[2])

        obs1 = full_coef(b_fc)
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
                  b_fc[~mC[mFC]])), 4),
              'med_B_C': round(float(np.median(
                  b_fc[mC[mFC]])), 4)}
        log('T1 FL coef %+.4f p %.3e | medB F %.4f C %.4f'
            % (obs1, p1, t1['med_B_F'], t1['med_B_C']),
            lines)

        # ---------- T2 content frequency gradient --
        ix_c = np.nonzero(mC | mR)[0]
        rho2 = spearman(B_all[ix_c], tid_test[ix_c])
        rng2 = np.random.default_rng(RNG_T2)
        cnt2 = 0
        for _ in range(N_PERM):
            v2 = abs(spearman(
                B_all[ix_c],
                rng2.permutation(tid_test[ix_c])))
            if v2 >= abs(rho2) - 1e-12:
                cnt2 += 1
        p2 = (cnt2 + 1) / (N_PERM + 1)
        t2 = {'rho_B_tid_content': round(float(rho2), 4),
              'p': float('%.3e' % p2),
              'strong': bool(abs(rho2) >= STRONG_RHO
                             and p2 <= P_TH)}
        log('T2 content rho(B,tid) %.4f p %.3e strong=%s'
            % (rho2, p2, t2['strong']), lines)

        # ---------- T3 stratified block perm --------
        order = np.argsort(tid_test[mFC])
        blocks = [order[k * 6:(k + 1) * 6]
                  for k in range(5)]
        n_mixed = sum(1 for bl in blocks
                      if mC[mFC][bl].any()
                      and (~mC[mFC][bl]).any())
        t3_valid = bool(n_mixed >= 4)
        p3 = None
        obs3 = float(np.median(b_fc[~mC[mFC]])
                     - np.median(b_fc[mC[mFC]]))
        if t3_valid:
            rng3 = np.random.default_rng(RNG_T3)
            cnt3 = 0
            for _ in range(N_PERM):
                bl = mC[mFC].copy()
                for blo in blocks:
                    bl[blo] = rng3.permutation(
                        bl[blo])
                v = float(np.median(
                    b_fc[~bl]) - np.median(b_fc[bl]))
                if abs(v) >= abs(obs3) - 1e-12:
                    cnt3 += 1
            p3 = (cnt3 + 1) / (N_PERM + 1)
        t3 = {'valid': t3_valid, 'n_mixed_blocks':
              int(n_mixed),
              'obs_med_gap': round(float(obs3), 4),
              'p': (float('%.3e' % p3)
                    if p3 is not None else None)}
        log('T3 blocks mixed %d/5 valid=%s gap %+.4f p %s'
            % (n_mixed, t3_valid, obs3,
               ('%.3e' % p3) if p3 is not None
               else 'NA'), lines)

        # ---------- T4 descriptives ----------
        xc = tid_test[ix_c].astype(np.float64)
        xc = (xc - xc.mean()) / xc.std()
        beta_c, *_ = np.linalg.lstsq(
            np.stack([np.ones(len(xc)), xc], axis=1),
            B_all[ix_c], rcond=None)
        xfp = (tid_test[mF].astype(np.float64)
               - tid_test[ix_c].mean()) \
            / tid_test[ix_c].std()
        pred_F = beta_c[0] + beta_c[1] * xfp
        t4 = {'med_B_R': round(float(np.median(
                  B_all[mR])), 4),
              'pred_B_F_from_content': [
                  round(float(v), 4) for v in pred_F],
              'obs_B_F': [round(float(v), 4)
                          for v in B_all[mF]],
              'extrapolation_caveat':
                  'F tid range %d-%d vs content support '
                  '%d-%d' % (tid_test[mF].min(),
                             tid_test[mF].max(),
                             tid_test[ix_c].min(),
                             tid_test[ix_c].max()),
              'rho_tokid_B_all45': round(float(
                  spearman(tid_test, B_all)), 4),
              'anchor_B': {w: round(float(B_anchor[w]), 4)
                           for w in ANCHOR_WORDS}}
        log('T4 medB R %.4f | rho(tokid,B)45 %.4f'
            % (t4['med_B_R'],
               t4['rho_tokid_B_all45']), lines)

        # ---------- verdict ----------
        if p1 <= P_TH and not t2['strong']:
            verdict = 'band_class_effect_beyond_frequency'
        elif p1 <= P_TH and t2['strong']:
            verdict = 'band_gap_frequency_confounded'
        elif p1 > P_NULL_TH:
            verdict = 'band_gap_frequency_controlled_away'
        else:
            verdict = 'band_gap_mixed_registered'

        save = {'B': B_all, 'C17': C17,
                'tids': tid_test, 'labels': lab,
                'words': np.array(
                    ['%s:%s' % (g, w)
                     for g, w in test_groups]),
                'B_anchor': np.array(
                    [B_anchor[w]
                     for w in ANCHOR_WORDS])}

    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2963, 'model': 'qwen3-4b',
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
            OUT, 'freq_band.npz'), **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2963 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
