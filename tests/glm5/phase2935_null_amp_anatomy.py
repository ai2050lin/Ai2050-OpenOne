# -*- coding: utf-8 -*-
"""Phase 2935: null-amplification mechanism anatomy.

Why: 2934 found that null-context CI exceeds func-context CI
in both bands (LOAD x1.83, DEEP x2.00) with the lin_r-CI law
preserved - random-token contexts make the readout MORE
ablation-sensitive everywhere. Mechanism unresolved: is the
amplification driven by the SPECIFIC token identities in the
null set (H1 token-id effect -> amplification ratio unstable
across null resamples) or by the ABSENCE of semantic
constraint in the context (H2 context-statistics effect ->
amplification ratio stable across null resamples)?

Mode: one run (qwen3-4b), 2934 corrected protocol verbatim
(per-condition batch57 forwards, REAL ablation o_proj-input
head slice zeroed at pos 1, readout dirs_word[35] rebuilt
this run, 1120 gated cells, L0 excluded). Conditions: func
(the anchor + cross-phase reproduction) + 4 independently
resampled null-tid sets (seeds 2896 original / 2914 / 2915 /
2916; 2927 sampling rule verbatim). 5 x 1120 = 5600 ablation
forwards.

Anchors (frozen):
  a1 dirs_word rebuild vs 2927 npz max abs < 1e-5
  a2 func baseline determinism max rel < 1e-4
  a3 hook efficacy: L18 all-heads pos1 ablation func
     mean|ds| > 0.01 * scale_func
  a4 mask counts 469/382/295 (source consistency)
  a5 func baseline separation lab0-lab1 > 0
  a6 CROSS-PHASE reproduction: func CI_rel vs 2933 npz
     (1120 cells) max abs diff < 1e-4
  a7 func separation > each null-set separation (guard)

Main tests (frozen):
  P1 null-set generalization: for each null set r, med_l_r
     (per-layer median CI over 32 heads) vs lin_r Spearman +
     LOAD/DEEP exact band permutation (law replication per
     set); grand test = ALL 4 sets replicate (r<0, p<=0.01,
     p_band<=0.05, diff>0).
  P2 amplification stability: per-cell amplification ratio
     a_r(c) = ci_null_r(c) / ci_func(c); pairwise Spearman
     across the 4 sets over 1120 cells (median of 6 pairs).
  P3 CI-level cross-set consistency: pairwise Spearman of
     raw ci_null_r over 1120 cells (median of 6 pairs).

Verdict (frozen):
  anchor fail                       => anchor_fail_all_void
  P1 replicates in all 4 sets AND median pairwise
  Spearman >= 0.9 in BOTH P2 and P3
                                    => null_amp_context_general
  P1 replicates in all 4 sets AND median pairwise Spearman
  >= 0.5 in BOTH P2 and P3
                                    => null_amp_mixed
  else                              => null_amp_token_unstable

Output: phase2935/null_amp_anatomy/.
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
SRC_2929 = os.path.join(BASE, 'phase2929',
                        'response_structure_atlas',
                        'response_structure_atlas.npz')
SRC_2930 = os.path.join(BASE, 'phase2930', 'direction_flip_control',
                        'direction_flip_control.npz')
SRC_2931 = os.path.join(BASE, 'phase2931', 'skeleton_overlap_null',
                        'skeleton_overlap_null.npz')
SRC_2933 = os.path.join(BASE, 'phase2933', 'full_atlas_ci',
                        'full_atlas_ci.npz')
OUT = os.path.join(BASE, 'phase2935', 'null_amp_anatomy')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2935_run_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
SEED = 2896
NH, HD = 32, 128
NL = 36
VOCAB = 151936
N_GRP = 10000
RNG_LAY = 2912
NULL_SEEDS = (2896, 2914, 2915, 2916)
SURV = [(1, 6), (5, 6), (7, 19), (8, 2), (14, 9), (20, 8),
        (21, 6)]
EXP_N86, EXP_NWD, EXP_SHARED = 469, 382, 295
LOAD_BAND = list(range(6, 13))
DEEP_BAND = list(range(28, 36))
A6_TOL = 1e-4

PREREG = {
    'mode': 'REAL ablation pos1 x 5 conditions (func anchor '
            '+ 4 resampled null-tid sets, 2927 sampling rule '
            'verbatim), per-condition batch57 forwards, 1120 '
            'gated cells, readout dirs_word[35] rebuilt this '
            'run; 2934 corrected protocol verbatim otherwise',
    'correction_note': 'carries the 2934 run1 lesson: per-'
                       'condition batch57 forwards mandatory '
                       'for cross-phase CI anchors (bf16 '
                       'cross-batch-composition noise)',
    'question': 'is the 2934 null-context amplification driven '
                'by specific null token identities (H1) or by '
                'absence of semantic constraint (H2)?',
    'anchors': {
        'a1': 'dirs_word rebuild vs 2927 npz max abs < 1e-5',
        'a2': 'func baseline determinism max rel < 1e-4',
        'a3': 'L18 all-heads pos1 ablation func mean|ds| > '
              '0.01*scale_func',
        'a4': 'mask counts 469/382/295',
        'a5': 'func baseline separation lab0-lab1 > 0',
        'a6': 'func CI_rel vs 2933 npz max abs < 1e-4',
        'a7': 'func separation > every null-set separation',
    },
    'P1': 'per null set r in 4 sets: med_l_r vs lin_r '
          'Spearman (perm 10000 rng 2912, two-sided) + '
          'LOAD L6-L12 vs DEEP L28-L35 exact layer-label '
          'permutation C(15,7)=6435 one-sided',
    'P2': 'per-cell amplification ratio a_r = ci_null_r / '
          'ci_func; pairwise Spearman across 4 sets over '
          '1120 cells (median of 6 pairs)',
    'P3': 'pairwise Spearman of raw ci_null_r across 4 '
          'sets over 1120 cells (median of 6 pairs)',
    'verdict': 'anchor fail => anchor_fail_all_void; P1 '
               'replicates in all 4 sets AND median pairwise '
               'Spearman >= 0.9 in BOTH P2 and P3 => '
               'null_amp_context_general; P1 all 4 AND '
               'median >= 0.5 in both => null_amp_mixed; '
               'else null_amp_token_unstable',
}


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


def unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-30)


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


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2935,
                   'name': 'null_amp_anatomy',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2887': sha8(SRC_2887),
                               's2927': sha8(SRC_2927),
                               's2929': sha8(SRC_2929),
                               's2930': sha8(SRC_2930),
                               's2931': sha8(SRC_2931),
                               's2933': sha8(SRC_2933)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'seed': SEED, 'n_grp': N_GRP,
                   'rng_lay': RNG_LAY,
                   'null_seeds': list(NULL_SEEDS),
                   'load_band': LOAD_BAND,
                   'deep_band': DEEP_BAND,
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    # ---------- sources ----------
    z87 = np.load(SRC_2887, allow_pickle=True)
    words = [tuple(str(w).split(':')) for w in z87['words']]
    lab_lang = np.asarray(z87['labels_lang']).astype(int)
    n_words = len(words)
    assert n_words == 57

    z27 = np.load(SRC_2927, allow_pickle=True)
    dirs_word_27 = z27['dirs_word'].astype(np.float64)
    z29 = np.load(SRC_2929, allow_pickle=True)
    _ = z29['rho_grid'].astype(np.float64)  # source-chain pin
    z30 = np.load(SRC_2930, allow_pickle=True)
    lin_r = z30['lin_r_profile'].astype(np.float64)
    z31 = np.load(SRC_2931, allow_pickle=True)
    S29c = z31['S29_corrected'].astype(bool)
    Smirc = z31['Smir_corrected'].astype(bool)
    shared = S29c & Smirc
    n86, nwd, nsh = int(S29c.sum()), int(Smirc.sum()), \
        int(shared.sum())
    a4_ok = bool(n86 == EXP_N86 and nwd == EXP_NWD
                 and nsh == EXP_SHARED)
    log('a4 mask counts n86=%d nwd=%d shared=%d ok=%s'
        % (n86, nwd, nsh, a4_ok), lines)

    z33 = np.load(SRC_2933, allow_pickle=True)
    cells_33 = [tuple(int(v) for v in c)
                for c in z33['cells']]
    ci33 = z33['ci_rel'].astype(np.float64)
    ci33_map = dict(zip(cells_33, ci33))

    cells_all = [(h, li) for li in range(1, NL)
                 for h in range(NH)]
    assert len(cells_all) == 1120
    log('cells: full grid %d (L0 excluded)' % len(cells_all),
        lines)

    # ---------- model ----------
    import torch
    import sys
    sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
    from phase2662_symmetric_mapping_contract import load_native
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        MD, local_files_only=True, trust_remote_code=True,
        use_fast=True)
    tc = {}

    def tid(t):
        if t not in tc:
            ids = tok(' ' + t, add_special_tokens=False)[
                'input_ids']
            if len(ids) != 1:
                ids = tok(t, add_special_tokens=False)[
                    'input_ids']
            assert len(ids) == 1, '%s -> %s' % (t, ids)
            tc[t] = int(ids[0])
        return tc[t]

    tid_map = {}
    for lang, ck, w in words:
        tid_map[w] = tid(w)
        if lang == 'en':
            assert tid_map[w] == int(ck), 'key mismatch %s' % w
    func_tid = tid('the')

    # null sets: 2927 sampling rule verbatim, one rng per set
    word_tids = set(tid_map.values())

    def sample_null(seed):
        rng = np.random.default_rng(seed)
        out = []
        while len(out) < n_words:
            r = int(rng.integers(0, VOCAB))
            if r not in word_tids and r > 0:
                out.append(r)
        return out

    null_sets = {('null%d' % k): sample_null(s)
                 for k, s in enumerate(NULL_SEEDS)}
    log('null sets sampled: %s'
        % {k: v[:4] for k, v in null_sets.items()}, lines)

    model, _ = load_native('qwen4')
    model.eval()
    layers = model.model.layers
    log('model loaded (load_native full GPU)', lines)

    cap = {'attnin': {}}
    state_fin = {'on': False}
    abl = {'li': None, 'h': None, 'pos0': False, 'pos1': False}
    fin_cap = {}
    handles = []

    def pre_attn(li):
        def h(module, args, kwargs):
            x = args[0] if args else kwargs.get('hidden_states')
            if x is None or x.dim() < 2:
                return
            cap['attnin'].setdefault(li, []).append(
                x.detach().float().cpu().numpy())
        return h

    def make_opro_hook(li):
        def hk(module, args, kwargs):
            if abl['li'] != li:
                return
            x = args[0]
            hd = abl['h']
            if abl['pos0']:
                if hd is None:
                    x[:, 0, :] = 0
                else:
                    x[:, 0, hd * HD:(hd + 1) * HD] = 0
            if abl['pos1']:
                if hd is None:
                    x[:, 1, :] = 0
                else:
                    x[:, 1, hd * HD:(hd + 1) * HD] = 0
        return hk

    def pre_norm(module, args, kwargs):
        if state_fin['on']:
            fin_cap['x'] = args[0][:, -1, :].detach() \
                .float().cpu().numpy()

    for li in range(NL):
        handles.append(
            layers[li].self_attn.register_forward_pre_hook(
                pre_attn(li), with_kwargs=True))
        handles.append(
            layers[li].self_attn.o_proj.register_forward_pre_hook(
                make_opro_hook(li), with_kwargs=True))
    handles.append(model.model.norm.register_forward_pre_hook(
        pre_norm, with_kwargs=True))

    def clear_cap():
        for li in cap['attnin']:
            del cap['attnin'][li][:]

    def forward1(toks):
        clear_cap()
        with torch.no_grad():
            model(torch.tensor([toks], device='cuda'))
        return {li: cap['attnin'][li][0]
                for li in cap['attnin']}

    def forward_batch(toks_list):
        clear_cap()
        fin_cap.pop('x', None)
        state_fin['on'] = True
        with torch.no_grad():
            out = model(torch.tensor(toks_list, device='cuda'))
        state_fin['on'] = False
        return fin_cap['x'].astype(np.float64)

    # ---------- pass 1: dirs_word rebuild (func, verbatim) --
    attn_store = {}
    for i, (_, _, w) in enumerate(words):
        attnin_all = forward1([func_tid, tid_map[w]])
        for li in range(NL):
            attn_store[(i, li)] = \
                attnin_all[li].astype(np.float32)
        if (i + 1) % 20 == 0:
            log('pass1 words [%d/%d]' % (i + 1, n_words), lines)
    d_dim = attn_store[(0, 0)].shape[-1]
    diffs_w = np.zeros((NL, d_dim))
    for li in range(NL):
        X = np.stack([attn_store[(i, li)][0, 1]
                      for i in range(n_words)]) \
            .astype(np.float64)
        diffs_w[li] = X[lab_lang == 0].mean(0) \
            - X[lab_lang == 1].mean(0)
    dirs_word = np.stack([unit(diffs_w[li]) for li in range(NL)])
    a1_diff = float(np.abs(dirs_word - dirs_word_27).max())
    a1_ok = bool(a1_diff < 1e-5)
    log('a1 dirs_word rebuild diff %.2e ok=%s'
        % (a1_diff, a1_ok), lines)

    # ---------- per-cond baselines (func twice for a2) ------
    batch_func = [[func_tid, tid_map[words[i][2]]]
                  for i in range(n_words)]
    batch_null = {k: [[ntids[i], tid_map[words[i][2]]]
                      for i in range(n_words)]
                  for k, ntids in null_sets.items()}

    fin_f1 = forward_batch(batch_func)
    fin_f2 = forward_batch(batch_func)
    a2_rel = float(np.abs(fin_f1 - fin_f2).max()
                   / max(float(np.abs(fin_f1).max()), 1e-30))
    a2_ok = bool(a2_rel < 1e-4)
    log('a2 baseline determinism rel %.2e ok=%s'
        % (a2_rel, a2_ok), lines)

    u35 = dirs_word[NL - 1]
    s_base = {'func': fin_f1 @ u35}
    scale = {'func': float(np.mean(np.abs(s_base['func'])))}
    sep = {'func': float(s_base['func'][lab_lang == 0].mean()
                         - s_base['func'][lab_lang == 1]
                         .mean())}
    for k, bt in batch_null.items():
        s_base[k] = forward_batch(bt) @ u35
        scale[k] = float(np.mean(np.abs(s_base[k])))
        sep[k] = float(s_base[k][lab_lang == 0].mean()
                       - s_base[k][lab_lang == 1].mean())
    a5_ok = bool(sep['func'] > 0.0)
    a7_ok = bool(all(sep['func'] > sep[k]
                     for k in batch_null))
    log('a5 func separation %.4f (scale %.4f) ok=%s'
        % (sep['func'], scale['func'], a5_ok), lines)
    log('a7 separations: %s (func>all ok=%s)'
        % ({k: round(v, 2) for k, v in sep.items()}, a7_ok),
        lines)

    # ---------- a3: hook efficacy (pos1, L18, func) --------
    abl['li'], abl['h'] = 18, None
    abl['pos0'], abl['pos1'] = False, True
    fin_a3 = forward_batch(batch_func)
    abl['li'], abl['h'] = None, None
    abl['pos0'], abl['pos1'] = False, False
    a3_val = float(np.mean(np.abs(
        fin_a3 @ u35 - s_base['func'])))
    a3_ok = bool(a3_val > 0.01 * scale['func'])
    log('a3 L18 all-heads pos1 ablation mean|ds| %.4f '
        '(0.01*scale %.4f) ok=%s'
        % (a3_val, 0.01 * scale['func'], a3_ok), lines)

    # ---------- ablation sweep: 5 conds x 1120 cells -------
    conds_all = ['func'] + list(batch_null.keys())
    ci_raw = {cn: {} for cn in conds_all}
    for cn in conds_all:
        bt = batch_func if cn == 'func' else batch_null[cn]
        for k, (h, li) in enumerate(cells_all):
            abl['li'], abl['h'] = li, h
            abl['pos0'], abl['pos1'] = False, True
            fin_a = forward_batch(bt)
            abl['li'], abl['h'] = None, None
            abl['pos0'], abl['pos1'] = False, False
            d = fin_a @ u35 - s_base[cn]
            ci_raw[cn][(h, li)] = float(np.mean(np.abs(d)))
            if (k + 1) % 400 == 0:
                log('ablation %s [%d/%d]'
                    % (cn, k + 1, len(cells_all)), lines)
    ci_rel = {cn: {c: ci_raw[cn][c] / scale[cn]
                   for c in cells_all}
              for cn in conds_all}
    log('ablation sweep done (%d conds x %d cells, func max '
        '%.4f)'
        % (len(conds_all), len(cells_all),
           max(ci_rel['func'].values())), lines)

    # ---------- a6: cross-phase reproduction ----------
    a6_diff = 0.0
    for c in cells_33:
        a6_diff = max(a6_diff,
                      abs(ci_rel['func'][c] - ci33_map[c]))
    a6_ok = bool(a6_diff < A6_TOL)
    log('a6 CI_rel reproduction vs 2933 (func, %d cells) '
        'max abs diff %.2e ok=%s'
        % (len(cells_33), a6_diff, a6_ok), lines)

    anchor_ok = bool(a1_ok and a2_ok and a3_ok and a4_ok
                     and a5_ok and a6_ok and a7_ok)

    verdict = None
    p1 = p2 = p3 = p4 = None
    save = {}
    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        from math import comb

        def band_and_linr(ci_map):
            med_l = np.zeros(NL)
            for li in range(1, NL):
                med_l[li] = float(np.median(
                    [ci_map[(h, li)] for h in range(NH)]))
            load_m = float(med_l[LOAD_BAND].mean())
            deep_m = float(med_l[DEEP_BAND].mean())
            obs_diff = load_m - deep_m
            pool_l = list(LOAD_BAND) + list(DEEP_BAND)
            n_load = len(LOAD_BAND)
            n_perm_b = comb(len(pool_l), n_load)
            rng_l = np.random.default_rng(RNG_LAY)
            vals_l = med_l[pool_l]
            cnt_b = 0
            for _ in range(n_perm_b):
                pm = rng_l.permutation(len(pool_l))
                if vals_l[pm[:n_load]].mean() \
                        - vals_l[pm[n_load:]].mean() \
                        >= obs_diff:
                    cnt_b += 1
            p2b = float(cnt_b / n_perm_b)
            r_obs = spearman(med_l[1:], lin_r[1:])
            cnt_lay = 0
            for _ in range(N_GRP):
                if abs(spearman(
                        med_l[1:],
                        lin_r[1:][rng_l.permutation(
                            NL - 1)])) >= abs(r_obs):
                    cnt_lay += 1
            p1b = float((cnt_lay + 1) / (N_GRP + 1))
            return {'load_m': round(load_m, 6),
                    'deep_m': round(deep_m, 6),
                    'diff': round(obs_diff, 6),
                    'p_band': round(p2b, 6),
                    'n_perm_band': int(n_perm_b),
                    'spearman_med_l_lin_r': round(r_obs, 4),
                    'p_linr': float('%.3e' % p1b),
                    'stable': bool(r_obs < 0 and p1b <= 0.01
                                   and p2b <= 0.05
                                   and obs_diff > 0)}

        # ---------- P1: per null-set law replication -------
        p1 = {'func': band_and_linr(ci_rel['func'])}
        log('P1 func: rho %.4f p %.3e p_band %.4f stable=%s'
            % (p1['func']['spearman_med_l_lin_r'],
               p1['func']['p_linr'], p1['func']['p_band'],
               p1['func']['stable']), lines)
        for k in batch_null:
            p1[k] = band_and_linr(ci_rel[k])
            log('P1 %s: LOAD %.6f DEEP %.6f rho %.4f '
                'p %.3e p_band %.4f stable=%s'
                % (k, p1[k]['load_m'], p1[k]['deep_m'],
                   p1[k]['spearman_med_l_lin_r'],
                   p1[k]['p_linr'], p1[k]['p_band'],
                   p1[k]['stable']), lines)
        p1_all = all(p1[k]['stable'] for k in batch_null)

        # ---------- P2: amplification-ratio stability ------
        ci_func = np.array([ci_rel['func'][c]
                            for c in cells_all])
        pos_of = {c: i for i, c in enumerate(cells_all)}
        amp = {k: np.array([ci_rel[k][c]
                            / max(ci_func[pos_of[c]], 1e-30)
                            for c in cells_all])
               for k in batch_null}
        ks = list(batch_null)
        p2_pairs = []
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                p2_pairs.append(spearman(amp[ks[i]],
                                         amp[ks[j]]))
        p2 = {'pairs': [round(v, 4) for v in p2_pairs],
              'median': round(float(np.median(p2_pairs)), 4)}

        # ---------- P3: raw CI cross-set consistency ------
        p3_pairs = []
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                p3_pairs.append(spearman(
                    np.array([ci_rel[ks[i]][c]
                              for c in cells_all]),
                    np.array([ci_rel[ks[j]][c]
                              for c in cells_all])))
        p3 = {'pairs': [round(v, 4) for v in p3_pairs],
              'median': round(float(np.median(p3_pairs)), 4)}
        log('P2 amp-ratio pairwise: %s median %.4f'
            % (p2['pairs'], p2['median']), lines)
        log('P3 raw-CI pairwise: %s median %.4f'
            % (p3['pairs'], p3['median']), lines)

        # ---------- P4: descriptive ----------
        amp_med = {k: round(float(np.median(amp[k])), 4)
                   for k in ks}
        prof = []
        for li in range(1, NL):
            row = {}
            for k in ks:
                cl = [ci_rel[k][(h, li)] for h in range(NH)]
                cf = [ci_rel['func'][(h, li)]
                      for h in range(NH)]
                row[k] = round(float(np.median(cl))
                               / max(float(np.median(cf)),
                                     1e-30), 3)
            prof.append({'layer': li,
                         'amp_ratio_medians': row})
        surv_tab = []
        for c in SURV:
            surv_tab.append({
                'cell': list(c),
                'ci_func': round(float(ci_rel['func'][c]), 6),
                'amp_null0': round(float(amp[ks[0]][pos_of[c]]),
                                   3),
                'amp_null1': round(float(amp[ks[1]][pos_of[c]]),
                                   3)})
        top5 = sorted(cells_all,
                      key=lambda c: -ci_rel[ks[0]][c])[:5]
        p4 = {'amp_median_per_null_set': amp_med,
              'func_grand_median': round(
                  float(np.median(ci_func)), 6),
              'layer_amp_profile_top': prof[:8],
              'survivor': surv_tab,
              'top5_ci_null0': [{'cell': list(c),
                                 'ci': round(float(
                                     ci_rel[ks[0]][c]), 6)}
                                for c in top5]}
        log('P4: amp medians %s | func grand %.6f'
            % (amp_med, p4['func_grand_median']), lines)

        # ---------- verdict ----------
        if p1_all and p2['median'] >= 0.9 \
                and p3['median'] >= 0.9:
            verdict = 'null_amp_context_general'
        elif p1_all and p2['median'] >= 0.5 \
                and p3['median'] >= 0.5:
            verdict = 'null_amp_mixed'
        else:
            verdict = 'null_amp_token_unstable'

        save = {
            'cells': np.array(cells_all, dtype=np.int64),
            'ci_rel': np.array(
                [[ci_rel[cn][c] for c in cells_all]
                 for cn in conds_all], dtype=np.float64),
            'cond_names': np.array(conds_all),
            'amp': np.array([amp[k] for k in ks]),
            'amp_names': np.array(ks),
            's_base': np.array([s_base[cn]
                                for cn in conds_all],
                               dtype=np.float32),
            'scale': np.array([scale[cn] for cn in conds_all]),
            'sep': np.array([sep[cn] for cn in conds_all]),
            'dirs_word': dirs_word}

    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2935, 'model': 'qwen3-4b',
           'prereg': PREREG,
           'anchors': {'a1_diff': float('%.3e' % a1_diff),
                       'a1_ok': a1_ok,
                       'a2_rel': float('%.3e' % a2_rel),
                       'a2_ok': a2_ok,
                       'a3_val': round(a3_val, 6),
                       'a3_ok': a3_ok,
                       'a4_ok': a4_ok,
                       'a5_sep': round(sep['func'], 6),
                       'a5_ok': a5_ok,
                       'a6_diff': float('%.3e' % a6_diff),
                       'a6_ok': a6_ok,
                       'sep_null_sets': {k: round(v, 4)
                                         for k, v in
                                         sep.items()},
                       'a7_ok': a7_ok,
                       'ok': anchor_ok},
           'P1': p1, 'P2': p2, 'P3': p3, 'P4': p4,
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if save:
        np.savez_compressed(os.path.join(OUT,
                                         'null_amp_anatomy.npz'),
                            **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2935 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
