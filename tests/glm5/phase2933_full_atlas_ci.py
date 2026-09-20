# -*- coding: utf-8 -*-
"""Phase 2933: full-grid CI atlas (all 1120 non-degenerate cells).

Why: 2932 established (REAL causal ablation, 648 cells) that the
295-cell convention-invariant skeleton is functionally
load-bearing (x1.124 vs matched background, p=1.0e-4) with a
BANDED layer structure: top-half hits concentrate in L6-L12
(quasi-linear load band) while L28/L29/L34 reverse. But the
matched-background design only sampled 353 of 1120 cells -
band and reversal claims rest on partial coverage. Full
coverage (every gated non-degenerate cell, L0 excluded by the
2931 gate discipline) turns the band profile into a direct
measurement and enables the three-way coupling atlas
CI_rel x rho86 x rho_mirror x lin_r.

Mode: one run (qwen3-4b), 2932 protocol verbatim (REAL
ablation: o_proj-input head slice zeroed at pos 1 per cell,
57 func-condition prompts batched verbatim 2887, readout
final residual projected on dirs_word[35] rebuilt this run).
Cells: all 32 heads x layers 1-35 = 1120.

Anchors (frozen):
  a1 dirs_word rebuild vs 2927 npz max abs < 1e-5
  a2 batched baseline determinism max rel < 1e-4
  a3 hook efficacy: all-heads L18 ablation mean|ds| >
     0.01 * scale
  a4 mask counts 469/382/295 (source consistency)
  a5 baseline separation mean(lab0 s) - mean(lab1 s) > 0
  a6 CROSS-PHASE reproduction: CI_rel on the 648 cells of
     2932 (subset of 1120) vs 2932 npz max abs diff < 1e-4

P2 band structure (main, frozen):
  med_l = median CI_rel over 32 heads per layer (l=1..35).
  P2b: LOAD band = L6-L12 (7 layers, 2932 hit band) vs DEEP
  band = L28-L35 (8 layers, 2932 reversed band); exact
  permutation over layer labels (C(15,7)=6435), one-sided.
  P2a: Spearman(med_l, lin_r_l) over 35 layers, two-sided
  permutation (10000, rng 2911).
P3 grid coupling: Spearman(CI_rel, rho86) and
  (CI_rel, rho_mirror) over all 1120 cells, permutation p
  (5000, rng 2910).
P4 (descriptive): per-layer Spearman(CI, rho86) profile;
  top-10 CI cells; survivor 7 table; skeleton-vs-rest CI
  contrast with full coverage (no sampling).

Verdict (frozen):
  anchor fail                    => anchor_fail_all_void
  p3a <= 0.001 & positive & p2b <= 0.05 (right direction)
                                 => full_atlas_band_and_coupling_confirmed
  p3a <= 0.001 & positive & p2b > 0.05
                                 => full_atlas_coupling_only
  else                           => full_atlas_unstructured

Output: phase2933/full_atlas_ci/.
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
SRC_2932 = os.path.join(BASE, 'phase2932',
                        'skeleton_functional_ablation',
                        'skeleton_functional_ablation.npz')
OUT = os.path.join(BASE, 'phase2933', 'full_atlas_ci')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2933_run_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
SEED = 2896
NH, HD = 32, 128
NL = 36
N_RHO = 5000
N_GRP = 10000
RNG_RHO = 2910
RNG_LAY = 2911
SURV = [(1, 6), (5, 6), (7, 19), (8, 2), (14, 9), (20, 8),
        (21, 6)]
EXP_N86, EXP_NWD, EXP_SHARED = 469, 382, 295
LOAD_BAND = list(range(6, 13))
DEEP_BAND = list(range(28, 36))
A6_TOL = 1e-4

PREREG = {
    'mode': 'REAL causal ablation full-grid atlas: o_proj-input '
            'head slice zeroed at pos 1 per cell for ALL 1120 '
            'gated non-degenerate cells (L0 excluded), 57 '
            'func-condition prompts batched, readout final '
            'residual projection on dirs_word[35]; 2932 '
            'protocol verbatim',
    'question': 'does the 2932 band structure (load band '
                'L6-L12, reversed deep band) and the '
                'CI-rho-lin_r coupling hold with FULL cell '
                'coverage?',
    'anchors': {
        'a1': 'dirs_word rebuild vs 2927 npz max abs < 1e-5',
        'a2': 'batched baseline determinism max rel < 1e-4',
        'a3': 'all-heads L18 ablation mean|ds| > 0.01*scale',
        'a4': 'mask counts 469/382/295',
        'a5': 'baseline separation mean(lab0)-mean(lab1) > 0',
        'a6': 'CI_rel on 2932 648-cell subset vs 2932 npz '
              'max abs diff < 1e-4',
    },
    'P2': 'med_l = per-layer median CI_rel (32 heads); P2b '
          'LOAD band L6-L12 (7) vs DEEP band L28-L35 (8), '
          'exact layer-label permutation C(15,7)=6435 '
          'one-sided; P2a Spearman(med_l, lin_r_l) over 35 '
          'layers, two-sided perm 10000 rng 2911',
    'P3': 'Spearman(CI_rel, rho86) and (CI_rel, rho_mirror) '
          'over 1120 cells, perm 5000 rng 2910',
    'P4': 'per-layer Spearman(CI, rho86) profile; top-10 CI '
          'cells; survivor 7; skeleton-vs-rest CI contrast '
          '(full coverage)',
    'verdict': 'anchor fail => anchor_fail_all_void; p3a <= '
               '0.001 & positive & p2b <= 0.05 (right '
               'direction) => '
               'full_atlas_band_and_coupling_confirmed; '
               'p3a <= 0.001 & positive & p2b > 0.05 => '
               'full_atlas_coupling_only; else '
               'full_atlas_unstructured',
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
        json.dump({'phase': 2933,
                   'name': 'full_atlas_ci',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2887': sha8(SRC_2887),
                               's2927': sha8(SRC_2927),
                               's2929': sha8(SRC_2929),
                               's2930': sha8(SRC_2930),
                               's2931': sha8(SRC_2931),
                               's2932': sha8(SRC_2932)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'seed': SEED, 'n_rho': N_RHO,
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
    rho_grid_29 = z29['rho_grid'].astype(np.float64)
    z30 = np.load(SRC_2930, allow_pickle=True)
    lin_r = z30['lin_r_profile'].astype(np.float64)
    rho_mir_30 = z30['rho_mirror_grid'].astype(np.float64)
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

    z32 = np.load(SRC_2932, allow_pickle=True)
    cells_32 = [tuple(int(v) for v in c)
                for c in z32['cells_all']]
    ci32 = z32['ci_rel'].astype(np.float64)
    ci32_map = dict(zip(cells_32, ci32))

    cells_all = [(h, li) for li in range(1, NL)
                 for h in range(NH)]
    assert len(cells_all) == 1120
    log('cells: full grid %d (L0 excluded by gate discipline)'
        % len(cells_all), lines)

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

    model, _ = load_native('qwen4')
    model.eval()
    layers = model.model.layers
    log('model loaded (load_native full GPU)', lines)

    cap = {'attnin': {}}
    state_fin = {'on': False}
    abl = {'li': None, 'h': None, 'all': False}
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
        def h(module, args, kwargs):
            if abl['li'] != li:
                return
            x = args[0]
            if abl['all']:
                x[:, 1, :] = 0
            else:
                hh = abl['h']
                x[:, 1, hh * HD:(hh + 1) * HD] = 0
        return h

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
        fin = fin_cap['x'].astype(np.float64)
        lg = out.logits[:, -1, :].detach().float() \
            .cpu().numpy()
        return fin, lg

    # ---------- pass 1: word probe rebuild ----------
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

    # ---------- batched baseline (twice) ----------
    batch_toks = [[func_tid, tid_map[w]] for _, _, w in words]
    fin1, lg1 = forward_batch(batch_toks)
    fin2, _ = forward_batch(batch_toks)
    a2_rel = float(np.abs(fin1 - fin2).max()
                   / max(float(np.abs(fin1).max()), 1e-30))
    a2_ok = bool(a2_rel < 1e-4)
    log('a2 baseline determinism rel %.2e ok=%s'
        % (a2_rel, a2_ok), lines)

    u35 = dirs_word[NL - 1]
    s_base = fin1 @ u35
    scale = float(np.mean(np.abs(s_base)))
    sep = float(s_base[lab_lang == 0].mean()
                - s_base[lab_lang == 1].mean())
    a5_ok = bool(sep > 0.0)
    log('a5 baseline separation %.4f (scale %.4f) ok=%s'
        % (sep, scale, a5_ok), lines)

    # ---------- a3: hook efficacy ----------
    abl['li'], abl['h'], abl['all'] = 18, None, True
    fin_a3, _ = forward_batch(batch_toks)
    abl['li'], abl['h'], abl['all'] = None, None, False
    a3_val = float(np.mean(np.abs(fin_a3 @ u35 - s_base)))
    a3_ok = bool(a3_val > 0.01 * scale)
    log('a3 all-heads L18 ablation mean|ds| %.4f '
        '(0.01*scale %.4f) ok=%s'
        % (a3_val, 0.01 * scale, a3_ok), lines)

    # ---------- ablation sweep (full grid) ----------
    ci_raw = {}
    ci_logit = {}
    s_abl_mat = np.zeros((len(cells_all), n_words))
    tids_arr = [tid_map[w] for _, _, w in words]
    for k, (h, li) in enumerate(cells_all):
        abl['li'], abl['h'], abl['all'] = li, h, False
        fin_a, lg_a = forward_batch(batch_toks)
        abl['li'], abl['h'], abl['all'] = None, None, False
        s_a = fin_a @ u35
        d = s_a - s_base
        ci_raw[(h, li)] = float(np.mean(np.abs(d)))
        dlg = lg_a[np.arange(n_words), tids_arr] \
            - lg1[np.arange(n_words), tids_arr]
        ci_logit[(h, li)] = float(np.mean(np.abs(dlg)))
        s_abl_mat[k] = s_a
        if (k + 1) % 200 == 0:
            log('ablation [%d/%d]'
                % (k + 1, len(cells_all)), lines)
    ci_rel = {c: ci_raw[c] / scale for c in cells_all}
    log('ablation sweep done (%d cells, CI_rel max %.4f)'
        % (len(cells_all), max(ci_rel.values())), lines)

    # ---------- a6: cross-phase reproduction ----------
    a6_diff = 0.0
    for c in cells_32:
        a6_diff = max(a6_diff, abs(ci_rel[c] - ci32_map[c]))
    a6_ok = bool(a6_diff < A6_TOL)
    log('a6 CI_rel reproduction vs 2932 (648 cells) max abs '
        'diff %.2e ok=%s' % (a6_diff, a6_ok), lines)

    anchor_ok = bool(a1_ok and a2_ok and a3_ok and a4_ok
                     and a5_ok and a6_ok)

    verdict = None
    p2 = p3 = p4 = None
    save = {}
    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        ci_arr = np.array([ci_rel[c] for c in cells_all])
        # ---------- P2: band structure ----------
        med_l = np.zeros(NL)
        for li in range(1, NL):
            med_l[li] = float(np.median(
                [ci_rel[(h, li)] for h in range(NH)]))
        load_m = float(med_l[LOAD_BAND].mean())
        deep_m = float(med_l[DEEP_BAND].mean())
        obs_diff = load_m - deep_m
        pool_l = list(LOAD_BAND) + list(DEEP_BAND)
        n_load = len(LOAD_BAND)
        cnt_b = 0
        from math import comb
        n_perm_b = comb(len(pool_l), n_load)
        rng_l = np.random.default_rng(RNG_LAY)
        vals_l = med_l[pool_l]
        for _ in range(n_perm_b):
            pm = rng_l.permutation(len(pool_l))
            if vals_l[pm[:n_load]].mean() \
                    - vals_l[pm[n_load:]].mean() >= obs_diff:
                cnt_b += 1
        p2b = float(cnt_b / n_perm_b)
        r_lay_obs = spearman(med_l[1:], lin_r[1:])
        cnt_lay = 0
        for _ in range(N_GRP):
            if abs(spearman(med_l[1:],
                            lin_r[1:][rng_l.permutation(
                                NL - 1)])) >= abs(r_lay_obs):
                cnt_lay += 1
        p2a = float((cnt_lay + 1) / (N_GRP + 1))
        p2 = {'median_ci_load_band': round(load_m, 6),
              'median_ci_deep_band': round(deep_m, 6),
              'diff': round(obs_diff, 6),
              'p2b_exact': round(p2b, 6),
              'n_perm_band': int(n_perm_b),
              'spearman_med_l_lin_r': round(r_lay_obs, 4),
              'p2a': float('%.3e' % p2a),
              'layer_median_ci_rel':
                  [round(float(v), 6) for v in med_l[1:]]}
        log('P2: LOAD %.6f vs DEEP %.6f diff %.6f p2b %.4f | '
            'rho(med_l, lin_r) %.4f p %.3e'
            % (load_m, deep_m, obs_diff, p2b, r_lay_obs,
               p2a), lines)

        # ---------- P3: grid coupling ----------
        rho86_fl = np.array([rho_grid_29[c]
                             for c in cells_all])
        rhomir_fl = np.array([rho_mir_30[c]
                              for c in cells_all])
        rng_r = np.random.default_rng(RNG_RHO)

        def perm_p(x, y, r_obs):
            cnt = 0
            for _ in range(N_RHO):
                if abs(spearman(x, y[rng_r.permutation(
                        len(y))])) >= abs(r_obs):
                    cnt += 1
            return float((cnt + 1) / (N_RHO + 1))

        r86_obs = spearman(ci_arr, rho86_fl)
        rmir_obs = spearman(ci_arr, rhomir_fl)
        p3a = perm_p(ci_arr, rho86_fl, r86_obs)
        p3m = perm_p(ci_arr, rhomir_fl, rmir_obs)
        p3 = {'spearman_ci_rho86': round(r86_obs, 4),
              'p_86': float('%.3e' % p3a),
              'spearman_ci_rho_mirror': round(rmir_obs, 4),
              'p_mirror': float('%.3e' % p3m),
              'n_cells': len(cells_all)}
        log('P3: rho(ci, rho86) %.4f p %.3e | rho(ci, '
            'rho_mirror) %.4f p %.3e (n=%d)'
            % (r86_obs, p3a, rmir_obs, p3m, len(cells_all)),
            lines)

        # ---------- P4: descriptive ----------
        prof = []
        for li in range(1, NL):
            cl = np.array([ci_rel[(h, li)]
                           for h in range(NH)])
            rl = np.array([rho_grid_29[(h, li)]
                           for h in range(NH)])
            prof.append({'layer': li,
                         'med_ci': round(float(np.median(cl)),
                                         6),
                         'spearman_ci_rho86':
                             round(spearman(cl, rl), 4),
                         'lin_r': round(float(lin_r[li]), 4)})
        top10 = sorted(cells_all,
                       key=lambda c: -ci_rel[c])[:10]
        sk_ci = [ci_rel[c] for c in cells_all if shared[c]]
        rest_ci = [ci_rel[c] for c in cells_all
                   if not shared[c]]
        surv_tab = [{'cell': list(c),
                     'ci_rel': round(float(ci_rel[c]), 6),
                     'layer_rank': int(1 + sum(
                         1 for h2 in range(NH)
                         if ci_rel[(h2, c[1])]
                         > ci_rel[c]))}
                    for c in SURV]
        ci_lg = np.array([ci_logit[c] for c in cells_all])
        p4 = {'layer_profile': prof,
              'top10_ci_cells': [{'cell': list(c),
                                  'ci_rel': round(
                                      float(ci_rel[c]), 6)}
                                 for c in top10],
              'skeleton_median_ci': round(
                  float(np.median(sk_ci)), 6),
              'rest_median_ci': round(
                  float(np.median(rest_ci)), 6),
              'skeleton_vs_rest_ranksum_direction':
                  bool(np.median(sk_ci)
                       > np.median(rest_ci)),
              'survivor': surv_tab,
              'spearman_ci_rel_ci_logit':
                  round(spearman(ci_arr, ci_lg), 4)}
        log('P4: skeleton %.6f vs rest %.6f | survivor ranks %s'
            % (np.median(sk_ci), np.median(rest_ci),
               [(d['cell'], d['layer_rank'])
                for d in surv_tab]), lines)

        # ---------- verdict ----------
        if p3a <= 0.001 and r86_obs > 0 \
                and p2b <= 0.05 and obs_diff > 0:
            verdict = 'full_atlas_band_and_coupling_confirmed'
        elif p3a <= 0.001 and r86_obs > 0:
            verdict = 'full_atlas_coupling_only'
        else:
            verdict = 'full_atlas_unstructured'

        save = {'s_base': s_base.astype(np.float32),
                's_abl': s_abl_mat.astype(np.float32),
                'cells': np.array(cells_all, dtype=np.int64),
                'ci_rel': ci_arr,
                'ci_logit': ci_lg,
                'dirs_word': dirs_word}

    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2933, 'model': 'qwen3-4b', 'prereg': PREREG,
           'anchors': {'a1_diff': float('%.3e' % a1_diff),
                       'a1_ok': a1_ok,
                       'a2_rel': float('%.3e' % a2_rel),
                       'a2_ok': a2_ok,
                       'a3_val': round(a3_val, 6),
                       'a3_ok': a3_ok,
                       'a4_ok': a4_ok,
                       'a5_sep': round(sep, 6),
                       'a5_ok': a5_ok,
                       'a6_diff': float('%.3e' % a6_diff),
                       'a6_ok': a6_ok,
                       'ok': anchor_ok},
           'P2': p2, 'P3': p3, 'P4': p4,
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if save:
        np.savez_compressed(os.path.join(OUT, 'full_atlas_ci.npz'),
                            **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2933 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
