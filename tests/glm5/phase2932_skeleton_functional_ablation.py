# -*- coding: utf-8 -*-
"""Phase 2932: functional criterion for the convention-invariant skeleton.

Why: 2929 found a probe-invariant response-structure skeleton
(rho >= layer null p95), 2930 showed rho is convention-invariant
while maxT selection is convention-relative (lin_r-moderated),
2931 showed the skeleton overlap across conventions exceeds the
independence null 1.9x. Missing piece: is the convention-
invariant skeleton FUNCTIONALLY load-bearing, i.e. does causal
ablation of its cells move the behavioral readout more than
matched background cells - and does the lin_r stratification
(quasi-linear L2-L17 vs deep nonlinear L22-L28,30-35) predict
functional impact?

Mode: one run (qwen3-4b). REAL causal ablation, not analytic:
o_proj-input head slice zeroed at pos 1 (head h of layer l
writes nothing to the residual stream), full batched forward
(57 func-condition prompts verbatim 2887, tokens [the, w]).
Readout: final-position final residual projected on
dirs_word[35] (language axis, rebuilt this run, anchored to
2927 npz). CI_raw(cell) = mean_w |s_abl - s_base|;
CI_rel = CI_raw / mean_w |s_base|.

Cells: shared gated skeleton (2931 S29_corrected &
Smir_corrected, 295 cells) + matched background: per layer,
2 non-skeleton heads per skeleton cell dealt from a rng-2907
permutation of the same-layer pool (deduplicated for compute).

Anchors (frozen):
  a1 dirs_word(this run) vs 2927 npz max abs < 1e-5
  a2 batched baseline determinism: second run final residual
     max rel diff < 1e-4
  a3 hook efficacy: ablating ALL 32 heads at L18 gives
     mean |ds| > 0.01 * scale
  a4 mask counts: n86 == 469, nwd == 382, shared == 295
  a5 baseline separation mean(lab0 s) - mean(lab1 s) > 0

P1 (main, frozen): D_s = CI_rel(s) - mean(CI_rel of its 2
matched bg) over 295 skeleton cells; one-sided sign-permutation
(10000, rng 2908) on median D.
  anchor fail                          => anchor_fail_all_void
  median D > 0 and p1 <= 0.01          =>
      skeleton_functionally_load_bearing
  median D <= 0 or p1 > 0.05           => skeleton_epiphenomenal
  else                                 =>
      skeleton_partially_load_bearing

P2: skeleton CI_rel in QL layers (lin_r < 0.9) vs DEEP layers
(lin_r > 1.4), two-sided label permutation (10000, rng 2909).
P3: Spearman(CI_rel, rho_2929) and (CI_rel, rho_mirror) over
skeleton cells, permutation p (5000, rng 2910).
P4 (descriptive): per-layer median CI_rel; survivor 7 table;
CI_logit (target-token logit impact) vs CI_rel.

Output: phase2932/skeleton_functional_ablation/.
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
OUT = os.path.join(BASE, 'phase2932', 'skeleton_functional_ablation')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2932_run_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
SEED = 2896
NH, HD = 32, 128
NL = 36
N_SIGN = 10000
N_RHO = 5000
RNG_BG = 2907
RNG_SIGN = 2908
RNG_GRP = 2909
RNG_RHO = 2910
SURV = [(1, 6), (5, 6), (7, 19), (8, 2), (14, 9), (20, 8),
        (21, 6)]
EXP_N86, EXP_NWD, EXP_SHARED = 469, 382, 295

PREREG = {
    'mode': 'REAL causal ablation: o_proj-input head slice '
            'zeroed at pos 1 per cell (h, l), full batched '
            'forward over 57 func-condition prompts; readout '
            'final residual projected on dirs_word[35]; '
            '2917/2927 protocol verbatim for word probe rebuild',
    'question': 'is the convention-invariant response-structure '
                'skeleton (2929/2930/2931) functionally '
                'load-bearing under causal ablation, and does '
                'the lin_r stratification predict functional '
                'impact?',
    'cells': 'shared gated skeleton 295 cells + per-layer '
             'matched background (2 non-skeleton heads per '
             'skeleton cell, rng 2907 permutation, dedup)',
    'readout': 's_w = <fin_residual[w, -1], dirs_word[35]>; '
               'CI_raw = mean_w |s_abl - s_base|; '
               'CI_rel = CI_raw / mean_w |s_base|; secondary '
               'CI_logit = mean_w |dlogit(target w)|',
    'anchors': {
        'a1': 'dirs_word rebuild vs 2927 npz max abs < 1e-5',
        'a2': 'batched baseline determinism max rel < 1e-4',
        'a3': 'all-heads L18 ablation mean|ds| > 0.01*scale',
        'a4': 'n86==469 nwd==382 shared==295',
        'a5': 'baseline separation mean(lab0)-mean(lab1) > 0',
    },
    'P1': 'D_s = CI_rel(s) - mean(matched bg) over 295 cells; '
          'one-sided sign-permutation 10000 rng 2908 on median; '
          'median D > 0 & p <= 0.01 => '
          'skeleton_functionally_load_bearing; median D <= 0 or '
          'p > 0.05 => skeleton_epiphenomenal; else '
          'skeleton_partially_load_bearing',
    'P2': 'QL (lin_r<0.9) vs DEEP (lin_r>1.4) skeleton CI_rel, '
          'two-sided label permutation 10000 rng 2909',
    'P3': 'Spearman(CI_rel, rho_2929) and (CI_rel, rho_mirror) '
          'on skeleton cells, perm 5000 rng 2910',
    'P4': 'per-layer median CI_rel; survivor 7; CI_logit vs '
          'CI_rel (descriptive)',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'median D > 0 & p1 <= 0.01 => '
               'skeleton_functionally_load_bearing; '
               'median D <= 0 or p1 > 0.05 => '
               'skeleton_epiphenomenal; else '
               'skeleton_partially_load_bearing',
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
        json.dump({'phase': 2932,
                   'name': 'skeleton_functional_ablation',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2887': sha8(SRC_2887),
                               's2927': sha8(SRC_2927),
                               's2929': sha8(SRC_2929),
                               's2930': sha8(SRC_2930),
                               's2931': sha8(SRC_2931)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'seed': SEED, 'n_sign': N_SIGN,
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    # ---------- sources (zero-forward cell plan) ----------
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

    ql_layers = [li for li in range(1, NL) if lin_r[li] < 0.9]
    deep_layers = [li for li in range(NL) if lin_r[li] > 1.4]
    mid_layers = [li for li in range(1, NL)
                  if 0.9 <= lin_r[li] <= 1.4]
    log('QL layers %s | DEEP %s | MID %s'
        % (ql_layers, deep_layers, mid_layers), lines)

    cells_skel = sorted((h, li) for h in range(NH)
                        for li in range(1, NL)
                        if shared[h, li])
    n_sk_by_layer = {}
    for (h, li) in cells_skel:
        n_sk_by_layer[li] = n_sk_by_layer.get(li, 0) + 1
    matched = {}
    rng_bg = np.random.default_rng(RNG_BG)
    unique_bg = set()
    for li in sorted(n_sk_by_layer):
        pool = [h for h in range(NH)
                if not S29c[h, li] and not Smirc[h, li]]
        perm = rng_bg.permutation(len(pool))
        sk_here = [c for c in cells_skel if c[1] == li]
        for i, c in enumerate(sk_here):
            m1 = pool[int(perm[(2 * i) % len(pool)])]
            m2 = pool[int(perm[(2 * i + 1) % len(pool)])]
            matched[c] = (m1, m2)
            unique_bg.add((m1, li))
            unique_bg.add((m2, li))
    cells_all = cells_skel + sorted(unique_bg)
    log('cells: skeleton %d | unique bg %d | total %d'
        % (len(cells_skel), len(unique_bg), len(cells_all)),
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

    anchor_ok = bool(a1_ok and a2_ok and a3_ok and a4_ok
                     and a5_ok)

    verdict = None
    p1 = p2 = p3 = p4 = None
    save = {}
    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        # ---------- ablation sweep ----------
        ci_raw = {}
        ci_logit = {}
        s_abl_mat = np.zeros((len(cells_all), n_words))
        tids_arr = [tid_map[w] for _, _, w in words]
        for k, (h, li) in enumerate(cells_all):
            abl['li'], abl['h'], abl['all'] = li, h, False
            fin_a, lg_a = forward_batch(batch_toks)
            abl['li'], abl['h'], abl['all'] = \
                None, None, False
            s_a = fin_a @ u35
            d = s_a - s_base
            ci_raw[(h, li)] = float(np.mean(np.abs(d)))
            dlg = lg_a[np.arange(n_words), tids_arr] \
                - lg1[np.arange(n_words), tids_arr]
            ci_logit[(h, li)] = float(np.mean(np.abs(dlg)))
            s_abl_mat[k] = s_a
            if (k + 1) % 100 == 0:
                log('ablation [%d/%d]' % (k + 1,
                                          len(cells_all)), lines)
        ci_rel = {c: ci_raw[c] / scale for c in cells_all}
        log('ablation sweep done (%d cells, CI_rel max %.4f)'
            % (len(cells_all), max(ci_rel.values())), lines)

        # ---------- P1 (main) ----------
        D = np.array([ci_rel[c] - 0.5 * (ci_rel[(
            matched[c][0], c[1])] + ci_rel[(matched[c][1],
                                           c[1])])
                      for c in cells_skel])
        med_D = float(np.median(D))
        rng_s = np.random.default_rng(RNG_SIGN)
        sgns = rng_s.choice(np.array([-1.0, 1.0]),
                            size=(N_SIGN, len(D)))
        med_perm = np.median(D[None, :] * sgns, axis=1)
        p1_val = float((np.sum(med_perm >= med_D) + 1)
                       / (N_SIGN + 1))
        ci_sk = np.array([ci_rel[c] for c in cells_skel])
        ci_bg = np.array([ci_rel[c] for c in sorted(unique_bg)])
        if med_D > 0 and p1_val <= 0.01:
            verdict = 'skeleton_functionally_load_bearing'
        elif med_D <= 0 or p1_val > 0.05:
            verdict = 'skeleton_epiphenomenal'
        else:
            verdict = 'skeleton_partially_load_bearing'
        p1 = {'n_skel': len(cells_skel),
              'median_D': round(med_D, 6),
              'p1': float('%.3e' % p1_val),
              'median_ci_rel_skel':
                  round(float(np.median(ci_sk)), 6),
              'median_ci_rel_bg':
                  round(float(np.median(ci_bg)), 6),
              'ratio_medians': round(
                  float(np.median(ci_sk))
                  / max(float(np.median(ci_bg)), 1e-30), 4)}
        log('P1: medD %.6f p %.3e | skel %.6f vs bg %.6f '
            '(x%.2f) | VERDICT-BASIS: %s'
            % (med_D, p1_val, np.median(ci_sk),
               np.median(ci_bg),
               np.median(ci_sk) / max(np.median(ci_bg), 1e-30),
               verdict), lines)

        # ---------- P2: QL vs DEEP ----------
        ci_ql = [ci_rel[c] for c in cells_skel
                 if c[1] in set(ql_layers)]
        ci_dp = [ci_rel[c] for c in cells_skel
                 if c[1] in set(deep_layers)]
        stat_obs = float(np.median(ci_ql) - np.median(ci_dp))
        rng_g = np.random.default_rng(RNG_GRP)
        pool2 = np.array(ci_ql + ci_dp, dtype=np.float64)
        nq = len(ci_ql)
        cnt2 = 0
        for _ in range(N_SIGN):
            pm = rng_g.permutation(len(pool2))
            st = float(np.median(pool2[pm[:nq]])
                       - np.median(pool2[pm[nq:]]))
            if abs(st) >= abs(stat_obs):
                cnt2 += 1
        p2_val = float((cnt2 + 1) / (N_SIGN + 1))
        p2 = {'n_ql': nq, 'n_deep': len(ci_dp),
              'median_ci_ql': round(float(np.median(ci_ql)), 6),
              'median_ci_deep':
                  round(float(np.median(ci_dp)), 6),
              'diff': round(stat_obs, 6),
              'p2': float('%.3e' % p2_val)}
        log('P2: QL %.6f (n=%d) vs DEEP %.6f (n=%d) diff %.6f '
            'p %.3e'
            % (np.median(ci_ql), nq, np.median(ci_dp),
               len(ci_dp), stat_obs, p2_val), lines)

        # ---------- P3: CI vs rho ----------
        rho86_sk = np.array([rho_grid_29[c] for c in cells_skel])
        rhomir_sk = np.array([rho_mir_30[c]
                              for c in cells_skel])
        r86_obs = spearman(ci_sk, rho86_sk)
        rmir_obs = spearman(ci_sk, rhomir_sk)
        rng_r = np.random.default_rng(RNG_RHO)

        def perm_p(x, y, r_obs):
            cnt = 0
            for _ in range(N_RHO):
                if abs(spearman(x, y[rng_r.permutation(
                        len(y))])) >= abs(r_obs):
                    cnt += 1
            return float((cnt + 1) / (N_RHO + 1))

        p3_86 = perm_p(ci_sk, rho86_sk, r86_obs)
        p3_mir = perm_p(ci_sk, rhomir_sk, rmir_obs)
        p3 = {'spearman_ci_rho86': round(r86_obs, 4),
              'p_86': float('%.3e' % p3_86),
              'spearman_ci_rho_mirror': round(rmir_obs, 4),
              'p_mirror': float('%.3e' % p3_mir)}
        log('P3: rho(ci, rho86) %.4f p %.3e | rho(ci, '
            'rho_mirror) %.4f p %.3e'
            % (r86_obs, p3_86, rmir_obs, p3_mir), lines)

        # ---------- P4: descriptive ----------
        layer_med = {}
        for li in range(1, NL):
            vals = [ci_rel[c] for c in cells_skel
                    if c[1] == li]
            if vals:
                layer_med[li] = round(float(np.median(vals)),
                                      6)
        surv_tab = [{'cell': list(c),
                     'ci_rel': round(float(ci_rel[c]), 6),
                     'rho86': round(
                         float(rho_grid_29[c]), 4),
                     'layer_lin_r': round(
                         float(lin_r[c[1]]), 4)}
                    for c in SURV if c in ci_rel]
        ci_lg = np.array([ci_logit[c] for c in cells_skel])
        p4 = {'layer_median_ci_rel': layer_med,
              'survivor': surv_tab,
              'spearman_ci_rel_ci_logit':
                  round(spearman(ci_sk, ci_lg), 4),
              'median_ci_logit_skel':
                  round(float(np.median(ci_lg)), 6)}
        log('P4 survivor: %s' % [(d['cell'], d['ci_rel'])
                                 for d in surv_tab], lines)

        save = {'s_base': s_base.astype(np.float32),
                's_abl': s_abl_mat.astype(np.float32),
                'cells_all': np.array(cells_all, dtype=np.int64),
                'cells_skel': np.array(cells_skel,
                                       dtype=np.int64),
                'ci_rel': np.array([ci_rel[c]
                                    for c in cells_all]),
                'ci_logit': np.array([ci_logit[c]
                                      for c in cells_all]),
                'dirs_word': dirs_word}

    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2932, 'model': 'qwen3-4b', 'prereg': PREREG,
           'anchors': {'a1_diff': float('%.3e' % a1_diff),
                       'a1_ok': a1_ok,
                       'a2_rel': float('%.3e' % a2_rel),
                       'a2_ok': a2_ok,
                       'a3_val': round(a3_val, 6),
                       'a3_ok': a3_ok,
                       'a4_ok': a4_ok,
                       'a5_sep': round(sep, 6),
                       'a5_ok': a5_ok,
                       'ok': anchor_ok},
           'P1': p1, 'P2': p2, 'P3': p3, 'P4': p4,
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if save:
        np.savez_compressed(
            os.path.join(OUT,
                         'skeleton_functional_ablation.npz'),
            **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2932 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
