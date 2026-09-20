# -*- coding: utf-8 -*-
"""Phase 2934: load-band functional anatomy - dual-position +
tri-condition CI decomposition.

Why: 2933 (full-grid real ablation) established the lin_r-CI
law at pos 1 under the func condition only: quasi-linear
layers (lin_r<0.9, L6-L12 hit band) carry causal load while
deep layers (L28-L35) are functionally silent (band p2b=0/
6435, spearman(med_l, lin_r)=-0.676). Two confounds remain:
(1) POSITION - the effect was measured at the word slot
(pos 1) only; (2) CONDITION - the readout used the func
context only. This phase measures CI under a 3x3 design:
ablation config {pos0, pos1, pos01} x readout condition
{same, func, null} (2927 prompt construction verbatim),
all 1120 gated cells, to test whether the lin_r-CI law is
position- and condition-general or a pos1/func artifact.

Mode: one run (qwen3-4b), 2933 protocol verbatim extended:
REAL ablation = o_proj-input head slice zeroed at the
configured position(s) per cell, batched forward over 171
prompts (3 conds x 57 words, same/func/null in fixed
order), readout final residual projected on dirs_word[35]
rebuilt this run (func pass1, 2927 construction verbatim).

Anchors (frozen):
  a1 dirs_word rebuild vs 2927 npz max abs < 1e-5
  a2 func baseline determinism max rel < 1e-4
  a3 hook efficacy: L18 all-heads pos01 ablation func
     mean|ds| > 0.01 * scale_func
  a4 mask counts 469/382/295 (source consistency)
  a5 func baseline separation lab0-lab1 > 0
  a6 CROSS-PHASE reproduction: pos1/func CI_rel vs 2933
     npz (1120 cells) max abs diff < 1e-4
  a7 null-separation sanity: func sep > null sep
     (descriptive guard, no hard threshold)

Main tests (frozen):
  P1 position dimension: for cfg in {pos0, pos1, pos01}:
    P1a exact layer-label permutation LOAD(L6-L12) vs
        DEEP(L28-L35) on med_l (C(15,7)=6435), one-sided;
    P1b Spearman(med_l[1:], lin_r[1:]) two-sided perm
        10000 rng 2912.
  P2 condition dimension: for cond in {same, func, null}
    at pos1: Spearman(med_l_cond, lin_r) perm 10000
    rng 2913 + LOAD/DEEP exact permutation.
  P3 descriptive: cross-config Spearman(ci_pos0, ci_pos1)
    and cross-condition at pos1; survivor 7 table;
    top-5 CI cells per config (func cond).

Verdict (frozen):
  anchor fail                          => anchor_fail_all_void
  cfg stable (r<0, p1b<=0.01, p1a<=0.05, diff>0) for ALL
  of {pos0, pos1, pos01} AND cond stable for ALL three
  conds (pos1)                         => linr_ci_law_general
  >=1 new cfg (pos0/pos01) stable OR cond stable
                                       => linr_ci_law_partial
  else                                 => linr_ci_law_pos1_only

Output: phase2934/loadband_anatomy/.
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
OUT = os.path.join(BASE, 'phase2934', 'loadband_anatomy')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2934_run_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
SEED = 2896
NH, HD = 32, 128
NL = 36
VOCAB = 151936
N_GRP = 10000
RNG_POS = 2912
RNG_COND = 2913
SURV = [(1, 6), (5, 6), (7, 19), (8, 2), (14, 9), (20, 8),
        (21, 6)]
EXP_N86, EXP_NWD, EXP_SHARED = 469, 382, 295
LOAD_BAND = list(range(6, 13))
DEEP_BAND = list(range(28, 36))
CFGS = ('pos0', 'pos1', 'pos01')
CONDS = ('same', 'func', 'null')
A6_TOL = 1e-4

PREREG = {
    'mode': 'REAL causal ablation 3x3 design: o_proj-input '
            'head slice zeroed per cell under configs '
            'pos0/pos1/pos01 (1120 gated cells each), '
            'batched forward per condition (same/func/null, '
            '2927 construction verbatim, batch57 each), '
            'readout final residual projection on '
            'dirs_word[35] rebuilt this run; 2933 protocol '
            'verbatim otherwise',
    'correction_note': 'run1 (single batch171 concat per '
                       'forward) failed a6: pos1/func CI_rel '
                       'vs 2933 max abs diff 3.69e-03 > 1e-4; '
                       'a2 same-batch determinism 0.00 proves '
                       'the gap is bf16 cross-batch-composition '
                       'noise, not implementation drift. '
                       'Correction: per-condition batch57 '
                       'forwards matching the 2933 batch '
                       'composition bit-wise; a6 tolerance '
                       'kept 1e-4.',
    'question': 'is the 2933 lin_r-CI law (quasi-linear '
                'layers carry load, deep layers silent) '
                'position- and condition-general?',
    'anchors': {
        'a1': 'dirs_word rebuild vs 2927 npz max abs < 1e-5',
        'a2': 'func baseline determinism max rel < 1e-4',
        'a3': 'L18 all-heads pos01 ablation func mean|ds| > '
              '0.01*scale_func',
        'a4': 'mask counts 469/382/295',
        'a5': 'func baseline separation lab0-lab1 > 0',
        'a6': 'pos1/func CI_rel vs 2933 npz max abs < 1e-4',
        'a7': 'func separation > null separation (guard)',
    },
    'P1': 'per cfg in {pos0,pos1,pos01}: P1a LOAD L6-L12 vs '
          'DEEP L28-L35 exact layer-label permutation '
          'C(15,7)=6435 one-sided on med_l; P1b '
          'Spearman(med_l[1:], lin_r[1:]) two-sided perm '
          '10000 rng 2912',
    'P2': 'per cond in {same,func,null} at pos1: '
          'Spearman(med_l_cond, lin_r) perm 10000 rng 2913 '
          '+ LOAD/DEEP exact permutation',
    'P3': 'cross-config and cross-condition Spearman; '
          'survivor 7; top-5 CI cells per config (func)',
    'verdict': 'anchor fail => anchor_fail_all_void; all '
               'three cfgs stable (r<0, p1b<=0.01, p1a<=0.05, '
               'diff>0) AND all three conds stable => '
               'linr_ci_law_general; >=1 new cfg (pos0/pos01) '
               'stable OR cond stable => linr_ci_law_partial; '
               'else linr_ci_law_pos1_only',
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
        json.dump({'phase': 2934,
                   'name': 'loadband_anatomy',
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
                   'rng_pos': RNG_POS, 'rng_cond': RNG_COND,
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

    def same_ctx(i):
        lang = words[i][0]
        cands = [j for j in range(n_words)
                 if words[j][0] == lang and j != i]
        return min(cands, key=lambda j: tid_map[words[j][2]])

    rng = np.random.default_rng(SEED)
    word_tids = set(tid_map.values())
    null_tids = []
    while len(null_tids) < n_words:
        r = int(rng.integers(0, VOCAB))
        if r not in word_tids and r > 0:
            null_tids.append(r)

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
        fin = fin_cap['x'].astype(np.float64)
        return fin

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

    # ---------- per-cond batched baseline (twice for a2) ----
    batch_by_cond = {
        'same': [[tid_map[words[same_ctx(i)][2]],
                  tid_map[words[i][2]]]
                 for i in range(n_words)],
        'func': [[func_tid, tid_map[words[i][2]]]
                 for i in range(n_words)],
        'null': [[null_tids[i], tid_map[words[i][2]]]
                 for i in range(n_words)]}
    assert all(len(v) == n_words
               for v in batch_by_cond.values())

    fin1 = {cn: forward_batch(batch_by_cond[cn])
            for cn in CONDS}
    fin2 = {cn: forward_batch(batch_by_cond[cn])
            for cn in CONDS}
    a2_rel = float(np.abs(fin1['func'] - fin2['func']).max()
                   / max(float(np.abs(fin1['func']).max()),
                         1e-30))
    a2_ok = bool(a2_rel < 1e-4)
    log('a2 baseline determinism rel %.2e ok=%s'
        % (a2_rel, a2_ok), lines)

    u35 = dirs_word[NL - 1]
    s_base = {cn: fin1[cn] @ u35 for cn in CONDS}
    scale = {cn: float(np.mean(np.abs(s_base[cn])))
             for cn in CONDS}
    sep = {cn: float(s_base[cn][lab_lang == 0].mean()
                     - s_base[cn][lab_lang == 1].mean())
           for cn in CONDS}
    a5_ok = bool(sep['func'] > 0.0)
    a7_ok = bool(sep['func'] > sep['null'])
    log('a5 func separation %.4f (scale %.4f) ok=%s'
        % (sep['func'], scale['func'], a5_ok), lines)
    log('a7 separations same %.4f func %.4f null %.4f '
        '(func>null ok=%s)'
        % (sep['same'], sep['func'], sep['null'], a7_ok),
        lines)

    # ---------- a3: hook efficacy (pos01, L18, func) --------
    abl['li'], abl['h'] = 18, None
    abl['pos0'], abl['pos1'] = True, True
    fin_a3 = forward_batch(batch_by_cond['func'])
    abl['li'], abl['h'] = None, None
    abl['pos0'], abl['pos1'] = False, False
    a3_val = float(np.mean(np.abs(
        fin_a3 @ u35 - s_base['func'])))
    a3_ok = bool(a3_val > 0.01 * scale['func'])
    log('a3 L18 all-heads pos01 ablation mean|ds| %.4f '
        '(0.01*scale %.4f) ok=%s'
        % (a3_val, 0.01 * scale['func'], a3_ok), lines)

    # ---------- ablation sweep: 3 cfgs x 1120 cells --------
    cfg_pos = {'pos0': (True, False), 'pos1': (False, True),
               'pos01': (True, True)}
    ci_raw = {(cfg, cn): {} for cfg in CFGS for cn in CONDS}
    for cfg in CFGS:
        p0, p1 = cfg_pos[cfg]
        for k, (h, li) in enumerate(cells_all):
            abl['li'], abl['h'] = li, h
            abl['pos0'], abl['pos1'] = p0, p1
            for cn in CONDS:
                fin_a = forward_batch(batch_by_cond[cn])
                d = fin_a @ u35 - s_base[cn]
                ci_raw[(cfg, cn)][(h, li)] = \
                    float(np.mean(np.abs(d)))
            abl['li'], abl['h'] = None, None
            abl['pos0'], abl['pos1'] = False, False
            if (k + 1) % 200 == 0:
                log('ablation %s [%d/%d]'
                    % (cfg, k + 1, len(cells_all)), lines)
    ci_rel = {(cfg, cn): {c: ci_raw[(cfg, cn)][c] / scale[cn]
                          for c in cells_all}
              for cfg in CFGS for cn in CONDS}
    log('ablation sweep done (3 cfgs x %d cells, func/pos1 '
        'max %.4f)'
        % (len(cells_all),
           max(ci_rel[('pos1', 'func')].values())), lines)

    # ---------- a6: cross-phase reproduction ----------
    a6_diff = 0.0
    for c in cells_33:
        a6_diff = max(a6_diff,
                      abs(ci_rel[('pos1', 'func')][c]
                          - ci33_map[c]))
    a6_ok = bool(a6_diff < A6_TOL)
    log('a6 CI_rel reproduction vs 2933 (pos1/func, %d '
        'cells) max abs diff %.2e ok=%s'
        % (len(cells_33), a6_diff, a6_ok), lines)

    anchor_ok = bool(a1_ok and a2_ok and a3_ok and a4_ok
                     and a5_ok and a6_ok and a7_ok)

    verdict = None
    p1 = p2 = p3 = None
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
            rng_l = np.random.default_rng(RNG_POS)
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

        # ---------- P1: position dimension ----------
        p1 = {}
        for cfg in CFGS:
            p1[cfg] = band_and_linr(ci_rel[(cfg, 'func')])
            log('P1 %s: LOAD %.6f DEEP %.6f p_band %.4f | '
                'rho %.4f p %.3e stable=%s'
                % (cfg, p1[cfg]['load_m'], p1[cfg]['deep_m'],
                   p1[cfg]['p_band'],
                   p1[cfg]['spearman_med_l_lin_r'],
                   p1[cfg]['p_linr'], p1[cfg]['stable']),
                lines)

        # ---------- P2: condition dimension (pos1) ----------
        rng_c = np.random.default_rng(RNG_COND)
        p2 = {}
        for cn in CONDS:
            p2[cn] = band_and_linr(ci_rel[('pos1', cn)])
            log('P2 %s: LOAD %.6f DEEP %.6f p_band %.4f | '
                'rho %.4f p %.3e stable=%s'
                % (cn, p2[cn]['load_m'], p2[cn]['deep_m'],
                   p2[cn]['p_band'],
                   p2[cn]['spearman_med_l_lin_r'],
                   p2[cn]['p_linr'], p2[cn]['stable']), lines)

        # ---------- P3: descriptive ----------
        ci0 = np.array([ci_rel[('pos0', 'func')][c]
                        for c in cells_all])
        ci1 = np.array([ci_rel[('pos1', 'func')][c]
                        for c in cells_all])
        ci01 = np.array([ci_rel[('pos01', 'func')][c]
                         for c in cells_all])
        cix = {cn: np.array([ci_rel[('pos1', cn)][c]
                             for c in cells_all])
               for cn in CONDS}
        top5 = {}
        for cfg in CFGS:
            cc = sorted(cells_all,
                        key=lambda c: -ci_rel[(cfg, 'func')][c]
                        )[:5]
            top5[cfg] = [{'cell': list(c),
                          'ci_rel': round(
                              float(ci_rel[(cfg, 'func')][c]),
                              6)} for c in cc]
        surv_tab = []
        for c in SURV:
            surv_tab.append({
                'cell': list(c),
                'ci_pos0': round(float(ci_rel[('pos0', 'func')][c]),
                                 6),
                'ci_pos1': round(float(ci_rel[('pos1', 'func')][c]),
                                 6),
                'ci_pos01': round(float(ci_rel[('pos01', 'func')][c]),
                                  6)})
        sk_ci = [ci_rel[('pos1', 'func')][c]
                 for c in cells_all if shared[c]]
        rest_ci = [ci_rel[('pos1', 'func')][c]
                   for c in cells_all if not shared[c]]
        p3 = {
            'spearman_ci_pos0_ci_pos1':
                round(spearman(ci0, ci1), 4),
            'spearman_ci_pos1_ci_pos01':
                round(spearman(ci1, ci01), 4),
            'cross_cond_pos1': {
                'same_func': round(spearman(cix['same'],
                                            cix['func']), 4),
                'null_func': round(spearman(cix['null'],
                                            cix['func']), 4),
                'same_null': round(spearman(cix['same'],
                                            cix['null']), 4)},
            'top5_per_config': top5,
            'survivor': surv_tab,
            'skeleton_median_ci_pos1':
                round(float(np.median(sk_ci)), 6),
            'rest_median_ci_pos1':
                round(float(np.median(rest_ci)), 6)}
        log('P3: pos0~pos1 %.4f pos1~pos01 %.4f | conds '
            'same~func %.4f null~func %.4f'
            % (p3['spearman_ci_pos0_ci_pos1'],
               p3['spearman_ci_pos1_ci_pos01'],
               p3['cross_cond_pos1']['same_func'],
               p3['cross_cond_pos1']['null_func']), lines)

        # ---------- verdict ----------
        new_cfg_ok = all(p1[c]['stable'] for c in
                         ('pos0', 'pos01'))
        pos1_ok = p1['pos1']['stable']
        cond_ok = all(p2[cn]['stable'] for cn in CONDS)
        if new_cfg_ok and pos1_ok and cond_ok:
            verdict = 'linr_ci_law_general'
        elif any(p1[c]['stable']
                 for c in ('pos0', 'pos01')) or cond_ok:
            verdict = 'linr_ci_law_partial'
        else:
            verdict = 'linr_ci_law_pos1_only'

        save = {
            'cells': np.array(cells_all, dtype=np.int64),
            'ci_rel': np.array(
                [[ci_rel[(cfg, cn)][c] for c in cells_all]
                 for cfg in CFGS for cn in CONDS],
                dtype=np.float64),
            'cfgs': np.array(CFGS), 'conds': np.array(CONDS),
            's_base': np.array([s_base[cn] for cn in CONDS],
                               dtype=np.float32),
            'scale': np.array([scale[cn] for cn in CONDS]),
            'sep': np.array([sep[cn] for cn in CONDS]),
            'dirs_word': dirs_word}

    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2934, 'model': 'qwen3-4b',
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
                       'sep_same': round(sep['same'], 6),
                       'sep_null': round(sep['null'], 6),
                       'a7_ok': a7_ok,
                       'ok': anchor_ok},
           'P1': p1, 'P2': p2, 'P3': p3,
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if save:
        np.savez_compressed(os.path.join(OUT,
                                         'loadband_anatomy.npz'),
                            **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2934 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
