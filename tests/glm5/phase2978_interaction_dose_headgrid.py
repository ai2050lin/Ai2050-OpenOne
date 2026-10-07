# -*- coding: utf-8 -*-
"""Phase 2978: Omega-B interaction dose-response grid +
per-head interaction carrier localization (preregistered).

Why: 2977 found a competitive sub-additive interaction at
the injection layer L17 (median I = -0.0236, p 2.4e-3,
only significant layer) with relative dose 0.1. Two open
questions: (1) is the sub-additive interaction robust
across the dose grid or confined to a dose window?
(2) which heads carry the interaction (head grid was
deferred in 2977 T3)?

Design (frozen before any observation):
  Stage 1: 74 base single forwards, 2973/2977 protocol
     verbatim (cells from 2977 execution.json); derive
     unit axis dirs d_lang, d_cls at L17 input (same
     recipe as 2977 -> identity anchor a5 vs 2977 npz).
  T1 dose grid: 24-word subset (rng 2980, stratified 6
     per cell) x 25 conditions (a,b) in {0,.05,.1,.2,.4}^2
     at L17 input pos1, injection v = a*n17_w*d_lang +
     b*n17_w*d_cls (relative dose, 2977 regime). 600
     forwards. Interaction per cell I(a,b) = R(a,b) -
     R(a,0) - R(0,b) + R(0,0), test at L17 (the only
     layer significant in 2977), sign-flip null rng 2981
     x10000, maxT family 24 nonzero cells, gate p<=0.01.
  T2 head grid: 74 words x 4 conditions {00,10,01,11} at
     authoritative dose 0.1 (2977 regime verbatim), 296
     forwards, capture per-head o_proj-input grid (36,32).
     I_grid = grid11-grid10-grid01+grid00, per (l,h)
     median_w, per-layer sign-flip maxT family 32,
     rng 2982, gate p<=0.01 (descriptive carrier layer;
     NOT part of main verdict - selection/structure
     separation discipline).
  T3 overlap calibration: L17 significant head set vs
     2964 T3.sig_heads (L34 carrier heads, quasi-post-hoc
     reference); null = random head draws matched in size,
     rng 2983 x2000 (discipline 11: selection-quantity
     overlap is chance-level by default, must calibrate).

Tests (frozen):
  T1 (PRIMARY): nonzero cells with p<=0.01 at L17;
     >=12 of 24 AND all their median I < 0 =>
     subadditive_dose_robust; 1..11 => subadditive_
     dose_windowed; 0 => dose_grid_interaction_ns.
  Dose-law descriptive (quasi-post-hoc, separated per
     discipline: monotonic vs peak tests are different
     claims): median I(L17) vs min(a,b) and vs a+b.
  T2: significant (l,h) cells registered; L17 head set.
  T3: overlap obs vs null p.

Anchors (frozen):
  a1 Vt8 rebuild vs 2939 npz < 1e-6
  a2 determinism < 1e-4
  a3 identity vs 2973 npz norms rel<1e-4, coss abs<1e-6
  a4 single-token 74/74
  a5 axis dirs unit-norm rebuild vs 2977 npz max|d|<1e-6
  a6 grid sham (0,0) == stage1 base bit-level (24 words)
  a7 injection locality (C_en[0], lang 0.1): layers<17
     identical, layers>=17 differ
  a9 cross-phase injection identity: subset 24-word
     R(0.1,0.1) at L17 vs 2977 prof11 subset max|d|<1e-12
     (same forward protocol; 1e-12 allows ulp-level
     association-order float difference)

Verdict (frozen):
  anchor fail => anchor_fail_all_void
  T1 nsig>=12 and all neg => subadditive_dose_robust
  T1 1<=nsig => subadditive_dose_windowed
  else => dose_grid_interaction_ns
"""
import hashlib
import io
import json
import os
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC_2939 = os.path.join(BASE, 'phase2939', 'rotation_target',
                        'rotation_target.npz')
SRC_2927 = os.path.join(BASE, 'phase2927', 'probe_relativity',
                        'probe_relativity.npz')
SRC_2973 = os.path.join(BASE, 'phase2973', 'fr_scale_audit',
                        'fr_scale_audit.npz')
SRC_2977_NPZ = os.path.join(BASE, 'phase2977',
                            'two_axis_fusion_injection',
                            'two_axis_fusion_injection.npz')
EXEC_2977 = os.path.join(BASE, 'phase2977',
                         'two_axis_fusion_injection',
                         'execution.json')
RES_2964 = os.path.join(BASE, 'phase2964',
                        'carrier_anatomy', 'result.json')
OUT = os.path.join(BASE, 'phase2978',
                   'interaction_dose_headgrid')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2978_run_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NH, HD = 32, 128
NL = 36
N_PERM = 10000
P_TH = 0.01
L_INJ = 17
S_GRID = [0.0, 0.05, 0.1, 0.2, 0.4]
N_SUB = 24
RNG_SUB = 2980
RNG_T1 = 2981
RNG_T2H = 2982
RNG_OVL = 2983
N_OVL = 2000

PREREG = {
    'mode': 'stage1: 74 base forwards (2973/2977 '
            'protocol), derive unit axis dirs; T1: 24-word '
            'subset x 25-condition dose grid (a,b) in '
            '{0,.05,.1,.2,.4}^2 relative dose at L17 '
            'input; T2: 74x4 factorial at dose 0.1 with '
            'per-head grid capture; T3 overlap calibration',
    'question': 'is the L17 competitive sub-additive '
                'interaction (2977) robust across the '
                'dose grid, and which heads carry it?',
    'subset': '24 words, stratified 6 per cell, rng 2980 '
              '(design-time, preregistered below)',
    'anchors': {
        'a1': 'Vt8 rebuild vs 2939 npz < 1e-6',
        'a2': 'determinism < 1e-4',
        'a3': 'identity vs 2973 npz norms rel<1e-4, '
              'coss abs<1e-6 (74x36)',
        'a4': 'single-token 74/74',
        'a5': 'unit axis dirs rebuild vs 2977 npz '
              'max|d|<1e-6',
        'a6': 'grid sham (0,0) == base bit-level '
              '(24 words x 36 prof)',
        'a7': 'injection locality L17 (C_en[0])',
        'a9': 'R(0.1,0.1) L17 subset vs 2977 prof11 '
              'max|d|<1e-12',
    },
    'T1': 'interaction I(a,b)=R(a,b)-R(a,0)-R(0,b)+R(0,0) '
          'at L17, median_w; nonzero cells family 24, '
          'sign-flip null rng 2981 x10000 maxT; gate '
          'p<=0.01; verdict robust if >=12 sig and all '
          'negative medians, windowed if 1..11, else ns',
    'T2': 'head grid I per (l,h), median_w, per-layer '
          'sign-flip maxT family 32 rng 2982, gate '
          'p<=0.01; descriptive carrier registration, '
          'NOT in main verdict',
    'T3': 'L17 sig heads vs 2964 T3.sig_heads overlap, '
          'random-head null rng 2983 x2000 (discipline 11)',
    'dose_law': 'descriptive quasi-post-hoc: median '
                'I(L17) vs min(a,b) and vs a+b; monotonic '
                'vs peak are separate claims',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'T1 nsig>=12 all neg => '
               'subadditive_dose_robust; T1 1<=nsig => '
               'subadditive_dose_windowed; else => '
               'dose_grid_interaction_ns',
    'correction_note': '',
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


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)

    # ---------- cells + subset (design-time) ----------
    e77 = json.load(open(EXEC_2977, encoding='utf-8'))
    F_EN = e77['cells']['F_en']
    F_FR = e77['cells']['F_fr']
    C_EN = e77['cells']['C_en']
    C_FR = e77['cells']['C_fr']
    assert (len(F_EN), len(F_FR), len(C_EN), len(C_FR)) \
        == (15, 15, 22, 22), 'cell size drift'
    cells = [('F', 'en', w) for w in F_EN] \
        + [('F', 'fr', w) for w in F_FR] \
        + [('C', 'en', w) for w in C_EN] \
        + [('C', 'fr', w) for w in C_FR]
    words77 = ['%s:%s:%s' % c for c in cells]
    n_test = len(cells)
    rng_sub = np.random.default_rng(RNG_SUB)
    sub_idx = []
    for seg, lo, hi in ((0, 0, 15), (1, 15, 30),
                        (2, 30, 52), (3, 52, 74)):
        pick = rng_sub.choice(np.arange(lo, hi),
                              size=6, replace=False)
        sub_idx.extend(int(v) for v in sorted(pick))
    assert len(sub_idx) == N_SUB
    sub_words = [words77[i] for i in sub_idx]

    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2978,
                   'name': 'interaction_dose_headgrid',
                   'created': time.strftime(
                       '%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2939': sha8(SRC_2939),
                               's2927': sha8(SRC_2927),
                               's2973': sha8(SRC_2973),
                               's2977npz': sha8(SRC_2977_NPZ),
                               's2964res': sha8(RES_2964)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'n_perm': N_PERM, 'n_overlap_null': N_OVL,
                   'rng': {'subset': RNG_SUB, 'T1': RNG_T1,
                           'T2head': RNG_T2H,
                           'overlap': RNG_OVL},
                   'p_threshold': P_TH,
                   's_grid': S_GRID,
                   'subset_idx': sub_idx,
                   'subset_words': sub_words,
                   'cells': {'F_en': F_EN, 'F_fr': F_FR,
                             'C_en': C_EN, 'C_fr': C_FR},
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen (subset %s, grid %s)'
        % (N_SUB, len(S_GRID) ** 2), lines)

    # ---------- sources ----------
    z39 = np.load(SRC_2939, allow_pickle=True)
    z27 = np.load(SRC_2927, allow_pickle=True)
    dirs27 = z27['dirs_word'].astype(np.float64)
    _, _, Vt_loc = np.linalg.svd(dirs27,
                                 full_matrices=False)
    a1_diff = float(np.abs(Vt_loc[:8]
                           - z39['Vt8']).max())
    a1_ok = bool(a1_diff < 1e-6)
    log('a1 Vt8 rebuild diff %.2e ok=%s'
        % (a1_diff, a1_ok), lines)
    u35 = dirs27[NL - 1]
    z73 = np.load(SRC_2973, allow_pickle=True)
    norms73 = z73['norms'].astype(np.float64)
    coss73 = z73['coss'].astype(np.float64)
    words73 = [str(w) for w in z73['words']]
    z77 = np.load(SRC_2977_NPZ, allow_pickle=True)
    d77_l = z77['d_lang'].astype(np.float64)
    d77_c = z77['d_cls'].astype(np.float64)
    prof11_77 = z77['prof11'].astype(np.float64)
    r64 = json.load(open(RES_2964, encoding='utf-8'))
    ref_heads = [int(h) for h
                 in r64['T3']['sig_heads']]
    log('ref heads (2964 T3.sig_heads) = %s'
        % ref_heads, lines)

    # ---------- model ----------
    import sys
    sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
    from phase2662_symmetric_mapping_contract import \
        load_native
    from transformers import AutoTokenizer
    import torch

    tok = AutoTokenizer.from_pretrained(
        MD, local_files_only=True, trust_remote_code=True,
        use_fast=True)
    assert words77 == words73, 'word order drift vs 2973'
    tid_map = {}
    n_single = 0
    for _, _, w in cells:
        ids = tok(' ' + w, add_special_tokens=False)[
            'input_ids']
        if len(ids) != 1:
            ids = tok(w, add_special_tokens=False)[
                'input_ids']
        if len(ids) == 1:
            n_single += 1
        tid_map[w] = int(ids[0]) if len(ids) == 1 else -1
    a4_ok = bool(n_single == n_test)
    log('a4 single-token %d/%d ok=%s'
        % (n_single, n_test, a4_ok), lines)
    ids_the = tok(' the', add_special_tokens=False)[
        'input_ids']
    assert len(ids_the) == 1
    func_tid = int(ids_the[0])

    model, _ = load_native('qwen4')
    model.eval()
    layers = model.model.layers
    log('model loaded', lines)

    cap_op = {li: [] for li in range(NL)}
    cap_x17 = []
    inj_state = {'dirs': None}
    handles = []

    def hs_of(args, kwargs):
        if args and args[0] is not None:
            return args[0]
        return kwargs.get('hidden_states')

    def hook_inj(module, args, kwargs):
        d = inj_state['dirs']
        if d is not None:
            x = hs_of(args, kwargs)
            if x is not None:
                dt = torch.as_tensor(
                    d, device=x.device, dtype=x.dtype)
                x[:, 1, :] += dt
        return None

    def hook_x17(module, args, kwargs):
        x = hs_of(args, kwargs)
        if x is None or x.dim() < 2:
            return None
        cap_x17.append(
            x[:, 1, :].detach().float().cpu().numpy())
        return None

    def hook_op(li):
        def h(module, args, kwargs):
            x = args[0] if args else kwargs.get('input')
            if x is None or x.dim() < 2:
                return None
            cap_op[li].append(
                x[:, 1, :].detach().float().cpu().numpy())
            return None
        return h

    handles.append(
        layers[L_INJ].self_attn
        .register_forward_pre_hook(
            hook_inj, with_kwargs=True))
    handles.append(
        layers[L_INJ].self_attn
        .register_forward_pre_hook(
            hook_x17, with_kwargs=True))
    for li in range(NL):
        handles.append(
            layers[li].self_attn.o_proj
            .register_forward_pre_hook(
                hook_op(li), with_kwargs=True))

    def clear_cap():
        for li in cap_op:
            del cap_op[li][:]
        del cap_x17[:]

    def forward1(toks, dirs=None):
        clear_cap()
        inj_state['dirs'] = dirs
        with torch.no_grad():
            model(torch.tensor([toks], device='cuda'))
        inj_state['dirs'] = None
        return {li: cap_op[li][0].astype(np.float64)
                for li in range(NL)}

    M = np.zeros((NL, NH * HD))
    Mnorm = np.zeros(NL)
    for li in range(NL):
        Wo = layers[li].self_attn.o_proj.weight.detach() \
            .float().cpu().numpy()
        M[li] = u35 @ Wo
        Mnorm[li] = float(np.linalg.norm(M[li]))

    def profile(op, want_grid=False):
        prof = np.zeros(NL)
        grid = np.zeros((NL, NH)) if want_grid else None
        for li in range(NL):
            x = op[li].reshape(-1)
            xm = (x * M[li]).reshape(NH, HD)
            hs = xm.sum(axis=1)
            if want_grid:
                grid[li] = hs
            prof[li] = float(hs.sum())
        return prof, grid

    # ---------- stage 1: base sweep ----------
    prof_base = np.zeros((n_test, NL))
    norms = np.zeros((n_test, NL))
    coss = np.zeros((n_test, NL))
    X17 = np.zeros((n_test, 2560))
    for i, (_, _, w) in enumerate(cells):
        op = forward1([func_tid, tid_map[w]])
        X17[i] = cap_x17[0].astype(np.float64).reshape(-1)
        for li in range(NL):
            x = op[li].reshape(-1)
            norms[i, li] = float(np.linalg.norm(x))
            coss[i, li] = float(np.dot(x, M[li])) \
                / max(norms[i, li] * Mnorm[li], 1e-30)
        prof_base[i], _ = profile(op)
        if (i + 1) % 20 == 0:
            log('base sweep [%d/%d]' % (i + 1, n_test),
                lines)

    # ---------- a3 identity vs 2973 ----------
    nrm_rel = float(np.max(np.abs(norms - norms73)
                           / np.maximum(norms73, 1e-30)))
    cos_abs = float(np.max(np.abs(coss - coss73)))
    a3_ok = bool(nrm_rel < 1e-4 and cos_abs < 1e-6)
    log('a3 identity vs 2973: norms rel %.2e coss abs '
        '%.2e ok=%s' % (nrm_rel, cos_abs, a3_ok), lines)

    # ---------- axis directions + a5 ----------
    lang = np.array([0 if c[1] == 'en' else 1
                     for c in cells])
    cls = np.array([0 if c[0] == 'F' else 1
                    for c in cells])
    d_lang = X17[lang == 1].mean(axis=0) \
        - X17[lang == 0].mean(axis=0)
    d_cls = X17[cls == 1].mean(axis=0) \
        - X17[cls == 0].mean(axis=0)
    n_lang = float(np.linalg.norm(d_lang))
    n_cls = float(np.linalg.norm(d_cls))
    cos_axes = float(np.dot(d_lang, d_cls)
                     / max(n_lang * n_cls, 1e-30))
    assert n_lang > 0 and n_cls > 0, 'degenerate axis'
    d_lang_u = d_lang / n_lang
    d_cls_u = d_cls / n_cls
    a5_diff = float(max(
        np.abs(d_lang_u - d77_l / np.linalg.norm(d77_l)).max(),
        np.abs(d_cls_u - d77_c / np.linalg.norm(d77_c)).max()))
    a5_ok = bool(a5_diff < 1e-6)
    log('a5 axis dirs vs 2977: max|d| %.2e cos=%.4f ok=%s'
        % (a5_diff, cos_axes, a5_ok), lines)

    # ---------- a2 determinism ----------
    op_a = forward1([func_tid, tid_map[C_EN[0]]])
    op_b = forward1([func_tid, tid_map[C_EN[0]]])
    a2_rel = float(np.abs(op_a[30] - op_b[30]).max()
                   / max(float(np.abs(op_a[30]).max()),
                         1e-30))
    a2_ok = bool(a2_rel < 1e-4)
    log('a2 determinism rel %.2e ok=%s'
        % (a2_rel, a2_ok), lines)

    # ---------- T1 dose grid (24-word subset) ----------
    n17_all = np.linalg.norm(X17, axis=1)
    G = len(S_GRID)
    R_grid = np.zeros((G, G, N_SUB, NL))
    for ia in range(G):
        for ib in range(G):
            a, b = S_GRID[ia], S_GRID[ib]
            for j, i in enumerate(sub_idx):
                w = cells[i][2]
                toks = [func_tid, tid_map[w]]
                if a == 0.0 and b == 0.0:
                    op = forward1(toks, None)
                else:
                    dv = (a * n17_all[i]) * d_lang_u \
                        + (b * n17_all[i]) * d_cls_u
                    op = forward1(toks, dv)
                R_grid[ia, ib, j], _ = profile(op)
            log('grid [%d,%d] done (a=%.2f b=%.2f)'
                % (ia, ib, a, b), lines)

    # ---------- a6 grid sham == base ----------
    a6_diff = 0.0
    for j, i in enumerate(sub_idx):
        a6_diff = max(a6_diff, float(np.max(np.abs(
            R_grid[0, 0, j] - prof_base[i]))))
    a6_ok = bool(a6_diff == 0.0)
    log('a6 sham vs base max|d| %.2e ok=%s'
        % (a6_diff, a6_ok), lines)

    # ---------- a9 cross-phase injection identity ----------
    a9_diff = 0.0
    for j, i in enumerate(sub_idx):
        a9_diff = max(a9_diff, float(abs(
            R_grid[2, 2, j, L_INJ]
            - prof11_77[i, L_INJ])))
    a9_ok = bool(a9_diff < 1e-12)
    log('a9 R(.1,.1) L17 vs 2977 prof11: max|d| %.2e '
        'ok=%s' % (a9_diff, a9_ok), lines)

    # ---------- a7 injection locality ----------
    w0 = C_EN[0]
    toks0 = [func_tid, tid_map[w0]]
    i0 = words77.index('%s:%s:%s' % ('C', 'en', w0))
    op_s = forward1(toks0, None)
    op_l = forward1(toks0, (0.1 * n17_all[i0]) * d_lang_u)
    pre_same = all(float(np.abs(
        op_s[li] - op_l[li]).max()) == 0.0
        for li in range(L_INJ))
    post_diff = any(float(np.abs(
        op_s[li] - op_l[li]).max()) > 0.0
        for li in range(L_INJ, NL))
    a7_ok = bool(pre_same and post_diff)
    log('a7 locality: pre17 identical=%s post17 '
        'differ=%s ok=%s'
        % (pre_same, post_diff, a7_ok), lines)

    anchor_ok = bool(a1_ok and a2_ok and a3_ok and a4_ok
                     and a5_ok and a6_ok and a7_ok
                     and a9_ok)
    verdict = None
    t1 = t2 = t3 = dose_law = None
    sig_cells = []
    I_cells = None
    p_cells = None
    I_grid_med = None
    p_head = None
    sig_heads_L17 = []
    ovl = None

    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        # ---------- T1 interaction cells ----------
        I_cells = np.zeros((G, G, N_SUB, NL))
        for ia in range(G):
            for ib in range(G):
                I_cells[ia, ib] = (
                    R_grid[ia, ib] - R_grid[ia, 0]
                    - R_grid[0, ib] + R_grid[0, 0])
        cells_nz = [(ia, ib) for ia in range(G)
                    for ib in range(G)
                    if not (S_GRID[ia] == 0.0
                            and S_GRID[ib] == 0.0)]
        nc = len(cells_nz)
        stat_cells = np.array([
            np.median(I_cells[ia, ib, :, L_INJ])
            for ia, ib in cells_nz])
        rng1 = np.random.default_rng(RNG_T1)
        stack = np.stack([I_cells[ia, ib, :, L_INJ]
                          for ia, ib in cells_nz])
        signs = np.where(rng1.random(
            (N_PERM, N_SUB)) < 0.5, -1.0, 1.0)
        fam1 = np.zeros(N_PERM)
        for k in range(N_PERM):
            fam1[k] = float(np.abs(np.median(
                stack * signs[k][None, :],
                axis=1)).max())
        p_cells = np.array([
            (np.sum(fam1 >= abs(v) - 1e-12) + 1)
            / (N_PERM + 1) for v in stat_cells])
        sig_cells = [(cells_nz[c][0], cells_nz[c][1])
                     for c in range(nc)
                     if p_cells[c] <= P_TH]
        med_sig = [float(stat_cells[c])
                   for c in range(nc)
                   if p_cells[c] <= P_TH]
        nsig = len(sig_cells)
        all_neg = all(v < 0 for v in med_sig)
        log('T1 L17 sig cells %d/24: %s medians %s'
            % (nsig, ['(%.2f,%.2f)' % (S_GRID[a],
                                       S_GRID[b])
                      for a, b in sig_cells],
               ['%.4f' % v for v in med_sig]), lines)
        # dose law descriptive (quasi-post-hoc)
        dose_law = {}
        for mkey, mfun in (
                ('min_ab',
                 lambda a, b: min(a, b)),
                ('sum_ab', lambda a, b: a + b)):
            prof_m = {}
            for ia, ib in cells_nz:
                key = round(mfun(S_GRID[ia],
                                 S_GRID[ib]), 3)
                prof_m.setdefault(key, []).append(
                    float(stat_cells[cells_nz.index(
                        (ia, ib))]))
            dose_law[mkey] = {
                str(k): round(float(np.median(v)), 4)
                for k, v in sorted(prof_m.items())}
        log('T1 dose law %s' % json.dumps(dose_law),
            lines)

        # ---------- T2 head grid ----------
        grids = {c: np.zeros((n_test, NL, NH))
                 for c in ('00', '10', '01', '11')}
        for i, (_, _, w) in enumerate(cells):
            toks = [func_tid, tid_map[w]]
            s_w = 0.1 * n17_all[i]
            for key, dv in (
                    ('00', None),
                    ('10', s_w * d_lang_u),
                    ('01', s_w * d_cls_u),
                    ('11', s_w * d_lang_u
                     + s_w * d_cls_u)):
                op = forward1(toks, dv)
                _, g = profile(op, want_grid=True)
                grids[key][i] = g
            if (i + 1) % 20 == 0:
                log('headgrid [%d/74]' % (i + 1), lines)
        I_grid = (grids['11'] - grids['10']
                  - grids['01'] + grids['00'])
        I_grid_med = np.median(I_grid, axis=0)
        rng2 = np.random.default_rng(RNG_T2H)
        signs2 = np.where(rng2.random(
            (N_PERM, n_test)) < 0.5, -1.0, 1.0)
        p_head = np.zeros((NL, NH))
        for li in range(NL):
            Il = I_grid[:, li, :]
            fam2 = np.zeros(N_PERM)
            for k in range(N_PERM):
                fam2[k] = float(np.abs(np.median(
                    Il * signs2[k][:, None],
                    axis=0)).max())
            for h in range(NH):
                p_head[li, h] = (
                    np.sum(fam2
                           >= abs(I_grid_med[li, h])
                           - 1e-12) + 1) / (N_PERM + 1)
        sig_heads_L17 = [h for h in range(NH)
                         if p_head[L_INJ, h] <= P_TH]
        nsig_grid = int(np.sum(p_head <= P_TH))
        log('T2 sig (l,h) cells total %d; L17 sig '
            'heads %s' % (nsig_grid, sig_heads_L17),
            lines)

        # ---------- T3 overlap calibration ----------
        S17 = sig_heads_L17
        obs = len(set(S17) & set(ref_heads))
        rng3 = np.random.default_rng(RNG_OVL)
        cnt = 0
        for _ in range(N_OVL):
            draw = rng3.choice(NH, size=len(S17),
                               replace=False)
            if len(set(int(x) for x in draw)
                   & set(ref_heads)) >= obs:
                cnt += 1
        ovl = {'S17_heads': S17,
               'ref_heads_2964': ref_heads,
               'obs_overlap': obs,
               'null_p': (cnt + 1) / (N_OVL + 1)}
        log('T3 overlap obs=%d null p=%.4f'
            % (obs, ovl['null_p']), lines)

        # ---------- verdict ----------
        if nsig >= 12 and all_neg:
            verdict = 'subadditive_dose_robust'
        elif nsig >= 1:
            verdict = 'subadditive_dose_windowed'
        else:
            verdict = 'dose_grid_interaction_ns'
        save = {
            'R_grid': R_grid,
            'I_cells': I_cells,
            'stat_cells': stat_cells,
            'p_cells': p_cells,
            'sig_cells': np.array(sig_cells),
            'dose_law_min': np.array(
                [dose_law['min_ab'].get(str(round(v, 3)),
                                        np.nan)
                 for v in [0.0, 0.05, 0.1, 0.2, 0.4]]),
            'I_grid_med': I_grid_med,
            'p_head': p_head,
            'sig_heads_L17': np.array(sig_heads_L17),
            'ref_heads': np.array(ref_heads),
            'ovl_p': np.array([ovl['null_p']]),
            'd_lang_u': d_lang_u, 'd_cls_u': d_cls_u,
            'axes_cos': np.array([cos_axes]),
            'n17': n17_all, 'words': np.array(words77),
            'subset_idx': np.array(sub_idx)}

    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2978, 'model': 'qwen3-4b',
           'prereg': PREREG,
           'anchors': {
               'a1_diff': float('%.3e' % a1_diff),
               'a1_ok': a1_ok,
               'a2_rel': float('%.3e' % a2_rel),
               'a2_ok': a2_ok,
               'a3_nrm_rel': float('%.3e' % nrm_rel),
               'a3_cos_abs': float('%.3e' % cos_abs),
               'a3_ok': a3_ok, 'a4_ok': a4_ok,
               'a5_diff': float('%.3e' % a5_diff),
               'a5_ok': a5_ok,
               'a6_diff': float('%.3e' % a6_diff),
               'a6_ok': a6_ok, 'a7_ok': a7_ok,
               'a9_diff': float('%.3e' % a9_diff),
               'a9_ok': a9_ok,
               'axes_cos': round(cos_axes, 4),
               'ok': anchor_ok},
           'T1': {'sig_cells':
                      ['(%.2f,%.2f)' % (S_GRID[a],
                                        S_GRID[b])
                       for a, b in sig_cells],
                  'p_cells':
                      [float('%.3e' % v)
                       for v in p_cells]
                      if p_cells is not None else None,
                  'stat_cells_L17':
                      np.round(stat_cells, 4).tolist()
                      if stat_cells is not None
                      else None},
           'dose_law': dose_law,
           'T2': {'L17_sig_heads': sig_heads_L17,
                  'n_sig_grid_cells':
                      int(np.sum(p_head <= P_TH))
                      if p_head is not None else None,
                  'L17_head_stats':
                      [[int(h),
                        round(float(I_grid_med[17, h]), 5),
                        float('%.3e'
                              % p_head[17, h])]
                       for h in np.argsort(
                           I_grid_med[17])[:8]]
                      if I_grid_med is not None
                      else None},
           'T3': ovl,
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if verdict != 'anchor_fail_all_void':
        np.savez_compressed(os.path.join(
            OUT, 'interaction_dose_headgrid.npz'), **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2978 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    import torch  # noqa: E402
    main()
