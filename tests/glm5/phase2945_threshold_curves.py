# -*- coding: utf-8 -*-
"""Phase 2945: concentration threshold curves.

Why: 2944 localized the regime switch to a redundant
concentration-driven threshold in L14-L18: any single layer
triggers at s=2 while the same total dose split over 5
layers does nothing. Open question: where exactly is the
per-layer threshold s_c, and is the threshold the point
where the PROPAGATED U8 displacement first reaches the
actual null displacement magnitude (ratio ~ 0.86 at s=2
per 2942 calibration)?

Mode: ONE model (qwen3-4b), forward family. Layers L15, L17
(2944 most sensitive) + L16 (2942 anchor reference), single-
layer coef 1.0; s grid {0.25,0.5,0.75,1.0,1.25,1.5,2.0};
K=3 same-session repeats, median readouts (2944 a8 mode).

Anchors (frozen; 2944 run values in parentheses):
  a1 dirs_word rebuild vs 2927 npz < 1e-5     (2.17e-08)
  a2 func baseline determinism < 1e-4         (0.0)
  a3 Vt8 rebuild vs 2939 npz < 1e-6           (0.0)
  a4 proj_func vs 2935 s_base[func] < 1e-4    (7.2e-06)
  a5 proj_null0 vs 2935 s_base[null0] < 1e-4  (6.3e-06)
  a6 func separation > 0                      (185.70)
  a7 injection construction self-check < 1e-9 (9.9e-14)
  a8 same-session repeat determinism < 1e-6   (2.8e-14)

Main tests (frozen):
  T1 threshold curves: sep_med(layer, s) monotone decreasing
     (Spearman(sep, s) < -0.9) AND steepest adjacent drop
     >= 40 for each of the 3 layers.
  T2 threshold magnitude identity: s_c = smallest s whose
     sep_med < 100 (with linear interpolation between grid
     neighbors); ratio_c = propagated-U8-ratio at s_c
     (linear in s from the measured ratio grid via the
     ratio(s) curve, interpolated). Pass iff
     max_c |ratio_c - 0.86| < 0.3 => the switch fires when
     the propagated displacement first reaches the actual
     null displacement magnitude; else
     threshold_magnitude_decoupled.

Verdict (frozen):
  anchor fail => anchor_fail_all_void
  T1 pass AND T2 pass => threshold_at_null_magnitude
  T1 pass else        => threshold_magnitude_decoupled
  T1 fail             => threshold_curve_nonmonotone

Descriptive: D1 sep(s) full curves; D2 rho(s); D3 ratio(s);
D4 s_c per layer + interpolated; D5 sep at s_c vs null0
sep 77.3 (linear-shell band per 2943).

Output: phase2945/threshold_curves/.
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
SRC_2935 = os.path.join(BASE, 'phase2935', 'null_amp_anatomy',
                        'null_amp_anatomy.npz')
SRC_2939 = os.path.join(BASE, 'phase2939', 'rotation_target',
                        'rotation_target.npz')
OUT = os.path.join(BASE, 'phase2945', 'threshold_curves')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2945_run_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
SEED = 2896
NH, HD = 32, 128
NL = 36
VOCAB = 151936
LAYERS = (15, 16, 17)
S_IDX = (0, 1, 4)
S_GRID = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
K_REPEAT = 3
SEP_THRESHOLD = 100.0
MONO_MIN = -0.9
STEEP_MIN = 40.0
RATIO_REF = 0.86
RATIO_TOL = 0.3

PREREG = {
    'mode': 'forward family: single-layer injection of xdir '
            '(2942 verbatim) at L15/L16/L17 coef 1.0, s grid '
            '{0.25..2.0}, K=3 same-session repeats, median '
            'readouts',
    'question': 'where is the per-layer switch threshold '
                's_c, and does the switch fire when the '
                'propagated U8 displacement first reaches '
                'the actual null displacement magnitude '
                '(ratio ~ 0.86)?',
    'anchors': {
        'a1': 'dirs_word rebuild vs 2927 npz < 1e-5',
        'a2': 'func baseline determinism < 1e-4',
        'a3': 'Vt8 rebuild vs 2939 npz < 1e-6',
        'a4': 'proj_func vs 2935 s_base[func] < 1e-4',
        'a5': 'proj_null0 vs 2935 s_base[null0] < 1e-4',
        'a6': 'func separation > 0',
        'a7': 'injection construction self-check < 1e-9',
        'a8': 'same-session repeat determinism < 1e-6',
    },
    'T1': 'sep curve monotone (spearman < -0.9) AND '
          'steepest adjacent drop >= 40 per layer',
    'T2': 's_c = first s with sep_med < 100 (grid + '
          'interpolation); ratio_c interpolated at s_c; '
          'pass iff max_c |ratio_c - 0.86| < 0.3',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'T1 pass AND T2 pass => '
               'threshold_at_null_magnitude; T1 pass else '
               '=> threshold_magnitude_decoupled; T1 fail '
               '=> threshold_curve_nonmonotone',
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


def rankdata(x):
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x), dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
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
        json.dump({'phase': 2945,
                   'name': 'threshold_curves',
                   'created':
                       time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2887': sha8(SRC_2887),
                               's2927': sha8(SRC_2927),
                               's2935': sha8(SRC_2935),
                               's2939': sha8(SRC_2939)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'seed': SEED, 'layers': list(LAYERS),
                   's_idx': list(S_IDX),
                   's_grid': list(S_GRID),
                   'k_repeat': K_REPEAT,
                   'sep_threshold': SEP_THRESHOLD,
                   'ratio_ref': RATIO_REF,
                   'ratio_tol': RATIO_TOL,
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

    z35 = np.load(SRC_2935, allow_pickle=True)
    conds35 = [str(s) for s in z35['cond_names']]
    s_base_35 = z35['s_base'].astype(np.float64)
    ifu35 = conds35.index('func')
    in035 = conds35.index('null0')

    z39 = np.load(SRC_2939, allow_pickle=True)
    Vt8_39 = z39['Vt8'].astype(np.float64)
    coords_39 = z39['coords'].astype(np.float64)
    conds39 = [str(s) for s in z39['cond_names']]
    dcks_39 = coords_39[conds39.index('null0')] \
        - coords_39[conds39.index('func')]
    log('sources ok', lines)

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

    word_tids = set(tid_map.values())

    def sample_null(seed):
        rng = np.random.default_rng(seed)
        out = []
        while len(out) < n_words:
            r = int(rng.integers(0, VOCAB))
            if r not in word_tids and r > 0:
                out.append(r)
        return out

    null0_tids = sample_null(2896)
    batch = {'func': [[func_tid, tid_map[words[i][2]]]
                      for i in range(n_words)],
             'null0': [[null0_tids[i], tid_map[words[i][2]]]
                       for i in range(n_words)]}

    model, _ = load_native('qwen4')
    model.eval()
    layers = model.model.layers
    log('model loaded (load_native full GPU)', lines)

    cap = {'attnin': {}}
    state_fin = {'on': False}
    fin_cap = {}
    inj = {'coef': None, 'scale': 0.0, 'vec': None}
    handles = []

    def pre_attn(li):
        def h(module, args, kwargs):
            x = args[0] if args else kwargs.get('hidden_states')
            if x is None or x.dim() < 2:
                return
            if inj['coef'] is not None \
                    and li in inj['coef']:
                c = inj['coef'][li]
                if c != 0.0:
                    x = x.clone()
                    x[:, 1, :] = x[:, 1, :] \
                        + c * inj['scale'] * inj['vec']
                if args:
                    nargs = (x,) + tuple(args[1:])
                    return nargs, kwargs
                nkw = dict(kwargs)
                nkw['hidden_states'] = x
                return args, nkw
            cap['attnin'].setdefault(li, []).append(
                x.detach().float().cpu().numpy())
            return None
        return h

    def pre_norm(module, args, kwargs):
        if state_fin['on']:
            fin_cap['x'] = args[0][:, -1, :].detach() \
                .float().cpu().numpy()

    for li in range(NL):
        handles.append(layers[li].self_attn
                       .register_forward_pre_hook(
                           pre_attn(li), with_kwargs=True))
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

    def forward_batch(toks_list, coef=None, scale=0.0,
                      vec=None):
        clear_cap()
        fin_cap.pop('x', None)
        state_fin['on'] = True
        inj['coef'] = coef
        inj['scale'] = float(scale)
        inj['vec'] = vec
        with torch.no_grad():
            model(torch.tensor(toks_list, device='cuda'))
        inj['coef'] = None
        state_fin['on'] = False
        fin = fin_cap['x'].astype(np.float64)
        return fin

    # ---------- pass 1: dirs_word rebuild ----------
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

    _, _, Vt = np.linalg.svd(dirs_word, full_matrices=False)
    Vt8 = Vt[:8]
    a3_diff = float(np.abs(Vt8 - Vt8_39).max())
    a3_ok = bool(a3_diff < 1e-6)
    log('a3 Vt8 rebuild vs 2939 max diff %.2e ok=%s'
        % (a3_diff, a3_ok), lines)

    u35 = dirs_word[NL - 1]

    dcks_S = dcks_39[:, list(S_IDX)]
    Vt8_S = Vt8[list(S_IDX)]
    xdir = dcks_S @ Vt8_S
    a7_diff = float(np.abs(xdir @ Vt8_S.T - dcks_S).max())
    a7_ok = bool(a7_diff < 1e-9)
    log('a7 construction self-check max abs %.2e ok=%s'
        % (a7_diff, a7_ok), lines)
    xdir_t = torch.tensor(xdir, device='cuda',
                          dtype=torch.bfloat16)
    med_dS = float(np.median(np.linalg.norm(dcks_S, axis=1)))

    # ---------- baselines ----------
    fin_f1 = forward_batch(batch['func'])
    fin_f2 = forward_batch(batch['func'])
    a2_rel = float(np.abs(fin_f1 - fin_f2).max()
                   / max(float(np.abs(fin_f1).max()), 1e-30))
    a2_ok = bool(a2_rel < 1e-4)
    log('a2 baseline determinism rel %.2e ok=%s'
        % (a2_rel, a2_ok), lines)

    def reads(fin):
        return fin @ u35, fin @ Vt8.T, \
            np.linalg.norm(fin, axis=1)

    proj_f0, c8_f0, nfin_f0 = reads(fin_f1)
    fin_n0 = forward_batch(batch['null0'])
    proj_n0, c8_n0, nfin_n0 = reads(fin_n0)

    a4_diff = float(np.abs(proj_f0 - s_base_35[ifu35]).max())
    a4_ok = bool(a4_diff < 1e-4)
    log('a4 proj_func vs 2935 max abs diff %.2e ok=%s'
        % (a4_diff, a4_ok), lines)
    a5_diff = float(np.abs(proj_n0 - s_base_35[in035]).max())
    a5_ok = bool(a5_diff < 1e-4)
    log('a5 proj_null0 vs 2935 max abs diff %.2e ok=%s'
        % (a5_diff, a5_ok), lines)
    sep_f = float(proj_f0[lab_lang == 0].mean()
                  - proj_f0[lab_lang == 1].mean())
    a6_ok = bool(sep_f > 0.0)
    log('a6 func separation %.4f ok=%s' % (sep_f, a6_ok), lines)
    sep_n = float(proj_n0[lab_lang == 0].mean()
                  - proj_n0[lab_lang == 1].mean())
    log('sep null0 %.3f (func %.3f)' % (sep_n, sep_f), lines)

    anchor_prelim = bool(a1_ok and a2_ok and a3_ok and a4_ok
                         and a5_ok and a6_ok and a7_ok)
    verdict = None
    t1 = t2 = d1 = d2 = d3 = d4 = d5 = None
    save = {}

    if not anchor_prelim:
        verdict = 'anchor_fail_all_void'
    else:
        sep_curves = {}
        rho_curves = {}
        ratio_curves = {}
        spreads = {}
        store_proj = {}
        for li in LAYERS:
            coef = {li: 1.0}
            for s in S_GRID:
                projs = []
                ratios = []
                for _ in range(K_REPEAT):
                    fin = forward_batch(
                        batch['func'], coef=coef, scale=s,
                        vec=xdir_t)
                    p, c8, nfin = reads(fin)
                    projs.append(p)
                    cs = c8[:, list(S_IDX)] \
                        - c8_f0[:, list(S_IDX)]
                    ratios.append(float(np.median(
                        np.linalg.norm(cs, axis=1))))
                P = np.stack(projs)
                key = (li, s)
                spreads['L%d|%.2f' % key] = \
                    float('%.2e' % float(
                        np.abs(P - P.mean(0)).max()))
                p_med = np.median(P, axis=0)
                store_proj[str(key)] = p_med
                sep_curves[str(key)] = float(
                    p_med[lab_lang == 0].mean()
                    - p_med[lab_lang == 1].mean())
                rho_curves[str(key)] = spearman(
                    p_med, proj_f0)
                ratio_curves[str(key)] = \
                    float(np.median(ratios)) / max(med_dS,
                                                   1e-30)
            log('L%d sep: %s' % (li, {
                ('%.2f' % s): round(sep_curves[str((li, s))], 1)
                for s in S_GRID}), lines)
            log('L%d ratio: %s' % (li, {
                ('%.2f' % s): round(ratio_curves[str((li, s))], 3)
                for s in S_GRID}), lines)

        a8_diff = max(float(v) for v in spreads.values())
        a8_ok = bool(a8_diff < 1e-6)
        log('a8 same-session determinism max spread %.2e '
            'ok=%s' % (a8_diff, a8_ok), lines)

        if not (anchor_prelim and a8_ok):
            verdict = 'anchor_fail_all_void'
        else:
            # ---------- T1: monotone + steep ----------
            t1_rows = {}
            for li in LAYERS:
                seps = np.array([sep_curves[str((li, s))]
                                 for s in S_GRID])
                rho_sp = spearman(seps, np.array(S_GRID))
                drops = np.abs(np.diff(seps))
                t1_rows['L%d' % li] = {
                    'spearman_sep_s': round(rho_sp, 4),
                    'steepest_drop':
                        round(float(drops.max()), 1)}
            t1_pass = bool(all(
                r['spearman_sep_s'] < MONO_MIN
                and r['steepest_drop'] >= STEEP_MIN
                for r in t1_rows.values()))
            t1 = {'rows': t1_rows, 'mono_min': MONO_MIN,
                  'steep_min': STEEP_MIN, 'pass': t1_pass}
            log('T1 %s pass=%s' % (t1_rows, t1_pass), lines)

            # ---------- T2: thresholds ----------
            t2_rows = {}
            for li in LAYERS:
                seps = [(s, sep_curves[str((li, s))])
                        for s in S_GRID]
                cross = [i for i, (s, sp) in
                         enumerate(seps)
                         if sp < SEP_THRESHOLD]
                if not cross:
                    t2_rows['L%d' % li] = {
                        's_c': None, 'ratio_c': None}
                    continue
                i0 = cross[0]
                if i0 == 0:
                    s_c = S_GRID[0]
                else:
                    s0, sp0 = seps[i0 - 1]
                    s1_, sp1 = seps[i0]
                    w = (sp0 - SEP_THRESHOLD) \
                        / max(sp0 - sp1, 1e-30)
                    s_c = s0 + w * (s1_ - s0)
                r0 = ratio_curves[str((li, S_GRID[i0 - 1]))] \
                    if i0 > 0 else 0.0
                r1_ = ratio_curves[str((li, S_GRID[i0]))]
                if i0 == 0:
                    ratio_c = r1_ * s_c / S_GRID[0]
                else:
                    s0 = S_GRID[i0 - 1]
                    s1_ = S_GRID[i0]
                    w = (s_c - s0) / (s1_ - s0)
                    ratio_c = r0 + w * (r1_ - r0)
                t2_rows['L%d' % li] = {
                    's_c': round(float(s_c), 4),
                    'ratio_c': round(float(ratio_c), 4),
                    'sep_at_s_c':
                        round(float(np.interp(
                            s_c, S_GRID,
                            [sep_curves[str((li, s))]
                             for s in S_GRID])), 1)}
            ratios_c = [v['ratio_c']
                        for v in t2_rows.values()
                        if v['ratio_c'] is not None]
            t2_pass = bool(len(ratios_c) == 3 and all(
                abs(r - RATIO_REF) < RATIO_TOL
                for r in ratios_c))
            t2 = {'rows': t2_rows, 'ratio_ref': RATIO_REF,
                  'ratio_tol': RATIO_TOL, 'pass': t2_pass}
            log('T2 %s pass=%s' % (t2_rows, t2_pass), lines)

            if t1_pass and t2_pass:
                verdict = 'threshold_at_null_magnitude'
            elif t1_pass:
                verdict = 'threshold_magnitude_decoupled'
            else:
                verdict = 'threshold_curve_nonmonotone'

            d1 = {'L%d' % li:
                  {'%.2f' % s:
                   round(sep_curves[str((li, s))], 2)
                   for s in S_GRID}
                  for li in LAYERS}
            d2 = {'L%d' % li:
                  {'%.2f' % s:
                   round(rho_curves[str((li, s))], 4)
                   for s in S_GRID}
                  for li in LAYERS}
            d3 = {'L%d' % li:
                  {'%.2f' % s:
                   round(ratio_curves[str((li, s))], 4)
                   for s in S_GRID}
                  for li in LAYERS}
            d4 = t2_rows
            d5 = {'sep_null0': round(sep_n, 2),
                  'sep_func': round(sep_f, 2),
                  'note': 'sep at s_c should cross the '
                          '2943 linear-shell band '
                          '(77-106) if the switch is the '
                          'concentration-gated version '
                          'of the shell'}

            save = {
                'words': np.array(['%s:%s:%s' % w
                                   for w in words],
                                  dtype=object),
                'labels_lang': lab_lang,
                's_grid': np.array(S_GRID),
                'layers': np.array(LAYERS),
                'proj_med': np.stack(
                    [store_proj[str((li, s))]
                     for li in LAYERS for s in S_GRID]),
                'proj_func0': proj_f0,
                'proj_null0': proj_n0,
                'sep_curves': np.array(
                    [sep_curves[str((li, s))]
                     for li in LAYERS for s in S_GRID]),
                'Vt8': Vt8, 'dirs_word': dirs_word,
            }

    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2945, 'model': 'qwen3-4b',
           'prereg': PREREG,
           'anchors': {'a1_diff': float('%.3e' % a1_diff),
                       'a1_ok': a1_ok,
                       'a2_rel': float('%.3e' % a2_rel),
                       'a2_ok': a2_ok,
                       'a3_diff': float('%.3e' % a3_diff),
                       'a3_ok': a3_ok,
                       'a4_diff': float('%.3e' % a4_diff),
                       'a4_ok': a4_ok,
                       'a5_diff': float('%.3e' % a5_diff),
                       'a5_ok': a5_ok,
                       'a6_sep_func': round(sep_f, 4),
                       'a6_ok': a6_ok,
                       'a7_diff': float('%.3e' % a7_diff),
                       'a7_ok': a7_ok,
                       'a8_max_spread':
                           float('%.3e' % a8_diff),
                       'a8_ok': a8_ok,
                       'ok': bool(anchor_prelim and a8_ok)},
           'T1': t1, 'T2': t2,
           'D1_sep': d1, 'D2_rho': d2, 'D3_ratio': d3,
           'D4_thresholds': d4, 'D5_shell_note': d5,
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if save:
        np.savez_compressed(os.path.join(
            OUT, 'threshold_curves.npz'), **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2945 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
