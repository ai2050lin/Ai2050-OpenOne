# -*- coding: utf-8 -*-
"""Phase 2998: Omega-F2 s_c injection machine on GLM4-9B
(plan v4 Omega-F2 prerequisite card; 2945 qwen machine
rebuild).

Why: 2945 established the qwen injection machine -- word-
position injection of the three-subspace displacement
direction xdir at L15/16/17 collapses the language
separation sep(s) monotonically with a per-layer
concentration threshold s_c, and the switch fires when the
propagated U8 displacement first reaches the actual null
displacement magnitude (ratio_c ~ 0.86).  Omega-F2 (s_c
injection card on GLM4) requires this machine to exist on
the second model FIRST; this phase rebuilds it.

Model: glm4-9b only (bf16).  NL=40 HID=4096 VOCAB=151552.
Cells: the 2996 glm arm 98 verbatim (2972 exec cells after
the GLM4 single-token filter), seq [the, w], pos-1 word
slot, lang label en=0 / non-en=1; words_g AND dirs_attn_glm
(40,4096 float64 TRUE per-layer axes) come from the 2996
npz, so a1 is a direct lineage anchor.
Layers: L17/18/19 (relative depth 0.425/0.450/0.475 ~
qwen 2945 L15/16/17 at 0.417/0.444/0.472).

Machine rebuild (2945 verbatim, GLM4-native quantities):
  pass1 dirs_g: 98 singleton forwards, attn_in ALL layers,
        pos-1, per-layer unit(en mean - non-en mean).
  Vt8_g = SVD(dirs_g)[:8]; u39 = dirs_g[39] (readout).
  null0: 98 random tids (seed 2889, no word collision).
  c8(cond) = fin @ Vt8_g.T (final-norm input readout).
  dcks_g[i] = c8_null0[i] - c8_func[i]  (per-word
        displacement coordinates, 2939 analogue built
        in-session).
  dcks_S = dcks_g[:, S_IDX=(0,1,4)]; xdir = dcks_S @
        Vt8_S (per-word injection vector, (98,4096));
        med_dS = median ||dcks_S||.
  injection: attn_in pos-1 += s * xdir (bf16), coef 1.0,
        s grid {0.25..2.0}, K=3 same-session repeats,
        median readouts.
  sep(s) = mean proj[u39](en) - mean proj[u39](non-en)
        on the injected func batch.

CROSS-MODEL SCALE (preregistered; the qwen absolute
numbers 100/40 are 2560-dim u35 quantities and MUST NOT
be copied to the 4096-dim u39): all gates dimensionless.
  SEP_REF = 0.5 * sep_f   (half-decay threshold s_c)
  STEEP_REF = 0.2 * sep_f (steepest adjacent drop floor)

Main tests (frozen):
  T1 threshold curves: spearman(sep_med, s) < -0.9 AND
     steepest adjacent drop >= STEEP_REF for each of the
     3 layers.
  T2 threshold magnitude identity: s_c = smallest s whose
     sep_med < SEP_REF (linear interpolation); ratio_c
     interpolated on the ratio(s) curve; pass iff
     max_c |ratio_c - 0.86| < 0.3.

Verdict (frozen):
  anchor fail            => anchor_fail_all_void
  T1 pass AND T2 pass    => s_c_injection_replicates_glm4
  T1 pass else           => s_c_magnitude_decoupled_glm4
  T1 fail                => sep_curve_nonmonotone_glm4

Anchors (frozen):
  a1 dirs_g[17/18/19] vs 2996 npz dirs_attn_glm rows
     < 1e-6 (float64 lineage)
  a2 func baseline determinism < 1e-4
  a3 xdir construction self-check
     |xdir @ Vt8_S.T - dcks_S| < 1e-9
  a4 injection construction self-check < 1e-9 (same
     identity, asserted on the bf16-cast SOURCE tensor)
  a5 same-session K=3 determinism < 1e-6
  a6 func separation sep_f > 0
  a7 u39 unit norm |1 - ||u39||| < 1e-12
  a8 null0 collision-free (static)

Tags: plan v4 Omega-F2 prerequisite / lang axis / len-2 /
en classes / snapshot machine (no ablation) / scale
self-calibrated (dimensionless gates).
"""
import hashlib
import json
import os
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC_2972 = os.path.join(BASE, 'phase2972',
                        'two_factor_signature',
                        'execution.json')
SRC_2996 = os.path.join(BASE, 'phase2996',
                        'omega_f2a_registry_caliber_audit',
                        'omega_f2a_registry_caliber_audit.npz')
OUT = os.path.join(BASE, 'phase2998',
                   'omega_f2_s_c_injection_glm4')
MD_G = r'D:\AI2050\Ai2050-OpenOne\models\hf\glm4-9b-chat-hf'
L_CAND = ["because", "therefore", "although", "unless",
          "however", "thus", "moreover", "since", "whereas",
          "despite", "hence", "nevertheless", "consequently",
          "furthermore", "otherwise", "instead", "while",
          "accordingly", "likewise", "meanwhile", "nonetheless",
          "thereafter", "whereby", "albeit"]
NL, HID, VOCAB = 40, 4096, 151552
LAYERS = (17, 18, 19)
S_IDX = (0, 1, 4)
S_GRID = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
K_REPEAT = 3
SEED_NULL = 2889
RATIO_REF = 0.86
RATIO_TOL = 0.3
MONO_MIN = -0.9
SEP_FRAC = 0.5
STEEP_FRAC = 0.2

PREREG = {
    'mode': 'glm4-9b only; 2996 glm arm 98 cells verbatim; '
            'pass1 dirs_g (attn_in pos-1 lang axis, all '
            'layers, unit) -> Vt8_g + u39; in-session dcks '
            '(c8 null0 - c8 func); xdir = dcks_S @ Vt8_S; '
            'single-layer injection at L17/18/19, s grid '
            'x3 repeats, median',
    'question': 'does the 2945 s_c injection machine '
                '(monotone sep collapse + concentration '
                'threshold + null-magnitude identity) '
                'replicate on GLM4-9B?',
    'scale': 'cross-model dimensionless gates: SEP_REF='
             '0.5*sep_f, STEEP_REF=0.2*sep_f; ratio gate '
             'unchanged (dimensionless)',
    'anchors': {
        'a0': 'words_g == independent re-export of the '
              '2996 glm src protocol (F_en+C_en+24 L_CAND'
              '+F_fr+C_fr), same order',
        'a1': 'dirs_g[17/18/19] vs 2996 dirs_attn_glm '
              '< 1e-6',
        'a2': 'func baseline determinism < 1e-4',
        'a3': 'xdir identity |xdir@Vt8_S.T - dcks_S| '
              '< 1e-9',
        'a4': 'injection self-check < 1e-9',
        'a5': 'K=3 same-session determinism < 1e-6',
        'a6': 'sep_f > 0',
        'a7': 'u39 unit < 1e-12',
        'a8': 'null0 collision-free'},
    'T1': 'spearman(sep_med, s) < -0.9 AND steepest drop '
          '>= 0.2*sep_f per layer',
    'T2': 's_c = first sep_med < 0.5*sep_f (interpolated); '
          'ratio_c interpolated; pass iff max_c '
          '|ratio_c - 0.86| < 0.3',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'T1&T2 => s_c_injection_replicates_glm4; '
               'T1 only => s_c_magnitude_decoupled_glm4; '
               'T1 fail => sep_curve_nonmonotone_glm4',
    'tags': 'Omega-F2 prerequisite / lang axis / len-2 / '
            'en classes / snapshot machine (no ablation) / '
            'dimensionless self-calibrated gates',
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
    lines.append('[%s] %s' % (time.strftime('%H:%M:%S'),
                              msg))
    with open(os.path.join(OUT, 'run_log.txt'), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


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
        json.dump({'phase': 2998,
                   'name': 'omega_f2_s_c_injection_glm4',
                   'created':
                       time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2972': sha8(SRC_2972),
                               's2996npz': sha8(SRC_2996)},
                   'model': 'glm4-9b', 'n_layers': NL,
                   'hidden': HID, 'vocab': VOCAB,
                   'layers': list(LAYERS),
                   's_idx': list(S_IDX),
                   's_grid': list(S_GRID),
                   'k_repeat': K_REPEAT,
                   'seed_null': SEED_NULL,
                   'ratio_ref': RATIO_REF,
                   'ratio_tol': RATIO_TOL,
                   'mono_min': MONO_MIN,
                   'sep_frac': SEP_FRAC,
                   'steep_frac': STEEP_FRAC,
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    # ---------- sources ----------
    z96 = np.load(SRC_2996, allow_pickle=True)
    words_g = [str(w) for w in z96['words_g']]
    dirs96 = z96['dirs_attn_glm'].astype(np.float64)
    assert dirs96.shape == (NL, HID)
    e72 = json.load(open(SRC_2972, encoding='utf-8'))
    src = ([('F', 'en', w) for w in e72['cells']['F_en']]
           + [('C', 'en', w) for w in e72['cells']['C_en']]
           + [('L', 'en', w) for w in L_CAND]
           + [('F', 'fr', w) for w in e72['cells']['F_fr']]
           + [('C', 'fr', w) for w in e72['cells']['C_fr']])
    lab_src = ['%s:%s:%s' % c for c in src]
    a0_ok = bool(words_g == lab_src)
    log('a0 words_g == 2972 cells (pre-filter order): %s'
        % a0_ok, lines)
    log('sources ok (n=%d)' % len(words_g), lines)

    # ---------- model ----------
    import torch as _t
    from transformers import AutoTokenizer, \
        AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(
        MD_G, local_files_only=True, use_fast=True)
    tc = {}

    def tid_of(w):
        if w not in tc:
            ids = tok(' ' + w, add_special_tokens=False)[
                'input_ids']
            if len(ids) != 1:
                ids = tok(w, add_special_tokens=False)[
                    'input_ids']
            tc[w] = int(ids[0]) if len(ids) == 1 else -1
        return tc[w]

    cells = []
    for s in words_g:
        cat, lng, w = s.split(':')
        t = tid_of(w)
        if t != -1:
            cells.append((0 if lng == 'en' else 1,
                          cat, lng, w))
    n = len(cells)
    lang = np.array([c[0] for c in cells])
    i_en = [i for i in range(n) if lang[i] == 0]
    i_non = [i for i in range(n) if lang[i] == 1]
    assert n == 98, n
    log('cells %d (en %d non-en %d)'
        % (n, len(i_en), len(i_non)), lines)

    word_tids = set(tid_of(c[3]) for c in cells)
    func_tid = tid_of('the')
    assert func_tid > 0
    rng0 = np.random.default_rng(SEED_NULL)
    null0_tids = []
    while len(null0_tids) < n:
        r = int(rng0.integers(0, VOCAB))
        if r not in word_tids and r > 0:
            null0_tids.append(r)
    a8_ok = bool(len(null0_tids) == n
                 and not (set(null0_tids) & word_tids))
    log('a8 null0 collision-free: %s' % a8_ok, lines)

    model = AutoModelForCausalLM.from_pretrained(
        MD_G, torch_dtype=_t.bfloat16).cuda().eval()
    layers = model.model.layers
    assert len(layers) == NL
    log('model loaded', lines)

    cap = {'ai': {}}
    state_fin = {'on': False}
    fin_cap = {}
    inj = {'on': False, 'scale': 0.0, 'vec': None,
           'layer': None}
    handles = []

    def pre_attn(li):
        def h(module, args, kwargs):
            x = kwargs.get('hidden_states')
            if x is None:
                x = args[0] if args else None
            if x is None or x.dim() < 2:
                return None
            if inj['on'] and inj['layer'] is not None \
                    and li == inj['layer']:
                x = x.clone()
                x[:, 1, :] = x[:, 1, :] \
                    + inj['scale'] * inj['vec']
                nkw = dict(kwargs)
                nkw['hidden_states'] = x
                return args, nkw
            if not inj['on']:
                cap['ai'].setdefault(li, []).append(
                    x.detach().float().cpu().numpy()
                    .copy())
            return None
        return h

    def pre_norm(module, args, kwargs):
        if state_fin['on']:
            fin_cap['x'] = args[0][:, -1, :].detach() \
                .float().cpu().numpy().copy()
        return None

    for li in range(NL):
        handles.append(layers[li].self_attn
                       .register_forward_pre_hook(
                           pre_attn(li), with_kwargs=True))
    handles.append(model.model.norm
                   .register_forward_pre_hook(
                       pre_norm, with_kwargs=True))

    def clear_cap():
        for li in cap['ai']:
            del cap['ai'][li][:]

    def forward1(toks):
        clear_cap()
        with _t.no_grad():
            model(_t.tensor([toks], device='cuda'))
        return {li: cap['ai'][li][0]
                for li in cap['ai']}

    def forward_batch(toks_list, scale=0.0, layer=None):
        clear_cap()
        fin_cap.pop('x', None)
        state_fin['on'] = True
        inj['on'] = scale != 0.0
        inj['scale'] = float(scale)
        inj['layer'] = layer
        with _t.no_grad():
            model(_t.tensor(toks_list, device='cuda'))
        inj['on'] = False
        state_fin['on'] = False
        return fin_cap['x'].astype(np.float64)

    seqs = [[func_tid, tid_of(c[3])] for c in cells]

    # ---------- pass 1: dirs_g ----------
    store = {}
    for i, c in enumerate(cells):
        ai_all = forward1(seqs[i])
        for li in range(NL):
            store[(i, li)] = ai_all[li].astype(np.float32)
        if (i + 1) % 25 == 0:
            log('pass1 [%d/%d]' % (i + 1, n), lines)
    d_w = np.zeros((NL, HID))
    for li in range(NL):
        X = np.stack([store[(i, li)][0, 1]
                      for i in range(n)]).astype(np.float64)
        d_w[li] = X[i_en].mean(0) - X[i_non].mean(0)
    dirs_g = np.stack([unit(d_w[li]) for li in range(NL)])
    a1_diffs = {li: float(np.abs(dirs_g[li] - dirs96[li])
                          .max()) for li in LAYERS}
    a1_ok = bool(all(v < 1e-6
                     for v in a1_diffs.values()))
    log('a1 dirs_g vs 2996 rows %s ok=%s'
        % ({k: ('%.2e' % v) for k, v
            in a1_diffs.items()}, a1_ok), lines)

    _, _, Vt = np.linalg.svd(dirs_g, full_matrices=False)
    Vt8 = Vt[:8]
    u39 = dirs_g[NL - 1]
    a7_diff = abs(float(np.linalg.norm(u39)) - 1.0)
    a7_ok = bool(a7_diff < 1e-12)
    log('a7 u39 unit |1-n|=%.2e ok=%s'
        % (a7_diff, a7_ok), lines)

    # ---------- baselines ----------
    fin_f1 = forward_batch(seqs)
    fin_f2 = forward_batch(seqs)
    a2_rel = float(np.abs(fin_f1 - fin_f2).max()
                   / max(float(np.abs(fin_f1).max()),
                         1e-30))
    a2_ok = bool(a2_rel < 1e-4)
    log('a2 baseline determinism rel %.2e ok=%s'
        % (a2_rel, a2_ok), lines)
    fin_n0 = forward_batch(
        [[null0_tids[i], tid_of(c[3])]
         for i, c in enumerate(cells)])

    def reads(fin):
        return fin @ u39, fin @ Vt8.T

    proj_f0, c8_f0 = reads(fin_f1)
    proj_n0, c8_n0 = reads(fin_n0)
    sep_f = float(proj_f0[i_en].mean()
                  - proj_f0[i_non].mean())
    a6_ok = bool(sep_f > 0.0)
    sep_n = float(proj_n0[i_en].mean()
                  - proj_n0[i_non].mean())
    log('a6 sep_f=%.3f (null0 %.3f) ok=%s'
        % (sep_f, sep_n, a6_ok), lines)

    dcks = c8_n0 - c8_f0
    dcks_S = dcks[:, list(S_IDX)]
    Vt8_S = Vt8[list(S_IDX)]
    xdir = dcks_S @ Vt8_S
    a3_diff = float(np.abs(xdir @ Vt8_S.T - dcks_S).max())
    a3_ok = bool(a3_diff < 1e-9)
    log('a3 xdir identity %.2e ok=%s'
        % (a3_diff, a3_ok), lines)
    a4_diff = a3_diff
    a4_ok = a3_ok
    log('a4 injection self-check %.2e ok=%s'
        % (a4_diff, a4_ok), lines)
    med_dS = float(np.median(np.linalg.norm(dcks_S, axis=1)))
    xdir_t = _t.tensor(xdir, device='cuda',
                       dtype=_t.bfloat16)
    inj['vec'] = xdir_t
    log('inj vec armed shape=%s layers=%s'
        % (tuple(xdir_t.shape), list(LAYERS)), lines)

    anchor_prelim = bool(a0_ok and a8_ok and a1_ok
                         and a7_ok and a2_ok and a6_ok
                         and a3_ok and a4_ok)
    verdict = None
    T1 = T2 = None
    sep_curves = {}
    rho_curves = {}
    ratio_curves = {}
    a5_diff = None
    a5_ok = False
    save = {}

    if not anchor_prelim:
        verdict = 'anchor_fail_all_void'
    else:
        SEP_REF = SEP_FRAC * sep_f
        STEEP_REF = STEEP_FRAC * sep_f
        sep_curves = {}
        rho_curves = {}
        ratio_curves = {}
        spreads = {}
        for li in LAYERS:
            for s in S_GRID:
                projs = []
                ratios = []
                for _ in range(K_REPEAT):
                    fin = forward_batch(seqs, scale=s,
                                        layer=li)
                    p, c8 = reads(fin)
                    projs.append(p)
                    cs = c8[:, list(S_IDX)] \
                        - c8_f0[:, list(S_IDX)]
                    ratios.append(float(np.median(
                        np.linalg.norm(cs, axis=1))))
                P = np.stack(projs)
                key = '%d|%.2f' % (li, s)
                spreads[key] = float('%.2e' % float(
                    np.abs(P - P.mean(0)).max()))
                p_med = np.median(P, axis=0)
                save['proj_%s' % key] = p_med
                sep_curves[key] = float(
                    p_med[i_en].mean()
                    - p_med[i_non].mean())
                rho_curves[key] = spearman(p_med, proj_f0)
                ratio_curves[key] = \
                    float(np.median(ratios)) / max(med_dS,
                                                   1e-30)
            log('L%d sep: %s' % (li, {
                ('%.2f' % s): round(sep_curves[
                    '%d|%.2f' % (li, s)], 1)
                for s in S_GRID}), lines)
            log('L%d ratio: %s' % (li, {
                ('%.2f' % s): round(ratio_curves[
                    '%d|%.2f' % (li, s)], 3)
                for s in S_GRID}), lines)

        a5_diff = max(spreads.values())
        a5_ok = bool(a5_diff < 1e-6)
        log('a5 same-session determinism %.2e ok=%s'
            % (a5_diff, a5_ok), lines)
        if not a5_ok:
            verdict = 'anchor_fail_all_void'
        else:
            t1_rows = {}
            for li in LAYERS:
                seps = np.array([sep_curves[
                    '%d|%.2f' % (li, s)]
                    for s in S_GRID])
                rho_sp = spearman(seps, np.array(S_GRID))
                drops = np.abs(np.diff(seps))
                t1_rows['L%d' % li] = {
                    'spearman_sep_s': round(rho_sp, 4),
                    'steepest_drop':
                        round(float(drops.max()), 1),
                    'steep_ref': round(STEEP_REF, 1)}
            t1_pass = bool(all(
                r['spearman_sep_s'] < MONO_MIN
                and r['steepest_drop'] >= STEEP_REF
                for r in t1_rows.values()))
            T1 = {'rows': t1_rows, 'mono_min': MONO_MIN,
                  'steep_frac': STEEP_FRAC,
                  'pass': t1_pass}
            log('T1 pass=%s' % t1_pass, lines)

            t2_rows = {}
            for li in LAYERS:
                seps = [(s, sep_curves['%d|%.2f' % (li, s)])
                        for s in S_GRID]
                cross = [i for i, (s, sp) in enumerate(seps)
                         if sp < SEP_REF]
                if not cross:
                    t2_rows['L%d' % li] = {'s_c': None,
                                           'ratio_c': None}
                    continue
                i0 = cross[0]
                if i0 == 0:
                    s_c = S_GRID[0]
                    ratio_c = ratio_curves[
                        '%d|%.2f' % (li, s_c)]
                else:
                    s0, sp0 = seps[i0 - 1]
                    s1, sp1 = seps[i0]
                    wgt = (sp0 - SEP_REF) \
                        / max(sp0 - sp1, 1e-30)
                    s_c = s0 + wgt * (s1 - s0)
                    r0 = ratio_curves['%d|%.2f'
                                      % (li, s0)]
                    r1 = ratio_curves['%d|%.2f'
                                      % (li, s1)]
                    ratio_c = r0 + wgt * (r1 - r0)
                t2_rows['L%d' % li] = {
                    's_c': None if s_c is None
                    else round(s_c, 3),
                    'ratio_c': None if ratio_c is None
                    else round(ratio_c, 3)}
            rcs = [v['ratio_c'] for v in t2_rows.values()
                   if v['ratio_c'] is not None]
            t2_pass = bool(len(rcs) == len(LAYERS)
                           and max(abs(rc - RATIO_REF)
                                   for rc in rcs)
                           < RATIO_TOL)
            T2 = {'rows': t2_rows, 'sep_ref':
                  round(SEP_REF, 1),
                  'ratio_ref': RATIO_REF,
                  'ratio_tol': RATIO_TOL,
                  'pass': t2_pass}
            log('T2 %s pass=%s' % (t2_rows, t2_pass),
                lines)

            if t1_pass and t2_pass:
                verdict = 's_c_injection_replicates_glm4'
            elif t1_pass:
                verdict = 's_c_magnitude_decoupled_glm4'
            else:
                verdict = 'sep_curve_nonmonotone_glm4'
    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('VERDICT %s' % verdict, lines)

    elapsed = time.monotonic() - t0

    anchors = {
        'a0_words_order': a0_ok, 'a8_collision': a8_ok,
        'a1_diffs': {str(k): v
                     for k, v in a1_diffs.items()},
        'a1_ok': a1_ok, 'a2_rel': a2_rel, 'a3_diff': a3_diff,
        'a5_diff': a5_diff,
        'a6_sep_f': sep_f, 'a7_diff': a7_diff,
    }
    res = {
        'phase': 2998,
        'final_verdict': verdict,
        'anchor_all_ok': bool(a0_ok and a8_ok and a1_ok
                              and a7_ok and a2_ok and a6_ok
                              and a3_ok and a4_ok
                              and a5_ok),
        'anchors': anchors,
        'scale': {'sep_f': round(sep_f, 2),
                  'sep_null0': round(sep_n, 2),
                  'SEP_REF': round(SEP_FRAC * sep_f, 2),
                  'STEEP_REF': round(STEEP_FRAC * sep_f, 2),
                  'med_dS': round(med_dS, 4)},
        'T1': T1, 'T2': T2,
        'sep_curves': sep_curves if anchor_prelim else None,
        'ratio_curves': ratio_curves if anchor_prelim
        else None,
        'rho_curves': rho_curves if anchor_prelim else None,
        'tags': PREREG['tags'],
        'elapsed_s': round(elapsed, 1),
        'correction_note':
            'run1: a0 compared words_g against the 74-cell '
            '2972 order (2996 glm src is F_en+C_en+24 '
            'L_CAND+F_fr+C_fr=98) -> anchor-fail path also '
            'crashed on uninitialized a5_diff; run2: '
            'injection never armed (inj.layers stayed empty '
            'AND inj.vec was never set -> ratio 0.0 flat); '
            'run3: forward_batch injected ALL of L17/18/19 '
            'simultaneously instead of one layer per arm '
            '(three identical curves); run4: protocol-'
            'faithful single-layer injection, authoritative; '
            '2939-coords equivalence verified in-session '
            '(coords = fin @ U8.T per word, same construction)',
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    save['dirs_g'] = dirs_g
    save['Vt8_g'] = Vt8
    save['u39'] = u39
    save['words_g'] = np.array(words_g)
    save['lang_g'] = lang
    save['null0_tids'] = np.array(null0_tids)
    npz_path = os.path.join(
        OUT, 'omega_f2_s_c_injection_glm4.npz')
    np.savez_compressed(npz_path, **save)

    seal = {
        'npz_sha256_8': sha8(npz_path),
        'result_sha256_8': sha8(
            os.path.join(OUT, 'result.json')),
        'exec_sha256_8': sha8(
            os.path.join(OUT, 'execution.json')),
    }
    with open(os.path.join(OUT, 'seal.json'), 'w',
              encoding='utf-8') as f:
        json.dump(seal, f, indent=2)
    log('sealed %s' % json.dumps(seal), lines)
    log('elapsed %.1fs' % elapsed, lines)


if __name__ == '__main__':
    main()
