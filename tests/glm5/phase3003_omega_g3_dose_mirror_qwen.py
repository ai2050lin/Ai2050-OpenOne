# -*- coding: utf-8 -*-
"""Phase 3003: Omega-G3 qwen amplification band - dose
law + mirror symmetry (GLM4 erasure-band contrast).

Why: 3002 (qwen mirror of 3001) found context_entangled_
qwen AND a mid-band xdir-specific amplification: ratio
1.21/1.50 (L15/L17) vs span(Vt8)-orthogonal random
0.011/0.0098 (~150x specificity), with L17 injection
propagating to L35 (eraser=None).  Open question: is the
amplification band a LINEAR SYMMETRIC channel gain (both
signs amplified, dose-proportional - a passive broadband
property of the mid-band residual geometry) or a
RECTIFYING/nonlinear mechanism (one sign only,
dose-supralinear - an active direction-selective
circuit)?  This decides what the qwen "sensitivity" IS:
linear symmetric gain would mean the mid-band simply
FAILS to erase (contrast with GLM4 L20 kill); rectifying
amplification would mean an active xdir-aligned circuit.

Design (3002/3000 machine verbatim: 57 words 2887,
dirs_word a1 vs 2927, Vt8 a3 vs 2939, u35 readout, dcks
from 2939 coords, xdir = dcks_S @ Vt8_S per-cell, attn_in
pos-1 single-layer coef injection, bf16):
  T1 dose law at L15/L17: s in {0.5, 1.0, 2.0, 3.0},
     K=3; per-s ratio_med; linearity = through-origin
     fit over all 4 points, max relative residual
     <= LIN_TOL.  sub_rnd0 control at s={1,2} L17
     (descriptive: does the orthogonal random direction
     ALSO grow with s?).
  T2 mirror symmetry: -xdir at s=2, K=2 at L15/L17;
     symmetric := BOTH layers |r_minus/r_plus - 1|
     <= MIRROR_TOL; rectifying := BOTH layers
     r_minus/r_plus <= RECT_FRAC.
  T3 (DESCRIPTIVE) cross-model contrast table from the
     3001 sealed result (GLM4 xdir/sub/mirror at L19) -
     no verdict branch.

Verdict (frozen):
  anchor fail                    => anchor_fail_all_void
  symmetric AND linear           => linear_symmetric_
                                    amplification_qwen
  symmetric AND NOT linear       => symmetric_nonlinear_
                                    amplification_qwen
  rectifying                     => rectifying_
                                    amplification_qwen
  else                           => asymmetric_
                                    amplification_qwen

Anchors (frozen):
  a0 words == 2887 re-export (57)
  a1 dirs_word vs 2927 < 1e-5
  a2 baseline determinism < 1e-4
  a3 Vt8 vs 2939 < 1e-6
  a4 proj_func vs 2935 s_base[func] < 1e-4 (bit-level)
  a5 proj_null0 vs 2935 s_base[null0] < 1e-4 (bit-level)
  a6 sep_f > 0
  a7 xdir identity < 1e-9
  a8 null0 collision-free
  a9 2945 repro (raw ratio, NOT rounded): |ratio_L15 -
     1.2149| < 0.1 AND |ratio_L17 - 1.5004| < 0.1
     (from the s=2.0 dose points, K=3 median)
  a10 3002 source integrity: result hash == seal AND
      verdict == context_entangled_qwen
  a11 same-session determinism < 1e-6 (all K-repeat
      arms)

Tags: Omega-G3 / lang axis / len-2 / en classes /
snapshot machine (no ablation) / dimensionless gates /
descriptive T3 / dose + mirror symmetry.
"""
import hashlib
import json
import os
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC_2887 = os.path.join(BASE, 'phase2887',
                        'language_axis_mlp',
                        'language_axis_mlp.npz')
SRC_2927 = os.path.join(BASE, 'phase2927',
                        'probe_relativity',
                        'probe_relativity.npz')
SRC_2935 = os.path.join(BASE, 'phase2935',
                        'null_amp_anatomy',
                        'null_amp_anatomy.npz')
SRC_2939 = os.path.join(BASE, 'phase2939',
                        'rotation_target',
                        'rotation_target.npz')
SRC_2945 = os.path.join(BASE, 'phase2945',
                        'threshold_curves',
                        'result.json')
D_3002 = os.path.join(BASE, 'phase3002',
                      'omega_g2_robustness_source_qwen')
OUT = os.path.join(BASE, 'phase3003',
                   'omega_g3_dose_mirror_qwen')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NL, HID, VOCAB = 36, 2560, 151936
DOSE_LAYERS = (15, 17)
S_GRID = (0.5, 1.0, 2.0, 3.0)
K_DOSE = 3
K_SCAN = 2
SEED_NULL = 2896          # 3000/3002 verbatim
SEED_RND = 3004           # 3002 verbatim (same sub_rnd0)
S_IDX = (0, 1, 4)
REF_2945 = {'15': 1.2149, '17': 1.5004}
A9_TOL = 0.1
LIN_TOL = 0.15
MIRROR_TOL = 0.2
RECT_FRAC = 0.3
N_SUB_RND = 2

PREREG = {
    'mode': 'qwen3-4b only; 3002 machine verbatim (57 '
            'words 2887, dirs_word a1 vs 2927, Vt8 a3 vs '
            '2939, u35, dcks 2939, xdir per-cell, attn_in '
            'pos-1 coef, bf16); dose grid + mirror at '
            'the 3002 amplification layers L15/L17',
    'question': 'is the qwen mid-band amplification a '
                'linear symmetric channel gain (passive '
                'no-erase) or a rectifying/nonlinear '
                'mechanism (active direction-selective '
                'circuit)?',
    'T1': 'dose law at L15/L17: s in {0.5,1,2,3} K=3; '
          'ratio_med(s); linear := through-origin fit '
          'over all 4 points with max relative residual '
          '<= 0.15 (per layer); sub_rnd0 at s={1,2} L17 '
          'descriptive (does orthogonal random also '
          'grow with s?)',
    'T2': 'mirror symmetry: -xdir s=2 K=2 at L15/L17; '
          'symmetric := BOTH layers |r-/r+ - 1| <= 0.2; '
          'rectifying := BOTH layers r-/r+ <= 0.3',
    'T3': 'DESCRIPTIVE cross-model contrast from 3001 '
          'sealed result; no verdict branch',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'symmetric AND linear => '
               'linear_symmetric_amplification_qwen; '
               'symmetric AND NOT linear => '
               'symmetric_nonlinear_amplification_qwen; '
               'rectifying => '
               'rectifying_amplification_qwen; '
               'else => asymmetric_amplification_qwen',
    'tags': 'Omega-G3 / lang axis / len-2 / en classes / '
            'snapshot machine (no ablation) / '
            'dimensionless gates / descriptive T3 / '
            'dose + mirror symmetry',
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


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 3003,
                   'name': 'omega_g3_dose_mirror_qwen',
                   'created':
                       time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {
                       's2887': sha8(SRC_2887),
                       's2927': sha8(SRC_2927),
                       's2935': sha8(SRC_2935),
                       's2939': sha8(SRC_2939),
                       's2945': sha8(SRC_2945),
                       's3002result': sha8(
                           D_3002 + r'\result.json'),
                       's3002seal': sha8(
                           D_3002 + r'\seal.json')},
                   'model': 'qwen3-4b',
                   'n_layers': NL, 'hidden': HID,
                   'vocab': VOCAB,
                   'dose_layers': list(DOSE_LAYERS),
                   's_grid': list(S_GRID),
                   'k_dose': K_DOSE, 'k_scan': K_SCAN,
                   's_idx': list(S_IDX),
                   'seed_null': SEED_NULL,
                   'seed_rnd': SEED_RND,
                   'ref_2945': REF_2945,
                   'a9_tol': A9_TOL,
                   'lin_tol': LIN_TOL,
                   'mirror_tol': MIRROR_TOL,
                   'rect_frac': RECT_FRAC,
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    # ---------- sources ----------
    z87 = np.load(SRC_2887, allow_pickle=True)
    words = [tuple(str(w).split(':'))
             for w in z87['words']]
    lab_lang = np.asarray(z87['labels_lang']).astype(int)
    n_words = len(words)
    a0_ok = bool(n_words == 57)
    log('a0 words == 2887 re-export: %s (n=%d)'
        % (a0_ok, n_words), lines)

    z27 = np.load(SRC_2927, allow_pickle=True)
    dirs27 = z27['dirs_word'].astype(np.float64)
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
    r45 = json.load(open(SRC_2945, encoding='utf-8'))

    # a10 3002 source integrity
    seal02 = json.load(open(D_3002 + r'\seal.json',
                            encoding='utf-8'))
    a10_ok = bool(seal02['result_sha256_8']
                  == sha8(D_3002 + r'\result.json'))
    r02 = json.load(open(D_3002 + r'\result.json',
                         encoding='utf-8'))
    a10_ok = a10_ok and bool(
        r02['final_verdict'] == 'context_entangled_qwen'
        and r02['anchor_all_ok'] is True)
    log('a10 3002 integrity %s' % a10_ok, lines)

    # ---------- model ----------
    import torch
    import sys
    sys.path.insert(
        0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
    from phase2662_symmetric_mapping_contract \
        import load_native
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
            assert tid_map[w] == int(ck), \
                'key mismatch %s' % w
    func_tid = tid('the')
    word_tids = set(tid_map.values())

    rng0 = np.random.default_rng(SEED_NULL)
    null0_tids = []
    while len(null0_tids) < n_words:
        r = int(rng0.integers(0, VOCAB))
        if r not in word_tids and r > 0:
            null0_tids.append(r)
    a8_ok = bool(len(null0_tids) == n_words
                 and not (set(null0_tids) & word_tids))
    log('a8 null0 collision-free: %s' % a8_ok, lines)

    batch = {'func': [[func_tid, tid_map[words[i][2]]]
                      for i in range(n_words)],
             'null0': [[null0_tids[i],
                        tid_map[words[i][2]]]
                       for i in range(n_words)]}

    model, _ = load_native('qwen4')
    model.eval()
    layers = model.model.layers
    assert len(layers) == NL
    log('model loaded', lines)

    cap = {'ai': {}}
    state_fin = {'on': False}
    fin_cap = {}
    inj = {'coef': None, 'scale': 0.0, 'vec': None}
    handles = []

    def pre_attn(li):
        def h(module, args, kwargs):
            x = args[0] if args \
                else kwargs.get('hidden_states')
            if x is None or x.dim() < 2:
                return None
            if inj['coef'] is not None \
                    and li in inj['coef']:
                c = inj['coef'][li]
                if c != 0.0:
                    x = x.clone()
                    x[:, 1, :] = x[:, 1, :] \
                        + c * inj['scale'] * inj['vec']
                if args:
                    return ((x,) + tuple(args[1:]),
                            kwargs)
                nkw = dict(kwargs)
                nkw['hidden_states'] = x
                return (args, nkw)
            cap['ai'].setdefault(li, []).append(
                x.detach().float().cpu().numpy())
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
        with torch.no_grad():
            model(torch.tensor([toks], device='cuda'))
        return {li: cap['ai'][li][0]
                for li in cap['ai']}

    def forward_batch(toks_list, coef=None, scale=0.0):
        clear_cap()
        fin_cap.pop('x', None)
        state_fin['on'] = True
        inj['coef'] = coef
        inj['scale'] = float(scale)
        with torch.no_grad():
            model(torch.tensor(toks_list, device='cuda'))
        inj['coef'] = None
        state_fin['on'] = False
        return fin_cap['x'].astype(np.float64)

    # ---------- pass 1: dirs_word rebuild ----------
    attn_store = {}
    for i, (_, _, w) in enumerate(words):
        attnin_all = forward1(
            [func_tid, tid_map[w]])
        for li in range(NL):
            attn_store[(i, li)] = \
                attnin_all[li].astype(np.float32)
        if (i + 1) % 20 == 0:
            log('pass1 [%d/%d]' % (i + 1, n_words), lines)
    d_w = np.zeros((NL, HID))
    for li in range(NL):
        X = np.stack([attn_store[(i, li)][0, 1]
                      for i in range(n_words)]) \
            .astype(np.float64)
        d_w[li] = X[lab_lang == 0].mean(0) \
            - X[lab_lang == 1].mean(0)
    dirs_word = np.stack([unit(d_w[li])
                          for li in range(NL)])
    a1_diff = float(np.abs(dirs_word - dirs27).max())
    a1_ok = bool(a1_diff < 1e-5)
    log('a1 dirs_word vs 2927 %.2e ok=%s'
        % (a1_diff, a1_ok), lines)

    _, _, Vt = np.linalg.svd(dirs_word,
                             full_matrices=False)
    Vt8 = Vt[:8]
    a3_diff = float(np.abs(Vt8 - Vt8_39).max())
    a3_ok = bool(a3_diff < 1e-6)
    log('a3 Vt8 vs 2939 %.2e ok=%s'
        % (a3_diff, a3_ok), lines)
    u35 = dirs_word[NL - 1]

    dcks_S = dcks_39[:, list(S_IDX)]
    Vt8_S = Vt8[list(S_IDX)]
    xdir = dcks_S @ Vt8_S
    a7_diff = float(np.abs(xdir @ Vt8_S.T - dcks_S).max())
    a7_ok = bool(a7_diff < 1e-9)
    log('a7 xdir identity %.2e ok=%s'
        % (a7_diff, a7_ok), lines)
    med_dS = float(np.median(np.linalg.norm(dcks_S,
                                            axis=1)))
    xdir_t = torch.tensor(xdir, device='cuda',
                          dtype=torch.bfloat16)
    inj['vec'] = xdir_t
    log('inj vec armed n=%d' % xdir.shape[0], lines)

    # ---------- baselines ----------
    fin_f1 = forward_batch(batch['func'])
    fin_f2 = forward_batch(batch['func'])
    a2_rel = float(np.abs(fin_f1 - fin_f2).max()
                   / max(float(np.abs(fin_f1).max()),
                         1e-30))
    a2_ok = bool(a2_rel < 1e-4)
    log('a2 baseline determinism rel %.2e ok=%s'
        % (a2_rel, a2_ok), lines)

    def reads(fin):
        return fin @ u35, fin @ Vt8.T

    proj_f0, c8_f0 = reads(fin_f1)
    a4_diff = float(np.abs(proj_f0 - s_base_35[ifu35])
                    .max())
    a4_ok = bool(a4_diff < 1e-4)
    fin_n0 = forward_batch(batch['null0'])
    proj_n0, _ = reads(fin_n0)
    a5_diff = float(np.abs(proj_n0 - s_base_35[in035])
                    .max())
    a5_ok = bool(a5_diff < 1e-4)
    sep_f = float(proj_f0[lab_lang == 0].mean()
                  - proj_f0[lab_lang == 1].mean())
    a6_ok = bool(sep_f > 0.0)
    sep_n = float(proj_n0[lab_lang == 0].mean()
                  - proj_n0[lab_lang == 1].mean())
    log('a4 %.2e a5 %.2e ok=%s/%s; a6 sep_f=%.2f '
        '(null0 %.2f) ok=%s'
        % (a4_diff, a5_diff, a4_ok, a5_ok, sep_f,
           sep_n, a6_ok), lines)

    def arm(vec_t, scale, layer, k, tag, spreads):
        projs = []
        ratios = []
        coef = {layer: 1.0}
        inj['vec'] = vec_t
        for _ in range(k):
            fin = forward_batch(batch['func'],
                                coef=coef, scale=scale)
            p, c8 = reads(fin)
            projs.append(p)
            cs = c8[:, list(S_IDX)] \
                - c8_f0[:, list(S_IDX)]
            ratios.append(float(np.median(
                np.linalg.norm(cs, axis=1))))
        inj['vec'] = xdir_t
        P = np.stack(projs)
        spreads[tag] = float('%.2e' % float(
            np.abs(P - P.mean(0)).max()))
        p_med = np.median(P, axis=0)
        sep_m = float(p_med[lab_lang == 0].mean()
                      - p_med[lab_lang == 1].mean())
        ratio_m = float(np.median(ratios)) \
            / max(med_dS, 1e-30)
        return p_med, sep_m, ratio_m

    anchor_prelim = bool(a0_ok and a8_ok and a1_ok
                         and a3_ok and a2_ok and a4_ok
                         and a5_ok and a6_ok and a7_ok
                         and a10_ok)
    verdict = None
    T1 = T2 = T3 = None
    spreads = {}
    a11_diff = None
    save = {}
    a9_diffs = None
    a9_ok = False
    linear = None
    symmetric = None
    rectifying = None
    i_en = lab_lang == 0
    i_non = lab_lang == 1

    if anchor_prelim:
        # ---------- T2 random directions (3002 verbatim
        # rng order: rnd_sub built BEFORE any arm call in
        # 3002 - replicate exactly with SEED_RND) ----------
        rng = np.random.default_rng(SEED_RND)
        xc = (Vt8 @ xdir.T).T
        xh = xc / np.linalg.norm(
            xc, axis=1, keepdims=True)
        rnd_sub = []
        for _ in range(N_SUB_RND):
            g = rng.standard_normal((n_words, 8))
            g = g - np.sum(g * xh, axis=1,
                           keepdims=True) * xh
            g = g / np.linalg.norm(g, axis=1,
                                   keepdims=True)
            rnd_sub.append(g @ Vt8)
        sub0_t = torch.tensor(
            rnd_sub[0], device='cuda',
            dtype=torch.bfloat16)

        # ---------- T1 dose law ----------
        dose = {}
        raw_s2 = {}
        for li in DOSE_LAYERS:
            curve = {}
            for s in S_GRID:
                _, sep_m, ratio_m = arm(
                    xdir_t, s, li, K_DOSE,
                    'dose|%d|%.2f' % (li, s),
                    spreads)
                curve['%.2f' % s] = {
                    'sep': round(sep_m, 2),
                    'ratio': round(ratio_m, 4)}
                if abs(s - 2.0) < 1e-9:
                    raw_s2[li] = ratio_m
            ss = np.array(S_GRID, dtype=np.float64)
            rr = np.array([curve['%.2f' % s]['ratio']
                           for s in S_GRID])
            c_lin = float(np.sum(ss * rr)
                          / max(np.sum(ss * ss), 1e-30))
            pred = c_lin * ss
            denom = max(float(np.max(rr)), 1e-30)
            max_rel = float(np.max(np.abs(rr - pred))
                            / denom)
            curve['lin_coef'] = round(c_lin, 4)
            curve['lin_max_rel'] = round(max_rel, 4)
            curve['linear'] = bool(max_rel <= LIN_TOL)
            dose['L%d' % li] = curve
            log('T1 dose L%d: %s'
                % (li, json.dumps(curve)), lines)
        linear = bool(all(dose['L%d' % li]['linear']
                          for li in DOSE_LAYERS))

        # a9 2945 repro from the s=2.0 dose medians
        a9_diffs = {
            '15': abs(raw_s2[15] - REF_2945['15']),
            '17': abs(raw_s2[17] - REF_2945['17'])}
        a9_ok = bool(a9_diffs['15'] < A9_TOL
                     and a9_diffs['17'] < A9_TOL)
        log('a9 2945 repro diffs %s ok=%s'
            % ({k: ('%.2e' % v)
                for k, v in a9_diffs.items()}, a9_ok),
            lines)

        # sub_rnd0 dose control (descriptive)
        subdose = {}
        for s in (1.0, 2.0):
            _, _, ratio_sub = arm(
                sub0_t, s, 17, K_DOSE,
                'subdose|17|%.2f' % s, spreads)
            subdose['%.2f' % s] = round(ratio_sub, 4)
        log('T1 sub_rnd0 dose control L17: %s'
            % json.dumps(subdose), lines)

        # ---------- T2 mirror symmetry ----------
        mir = {}
        ratios_m = {}
        for li in DOSE_LAYERS:
            p_mm, sep_mm, ratio_mm = arm(
                xdir_t, -2.0, li, K_SCAN,
                'mirror|%d' % li, spreads)
            save['proj_mirror_%d' % li] = p_mm
            ratios_m[li] = ratio_mm
            mir['L%d' % li] = {
                'sep': round(sep_mm, 2),
                'ratio': round(ratio_mm, 4)}
        sym_parts = {}
        for li in DOSE_LAYERS:
            sym_parts[li] = abs(
                ratios_m[li]
                / max(raw_s2[li], 1e-30) - 1.0)
        symmetric = bool(all(sym_parts[li]
                             <= MIRROR_TOL
                             for li in DOSE_LAYERS))
        rectifying = bool(all(
            ratios_m[li] / max(raw_s2[li], 1e-30)
            <= RECT_FRAC for li in DOSE_LAYERS))
        T1 = {'dose': dose,
              'sub_rnd0_dose_L17': subdose,
              'linear': linear,
              'lin_tol': LIN_TOL}
        T2 = {'mirror': mir,
              'r_pos_s2': {str(li): round(raw_s2[li], 4)
                           for li in DOSE_LAYERS},
              'sym_parts': {str(li): round(v, 4)
                            for li, v
                            in sym_parts.items()},
              'symmetric': symmetric,
              'rectifying': rectifying,
              'mirror_tol': MIRROR_TOL,
              'rect_frac': RECT_FRAC}
        log('T2 mirror: %s sym_parts=%s symmetric=%s '
            'rectifying=%s'
            % (json.dumps(mir),
               json.dumps({str(li): round(v, 4)
                           for li, v
                           in sym_parts.items()}),
               symmetric, rectifying), lines)

        # ---------- T3 cross-model (descriptive) ----------
        t1g = r02['T1']
        T3 = {'qwen_3002': {
                  'xdir_ratio_L15':
                      r02['T2']['L15']['xdir']['ratio'],
                  'xdir_ratio_L17':
                      r02['T2']['L17']['xdir']['ratio'],
                  'sub_rnd_L17':
                      r02['T2']['L17'][
                          'sub_rnd_median'],
                  'carry_index': t1g['carry_index']},
              'glm4_3001': {
                  'xdir_ratio_L19':
                      r02['T1']['glm4_3001_mirror'][
                          'word_eff'],
                  'note': 'GLM4 contrast from 3001 '
                          'sealed result: xdir 0.0166 '
                          'vs sub 0.0094 at L19 '
                          '(non-selective erasure), '
                          'mirror sep ~ sep_f '
                          '(erased both signs)'},
              'note': 'descriptive; no verdict branch'}

        a11_diff = max(spreads.values())
        a11_ok = bool(a11_diff < 1e-6)
        log('a11 same-session determinism %.2e ok=%s'
            % (a11_diff, a11_ok), lines)

        if not (a9_ok and a11_ok):
            verdict = 'anchor_fail_all_void'
        elif symmetric and linear:
            verdict = \
                'linear_symmetric_amplification_qwen'
        elif symmetric:
            verdict = \
                'symmetric_nonlinear_amplification_qwen'
        elif rectifying:
            verdict = 'rectifying_amplification_qwen'
        else:
            verdict = 'asymmetric_amplification_qwen'
    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('VERDICT %s' % verdict, lines)

    elapsed = time.monotonic() - t0

    anchors = {
        'a0_words': a0_ok, 'a8_collision': a8_ok,
        'a1_diff': a1_diff, 'a2_rel': a2_rel,
        'a3_diff': a3_diff, 'a4_diff': a4_diff,
        'a5_diff': a5_diff, 'a6_sep_f': sep_f,
        'a7_diff': a7_diff,
        'a9_diffs': {k: v
                     for k, v in a9_diffs.items()}
        if a9_diffs is not None else None,
        'a9_ok': a9_ok if anchor_prelim else None,
        'a10_ok': a10_ok, 'a11_diff': a11_diff,
    }
    res = {
        'phase': 3003,
        'final_verdict': verdict,
        'anchor_all_ok': bool(
            a0_ok and a8_ok and a1_ok and a3_ok
            and a2_ok and a4_ok and a5_ok and a6_ok
            and a7_ok and a10_ok and a9_ok
            and a11_ok),
        'anchors': anchors,
        'scale': {'sep_f': round(sep_f, 2),
                  'sep_null0': round(sep_n, 2),
                  'med_dS': round(med_dS, 4)},
        'T1': T1, 'T2': T2, 'T3': T3,
        'tags': PREREG['tags'],
        'elapsed_s': round(elapsed, 1),
        'correction_note': 'first run',
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    save['dirs_word'] = dirs_word
    save['Vt8'] = Vt8
    save['u35'] = u35
    save['xdir'] = xdir
    save['words'] = np.array(['%s:%s:%s' % w
                              for w in words],
                             dtype=object)
    save['labels_lang'] = lab_lang
    save['null0_tids'] = np.array(null0_tids)
    save['proj_base'] = proj_f0
    npz_path = os.path.join(
        OUT, 'omega_g3_dose_mirror_qwen.npz')
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
