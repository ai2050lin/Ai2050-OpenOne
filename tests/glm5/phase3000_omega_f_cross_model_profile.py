# -*- coding: utf-8 -*-
"""Phase 3000: Omega-F cross-model propagation profile
(qwen full-depth scan vs GLM4 2999 sealed scan).

Why: 2945 established the qwen s_c machine (ratio reaches
~1.2-1.5 at s=2 in L15/17); 2998/2999 showed GLM4 erases
the same machine globally (max ratio 0.128 @L4, s_c>6).
Open question (plan v4 Omega-F close): quantify the
CROSS-MODEL propagation profiles on the SAME dimensionless
protocol - single-layer xdir injection, ratio =
median||dc8_S||/med_dS at the final-norm readout - so the
defense gap is a number, not a narrative.

Design: qwen3-4b arm only (GLM4 side reused from the 2999
sealed artifact, verified by hash).  2945 machine
verbatim: 57 words (2887), dirs_word rebuild (a1 vs 2927),
Vt8 (a3 vs 2939), u35 readout, dcks from 2939 coords
(null0 - func), xdir = dcks_S @ Vt8_S, S_IDX=(0,1,4),
attn_in pos-1 single-layer coef injection, bf16.
  T1 scan: s=2.0 K=2 at every layer L2..L29 (relative
     depth 0.083..0.833, paralleling GLM4 L2..L33
     0.075..0.85); per-layer sep_med, ratio_med.
  T2 dose: top-2 layers by ratio_med, s in {0.5,1,1.5,2}
     K=3; s_c = first sep_med < 100 (qwen-native absolute
     from 2945 prereg; INTERNAL descriptive only - the
     cross-model gates stay dimensionless).
  T3 mirror: -xdir s=2 at argmax layer.
  T4 cross-model (dimensionless, from scan profiles):
     max_ratio, band_n (#layers ratio>=0.1), area (mean
     ratio over scan), argmax relative depth - qwen vs
     GLM4(2999).

Verdict (frozen):
  anchor fail => anchor_fail_all_void
  qwen max_ratio >= 0.5 AND qwen area > 3x glm_area
  AND qwen band_n > 2x glm_band_n
                          => cross_model_scale_gap_confirmed
  qwen max_ratio >= 0.5   => profiles_converged_mid
  qwen max_ratio < 0.1    => qwen_band_absent_discrepant
  else                    => weak_partial_band_qwen

Anchors (frozen):
  a0 words == 2887 re-export
  a1 dirs_word vs 2927 < 1e-5
  a2 baseline determinism < 1e-4
  a3 Vt8 vs 2939 < 1e-6
  a4 proj_func vs 2935 < 1e-4
  a5 proj_null0 vs 2935 < 1e-4
  a6 sep_f > 0
  a7 xdir identity < 1e-9
  a8 same-session determinism < 1e-6 (all arms)
  a9 2945 reproduction: ratio_med(L15,s=2) within 0.1 of
     1.2149 AND ratio_med(L17,s=2) within 0.1 of 1.5004
  a10 GLM4 source integrity: 2999 npz sha256_8 == its
      seal value AND verdict/scan fields present

Tags: Omega-F close / lang axis / len-2 / en classes /
snapshot machine (no ablation) / dimensionless
cross-model gates.
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
SRC_2999D = os.path.join(BASE, 'phase2999',
                         'omega_f2b_sensitivity_band_glm4')
OUT = os.path.join(BASE, 'phase3000',
                   'omega_f_cross_model_profile')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NL, VOCAB = 36, 151936
SCAN_LAYERS = tuple(range(2, 30))      # L2..L29
S_SCAN = 2.0
K_SCAN = 2
S_GRID_DOSE = (0.5, 1.0, 1.5, 2.0)
K_DOSE = 3
DOSE_TOP = 2
S_IDX = (0, 1, 4)
SEP_THRESH_Q = 100.0                   # qwen-native, 2945
RATIO_STRONG = 0.5
RATIO_DEAD = 0.1
A2945_L15 = 1.2149
A2945_L17 = 1.5004
A9_TOL = 0.1
AREA_GAP = 3.0
BAND_GAP = 2.0

PREREG = {
    'mode': 'qwen3-4b only; 2945 machine verbatim (57 '
            'words 2887, dirs a1 vs 2927, Vt8 a3 vs '
            '2939, u35, dcks from 2939 coords, xdir = '
            'dcks_S@Vt8_S, attn_in pos-1 single-layer '
            'coef); scan s=2 K=2 at L2..L29; dose top-2 '
            's{0.5..2} K3; mirror -xdir at argmax; '
            'GLM4 side from 2999 sealed npz (a10 hash)',
    'question': 'how large is the cross-model '
                'propagation gap on the same '
                'dimensionless protocol (ratio '
                'profile, band_n, area)?',
    'scale': 'dimensionless: ratio = median||dc8_S||/'
             'med_dS; gates RATIO_STRONG=0.5, '
             'RATIO_DEAD=0.1, AREA_GAP=3, BAND_GAP=2; '
             's_c via qwen-native sep<100 INTERNAL '
             'descriptive only',
    'anchors': {
        'a0': 'words == 2887 re-export',
        'a1': 'dirs_word vs 2927 < 1e-5',
        'a2': 'baseline determinism < 1e-4',
        'a3': 'Vt8 vs 2939 < 1e-6',
        'a4': 'proj_func vs 2935 < 1e-4',
        'a5': 'proj_null0 vs 2935 < 1e-4',
        'a6': 'sep_f > 0',
        'a7': 'xdir identity < 1e-9',
        'a8': 'same-session determinism < 1e-6',
        'a9': '2945 repro: ratio L15/L17 s=2 within '
              '0.1 of 1.2149/1.5004',
        'a10': '2999 npz hash == seal + fields present'},
    'T1': 'qwen scan profile: per-layer sep_med/'
          'ratio_med at s=2; argmax + band (ratio>=0.1)',
    'T2': 'qwen dose: s_c = first sep_med < 100 '
          '(interpolated) at top-2 layers',
    'T3': 'mirror direction specificity at argmax',
    'T4': 'cross-model: qwen vs GLM4(2999) max_ratio, '
          'band_n, area (dimensionless)',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'max>=0.5 AND area>3x AND band>2x => '
               'cross_model_scale_gap_confirmed; '
               'max>=0.5 => profiles_converged_mid; '
               'max<0.1 => qwen_band_absent_discrepant; '
               'else => weak_partial_band_qwen',
    'tags': 'Omega-F close / lang axis / len-2 / en '
            'classes / snapshot machine (no ablation) / '
            'dimensionless cross-model gates',
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
        json.dump({'phase': 3000,
                   'name': 'omega_f_cross_model_profile',
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
                       's2999result': sha8(
                           SRC_2999D + r'\result.json'),
                       's2999seal': sha8(
                           SRC_2999D + r'\seal.json')},
                   'model': 'qwen3-4b', 'n_layers': NL,
                   'vocab': VOCAB,
                   'scan_layers': list(SCAN_LAYERS),
                   's_scan': S_SCAN, 'k_scan': K_SCAN,
                   's_grid_dose': list(S_GRID_DOSE),
                   'k_dose': K_DOSE,
                   'dose_top': DOSE_TOP,
                   's_idx': list(S_IDX),
                   'sep_thresh_q': SEP_THRESH_Q,
                   'a2945_l15': A2945_L15,
                   'a2945_l17': A2945_L17,
                   'a9_tol': A9_TOL,
                   'area_gap': AREA_GAP,
                   'band_gap': BAND_GAP,
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    # ---------- sources ----------
    z87 = np.load(SRC_2887, allow_pickle=True)
    words = [tuple(str(w).split(':'))
             for w in z87['words']]
    lab_lang = np.asarray(z87['labels_lang']).astype(int)
    n_words = len(words)
    assert n_words == 57

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
    d3 = r45['D3_ratio']
    a9_ref = {'15': d3['L15']['2.00'],
              '17': d3['L17']['2.00']}
    log('a9 refs from 2945: %s' % a9_ref, lines)

    # a10 GLM4 source integrity
    seal99 = json.load(open(SRC_2999D + r'\seal.json',
                            encoding='utf-8'))
    npz99_path = (SRC_2999D
                  + r'\omega_f2b_sensitivity_band_glm4.npz')
    a10_ok = bool(seal99['npz_sha256_8']
                  == sha8(npz99_path))
    r99 = json.load(open(SRC_2999D + r'\result.json',
                         encoding='utf-8'))
    a10_ok = a10_ok and bool(
        r99['final_verdict'] == 'weak_partial_band_glm4'
        and 'scan' in r99 and r99['anchor_all_ok'] is True)
    scan_g = {int(k): v['ratio']
              for k, v in r99['scan'].items()}
    glm_area = float(np.mean([scan_g[k]
                              for k in sorted(scan_g)]))
    glm_band_n = int(sum(1 for k in scan_g
                         if scan_g[k] >= RATIO_DEAD))
    glm_max = max(scan_g.values())
    glm_argmax_rel = (max(scan_g,
                          key=lambda k: scan_g[k]) + 1) / 40.0
    log('a10 2999 integrity %s (glm area=%.4f band_n=%d '
        'max=%.4f argmax_rel=%.3f)'
        % (a10_ok, glm_area, glm_band_n, glm_max,
           glm_argmax_rel), lines)

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
    a0_ok = True
    func_tid = tid('the')
    word_tids = set(tid_map.values())

    rng0 = np.random.default_rng(2896)
    null0_tids = []
    while len(null0_tids) < n_words:
        r = int(rng0.integers(0, VOCAB))
        if r not in word_tids and r > 0:
            null0_tids.append(r)
    batch = {'func': [[func_tid, tid_map[words[i][2]]]
                      for i in range(n_words)],
             'null0': [[null0_tids[i],
                        tid_map[words[i][2]]]
                       for i in range(n_words)]}
    log('a0 words %d ok; null0 collision-free %s'
        % (n_words, not (set(null0_tids) & word_tids)),
        lines)

    model, _ = load_native('qwen4')
    model.eval()
    layers = model.model.layers
    assert len(layers) == NL
    log('model loaded', lines)

    cap = {'attnin': {}}
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
    handles.append(model.model.norm
                   .register_forward_pre_hook(
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
    d_dim = attn_store[(0, 0)].shape[-1]
    diffs_w = np.zeros((NL, d_dim))
    for li in range(NL):
        X = np.stack([attn_store[(i, li)][0, 1]
                      for i in range(n_words)]) \
            .astype(np.float64)
        diffs_w[li] = X[lab_lang == 0].mean(0) \
            - X[lab_lang == 1].mean(0)
    dirs_word = np.stack([unit(diffs_w[li])
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
    fin_n0 = forward_batch(batch['null0'])
    proj_n0, _ = reads(fin_n0)
    a4_diff = float(np.abs(proj_f0 - s_base_35[ifu35])
                    .max())
    a4_ok = bool(a4_diff < 1e-4)
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

    def arm(coef_scale, layer, k, tag, spreads):
        projs = []
        ratios = []
        coef = {layer: 1.0}
        for _ in range(k):
            fin = forward_batch(batch['func'],
                                coef=coef,
                                scale=coef_scale)
            p, c8 = reads(fin)
            projs.append(p)
            cs = c8[:, list(S_IDX)] \
                - c8_f0[:, list(S_IDX)]
            ratios.append(float(np.median(
                np.linalg.norm(cs, axis=1))))
        P = np.stack(projs)
        spreads[tag] = float('%.2e' % float(
            np.abs(P - P.mean(0)).max()))
        p_med = np.median(P, axis=0)
        sep_m = float(p_med[lab_lang == 0].mean()
                      - p_med[lab_lang == 1].mean())
        ratio_m = float(np.median(ratios)) \
            / max(med_dS, 1e-30)
        return p_med, sep_m, ratio_m

    anchor_prelim = bool(a0_ok and a1_ok and a2_ok
                         and a3_ok and a4_ok and a5_ok
                         and a6_ok and a7_ok and a10_ok)
    verdict = None
    T1 = T2 = T3 = T4 = None
    scan = {}
    dose = {}
    spreads = {}
    a8_diff = None
    a9_diffs = None
    save = {}

    if anchor_prelim:
        # ---------- T1 scan ----------
        for li in SCAN_LAYERS:
            p_med, sep_m, ratio_m = arm(
                S_SCAN, li, K_SCAN, 'scan|%d' % li,
                spreads)
            scan[li] = {'sep': sep_m, 'ratio': ratio_m}
            save['proj_scan_%d' % li] = p_med
        log('scan done', lines)
        a9_diffs = {
            '15': abs(scan[15]['ratio'] - a9_ref['15']),
            '17': abs(scan[17]['ratio'] - a9_ref['17'])}
        a9_ok = bool(a9_diffs['15'] < A9_TOL
                     and a9_diffs['17'] < A9_TOL)
        log('a9 2945 repro diffs %s ok=%s'
            % ({k: round(v, 4)
                for k, v in a9_diffs.items()}, a9_ok),
            lines)
        prof = {li: (round(scan[li]['sep'], 1),
                     round(scan[li]['ratio'], 3))
                for li in SCAN_LAYERS}
        log('scan (sep, ratio): %s'
            % json.dumps({str(k): v
                          for k, v in prof.items()}),
            lines)

        if not a9_ok:
            verdict = 'anchor_fail_all_void'
        else:
            li_best = max(SCAN_LAYERS,
                          key=lambda li: scan[li]['ratio'])
            max_ratio = scan[li_best]['ratio']
            band = [li for li in SCAN_LAYERS
                    if scan[li]['ratio'] >= RATIO_DEAD]
            q_area = float(np.mean(
                [scan[li]['ratio']
                 for li in SCAN_LAYERS]))
            T1 = {'argmax_layer': li_best,
                  'max_ratio': round(max_ratio, 4),
                  'band': band,
                  'band_n': len(band),
                  'area': round(q_area, 4),
                  'argmax_rel_depth':
                      round((li_best + 1)
                            / float(NL), 3)}
            log('T1 argmax=L%d ratio=%.3f band_n=%d '
                'area=%.3f'
                % (li_best, max_ratio, len(band),
                   q_area), lines)

            # ---------- T3 mirror ----------
            p_mm, sep_mm, ratio_mm = arm(
                -S_SCAN, li_best, K_SCAN,
                'mirror|%d' % li_best, spreads)
            plus_drop = sep_f - scan[li_best]['sep']
            mirror_drop = sep_f - sep_mm
            t3_specific = bool(sep_mm > sep_f
                               or mirror_drop
                               < 0.5 * plus_drop)
            T3 = {'layer': li_best,
                  'mirror_sep': round(sep_mm, 2),
                  'plus_sep': round(
                      scan[li_best]['sep'], 2),
                  'direction_specific': t3_specific}
            log('T3 mirror L%d sep=%.1f (plus %.1f, '
                'base %.1f) specific=%s'
                % (li_best, sep_mm,
                   scan[li_best]['sep'], sep_f,
                   t3_specific), lines)

            # ---------- T2 dose ----------
            top = sorted(SCAN_LAYERS,
                         key=lambda li: scan[li]['ratio'],
                         reverse=True)[:DOSE_TOP]
            t2_rows = {}
            for li in top:
                curve = {}
                for s in S_GRID_DOSE:
                    p_med, sep_m, ratio_m = arm(
                        s, li, K_DOSE,
                        'dose|%d|%.2f' % (li, s),
                        spreads)
                    curve['%.2f' % s] = {
                        'sep': round(sep_m, 2),
                        'ratio': round(ratio_m, 4)}
                save['proj_dose_%d' % li] = p_med
                seps = [(s, curve['%.2f' % s]['sep'])
                        for s in S_GRID_DOSE]
                cross = [i for i, (s, sp)
                         in enumerate(seps)
                         if sp < SEP_THRESH_Q]
                if not cross:
                    s_c = None
                else:
                    i0 = cross[0]
                    if i0 == 0:
                        s_c = S_GRID_DOSE[0]
                    else:
                        s0, sp0 = seps[i0 - 1]
                        s1_, sp1 = seps[i0]
                        w = (sp0 - SEP_THRESH_Q) \
                            / max(sp0 - sp1, 1e-30)
                        s_c = s0 + w * (s1_ - s0)
                t2_rows['L%d' % li] = {
                    'curve': curve,
                    's_c': None if s_c is None
                    else round(float(s_c), 3)}
                log('dose L%d: %s'
                    % (li, json.dumps(curve)), lines)
            T2 = {'layers': top, 'rows': t2_rows,
                  'sep_thresh_q': SEP_THRESH_Q,
                  'note': 's_c uses qwen-native '
                          'absolute 100; INTERNAL '
                          'descriptive only'}

            # ---------- T4 cross-model ----------
            q_band_n = len(band)
            t4_pass = bool(max_ratio >= RATIO_STRONG
                           and q_area
                           > AREA_GAP * glm_area
                           and q_band_n
                           > BAND_GAP * glm_band_n)
            T4 = {'qwen': {
                      'max_ratio': round(max_ratio, 4),
                      'band_n': q_band_n,
                      'area': round(q_area, 4),
                      'argmax_rel_depth': T1[
                          'argmax_rel_depth']},
                  'glm4_2999': {
                      'max_ratio': glm_max,
                      'band_n': glm_band_n,
                      'area': round(glm_area, 4),
                      'argmax_rel_depth':
                          round(glm_argmax_rel, 3)},
                  'area_ratio': round(
                      q_area / max(glm_area, 1e-30), 1),
                  'pass': t4_pass}
            log('T4 qwen area=%.3f band_n=%d vs glm '
                'area=%.3f band_n=%d area_ratio=%.1f '
                'pass=%s'
                % (q_area, q_band_n, glm_area,
                   glm_band_n,
                   q_area / max(glm_area, 1e-30),
                   t4_pass), lines)

            a8_diff = max(spreads.values())
            a8_ok = bool(a8_diff < 1e-6)
            log('a8 same-session determinism %.2e ok=%s'
                % (a8_diff, a8_ok), lines)
            if not a8_ok:
                verdict = 'anchor_fail_all_void'
            elif t4_pass:
                verdict = 'cross_model_scale_gap_confirmed'
            elif max_ratio >= RATIO_STRONG:
                verdict = 'profiles_converged_mid'
            elif max_ratio < RATIO_DEAD:
                verdict = 'qwen_band_absent_discrepant'
            else:
                verdict = 'weak_partial_band_qwen'
    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('VERDICT %s' % verdict, lines)

    elapsed = time.monotonic() - t0

    anchors = {
        'a0_words': a0_ok, 'a1_diff': a1_diff,
        'a2_rel': a2_rel, 'a3_diff': a3_diff,
        'a4_diff': a4_diff, 'a5_diff': a5_diff,
        'a6_sep_f': sep_f, 'a7_diff': a7_diff,
        'a8_diff': a8_diff,
        'a9_diffs': None if a9_diffs is None
        else {k: round(v, 4)
              for k, v in a9_diffs.items()},
        'a10_ok': a10_ok,
    }
    res = {
        'phase': 3000,
        'final_verdict': verdict,
        'anchor_all_ok': bool(
            anchor_prelim and a8_ok
            and a9_diffs is not None
            and all(v < A9_TOL
                    for v in a9_diffs.values())),
        'anchors': anchors,
        'scale': {'sep_f': round(sep_f, 2),
                  'sep_null0': round(sep_n, 2),
                  'med_dS': round(med_dS, 4)},
        'T1': T1, 'T2': T2, 'T3': T3, 'T4': T4,
        'scan': {str(k): {'sep': round(v['sep'], 2),
                          'ratio': round(v['ratio'], 4)}
                 for k, v in scan.items()},
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
    save['words'] = np.array(['%s:%s:%s' % w
                              for w in words],
                             dtype=object)
    save['labels_lang'] = lab_lang
    save['null0_tids'] = np.array(null0_tids)
    npz_path = os.path.join(
        OUT, 'omega_f_cross_model_profile.npz')
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
