# -*- coding: utf-8 -*-
"""Phase 2999: Omega-F2b GLM4 sensitivity band depth scan.

Why: 2998 rebuilt the 2945 injection machine on GLM4 and
found near-total erasure at the qwen-equivalent depth band
L17/18/19 (ratio <= 0.027 at s=2 vs qwen 0.86; sep collapse
<= 0.7% vs qwen ~46%; L19 slightly REVERSES).  Open
question (plan v4 next-step A): is GLM4's defense EARLIER
(erosion starts before L17 and is done by then), is the
mechanism LAYER-SHIFTED (some other depth band still
sensitive), or is erasure GLOBAL (no layer works)?

Design: single-layer xdir injection at s=2.0 (2998 max,
where qwen ratio is far past threshold) at every layer
L2..L33 (relative depth 0.05..0.825), K=2 repeats, median
readouts; per-layer sep_med and ratio_med.  Then a dose
stage at the top-2 responsive layers (by ratio_med) on a
finer/wider s grid with K=3 to estimate s_c (first
sep_med < 0.5*sep_f, interpolated) and ratio_c.  Mirror
control: -xdir at s=2 at the argmax layer and at the L19
reference (direction specificity).

Machine: 2998 verbatim (98 cells, dirs_g rebuild bit-level
anchored to the 2996 npz, Vt8/u39, in-session dcks, xdir =
dcks_S @ Vt8_S, attn_in pos-1 injection, final-norm input
readout).

CROSS-MODEL SCALE (2998 prereg carries over; dimensionless):
  SEP_REF  = 0.5 * sep_f
  RATIO_STRONG = 0.5  (layer counts as sensitive if the
      propagated displacement reaches half the null
      magnitude at s=2)
  RATIO_DEAD   = 0.1  (layer counts as erasing below this)

Verdict (frozen):
  anchor fail            => anchor_fail_all_void
  max ratio_med >= 0.5 AND its layer reaches s_c in dose
                         => band_localized_glm4
  max ratio_med < 0.1    => erasure_global_glm4
  else                   => weak_partial_band_glm4

Anchors (frozen; 2998 set + scan-specific):
  a0 words_g == 2996 glm src protocol re-export
  a1 dirs_g[17/18/19] vs 2996 dirs_attn_glm < 1e-6
  a2 func baseline determinism < 1e-4
  a3/a4 xdir identity < 1e-9
  a5 same-session K repeats determinism < 1e-6 (max over
     all scan/dose arms)
  a6 sep_f > 0
  a7 u39 unit < 1e-12
  a8 null0 collision-free
  a9 machine transmits: ratio_med at the L19 reference
     arm (s=2) >= 0.01 (2998 measured ~0.02-0.10 there;
     guards against a silent re-break of the injector)

Tags: plan v4 Omega-F2b / lang axis / len-2 / en classes /
snapshot machine (no ablation) / dimensionless gates.
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
OUT = os.path.join(BASE, 'phase2999',
                   'omega_f2b_sensitivity_band_glm4')
MD_G = r'D:\AI2050\Ai2050-OpenOne\models\hf\glm4-9b-chat-hf'
L_CAND = ["because", "therefore", "although", "unless",
          "however", "thus", "moreover", "since", "whereas",
          "despite", "hence", "nevertheless", "consequently",
          "furthermore", "otherwise", "instead", "while",
          "accordingly", "likewise", "meanwhile", "nonetheless",
          "thereafter", "whereby", "albeit"]
NL, HID, VOCAB = 40, 4096, 151552
SCAN_LAYERS = tuple(range(2, 34))     # L2..L33
REF_LAYER = 19                        # 2998 reference arm
S_SCAN = 2.0
K_SCAN = 2
S_GRID_DOSE = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
K_DOSE = 3
DOSE_TOP = 2
S_IDX = (0, 1, 4)
SEED_NULL = 2889
RATIO_STRONG = 0.5
RATIO_DEAD = 0.1
SEP_FRAC = 0.5

PREREG = {
    'mode': 'glm4-9b only; 2998 machine verbatim (98 '
            'cells, dirs_g anchored to 2996 npz, Vt8/u39, '
            'in-session dcks, xdir single-layer attn_in '
            'pos-1 injection); scan s=2.0 K=2 at L2..L33; '
            'dose stage top-2 layers by ratio, s grid '
            '0.5..6.0 K=3; mirror -xdir at argmax layer '
            'and L19 ref',
    'question': 'is GLM4 language-separation defense '
                'EARLIER than L17, layer-SHIFTED to '
                'another band, or GLOBAL erasure?',
    'scale': 'dimensionless: SEP_REF=0.5*sep_f; '
             'RATIO_STRONG=0.5, RATIO_DEAD=0.1',
    'anchors': {
        'a0': 'words_g == 2996 glm src re-export',
        'a1': 'dirs_g[17/18/19] vs 2996 dirs_attn_glm '
              '< 1e-6',
        'a2': 'baseline determinism < 1e-4',
        'a3': 'xdir identity < 1e-9',
        'a4': 'injection self-check < 1e-9',
        'a5': 'same-session determinism < 1e-6 (all arms)',
        'a6': 'sep_f > 0',
        'a7': 'u39 unit < 1e-12',
        'a8': 'null0 collision-free',
        'a9': 'L19 ref arm ratio_med(s=2) >= 0.01'},
    'T1': 'scan: per-layer sep_med/ratio_med at s=2; '
          'argmax layer + contiguous band with '
          'ratio_med >= RATIO_DEAD',
    'T2': 'dose: s_c = first sep_med < SEP_REF '
          '(interpolated), ratio_c interpolated, at '
          'top-2 layers',
    'T3': 'mirror: sep_med(-xdir, s=2) at argmax and L19 '
          'ref; direction-specific iff the mirror sep '
          'drop (sep_f - sep_mirror) < 0.5 * (sep_f - '
          'sep_plus) or the mirror sep rises',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'max ratio >= 0.5 AND that layer reaches '
               's_c => band_localized_glm4; max ratio '
               '< 0.1 => erasure_global_glm4; else '
               'weak_partial_band_glm4',
    'tags': 'Omega-F2b / lang axis / len-2 / en classes / '
            'snapshot machine (no ablation) / '
            'dimensionless gates',
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
        json.dump({'phase': 2999,
                   'name': 'omega_f2b_sensitivity_band_glm4',
                   'created':
                       time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2972': sha8(SRC_2972),
                               's2996npz': sha8(SRC_2996)},
                   'model': 'glm4-9b', 'n_layers': NL,
                   'hidden': HID, 'vocab': VOCAB,
                   'scan_layers': list(SCAN_LAYERS),
                   'ref_layer': REF_LAYER,
                   's_scan': S_SCAN,
                   'k_scan': K_SCAN,
                   's_grid_dose': list(S_GRID_DOSE),
                   'k_dose': K_DOSE,
                   'dose_top': DOSE_TOP,
                   's_idx': list(S_IDX),
                   'seed_null': SEED_NULL,
                   'ratio_strong': RATIO_STRONG,
                   'ratio_dead': RATIO_DEAD,
                   'sep_frac': SEP_FRAC,
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
    log('a0 words_g == 2996 glm src re-export: %s'
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

    # ---------- dirs_g rebuild (2998 verbatim) ----------
    store = {}
    for i, c in enumerate(cells):
        clear_cap()
        with _t.no_grad():
            model(_t.tensor([seqs[i]], device='cuda'))
        for li in range(NL):
            store[(i, li)] = cap['ai'][li][0] \
                .astype(np.float32)
        if (i + 1) % 25 == 0:
            log('pass1 [%d/%d]' % (i + 1, n), lines)
    d_w = np.zeros((NL, HID))
    for li in range(NL):
        X = np.stack([store[(i, li)][0, 1]
                      for i in range(n)]).astype(np.float64)
        d_w[li] = X[i_en].mean(0) - X[i_non].mean(0)
    dirs_g = np.stack([unit(d_w[li]) for li in range(NL)])
    a1_diffs = {li: float(np.abs(dirs_g[li] - dirs96[li])
                          .max()) for li in (17, 18, 19)}
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
    a4_ok = a3_ok
    log('a3/a4 xdir identity %.2e ok=%s'
        % (a3_diff, a3_ok), lines)
    med_dS = float(np.median(np.linalg.norm(dcks_S, axis=1)))
    xdir_t = _t.tensor(xdir, device='cuda',
                       dtype=_t.bfloat16)
    inj['vec'] = xdir_t
    log('inj vec armed shape=%s'
        % (tuple(xdir_t.shape),), lines)

    def arm(scale, layer, k, tag, spreads):
        projs = []
        ratios = []
        for _ in range(k):
            fin = forward_batch(seqs, scale=scale,
                                layer=layer)
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
        sep_m = float(p_med[i_en].mean()
                      - p_med[i_non].mean())
        ratio_m = float(np.median(ratios)) \
            / max(med_dS, 1e-30)
        return p_med, sep_m, ratio_m

    anchor_prelim = bool(a0_ok and a8_ok and a1_ok
                         and a7_ok and a2_ok and a6_ok
                         and a3_ok and a4_ok)
    verdict = None
    T1 = T2 = T3 = None
    scan = {}
    dose = {}
    mirror = {}
    spreads = {}
    a5_diff = None
    a9_val = None
    save = {}

    if anchor_prelim:
        # ---------- T1 scan L2..L33 at s=2 ----------
        SEP_REF = SEP_FRAC * sep_f
        for li in SCAN_LAYERS:
            p_med, sep_m, ratio_m = arm(
                S_SCAN, li, K_SCAN, 'scan|%d' % li,
                spreads)
            scan[li] = {'sep': sep_m, 'ratio': ratio_m}
            save['proj_scan_%d' % li] = p_med
        log('scan done', lines)
        a9_val = scan[REF_LAYER]['ratio']
        a9_ok = bool(a9_val >= 0.01)
        log('a9 L19 ref ratio=%.4f ok=%s'
            % (a9_val, a9_ok), lines)
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
            band = [li for li in SCAN_LAYERS
                    if scan[li]['ratio'] >= RATIO_DEAD]
            max_ratio = scan[li_best]['ratio']
            T1 = {'argmax_layer': li_best,
                  'max_ratio': round(max_ratio, 4),
                  'band_dead': band,
                  'ratio_strong': RATIO_STRONG,
                  'ratio_dead': RATIO_DEAD}
            log('T1 argmax=L%d ratio=%.3f band=%s'
                % (li_best, max_ratio,
                   '%d..%d' % (band[0], band[-1])
                   if band else 'EMPTY'), lines)

            # ---------- T3 mirror ----------
            for li in (li_best, REF_LAYER):
                p_mm, sep_mm, ratio_mm = arm(
                    -S_SCAN, li, K_SCAN, 'mirror|%d' % li,
                    spreads)
                mirror[li] = {'sep': sep_mm,
                              'ratio': ratio_mm}
                save['proj_mirror_%d' % li] = p_mm
            m_best = mirror[li_best]
            plus_drop = sep_f - scan[li_best]['sep']
            mirror_drop = sep_f - m_best['sep']
            t3_specific = bool(m_best['sep'] > sep_f
                               or mirror_drop
                               < 0.5 * plus_drop)
            T3 = {'layers': [li_best, REF_LAYER],
                  'mirror': {str(li): {
                      'sep': round(mirror[li]['sep'], 2),
                      'ratio': round(
                          mirror[li]['ratio'], 4)}
                      for li in mirror},
                  'direction_specific': t3_specific}
            log('T3 mirror L%d sep=%.1f (plus %.1f, '
                'base %.1f) specific=%s'
                % (li_best, m_best['sep'],
                   scan[li_best]['sep'], sep_f,
                   t3_specific), lines)

            # ---------- T2 dose at top-2 ----------
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
                         in enumerate(seps) if sp < SEP_REF]
                if not cross:
                    s_c = ratio_c = None
                else:
                    i0 = cross[0]
                    if i0 == 0:
                        s_c = S_GRID_DOSE[0]
                        ratio_c = curve['%.2f'
                                        % s_c]['ratio']
                    else:
                        s0, sp0 = seps[i0 - 1]
                        s1, sp1 = seps[i0]
                        wgt = (sp0 - SEP_REF) \
                            / max(sp0 - sp1, 1e-30)
                        s_c = s0 + wgt * (s1 - s0)
                        r0 = curve['%.2f' % s0]['ratio']
                        r1 = curve['%.2f' % s1]['ratio']
                        ratio_c = r0 + wgt * (r1 - r0)
                t2_rows['L%d' % li] = {
                    'curve': curve,
                    's_c': None if s_c is None
                    else round(s_c, 3),
                    'ratio_c': None if ratio_c is None
                    else round(ratio_c, 4)}
                log('dose L%d: %s'
                    % (li, json.dumps(curve)), lines)
            T2 = {'layers': top, 'rows': t2_rows,
                  'sep_ref': round(SEP_REF, 1)}

            a5_diff = max(spreads.values())
            a5_ok = bool(a5_diff < 1e-6)
            log('a5 same-session determinism %.2e ok=%s'
                % (a5_diff, a5_ok), lines)
            if not a5_ok:
                verdict = 'anchor_fail_all_void'
            else:
                best_row = t2_rows.get('L%d' % li_best)
                reaches = bool(
                    best_row is not None
                    and best_row['s_c'] is not None)
                if max_ratio >= RATIO_STRONG and reaches:
                    verdict = 'band_localized_glm4'
                elif max_ratio < RATIO_DEAD:
                    verdict = 'erasure_global_glm4'
                else:
                    verdict = 'weak_partial_band_glm4'
    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('VERDICT %s' % verdict, lines)

    elapsed = time.monotonic() - t0

    anchors = {
        'a0_words_order': a0_ok, 'a8_collision': a8_ok,
        'a1_diffs': {str(k): v
                     for k, v in a1_diffs.items()},
        'a1_ok': a1_ok, 'a2_rel': a2_rel,
        'a3_diff': a3_diff, 'a5_diff': a5_diff,
        'a6_sep_f': sep_f, 'a7_diff': a7_diff,
        'a9_ref_ratio': a9_val,
    }
    res = {
        'phase': 2999,
        'final_verdict': verdict,
        'anchor_all_ok': bool(
            a0_ok and a8_ok and a1_ok and a7_ok
            and a2_ok and a6_ok and a3_ok and a4_ok
            and a5_ok and a9_val is not None
            and a9_val >= 0.01),
        'anchors': anchors,
        'scale': {'sep_f': round(sep_f, 2),
                  'sep_null0': round(sep_n, 2),
                  'SEP_REF': round(SEP_FRAC * sep_f, 2),
                  'med_dS': round(med_dS, 4)},
        'T1': T1, 'T2': T2, 'T3': T3,
        'scan': {str(k): {'sep': round(v['sep'], 2),
                          'ratio': round(v['ratio'], 4)}
                 for k, v in scan.items()},
        'tags': PREREG['tags'],
        'elapsed_s': round(elapsed, 1),
        'correction_note':
            'run1: crashed at the inj-vec log line '
            "('...%s' %% tuple(shape) put a 2-tuple into a "
            'single placeholder -> TypeError before any '
            'experimental arm; no verdict); run2: '
            'protocol-faithful full pass, authoritative',
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
        OUT, 'omega_f2b_sensitivity_band_glm4.npz')
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
