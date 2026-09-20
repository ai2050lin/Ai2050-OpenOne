# -*- coding: utf-8 -*-
"""Phase 2944: switch layer localization.

Why: 2942 injected the joint U8 displacement pattern
xdir(w) = sum_{k in S={v1,v2,v5}} dcks[w,k]*Vt8[k] at L16
(2927 site) and found a switch-like sep response (steep drop
176.8 -> 33.0 -> -36.2 for s=1..3, overshooting null0 77.3)
plus cross-session instability at s=2 (2942 probe: same-
session bit-deterministic, cross-session sep in {85,33,14}).
2941 showed single-direction v3 injection at L16 is damped
and 2940 localized v3 ownership to L14-L18. Open question:
WHICH layer(s) host the switch - is it localizable to one
layer of the L14-L18 band, or does it require joint
multi-layer drive?

Mode: ONE model (qwen3-4b), forward family. Injection vector
verbatim 2942 (xdir from 2939 dcks + Vt8, S={v1,v2,v5});
site: attn-input pos-1, layer-set injection (new); s in
{1,2,4}; K=3 same-session repeats per config (2942 lesson:
same-session deterministic; verdicts use the median, spread
registered; 2942 L16@2 values are cross-session REFERENCE
ONLY, not anchors).

Layer configs (frozen):
  single: L14, L15, L16, L17, L18 (coef 1.0 each)
  multi_split: L14-18 jointly, coef 0.2 each (total dose =
      s*xdir, comparable to single-layer)
  multi_full: L14-18 jointly, coef 1.0 each (descriptive)

Anchors (frozen; 2942 run values in parentheses):
  a1 dirs_word rebuild vs 2927 npz < 1e-5     (2.17e-08)
  a2 func baseline determinism < 1e-4         (0.0)
  a3 Vt8 rebuild vs 2939 npz < 1e-6           (0.0)
  a4 proj_func vs 2935 s_base[func] < 1e-4    (7.2e-06)
  a5 proj_null0 vs 2935 s_base[null0] < 1e-4  (6.3e-06)
  a6 func separation > 0                      (185.70)
  a7 injection construction self-check < 1e-9 (9.9e-14)
  a8 same-session repeat determinism: for every config/s,
     max spread over K=3 repeats of median proj < 1e-6

Main tests (frozen):
  T1 single-layer switch spectrum: sep_med(c, s=2) for c in
     {L14..L18}; triggered iff sep_med < SWITCH_THRESHOLD
     = 100.0 (transition band between func 185.7 and null0
     77.3).
  T2 multi-layer: sep_med(multi_split, s=2) < 100.
  Verdict (frozen):
    anchor fail => anchor_fail_all_void
    |T_single| >= 1 AND multi_sep < min_single_sep - 20
                            => switch_distributed_synergy
    |T_single| >= 1 (else) => switch_localized
    |T_single| == 0 AND T2 => switch_requires_multilayer
    else                    => switch_not_in_band

Descriptive: D1 sep(c,s) full spectrum; D2 rho(c,s); D3
ratio(c,s) calibration; D4 L16@2 vs 2942 npz reference
(cross-session divergence, NOT an anchor); D5 norm ratios.

Output: phase2944/switch_localization/.
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
SRC_2942 = os.path.join(BASE, 'phase2942', 'u8_joint_injection',
                        'u8_joint_injection.npz')
OUT = os.path.join(BASE, 'phase2944', 'switch_localization')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2944_run_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
SEED = 2896
NH, HD = 32, 128
NL = 36
VOCAB = 151936
SINGLE_LAYERS = (14, 15, 16, 17, 18)
BAND = (14, 15, 16, 17, 18)
S_IDX = (0, 1, 4)             # v1, v2, v5
SCALES = (1.0, 2.0, 4.0)
K_REPEAT = 3
SWITCH_THRESHOLD = 100.0
SYNERGY_MARGIN = 20.0

PREREG = {
    'mode': 'forward family: layer-set injection of xdir(w) '
            '= sum_{k in S={v1,v2,v5}} dcks[w,k]*Vt8[k] '
            '(2942 verbatim vector), site attn-input pos-1; '
            'single layers L14-L18 (coef 1.0), multi_split '
            'L14-18 coef 0.2 each, multi_full coef 1.0 '
            '(descriptive); s in {1,2,4}; K=3 same-session '
            'repeats, median readouts; 2942 L16@2 values '
            'cross-session reference only',
    'question': 'is the regime switch localizable to a '
                'single layer of the L14-L18 v3-ownership '
                'band, or does it require joint multi-layer '
                'drive?',
    'anchors': {
        'a1': 'dirs_word rebuild vs 2927 npz < 1e-5',
        'a2': 'func baseline determinism < 1e-4',
        'a3': 'Vt8 rebuild vs 2939 npz < 1e-6',
        'a4': 'proj_func vs 2935 s_base[func] < 1e-4',
        'a5': 'proj_null0 vs 2935 s_base[null0] < 1e-4',
        'a6': 'func separation > 0',
        'a7': 'injection construction self-check < 1e-9',
        'a8': 'same-session repeat determinism: max spread '
              'of median proj over K=3 < 1e-6 per config/s',
    },
    'T1': 'single-layer switch spectrum: sep_med(c,2) < 100 '
          'iff triggered',
    'T2': 'multi_split L14-18: sep_med < 100',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               '|T_single| >= 1 AND multi_sep < '
               'min_single_sep - 20 => '
               'switch_distributed_synergy; |T_single| >= 1 '
               'else => switch_localized; |T_single| == 0 '
               'AND T2 => switch_requires_multilayer; else '
               '=> switch_not_in_band',
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
        json.dump({'phase': 2944,
                   'name': 'switch_localization',
                   'created':
                       time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2887': sha8(SRC_2887),
                               's2927': sha8(SRC_2927),
                               's2935': sha8(SRC_2935),
                               's2939': sha8(SRC_2939),
                               's2942': sha8(SRC_2942)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'seed': SEED,
                   'single_layers': list(SINGLE_LAYERS),
                   'band': list(BAND),
                   's_idx': list(S_IDX),
                   'scales': list(SCALES),
                   'k_repeat': K_REPEAT,
                   'switch_threshold': SWITCH_THRESHOLD,
                   'synergy_margin': SYNERGY_MARGIN,
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

    z42 = np.load(SRC_2942, allow_pickle=True)
    proj_inj_42 = z42['proj_inj'].astype(np.float64)
    scales_42 = z42['scales'].astype(float)
    proj_f0_42 = z42['proj_func0'].astype(np.float64)
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

    # ---------- injection vector construction ----------
    dcks_S = dcks_39[:, list(S_IDX)]          # (57, 3)
    Vt8_S = Vt8[list(S_IDX)]                  # (3, 2560)
    xdir = dcks_S @ Vt8_S                     # (57, 2560)
    a7_diff = float(np.abs(xdir @ Vt8_S.T - dcks_S).max())
    a7_ok = bool(a7_diff < 1e-9)
    log('a7 construction self-check max abs %.2e ok=%s'
        % (a7_diff, a7_ok), lines)
    xdir_t = torch.tensor(xdir, device='cuda',
                          dtype=torch.bfloat16)

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
        # ---------- configs ----------
        coef_single = {li: {li: 1.0} for li in SINGLE_LAYERS}
        coef_split = {li: 0.2 for li in BAND}
        coef_full = {li: 1.0 for li in BAND}
        configs = [('L%d' % li, coef_single[li])
                   for li in SINGLE_LAYERS] \
            + [('multi_split', coef_split),
               ('multi_full', coef_full)]

        sep_med = {}
        rho_med = {}
        med_shift = {}
        ratio_cs = {}
        norm_r = {}
        spreads = {}
        store_proj = {}
        for cname, coef in configs:
            for s in SCALES:
                projs = []
                for _ in range(K_REPEAT):
                    fin = forward_batch(
                        batch['func'], coef=coef, scale=s,
                        vec=xdir_t)
                    p, c8, nfin = reads(fin)
                    projs.append(p)
                P = np.stack(projs)
                spread = float(np.abs(
                    P - P.mean(0)).max())
                key = (cname, s)
                spreads['%s|%.0f' % key] = \
                    float('%.2e' % spread)
                p_med = np.median(P, axis=0)
                store_proj[key] = p_med
                sep_med['%s|%.0f' % key] = round(float(
                    p_med[lab_lang == 0].mean()
                    - p_med[lab_lang == 1].mean()), 2)
                rho_med['%s|%.0f' % key] = round(
                    spearman(p_med, proj_f0), 4)
                med_shift['%s|%.0f' % key] = round(float(
                    np.median(p_med - proj_f0)), 3)
                cs = c8[:, list(S_IDX)] \
                    - c8_f0[:, list(S_IDX)]
                med_dS = float(np.median(
                    np.linalg.norm(dcks_S, axis=1)))
                ratio_cs['%s|%.0f' % key] = round(float(
                    np.median(np.linalg.norm(cs, axis=1)))
                    / max(med_dS, 1e-30), 4)
                norm_r['%s|%.0f' % key] = round(float(
                    np.median(nfin
                              / np.maximum(nfin_f0, 1e-30))),
                    5)
            log('config %s: sep %s' % (
                cname,
                {('%.0f' % s): sep_med['%s|%.0f' % (cname, s)]
                 for s in SCALES}), lines)

        a8_diff = max(float(v) for v in spreads.values())
        a8_ok = bool(a8_diff < 1e-6)
        log('a8 same-session determinism max spread %.2e '
            'ok=%s' % (a8_diff, a8_ok), lines)
        anchor_ok = bool(anchor_prelim and a8_ok)

        if not anchor_ok:
            verdict = 'anchor_fail_all_void'
        else:
            # ---------- T1/T2 ----------
            t1 = {'sep_med_s2': {'L%d' % li:
                                 sep_med['L%d|2' % li]
                                 for li in SINGLE_LAYERS},
                 'threshold': SWITCH_THRESHOLD}
            t1['triggered'] = [c for c in t1['sep_med_s2']
                               if t1['sep_med_s2'][c]
                               < SWITCH_THRESHOLD]
            t2 = {'sep_med': sep_med['multi_split|2'],
                  'threshold': SWITCH_THRESHOLD,
                  'triggered':
                      bool(sep_med['multi_split|2']
                           < SWITCH_THRESHOLD)}
            min_single = min(t1['sep_med_s2'].values())
            log('T1 single sep(s=2) %s | triggered %s'
                % (t1['sep_med_s2'], t1['triggered']), lines)
            log('T2 multi_split sep(s=2) %.2f triggered=%s'
                % (t2['sep_med'], t2['triggered']), lines)

            if len(t1['triggered']) >= 1 \
                    and t2['sep_med'] < min_single \
                    - SYNERGY_MARGIN:
                verdict = 'switch_distributed_synergy'
            elif len(t1['triggered']) >= 1:
                verdict = 'switch_localized'
            elif t2['triggered']:
                verdict = 'switch_requires_multilayer'
            else:
                verdict = 'switch_not_in_band'

            # ---------- D4: L16@2 vs 2942 reference ----
            i42 = int(np.argmin(np.abs(scales_42 - 2.0)))
            sep42 = float(
                proj_inj_42[i42][lab_lang == 0].mean()
                - proj_inj_42[i42][lab_lang == 1].mean())
            d4 = {'this_session_L16_s2':
                      sep_med['L16|2'],
                  'mean_shift_L16_s2':
                      med_shift['L16|2'],
                  'ref_2942_sep_s2': round(sep42, 2),
                  'ref_2942_median_shift': round(float(
                      np.median(proj_inj_42[i42]
                                - proj_f0_42)), 3),
                  'note': 'cross-session divergence is a '
                          'registered 2942 finding, not '
                          'an anchor'}
            log('D4 L16@2 this session sep %.2f vs 2942 ref '
                '%.2f' % (sep_med['L16|2'], sep42), lines)

            d1 = sep_med
            d2 = rho_med
            d3 = {'med_shift': med_shift,
                  'ratio': ratio_cs}
            d5 = norm_r

            save = {
                'words': np.array(['%s:%s:%s' % w
                                   for w in words],
                                  dtype=object),
                'labels_lang': lab_lang,
                'scales': np.array(SCALES),
                'configs': np.array([c for c, _ in configs]),
                'proj_med': np.stack(
                    [store_proj[(c, s)]
                     for c, _ in configs for s in SCALES]),
                'proj_func0': proj_f0,
                'proj_null0': proj_n0,
                'sep_med': np.array(
                    [sep_med['%s|%.0f' % (c, s)]
                     for c, _ in configs
                     for s in SCALES]),
                'Vt8': Vt8, 'dirs_word': dirs_word,
            }

    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2944, 'model': 'qwen3-4b',
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
                       'ok': anchor_ok},
           'T1': t1, 'T2': t2,
           'D1_sep': d1, 'D2_rho': d2, 'D3_shift_ratio': d3,
           'D4_L16_vs_2942': d4, 'D5_norm': d5,
           'spreads': spreads,
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if save:
        np.savez_compressed(os.path.join(
            OUT, 'switch_localization.npz'), **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2944 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
