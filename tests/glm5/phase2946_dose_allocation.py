# -*- coding: utf-8 -*-
"""Phase 2946: dose-allocation interaction (L17,L16 pair).

Why: 2944 showed the switch requires single-layer peak
concentration (same total dose split over 5 layers does
nothing); 2945 measured per-layer thresholds s_c(L17)=0.656,
s_c(L15)=0.845, s_c(L16)=1.843 and decoupled them from the
propagated-displacement magnitude. Open question: when two
layers are co-injected with a fixed total dose s split as
(alpha, 1-alpha), does the joint threshold follow the
LOCAL-concentration rule
    H_local: s_c(a) = min(s_c17/alpha17, s_c16/alpha16)
or the averaged-dose rule
    H_avg:  s_c(a) = alpha17*s_c17 + alpha16*s_c16?
The discriminating point is J25 (alpha17=0.25):
H_local 2.4573 vs H_avg 1.5463.

Mode: ONE model (qwen3-4b), forward family. Pair (L17,L16);
configs J75={17:0.75,16:0.25}, J50={17:0.5,16:0.5},
J25={17:0.25,16:0.75} + single-layer references
S17={17:1.0}, S16={16:1.0}; xdir injection (2942/2945
verbatim, S_IDX=(0,1,4) i.e. v1/v2/v5);
s grid {0.5,0.75,1.0,1.25,1.5,2.0,2.5,3.0}; K=3 same-session
repeats, median readouts.

Anchors (frozen; reference values from 2945):
  a1 dirs_word rebuild vs 2927 npz < 1e-5     (2.17e-08)
  a2 func baseline determinism < 1e-4         (0.0)
  a3 Vt8 rebuild vs 2939 npz < 1e-6           (0.0)
  a4 proj_func vs 2935 s_base[func] < 1e-4    (7.2e-06)
  a5 proj_null0 vs 2935 s_base[null0] < 1e-4  (6.3e-06)
  a6 func separation > 0                      (185.70)
  a7 injection construction self-check < 1e-9 (9.9e-14)
  a8 same-session repeat determinism < 1e-6   (2.8e-14)
  a9 S17 ref curve vs 2945 D1_sep[L17] at shared
     s in {0.5..2.0} < 1.0 abs (steep monotone layer,
     cross-session stable zone; L16 ref kept descriptive)

Main test (frozen):
  T1 joint threshold rule: for each of J75/J50/J25,
     s_c = first s with sep_med < 100 (linear interpolation);
     predicted_local = min(0.656/alpha17, 1.843/alpha16),
     predicted_avg   = alpha17*0.656 + alpha16*1.843
     (both frozen from 2945 T2 before any 2946 observation).
     pass_local  iff max_c |s_c - pred_local|  < 0.35
     pass_avg    iff max_c |s_c - pred_avg|    < 0.35

Verdict (frozen):
  anchor fail            => anchor_fail_all_void
  pass_local             => switch_local_concentration
  (pass_avg, not local)  => switch_avg_concentration
  neither                => switch_interaction_nonlinear
  (if both pass, verdict switch_local_concentration with
   note that avg is also within tolerance)

Descriptive: D1 sep curves per config; D2 rho(s); D3
ratio(s); D4 measured vs predicted table; D5 S16 ref curve
vs 2945 D1_sep[L16] (cross-session check, no gate).

Output: phase2946/dose_allocation/.
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
SRC_2945 = os.path.join(BASE, 'phase2945', 'threshold_curves',
                        'result.json')
OUT = os.path.join(BASE, 'phase2946', 'dose_allocation')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2946_run_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
SEED = 2896
NH, HD = 32, 128
NL = 36
VOCAB = 151936
LP, LQ = 17, 16
S_IDX = (0, 1, 4)
S_GRID = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
K_REPEAT = 3
SEP_THRESHOLD = 100.0
SC17_REF = 0.656     # frozen from 2945 T2
SC16_REF = 1.843     # frozen from 2945 T2
TOL = 0.35
A9_TOL = 1.0
CONFIGS = {'J75': {LP: 0.75, LQ: 0.25},
           'J50': {LP: 0.5, LQ: 0.5},
           'J25': {LP: 0.25, LQ: 0.75},
           'S17': {LP: 1.0},
           'S16': {LQ: 1.0}}

PREREG = {
    'mode': 'forward family: joint two-layer injection of '
            'xdir (2942/2945 verbatim) at (L17,L16) with '
            'allocations J75/J50/J25 + single-layer refs '
            'S17/S16, s grid {0.5..3.0}, K=3 same-session '
            'repeats, median readouts',
    'question': 'does the joint (L17,L16) threshold follow '
                'the local-concentration rule '
                'min(s_c17/a17, s_c16/a16) or the averaged '
                'rule a17*s_c17 + a16*s_c16?',
    'frozen_predictions': {
        's_c17': SC17_REF, 's_c16': SC16_REF,
        'pred_local': {'J75': round(min(SC17_REF / 0.75,
                                        SC16_REF / 0.25), 4),
                       'J50': round(min(SC17_REF / 0.5,
                                        SC16_REF / 0.5), 4),
                       'J25': round(min(SC17_REF / 0.25,
                                        SC16_REF / 0.75), 4)},
        'pred_avg': {'J75': round(0.75 * SC17_REF
                                  + 0.25 * SC16_REF, 4),
                     'J50': round(0.5 * SC17_REF
                                  + 0.5 * SC16_REF, 4),
                     'J25': round(0.25 * SC17_REF
                                  + 0.75 * SC16_REF, 4)}},
    'anchors': {
        'a1': 'dirs_word rebuild vs 2927 npz < 1e-5',
        'a2': 'func baseline determinism < 1e-4',
        'a3': 'Vt8 rebuild vs 2939 npz < 1e-6',
        'a4': 'proj_func vs 2935 s_base[func] < 1e-4',
        'a5': 'proj_null0 vs 2935 s_base[null0] < 1e-4',
        'a6': 'func separation > 0',
        'a7': 'injection construction self-check < 1e-9',
        'a8': 'same-session repeat determinism < 1e-6',
        'a9': 'S17 ref curve vs 2945 D1_sep[L17] shared s '
              '< 1.0 abs',
    },
    'T1': 'joint s_c (first s with sep_med<100, interp) '
          'within 0.35 of pred_local for all 3 allocs '
          '(=> local); else within 0.35 of pred_avg '
          '(=> avg); else nonlinear',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'T1 local => switch_local_concentration; '
               'T1 avg only => switch_avg_concentration; '
               'else => switch_interaction_nonlinear',
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


def sc_interp(seps_by_s, grid, thr):
    """First s with sep < thr, linear interpolation.

    seps_by_s uses '%.2f' string keys (lesson 18: keep one
    key convention end to end).
    """
    seps = [(s, seps_by_s['%.2f' % s]) for s in grid]
    cross = [i for i, (s, sp) in enumerate(seps)
             if sp < thr]
    if not cross:
        return None
    i0 = cross[0]
    if i0 == 0:
        return float(grid[0])
    s0, sp0 = seps[i0 - 1]
    s1, sp1 = seps[i0]
    w = (sp0 - thr) / max(sp0 - sp1, 1e-30)
    return float(s0 + w * (s1 - s0))


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2946,
                   'name': 'dose_allocation',
                   'created':
                       time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2887': sha8(SRC_2887),
                               's2927': sha8(SRC_2927),
                               's2935': sha8(SRC_2935),
                               's2939': sha8(SRC_2939),
                               's2945': sha8(SRC_2945)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'seed': SEED,
                   'pair': [LP, LQ],
                   'configs': {k: {str(a): c for a, c
                                   in v.items()}
                               for k, v in
                               CONFIGS.items()},
                   's_grid': list(S_GRID),
                   'k_repeat': K_REPEAT,
                   'sep_threshold': SEP_THRESHOLD,
                   'sc17_ref': SC17_REF, 'sc16_ref': SC16_REF,
                   'tol': TOL, 'a9_tol': A9_TOL,
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

    r45 = json.load(open(SRC_2945, encoding='utf-8'))
    sep45 = r45['D1_sep']
    log('sources ok (2945 ref loaded)', lines)

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
    t1 = d1 = d2 = d3 = d4 = d5 = None
    save = {}
    a9_diff = None
    a9_ok = False

    if not anchor_prelim:
        verdict = 'anchor_fail_all_void'
    else:
        sep_curves = {}
        rho_curves = {}
        ratio_curves = {}
        spreads = {}
        store_proj = {}
        for cn, coef in CONFIGS.items():
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
                key = '%s|%.2f' % (cn, s)
                spreads[key] = \
                    float('%.2e' % float(
                        np.abs(P - P.mean(0)).max()))
                p_med = np.median(P, axis=0)
                store_proj[key] = p_med
                sep_curves[key] = float(
                    p_med[lab_lang == 0].mean()
                    - p_med[lab_lang == 1].mean())
                rho_curves[key] = spearman(
                    p_med, proj_f0)
                ratio_curves[key] = \
                    float(np.median(ratios)) / max(med_dS,
                                                   1e-30)
            log('%s sep: %s' % (cn, {
                ('%.2f' % s):
                    round(sep_curves['%s|%.2f' % (cn, s)], 1)
                for s in S_GRID}), lines)

        a8_diff = max(float(v) for v in spreads.values())
        a8_ok = bool(a8_diff < 1e-6)
        log('a8 same-session determinism max spread %.2e '
            'ok=%s' % (a8_diff, a8_ok), lines)

        # a9: S17 ref vs 2945 D1_sep[L17] (shared grid)
        a9_diffs = []
        for s in S_GRID:
            if s > 2.0 + 1e-9:
                continue
            k45 = '%.2f' % s
            ref = sep45['L17'][k45]
            a9_diffs.append(abs(
                sep_curves['S17|%.2f' % s] - ref))
        a9_diff = float(max(a9_diffs))
        a9_ok = bool(a9_diff < A9_TOL)
        log('a9 S17 ref vs 2945 L17 max abs diff %.4f '
            'ok=%s' % (a9_diff, a9_ok), lines)

        if not (anchor_prelim and a8_ok and a9_ok):
            verdict = 'anchor_fail_all_void'
        else:
            # ---------- T1: threshold rule ----------
            t1_rows = {}
            meas_sc = {}
            for cn in ('J75', 'J50', 'J25'):
                seps = {'%.2f' % s:
                        sep_curves['%s|%.2f' % (cn, s)]
                        for s in S_GRID}
                s_c = sc_interp(seps, S_GRID, SEP_THRESHOLD)
                a17 = CONFIGS[cn][LP]
                a16 = CONFIGS[cn][LQ]
                pl = min(SC17_REF / a17, SC16_REF / a16)
                pa = a17 * SC17_REF + a16 * SC16_REF
                meas_sc[cn] = s_c
                t1_rows[cn] = {
                    's_c': round(s_c, 4),
                    'pred_local': round(pl, 4),
                    'pred_avg': round(pa, 4),
                    'err_local': round(abs(s_c - pl), 4),
                    'err_avg': round(abs(s_c - pa), 4),
                    'alpha17': a17}
            pass_local = bool(all(
                r['err_local'] < TOL
                for r in t1_rows.values()))
            pass_avg = bool(all(
                r['err_avg'] < TOL
                for r in t1_rows.values()))
            t1 = {'rows': t1_rows, 'tol': TOL,
                  'pass_local': pass_local,
                  'pass_avg': pass_avg}
            log('T1 %s local=%s avg=%s'
                % (t1_rows, pass_local, pass_avg), lines)

            if pass_local:
                verdict = 'switch_local_concentration'
            elif pass_avg:
                verdict = 'switch_avg_concentration'
            else:
                verdict = 'switch_interaction_nonlinear'

            d1 = {cn: {'%.2f' % s:
                       round(sep_curves['%s|%.2f' % (cn, s)], 2)
                       for s in S_GRID}
                  for cn in CONFIGS}
            d2 = {cn: {'%.2f' % s:
                       round(rho_curves['%s|%.2f' % (cn, s)], 4)
                       for s in S_GRID}
                  for cn in CONFIGS}
            d3 = {cn: {'%.2f' % s:
                       round(ratio_curves['%s|%.2f' % (cn, s)],
                             4)
                       for s in S_GRID}
                  for cn in CONFIGS}
            d4 = {'measured_vs_predicted': t1_rows,
                  'note': 'H_local discriminating point is '
                          'J25 (2.4573 vs 1.5463)'}
            d5 = {'S16_ref':
                  {'%.2f' % s:
                   round(sep_curves['S16|%.2f' % s], 2)
                   for s in S_GRID},
                  'ref_2945_L16': sep45['L16'],
                  'note': 'cross-session comparison, '
                          'descriptive only (mid-gain '
                          'zone known unstable)'}
            d5['max_abs_diff'] = round(float(max(
                abs(sep_curves['S16|%.2f' % s]
                    - sep45['L16']['%.2f' % s])
                for s in S_GRID if s <= 2.0 + 1e-9)), 2)

            save = {
                'words': np.array(['%s:%s:%s' % w
                                   for w in words],
                                  dtype=object),
                'labels_lang': lab_lang,
                's_grid': np.array(S_GRID),
                'configs': np.array(list(CONFIGS)),
                'proj_med': np.stack(
                    [store_proj['%s|%.2f' % (cn, s)]
                     for cn in CONFIGS for s in S_GRID]),
                'proj_func0': proj_f0,
                'proj_null0': proj_n0,
                'sep_curves': np.array(
                    [sep_curves['%s|%.2f' % (cn, s)]
                     for cn in CONFIGS for s in S_GRID]),
                'Vt8': Vt8, 'dirs_word': dirs_word,
            }

    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2946, 'model': 'qwen3-4b',
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
                       'a9_diff': None if a9_diff is None
                       else round(a9_diff, 4),
                       'a9_ok': a9_ok,
                       'ok': bool(anchor_prelim and a8_ok
                                  and a9_ok)},
           'T1': t1,
           'D1_sep': d1, 'D2_rho': d2, 'D3_ratio': d3,
           'D4_rule': d4, 'D5_s16_ref': d5,
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if save:
        np.savez_compressed(os.path.join(
            OUT, 'dose_allocation.npz'), **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2946 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
