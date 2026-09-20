# -*- coding: utf-8 -*-
"""Phase 2953: A11 s-response curves (sigmoid fit vs sep threshold).

Why: 2952 localized the head-level micro-carrier of the
concentration switch to an attention-routing jump (A11 x7-10).
2945 measured per-layer sep thresholds s_c (L17 0.656 /
L16 1.843).  Open question: is A11(s) a sigmoid jump, and does
its transition midpoint s_t coincide with the sep threshold
s_c (micro-replay of the 2945 threshold at the routing level)?

Mode: ONE model (qwen3-4b), forward family.  Single-layer xdir
injection (2942/2945/2952 verbatim, coef 1.0) at L17 / L16;
fine s grids densified near each layer's s_c:
  GRID17 = 0.25 0.375 0.5 0.625 0.75 0.875 1.0 1.25 1.5 2.0
  GRID16 = 0.75 1.0 1.25 1.5 1.625 1.75 1.8125 1.875 1.9375 2.0
plus the s=0 base point (11 curve points per layer).
K=3 same-session repeats per point: sep = median (2945 mode);
A11 from the first repeat (determinism anchor a8 covers the
rest).  Captures per forward: v_proj out pos0/pos1, o_proj in
pos1 (A11 recovery, 2952 recover verbatim), final-norm input
(sep, 2945 verbatim).

Anchors (frozen):
  a1 dirs_word rebuild vs 2927 < 1e-5
  a2 func baseline determinism rel < 1e-4
  a3 Vt8 rebuild vs 2939 < 1e-6
  a4 proj_func vs 2935 s_base[func] < 1e-4
  a5 proj_null0 vs 2935 s_base[null0] < 1e-4
  a6 func separation > 0
  a7 injection construction self-check < 1e-9
  a8 same-session repeat spread < 1e-6
  a9 GQA gates (v_proj out 1024, o_proj in 4096)
  a10 base A11 vs 2952 A11b (keep heads) < 1e-6, both layers
  a11 shared-point A11 vs 2952 A11n (L17 s=1.0, L16 s=2.0,
      keep heads) < 1e-6
  a12 A11 line-recovery residual < 0.3 (all s, both layers)
  a13 shared-point sep vs 2945 sep_curves < 0.05 (all shared
      s points, both layers)

Main tests (frozen; reachability pre-check per discipline 10:
A11 in [0,1] bounded, monotone rise expected from 2952
endpoints, so the logistic is identifiable on the grid; k-gate
added because a near-linear ramp also fits a logistic with
tiny k and unidentifiable s_t):
  T1 sigmoid structure: logistic fit of median-keep-head
     A11(s) (11 points incl s=0) has R2 >= 0.9 AND k >= 2.0
     for BOTH layers.
  T2 transition-threshold lock: |s_t - s_c| < 0.3 for BOTH
     layers, where s_c is recomputed in-session from the
     median sep curve by the 2945 rule (first crossing of
     100, linear interpolation between grid neighbors).
  T3 steepness dissociation: k_L17 > k_L16 (switch-type
     layer transitions more sharply), consistent with 2945's
     switch-type vs gradual-type split.

Verdict (frozen):
  anchor fail                    => anchor_fail_all_void
  T1 & T2 & T3                   => a11_sigmoid_threshold_locked
  T1 & T2 & ~T3                  => a11_sigmoid_threshold_coincident
  T1 & ~T2                       => a11_sigmoid_threshold_decoupled
  ~T1                            => a11_curve_not_sigmoid

Descriptive: D1 per-layer A11(s) median curves + sep curves;
D2 per-keep-head fit params (k, s_t, R2); D3 fit vs 2945
threshold comparison table.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import curve_fit

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
                        'threshold_curves.npz')
SRC_2951 = os.path.join(BASE, 'phase2951',
                        'rebalance_carrier_functional',
                        'rebalance_carrier_functional.npz')
SRC_2952 = os.path.join(BASE, 'phase2952', 'amplification_anatomy',
                        'amplification_anatomy.npz')
OUT = os.path.join(BASE, 'phase2953', 'a11_s_response')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2953_run_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NH, HD = 32, 128
NL = 36
VOCAB = 151936
SEED = 2896
LAYERS = (17, 16)
GRID17 = (0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.25,
          1.5, 2.0)
GRID16 = (0.75, 1.0, 1.25, 1.5, 1.625, 1.75, 1.8125, 1.875,
          1.9375, 2.0)
GRID = {17: GRID17, 16: GRID16}
S_IDX = (0, 1, 4)
K_REPEAT = 3
SEP_THRESHOLD = 100.0
R2_MIN = 0.9
K_MIN = 2.0
LOCK_TOL = 0.3
A13_TOL = 0.05

PREREG = {
    'mode': 'forward family: single-layer xdir injection at '
            'L17/L16 coef 1.0, fine s grids densified near '
            '2945 s_c, K=3 same-session repeats; captures '
            'v_proj pos0/pos1, o_proj in pos1, final-norm '
            'input; A11 recovered by line least squares '
            '(2952 verbatim), sep via final-norm @ u35 '
            '(2945 verbatim)',
    'question': 'is the attention-routing jump A11(s) a '
                'sigmoid, and does its transition midpoint '
                's_t coincide with the 2945 sep threshold '
                's_c?',
    'anchors': {
        'a1': 'dirs_word rebuild vs 2927 < 1e-5',
        'a2': 'func baseline determinism rel < 1e-4',
        'a3': 'Vt8 rebuild vs 2939 < 1e-6',
        'a4': 'proj_func vs 2935 < 1e-4',
        'a5': 'proj_null0 vs 2935 < 1e-4',
        'a6': 'func separation > 0',
        'a7': 'injection construction self-check < 1e-9',
        'a8': 'same-session repeat spread < 1e-6',
        'a9': 'GQA gates',
        'a10': 'base A11 vs 2952 A11b keep heads < 1e-6',
        'a11': 'shared-point A11 vs 2952 A11n keep heads '
               '< 1e-6 (L17@1.0, L16@2.0)',
        'a12': 'line-recovery residual < 0.3 all s',
        'a13': 'shared-point sep vs 2945 sep_curves < 0.05',
    },
    'T1': 'logistic fit R2 >= 0.9 AND k >= 2.0, median '
          'keep-head A11(s), both layers (k-gate per '
          'discipline 10 reachability: tiny-k logistic '
          'degenerates to unidentifiable linear ramp)',
    'T2': '|s_t - s_c| < 0.3 both; s_c recomputed '
          'in-session, 2945 rule (first crossing of 100, '
          'linear interpolation)',
    'T3': 'k_L17 > k_L16 (switch-type steeper)',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'T1&T2&T3 => a11_sigmoid_threshold_locked; '
               'T1&T2 => a11_sigmoid_threshold_coincident; '
               'T1 => a11_sigmoid_threshold_decoupled; '
               'else => a11_curve_not_sigmoid',
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


def skey(li, s):
    return 'L%d|%.4f' % (li, float(s))


def logistic(s, a_lo, a_hi, k, s_t):
    return a_lo + (a_hi - a_lo) / (1.0 + np.exp(-k * (s - s_t)))


def fit_logistic(s_grid, y, s_guess):
    best = None
    for k0 in (2.0, 5.0, 15.0):
        try:
            p, _ = curve_fit(
                logistic, s_grid, y,
                p0=[max(float(y.min()) * 0.5, 1e-3),
                    float(y.max()) * 1.1, k0, s_guess],
                bounds=([0.0, 0.05, 0.1, 0.0],
                        [0.2, 1.0, 100.0,
                         float(s_grid.max()) + 0.5]),
                maxfev=20000)
        except Exception:
            continue
        sse = float(((logistic(s_grid, *p) - y) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, p)
    if best is None:
        return None
    sse, p = best
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - sse / max(sst, 1e-30)
    return {'a_lo': float(p[0]), 'a_hi': float(p[1]),
            'k': float(p[2]), 's_t': float(p[3]),
            'r2': float(r2)}


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2953,
                   'name': 'a11_s_response',
                   'created':
                       time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2887': sha8(SRC_2887),
                               's2927': sha8(SRC_2927),
                               's2935': sha8(SRC_2935),
                               's2939': sha8(SRC_2939),
                               's2945': sha8(SRC_2945),
                               's2951': sha8(SRC_2951),
                               's2952': sha8(SRC_2952)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'seed': SEED, 'layers': list(LAYERS),
                   'grid17': list(GRID17),
                   'grid16': list(GRID16),
                   's_idx': list(S_IDX),
                   'k_repeat': K_REPEAT,
                   'sep_threshold': SEP_THRESHOLD,
                   'r2_min': R2_MIN, 'k_min': K_MIN,
                   'lock_tol': LOCK_TOL,
                   'a13_tol': A13_TOL,
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

    z45 = np.load(SRC_2945, allow_pickle=True)
    sep45 = z45['sep_curves'].astype(np.float64)
    grid45 = z45['s_grid'].astype(np.float64)
    layers45 = [int(x) for x in z45['layers']]

    z51 = np.load(SRC_2951, allow_pickle=True)
    z52 = np.load(SRC_2952, allow_pickle=True)
    log('sources ok', lines)

    # ---------- model ----------
    import torch
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
            assert len(ids) == 1
            tc[t] = int(ids[0])
        return tc[t]

    tid_map = {}
    for lang, ck, w in words:
        tid_map[w] = tid(w)
    func_tid = tid('the')
    batch = [[func_tid, tid_map[words[i][2]]]
             for i in range(n_words)]

    model, _ = load_native('qwen4')
    model.eval()
    layers = model.model.layers
    log('model loaded', lines)

    a9_ok = bool(
        layers[0].self_attn.v_proj.weight.shape[0] == 1024
        and layers[0].self_attn.o_proj.in_features == NH * HD)
    log('a9 GQA gates ok=%s' % a9_ok, lines)

    cap_v = {}
    cap_x = {}
    cap_in = {'on': False, 'store': {}}
    fin_cap = {}
    state_fin = {'on': False}
    inj = {'li': None, 'scale': 0.0, 'vec': None}
    handles = []

    def pre_attn(li):
        def h(module, args, kwargs):
            x = args[0] if args else kwargs.get('hidden_states')
            if x is None or x.dim() < 2:
                return
            if inj['li'] == li and inj['vec'] is not None:
                x = x.clone()
                x[:, 1, :] = x[:, 1, :] \
                    + inj['scale'] * inj['vec']
                if args:
                    return (x,) + tuple(args[1:]), kwargs
                nkw = dict(kwargs)
                nkw['hidden_states'] = x
                return args, nkw
            if cap_in['on']:
                cap_in['store'].setdefault(li, []).append(
                    x[:, 1, :].detach().float()
                    .cpu().numpy())
            return None
        return h

    def hook_v(li):
        def h(module, args, output):
            if li in LAYERS:
                o = output.detach().float().cpu().numpy()
                cap_v.setdefault(li, []).append(
                    (o[:, 0, :].copy(), o[:, 1, :].copy()))
            return None
        return h

    def hook_x(li):
        def h(module, args, kwargs):
            x = args[0] if args else kwargs.get('input')
            if x is None or x.dim() < 2:
                return None
            if li in LAYERS:
                cap_x.setdefault(li, []).append(
                    x[:, 1, :].detach().float()
                    .cpu().numpy())
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
    for li in LAYERS:
        handles.append(layers[li].self_attn.v_proj
                       .register_forward_hook(hook_v(li)))
        handles.append(layers[li].self_attn.o_proj
                       .register_forward_pre_hook(
                           hook_x(li), with_kwargs=True))
    handles.append(model.model.norm.register_forward_pre_hook(
        pre_norm, with_kwargs=True))

    # ---------- pass 1: dirs_word rebuild (a1/a3) ----------
    attn_store = {}
    cap_in['on'] = True
    for i, (_, _, w) in enumerate(words):
        cap_in['store'].clear()
        with torch.no_grad():
            model(torch.tensor([[func_tid,
                                 tid_map[words[i][2]]]],
                               device='cuda'))
        for li in range(NL):
            attn_store[(i, li)] = \
                cap_in['store'][li][0].astype(np.float32)
        if (i + 1) % 20 == 0:
            log('pass1 [%d/%d]' % (i + 1, n_words), lines)
    cap_in['on'] = False

    d_dim = attn_store[(0, 0)].shape[-1]
    diffs_w = np.zeros((NL, d_dim))
    for li in range(NL):
        X = np.stack([attn_store[(i, li)]
                      for i in range(n_words)]) \
            .astype(np.float64)
        diffs_w[li] = X[lab_lang == 0].mean(0) \
            - X[lab_lang == 1].mean(0)
    dirs_word = np.stack([unit(diffs_w[li]) for li in range(NL)])
    a1_diff = float(np.abs(dirs_word - dirs_word_27).max())
    a1_ok = bool(a1_diff < 1e-5)
    log('a1 dirs rebuild diff %.2e ok=%s' % (a1_diff, a1_ok),
        lines)
    _, _, Vt = np.linalg.svd(dirs_word, full_matrices=False)
    Vt8 = Vt[:8]
    a3_diff = float(np.abs(Vt8 - Vt8_39).max())
    a3_ok = bool(a3_diff < 1e-6)
    log('a3 Vt8 vs 2939 %.2e ok=%s' % (a3_diff, a3_ok), lines)
    u35 = dirs_word[NL - 1]
    xdir = dcks_39[:, list(S_IDX)] @ Vt8[list(S_IDX)]
    a7_diff = float(np.abs(
        xdir @ Vt8[list(S_IDX)].T
        - dcks_39[:, list(S_IDX)]).max())
    a7_ok = bool(a7_diff < 1e-9)
    log('a7 xdir self-check %.2e ok=%s' % (a7_diff, a7_ok),
        lines)
    xdir_t = torch.tensor(xdir, device='cuda',
                          dtype=torch.bfloat16)

    # ---------- baselines (a2/a4/a5/a6) ----------
    def forward_batch(scale=0.0, inj_li=None):
        cap_v.clear()
        cap_x.clear()
        fin_cap.pop('x', None)
        inj['li'] = inj_li
        inj['scale'] = float(scale)
        inj['vec'] = xdir_t if scale else None
        state_fin['on'] = True
        with torch.no_grad():
            model(torch.tensor(batch, device='cuda'))
        inj['li'] = None
        inj['scale'] = 0.0
        inj['vec'] = None
        state_fin['on'] = False
        fin = fin_cap['x'].astype(np.float64)
        v = {li: (np.stack([a for a, b in cap_v[li]]),
                  np.stack([b for a, b in cap_v[li]]))
             for li in cap_v}
        x = {li: np.stack(cap_x[li])[0] for li in cap_x}
        return fin, v, x

    fin_f1, _, _ = forward_batch()
    fin_f2, _, _ = forward_batch()
    a2_rel = float(np.abs(fin_f1 - fin_f2).max()
                   / max(float(np.abs(fin_f1).max()), 1e-30))
    a2_ok = bool(a2_rel < 1e-4)
    log('a2 baseline determinism rel %.2e ok=%s'
        % (a2_rel, a2_ok), lines)

    proj_f0 = fin_f1 @ u35
    a4_diff = float(np.abs(proj_f0 - s_base_35[ifu35]).max())
    a4_ok = bool(a4_diff < 1e-4)
    log('a4 proj_func vs 2935 %.2e ok=%s'
        % (a4_diff, a4_ok), lines)
    sep_f = float(proj_f0[lab_lang == 0].mean()
                  - proj_f0[lab_lang == 1].mean())
    a6_ok = bool(sep_f > 0.0)
    log('a6 sep_func %.4f ok=%s' % (sep_f, a6_ok), lines)

    # null0 batch (a5, 2945 verbatim)
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
    batch_n0 = [[null0_tids[i], tid_map[words[i][2]]]
                for i in range(n_words)]
    fin_n0, _, _ = forward_batch()
    # re-run with null0 tokens: reuse forward_batch by
    # temporarily swapping the batch
    def forward_toks(toks_list, scale=0.0, inj_li=None):
        cap_v.clear()
        cap_x.clear()
        fin_cap.pop('x', None)
        inj['li'] = inj_li
        inj['scale'] = float(scale)
        inj['vec'] = xdir_t if scale else None
        state_fin['on'] = True
        with torch.no_grad():
            model(torch.tensor(toks_list, device='cuda'))
        inj['li'] = None
        inj['scale'] = 0.0
        inj['vec'] = None
        state_fin['on'] = False
        return fin_cap['x'].astype(np.float64)

    fin_n0 = forward_toks(batch_n0)
    proj_n0 = fin_n0 @ u35
    a5_diff = float(np.abs(proj_n0 - s_base_35[in035]).max())
    a5_ok = bool(a5_diff < 1e-4)
    log('a5 proj_null0 vs 2935 %.2e ok=%s'
        % (a5_diff, a5_ok), lines)

    # ---------- curve forwards ----------
    NKV = 1024 // HD
    HPG = NH // NKV

    def recover(Xf, v0r, v1r):
        A = np.zeros((n_words, NH))
        res = 0.0
        for hh in range(NH):
            k = hh // HPG
            d = v1r[:, k, :] - v0r[:, k, :]
            den_w = (d * d).sum(1)
            num = ((Xf[:, hh, :] - v0r[:, k, :]) * d).sum(1)
            A[:, hh] = num / np.maximum(den_w, 1e-30)
            rec = v0r[:, k, :] + A[:, hh:hh + 1] * d
            res = max(res, float(np.abs(
                rec - Xf[:, hh, :]).max()))
        return A, res

    # base captures for A11b (both injected layers)
    _, v_base, x_base = forward_batch()
    A11b = {}
    res_b_max = 0.0
    for li in LAYERS:
        v0b, v1b = v_base[li]
        A, r = recover(
            x_base[li].reshape(n_words, NH, HD),
            v0b.reshape(n_words, NKV, HD),
            v1b.reshape(n_words, NKV, HD))
        A11b[li] = A
        res_b_max = max(res_b_max, r)
    log('base A11 recovered (recon residual %.2e)'
        % res_b_max, lines)

    sep_cur = {}
    A11_cur = {}
    spreads = []
    for li in LAYERS:
        for s in GRID[li]:
            fins = []
            v0 = v1 = x1 = None
            for rep in range(K_REPEAT):
                fin, vv, xx = forward_batch(
                    scale=s, inj_li=li)
                fins.append(fin)
                if rep == 0:
                    v0, v1 = vv[li]
                    x1 = xx[li]
            P = np.stack(fins) @ u35
            spreads.append(float(np.abs(P - P.mean(0)).max()))
            p_med = np.median(P, axis=0)
            sep_cur[skey(li, s)] = float(
                p_med[lab_lang == 0].mean()
                - p_med[lab_lang == 1].mean())
            A, r = recover(
                x1.reshape(n_words, NH, HD),
                v0.reshape(n_words, NKV, HD),
                v1.reshape(n_words, NKV, HD))
            A11_cur[skey(li, s)] = A
            res_b_max = max(res_b_max, r)
        log('L%d sep: %s' % (li, {
            '%.4f' % s: round(sep_cur[skey(li, s)], 1)
            for s in GRID[li]}), lines)

    a8_diff = max(spreads) if spreads else 0.0
    a8_ok = bool(a8_diff < 1e-6)
    a12_ok = bool(res_b_max < 0.3)
    log('a8 repeat spread %.2e ok=%s | a12 recon %.2e ok=%s'
        % (a8_diff, a8_ok, res_b_max, a12_ok), lines)

    # ---------- cross-phase anchors a10/a11/a13 ----------
    keep = {17: z51['keep_L17'], 16: z51['keep_L16']}
    a10_diff = 0.0
    for li in LAYERS:
        a10_diff = max(a10_diff, float(np.abs(
            A11b[li][:, keep[li]]
            - z52['A11b_L%d' % li][:, keep[li]]).max()))
    a10_ok = bool(a10_diff < 1e-6)
    a11_diff = 0.0
    a11_diff = max(a11_diff, float(np.abs(
        A11_cur[skey(17, 1.0)][:, keep[17]]
        - z52['A11n_L17'][:, keep[17]]).max()))
    a11_diff = max(a11_diff, float(np.abs(
        A11_cur[skey(16, 2.0)][:, keep[16]]
        - z52['A11n_L16'][:, keep[16]]).max()))
    a11_ok = bool(a11_diff < 1e-6)
    a13_diff = 0.0
    a13_pts = {}
    for li in LAYERS:
        i45 = layers45.index(li)
        for s in GRID[li]:
            hit = [j for j, s45 in enumerate(grid45)
                   if abs(float(s45) - float(s)) < 1e-9]
            if not hit:
                continue
            d = abs(sep_cur[skey(li, s)]
                    - float(sep45[i45 * len(grid45) + hit[0]]))
            a13_pts['%s@%.4f' % ('L%d' % li, s)] = round(d, 5)
            a13_diff = max(a13_diff, d)
    a13_ok = bool(a13_diff < A13_TOL)
    log('a10 base A11 vs 2952 %.2e ok=%s | a11 shared A11 '
        'vs 2952 %.2e ok=%s | a13 shared sep vs 2945 %.2e '
        'ok=%s (%d pts)'
        % (a10_diff, a10_ok, a11_diff, a11_ok, a13_diff,
           a13_ok, len(a13_pts)), lines)

    anchor_prelim = bool(a1_ok and a2_ok and a3_ok and a4_ok
                         and a5_ok and a6_ok and a7_ok
                         and a8_ok and a9_ok and a10_ok
                         and a11_ok and a12_ok and a13_ok)

    verdict = None
    t1 = t2 = t3 = d1 = d2 = d3 = None
    save = {}
    if not anchor_prelim:
        verdict = 'anchor_fail_all_void'
    else:
        # ---------- in-session s_c (2945 rule) ----------
        s_c = {}
        for li in LAYERS:
            pts = sorted([(float(s), sep_cur[skey(li, s)])
                          for s in GRID[li]])
            cross = [i for i, (s, sp) in enumerate(pts)
                     if sp < SEP_THRESHOLD]
            if not cross:
                s_c[li] = None
                continue
            i0 = cross[0]
            if i0 == 0:
                s_c[li] = pts[0][0]
            else:
                s0, sp0 = pts[i0 - 1]
                s1_, sp1 = pts[i0]
                w = (sp0 - SEP_THRESHOLD) \
                    / max(sp0 - sp1, 1e-30)
                s_c[li] = s0 + w * (s1_ - s0)
        log('in-session s_c: %s'
            % {'L%d' % li: (None if s_c[li] is None
                            else round(float(s_c[li]), 4))
               for li in LAYERS}, lines)

        # ---------- T1/T2/T3 ----------
        t1 = {}
        t2 = {}
        t3 = {}
        d1 = {}
        d2 = {}
        fits = {}
        for li in LAYERS:
            key = 'L%d' % li
            s_grid = np.array([0.0] + [float(s)
                                       for s in GRID[li]])
            curve_med = np.array(
                [float(np.median(A11b[li][:, keep[li]]))]
                + [float(np.median(
                    A11_cur[skey(li, s)][:, keep[li]]))
                    for s in GRID[li]])
            guess = float(s_c[li]) if s_c[li] is not None \
                else float(np.median(s_grid))
            fit = fit_logistic(s_grid, curve_med, guess)
            fits[key] = fit
            if fit is None:
                t1[key] = {'pass': False, 'fit_error': True}
                t2[key] = {'pass': False, 'fit_error': True}
                t3[key] = {'k': None, 'pass': False}
                continue
            t1_pass = bool(fit['r2'] >= R2_MIN
                           and fit['k'] >= K_MIN)
            st = fit['s_t']
            t2_pass = bool(s_c[li] is not None
                           and abs(st - float(s_c[li]))
                           < LOCK_TOL)
            t1[key] = {'r2': round(fit['r2'], 4),
                       'k': round(fit['k'], 3),
                       's_t': round(st, 4),
                       'a_lo': round(fit['a_lo'], 4),
                       'a_hi': round(fit['a_hi'], 4),
                       'pass': t1_pass}
            t2[key] = {'s_t': round(st, 4),
                       's_c_insession': (None if s_c[li]
                                         is None else
                                         round(float(s_c[li]),
                                               4)),
                       'abs_diff': (None if s_c[li] is None
                                    else round(abs(st
                                                   - float(s_c[li])),
                                               4)),
                       'tol': LOCK_TOL, 'pass': t2_pass}
            d1[key] = {
                'a11_curve': ['%.4f' % v for v in curve_med],
                'sep_curve': ['%.1f' % sep_cur[skey(li, s)]
                              for s in GRID[li]]}
            # per-head fits (descriptive D2)
            rows = []
            for hh in keep[li]:
                yh = np.array(
                    [float(np.median(A11b[li][:, hh]))]
                    + [float(np.median(
                        A11_cur[skey(li, s)][:, hh]))
                        for s in GRID[li]])
                fh = fit_logistic(s_grid, yh, guess)
                if fh is None:
                    rows.append({'head': int(hh),
                                 'fit_error': True})
                    continue
                rows.append({
                    'head': int(hh),
                    'k': round(fh['k'], 2),
                    's_t': round(fh['s_t'], 4),
                    'r2': round(fh['r2'], 4),
                    'lock': bool(abs(fh['s_t']
                                     - float(s_c[li]))
                                 < LOCK_TOL)})
            d2[key] = rows
            log('%s: fit k=%.3f s_t=%.4f r2=%.4f (T1 %s) | '
                's_c=%.4f |s_t-s_c|=%.4f (T2 %s)'
                % (key, fit['k'], st, fit['r2'], t1_pass,
                   float(s_c[li]) if s_c[li] is not None
                   else float('nan'),
                   abs(st - float(s_c[li]))
                   if s_c[li] is not None else float('nan'),
                   t2_pass), lines)

        k17 = fits.get('L17')
        k16 = fits.get('L16')
        t3_pass = bool(k17 is not None and k16 is not None
                       and k17['k'] > k16['k'])
        t3 = {'k_L17': (None if k17 is None
                        else round(k17['k'], 3)),
              'k_L16': (None if k16 is None
                        else round(k16['k'], 3)),
              'pass': t3_pass}
        log('T3 k_L17=%s vs k_L16=%s pass=%s'
            % (t3['k_L17'], t3['k_L16'], t3_pass), lines)

        t1_all = all(v.get('pass') for v in t1.values())
        t2_all = all(v.get('pass') for v in t2.values())
        if not t1_all:
            verdict = 'a11_curve_not_sigmoid'
        elif t2_all and t3_pass:
            verdict = 'a11_sigmoid_threshold_locked'
        elif t2_all:
            verdict = 'a11_sigmoid_threshold_coincident'
        else:
            verdict = 'a11_sigmoid_threshold_decoupled'

        d3 = {'L17': {'s_c_2945': 0.656,
                      's_c_insession':
                          round(float(s_c[17]), 4),
                      's_t': round(fits['L17']['s_t'], 4)
                      if fits['L17'] else None},
              'L16': {'s_c_2945': 1.843,
                      's_c_insession':
                          round(float(s_c[16]), 4),
                      's_t': round(fits['L16']['s_t'], 4)
                      if fits['L16'] else None}}

        save = {
            'words': np.array(['%s:%s:%s' % w
                               for w in words],
                              dtype=object),
            'labels_lang': lab_lang,
            'keep_L17': keep[17], 'keep_L16': keep[16],
            'grid17': np.array(GRID17),
            'grid16': np.array(GRID16),
            'A11b_L17': A11b[17], 'A11b_L16': A11b[16],
        }
        for li in LAYERS:
            for s in GRID[li]:
                save['A11_%s' % skey(li, s)] = \
                    A11_cur[skey(li, s)]
        save['sep_curves'] = np.array(
            [[sep_cur[skey(li, s)] for s in GRID[li]]
             for li in LAYERS])

    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2953, 'model': 'qwen3-4b',
           'prereg': PREREG,
           'anchors': {
               'a1_diff': float('%.3e' % a1_diff),
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
               'a8_diff': float('%.3e' % a8_diff),
               'a8_ok': a8_ok,
               'a9_ok': a9_ok,
               'a10_diff': float('%.3e' % a10_diff),
               'a10_ok': a10_ok,
               'a11_diff': float('%.3e' % a11_diff),
               'a11_ok': a11_ok,
               'a12_diff': float('%.3e' % res_b_max),
               'a12_ok': a12_ok,
               'a13_diff': float('%.3e' % a13_diff),
               'a13_ok': a13_ok,
               'a13_points': a13_pts,
               'ok': bool(anchor_prelim)},
           'T1_sigmoid_fit': t1, 'T2_threshold_lock': t2,
           'T3_steepness': t3,
           'D1_curves': d1, 'D2_per_head': d2,
           'D3_vs_2945': d3,
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if save:
        np.savez_compressed(os.path.join(
            OUT, 'a11_s_response.npz'), **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2953 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
