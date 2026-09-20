# -*- coding: utf-8 -*-
"""Phase 2966: is L34/h15 a member of the A11 routing
network? (downstream propagation test)

Why: 2964 localized the function-vs-content class carrier
to L34/h15 (unique maxT head); 2965 showed h15 is the
largest single-head causal contributor but the class
effect survives ablation (shared carrier). Open question:
is the L34 class readout DOWNSTREAM of the L17 attention
routing switch (2952/2953), i.e. does injecting the
language direction xdir at L17 move h15's contribution at
L34 with a threshold locked to the L17 switch s_c? If yes,
the class carrier is part of the routing network; if no,
it is an independent readout layer.

Mode: forward family, 57-word language batch VERBATIM
(2887 list / 2953 protocol). Injection at L17 coef 1.0,
s grid = 2953 GRID17 + s=0 (11 points), K=1 (determinism
anchor covers repeats; 2953 a8 spread < 1e-6). Captures:
self_attn input all layers (pass1 dirs rebuild), v_proj
pos0/pos1 at L17+L34, o_proj input pos1 all layers,
final-norm input.

Anchors (frozen):
  a1 dirs_word rebuild vs 2927 < 1e-5
  a2 baseline determinism rel < 1e-4
  a3 Vt8 rebuild vs 2939 < 1e-6
  a4 proj_func vs 2935 < 1e-4
  a5 proj_null0 vs 2935 < 1e-4
  a6 func separation > 0
  a7 xdir construction self-check < 1e-9
  a8 GQA gates (v_proj out 1024, o_proj in 4096)
  a9 A11_L17 all 11 s points vs 2953 npz < 1e-6
     (batch composition bit-identical: same 57 batch,
     same K=1 first repeat 2953 used)
  a10 line-recovery residual L17 < 0.3
  a11 sep shared grid points vs 2953 npz L17 row < 0.05
  NOTE: L34 A11 recovery residual is registered as a
  DESCRIPTIVE quality metric (not an anchor) - a recovery
  failure at L34 must not void the C-based tests
  (anchor-reachability discipline 10).

Tests (frozen):
  T1 (targeted confirmation of the 2964-selected head =>
      quasi-post-hoc, discipline 9): spearman of the
      median-word C[34,h15](s) curve over the 11 s points,
      two-sided label permutation (rng 2973, 10000);
      gate p <= 0.01; effect size |C15(2)-C15(0)| and
      rel-to-std reported (discipline 15). All-head family
      (32) registered descriptively with maxT note.
  T2 (only if T1 pass): free-asymptote logistic fit of the
      h15 curve (C is signed; 2953 bounds assumed [0,1]
      A11 and do NOT apply). Gates: R2 >= 0.9 AND k >= 2.0
      AND |s_t - s_c_L17_insession| < 0.3 => locked;
      R2/k pass but |s_t - s_c| >= 0.3 => rethresholded;
      else gradual. s_c recomputed in-session (2945 rule:
      first crossing of 100).
  T3 descriptive: B(s) = prof[6:13]-prof[28:36] dose
      response (spearman perm rng 2974); A11_L34 curves +
      recovery residual; sep curve.

Verdict (frozen):
  anchor fail => anchor_fail_all_void
  T1 fail => l34_h15_independent_of_routing
  T1 pass & T2 locked => l34_h15_routing_member_locked
  T1 pass & T2 rethresholded => l34_h15_downstream_rethresholded
  T1 pass & T2 gradual => l34_h15_gradual_responder
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
SRC_2953 = os.path.join(BASE, 'phase2953', 'a11_s_response',
                        'a11_s_response.npz')
OUT = os.path.join(BASE, 'phase2966', 'l34_routing_membership')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2966_run_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NH, HD = 32, 128
NL = 36
VOCAB = 151936
SEED = 2896
LI_INJ = 17
LI_TGT = 34
H_TGT = 15
GRID17 = (0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.25,
          1.5, 2.0)
S_IDX = (0, 1, 4)
N_PERM = 10000
RNG_T1, RNG_T3 = 2973, 2974
SEP_THRESHOLD = 100.0
R2_MIN = 0.9
K_MIN = 2.0
LOCK_TOL = 0.3
A9_TOL = 1e-6
A11_TOL = 0.05

PREREG = {
    'mode': 'forward family: L17 xdir injection coef 1.0, '
            '2953 GRID17 + s=0 (11 points), K=1, 57-word '
            'batch verbatim; captures self_attn input all '
            'layers (pass1), v_proj pos0/pos1 L17+L34, '
            'o_proj input pos1 all layers, final-norm input',
    'question': 'is the L34/h15 class-carrier contribution '
                'downstream of the L17 routing switch '
                '(threshold-locked dose response), or '
                'independent of it?',
    'anchors': {
        'a1': 'dirs_word rebuild vs 2927 < 1e-5',
        'a2': 'baseline determinism rel < 1e-4',
        'a3': 'Vt8 vs 2939 < 1e-6',
        'a4': 'proj_func vs 2935 < 1e-4',
        'a5': 'proj_null0 vs 2935 < 1e-4',
        'a6': 'sep_func > 0',
        'a7': 'xdir self-check < 1e-9',
        'a8': 'GQA gates',
        'a9': 'A11_L17 all 11 s vs 2953 npz < 1e-6',
        'a10': 'line-recovery residual L17 < 0.3',
        'a11': 'sep shared grid vs 2953 L17 row < 0.05',
        'note': 'L34 A11 recovery residual = descriptive '
                'only (anchor reachability, discipline 10)',
    },
    'T1': 'spearman(median-word C[34,h15](s) curve, s), '
          'two-sided perm rng 2973 x10000, gate p <= 0.01; '
          'quasi-post-hoc targeted (2964 selection, '
          'discipline 9); effect size reported',
    'T2': 'free-asymptote logistic fit (C signed): R2 >= 0.9 '
          '& k >= 2.0 & |s_t - s_c_insession| < 0.3 => '
          'locked; R2/k pass only => rethresholded; else '
          'gradual; s_c via 2945 rule (crossing 100)',
    'T3': 'descriptive: B(s) perm rng 2974; A11_L34 + '
          'residual; all-head C-response family (maxT note)',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'T1 fail => '
               'l34_h15_independent_of_routing; T1 pass & '
               'locked => l34_h15_routing_member_locked; '
               'T1 pass & rethresholded => '
               'l34_h15_downstream_rethresholded; T1 pass & '
               'gradual => l34_h15_gradual_responder',
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


def logistic2(s, a_lo, a_hi, k, s_t):
    return a_lo + (a_hi - a_lo) \
        / (1.0 + np.exp(-k * (s - s_t)))


def fit_logistic_free(s_grid, y, s_guess):
    best = None
    for k0 in (2.0, 5.0, 15.0):
        try:
            p, _ = curve_fit(
                logistic2, s_grid, y,
                p0=[float(y.min()), float(y.max()), k0,
                    s_guess],
                bounds=([-np.inf, -np.inf, 0.1, 0.0],
                        [np.inf, np.inf, 100.0,
                         float(s_grid.max()) + 0.5]),
                maxfev=20000)
        except Exception:
            continue
        sse = float(((logistic2(s_grid, *p) - y) ** 2).sum())
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
        json.dump({'phase': 2966,
                   'name': 'l34_routing_membership',
                   'created':
                       time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2887': sha8(SRC_2887),
                               's2927': sha8(SRC_2927),
                               's2935': sha8(SRC_2935),
                               's2939': sha8(SRC_2939),
                               's2953': sha8(SRC_2953)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'seed': SEED, 'inj_layer': LI_INJ,
                   'target_layer': LI_TGT,
                   'target_head': H_TGT,
                   'grid17': list(GRID17),
                   's_idx': list(S_IDX), 'k_repeat': 1,
                   'n_perm': N_PERM,
                   'rng': {'T1': RNG_T1, 'T3': RNG_T3},
                   'sep_threshold': SEP_THRESHOLD,
                   'r2_min': R2_MIN, 'k_min': K_MIN,
                   'lock_tol': LOCK_TOL,
                   'a9_tol': A9_TOL, 'a11_tol': A11_TOL,
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
    z53 = np.load(SRC_2953, allow_pickle=True)
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

    a8_ok = bool(
        layers[0].self_attn.v_proj.weight.shape[0] == 1024
        and layers[0].self_attn.o_proj.in_features == NH * HD)
    log('a8 GQA gates ok=%s' % a8_ok, lines)

    cap_in = {'on': False, 'store': {}}
    cap_v = {}
    cap_x = {}
    fin_cap = {}
    state_fin = {'on': False}
    inj = {'li': None, 'scale': 0.0, 'vec': None}
    handles = []

    V_HOOK_LAYERS = (LI_INJ, LI_TGT)

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
            if li in V_HOOK_LAYERS:
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
            cap_x.setdefault(li, []).append(
                x[:, 1, :].detach().float().cpu().numpy())
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
        handles.append(layers[li].self_attn.o_proj
                       .register_forward_pre_hook(
                           hook_x(li), with_kwargs=True))
    for li in V_HOOK_LAYERS:
        handles.append(layers[li].self_attn.v_proj
                       .register_forward_hook(hook_v(li)))
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

    M = np.zeros((NL, NH * HD))
    for li in range(NL):
        Wl = layers[li].self_attn.o_proj.weight.detach() \
            .float().cpu().numpy()
        M[li] = u35 @ Wl

    def forward_batch(toks_list, scale=0.0, inj_li=None):
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
        fin = fin_cap['x'].astype(np.float64)
        v = {li: (np.stack([a for a, b in cap_v[li]]),
                  np.stack([b for a, b in cap_v[li]]))
             for li in cap_v}
        x = {li: np.stack(cap_x[li])[0].astype(np.float64)
             for li in cap_x}
        return fin, v, x

    # ---------- baselines (a2/a4/a5/a6) ----------
    fin_f1, _, _ = forward_batch(batch)
    fin_f2, _, _ = forward_batch(batch)
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
    fin_n0, _, _ = forward_batch(batch_n0)
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

    _, v_base, x_base = forward_batch(batch)
    A11b = {}
    res17 = res34 = 0.0
    A, r = recover(
        x_base[LI_INJ].reshape(n_words, NH, HD),
        v_base[LI_INJ][0].reshape(n_words, NKV, HD),
        v_base[LI_INJ][1].reshape(n_words, NKV, HD))
    A11b[LI_INJ] = A
    res17 = r
    A, r = recover(
        x_base[LI_TGT].reshape(n_words, NH, HD),
        v_base[LI_TGT][0].reshape(n_words, NKV, HD),
        v_base[LI_TGT][1].reshape(n_words, NKV, HD))
    A11b[LI_TGT] = A
    res34 = r
    a10_ok = bool(res17 < 0.3)
    log('a10 L17 recovery residual %.2e ok=%s | L34 '
        'residual %.2e (descriptive)'
        % (res17, a10_ok, res34), lines)

    s_grid = np.array([0.0] + [float(s) for s in GRID17])
    sep_cur = {}
    A11_cur = {}
    C34_cur = {}
    B_cur = {}
    for s in s_grid:
        fin, vv, xx = forward_batch(
            batch, scale=float(s),
            inj_li=(LI_INJ if s > 0 else None))
        P = fin @ u35
        sep_cur[skey(LI_INJ, s)] = float(
            P[lab_lang == 0].mean() - P[lab_lang == 1].mean())
        if s > 0:
            A, r = recover(
                xx[LI_INJ].reshape(n_words, NH, HD),
                vv[LI_INJ][0].reshape(n_words, NKV, HD),
                vv[LI_INJ][1].reshape(n_words, NKV, HD))
            A11_cur[skey(LI_INJ, s)] = A
            res17 = max(res17, r)
        A, r = recover(
            xx[LI_TGT].reshape(n_words, NH, HD),
            vv[LI_TGT][0].reshape(n_words, NKV, HD),
            vv[LI_TGT][1].reshape(n_words, NKV, HD))
        A11_cur[skey(LI_TGT, s)] = A
        res34 = max(res34, r)
        C34 = np.zeros((n_words, NH))
        prof = np.zeros(NL)
        for li in range(NL):
            X = xx[li]
            xm = (X * M[li]).reshape(n_words, NH, HD)
            ch = xm.sum(axis=2)
            if li == LI_TGT:
                C34 = ch
            prof[li] = float(ch.sum(axis=1).mean())
        C34_cur[s] = C34
        B_cur[s] = float(prof[6:13].mean()
                         - prof[28:36].mean())
        log('s=%.4f sep=%.1f B=%.4f' % (s, sep_cur[
            skey(LI_INJ, s)], B_cur[s]), lines)
    a10_ok = bool(res17 < 0.3)
    log('a10 L17 recovery residual max %.2e ok=%s | L34 '
        'residual max %.2e (descriptive)'
        % (res17, a10_ok, res34), lines)

    # ---------- cross-phase anchors a9/a11 ----------
    a9_diff = float(np.abs(
        A11b[LI_INJ] - z53['A11b_L17']).max())
    for s in GRID17:
        a9_diff = max(a9_diff, float(np.abs(
            A11_cur[skey(LI_INJ, s)]
            - z53['A11_%s' % skey(LI_INJ, s)]).max()))
    a9_ok = bool(a9_diff < A9_TOL)
    log('a9 A11_L17 vs 2953 npz (all 11 s) %.2e ok=%s'
        % (a9_diff, a9_ok), lines)
    sep53 = z53['sep_curves'].astype(np.float64)
    a11_diff = 0.0
    a11_pts = {}
    for j, s in enumerate(GRID17):
        d = abs(sep_cur[skey(LI_INJ, s)] - float(sep53[0, j]))
        a11_pts['%.4f' % s] = round(d, 5)
        a11_diff = max(a11_diff, d)
    a11_ok = bool(a11_diff < A11_TOL)
    log('a11 sep shared grid vs 2953 L17 row %.2e ok=%s '
        '(%d pts)' % (a11_diff, a11_ok, len(a11_pts)), lines)

    anchor_ok = bool(a1_ok and a2_ok and a3_ok and a4_ok
                     and a5_ok and a6_ok and a7_ok and a8_ok
                     and a9_ok and a10_ok and a11_ok)
    verdict = None
    t1 = t2 = t3 = None
    save = {}
    s_c = None

    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        # in-session s_c (2945 rule)
        pts = sorted([(float(s), sep_cur[skey(LI_INJ, s)])
                      for s in s_grid])
        cross = [i for i, (sv, sp) in enumerate(pts)
                 if sp < SEP_THRESHOLD]
        if not cross:
            s_c = None
        else:
            i0 = cross[0]
            if i0 == 0:
                s_c = pts[0][0]
            else:
                s0, sp0 = pts[i0 - 1]
                s1_, sp1 = pts[i0]
                w = (sp0 - SEP_THRESHOLD) \
                    / max(sp0 - sp1, 1e-30)
                s_c = s0 + w * (s1_ - s0)
        log('in-session s_c(L17) = %s'
            % (None if s_c is None
               else round(float(s_c), 4)), lines)

        # ---------- T1 h15 C-dose response ----------
        curve15 = np.array([float(np.median(C34_cur[s][:, H_TGT]))
                            for s in s_grid])
        rho1 = spearman(s_grid, curve15)
        rng1 = np.random.default_rng(RNG_T1)
        cnt1 = 0
        for _ in range(N_PERM):
            if abs(spearman(rng1.permutation(s_grid),
                            curve15)) \
                    >= abs(rho1) - 1e-12:
                cnt1 += 1
        p1 = (cnt1 + 1) / (N_PERM + 1)
        t1_pass = bool(p1 <= 0.01)
        eff_abs = float(abs(curve15[-1] - curve15[0]))
        sd0 = float(C34_cur[0.0][:, H_TGT].std())
        t1 = {'rho': round(float(rho1), 4),
              'perm_p': float('%.3e' % p1),
              'pass': t1_pass,
              'effect_abs': round(eff_abs, 4),
              'effect_rel_std0': round(
                  eff_abs / max(sd0, 1e-30), 4),
              'note': 'quasi-post-hoc targeted '
                      '(discipline 9)'}
        log('T1 rho(s, C15) %.4f perm-p %.3e pass=%s | '
            'effect %.4f (rel std0 %.4f)'
            % (rho1, p1, t1_pass, eff_abs,
               eff_abs / max(sd0, 1e-30)), lines)

        # all-head family descriptive (maxT note)
        fam = []
        for h in range(NH):
            ch = np.array([float(np.median(C34_cur[s][:, h]))
                           for s in s_grid])
            fam.append(spearman(s_grid, ch))
        fam = np.array(fam)
        t1_fam_top5 = [
            (int(h), round(float(fam[h]), 4))
            for h in np.argsort(-np.abs(fam))[:5]]

        if t1_pass:
            # ---------- T2 logistic fit ----------
            guess = float(s_c) if s_c is not None \
                else float(np.median(s_grid))
            fit = fit_logistic_free(s_grid, curve15, guess)
            if fit is None:
                t2 = {'fit_error': True, 'pass': False}
                verdict = 'l34_h15_gradual_responder'
            else:
                lock = bool(s_c is not None
                            and abs(fit['s_t']
                                    - float(s_c)) < LOCK_TOL)
                shape = bool(fit['r2'] >= R2_MIN
                             and fit['k'] >= K_MIN)
                t2 = {'r2': round(fit['r2'], 4),
                      'k': round(fit['k'], 3),
                      's_t': round(fit['s_t'], 4),
                      's_c_insession': (None if s_c is None
                                        else round(
                                            float(s_c), 4)),
                      'lock': lock, 'shape': shape}
                if shape and lock:
                    verdict = \
                        'l34_h15_routing_member_locked'
                elif shape:
                    verdict = \
                        'l34_h15_downstream_rethresholded'
                else:
                    verdict = 'l34_h15_gradual_responder'
                log('T2 fit r2=%.4f k=%.3f s_t=%.4f '
                    's_c=%s shape=%s lock=%s'
                    % (fit['r2'], fit['k'], fit['s_t'],
                       None if s_c is None
                       else round(float(s_c), 4),
                       shape, lock), lines)
        else:
            t2 = None
            verdict = 'l34_h15_independent_of_routing'

        # ---------- T3 descriptive ----------
        B_curve = np.array([B_cur[s] for s in s_grid])
        rho3 = spearman(s_grid, B_curve)
        rng3 = np.random.default_rng(RNG_T3)
        cnt3 = 0
        for _ in range(N_PERM):
            if abs(spearman(rng3.permutation(s_grid),
                            B_curve)) \
                    >= abs(rho3) - 1e-12:
                cnt3 += 1
        p3 = (cnt3 + 1) / (N_PERM + 1)
        a34_curve = np.array(
            [float(np.median(A11_cur[skey(LI_TGT, s)][:, H_TGT]))
             for s in s_grid])
        t3 = {'rho_B_s': round(float(rho3), 4),
              'perm_p_B': float('%.3e' % p3),
              'B_curve': [round(float(v), 4)
                          for v in B_curve],
              'A11_L34_h15_curve': [round(float(v), 4)
                                    for v in a34_curve],
              'A11_L34_recovery_residual': float(
                  '%.3e' % res34),
              'C_response_family_top5': t1_fam_top5,
              'sep_curve': [round(
                  float(sep_cur[skey(LI_INJ, s)]), 1)
                  for s in s_grid]}
        log('T3 rho(s,B) %.4f p %.3e | A11_L34_h15 %s'
            % (rho3, p3, t3['A11_L34_h15_curve']), lines)

        save = {'s_grid': s_grid,
                'C34_curves': np.stack(
                    [C34_cur[s] for s in s_grid]),
                'B_curve': B_curve,
                'sep_curve': np.array(
                    [sep_cur[skey(LI_INJ, s)]
                     for s in s_grid]),
                'A11_L34': np.stack(
                    [A11_cur[skey(LI_TGT, s)]
                     for s in s_grid]),
                'A11_L17_base': A11b[LI_INJ],
                'labels_lang': lab_lang,
                'words': np.array(
                    ['%s:%s:%s' % w for w in words],
                    dtype=object)}

    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2966, 'model': 'qwen3-4b',
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
               'a8_ok': a8_ok,
               'a9_diff': float('%.3e' % a9_diff),
               'a9_ok': a9_ok,
               'a10_res17': float('%.3e' % res17),
               'a10_ok': a10_ok,
               'a11_diff': float('%.3e' % a11_diff),
               'a11_ok': a11_ok,
               'a11_points': a11_pts,
               'ok': anchor_ok},
           's_c_insession': (None if s_c is None
                             else round(float(s_c), 4)),
           'T1_h15_dose': t1, 'T2_logistic': t2,
           'T3_descriptive': t3,
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if verdict != 'anchor_fail_all_void':
        np.savez_compressed(os.path.join(
            OUT, 'l34_routing.npz'), **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2966 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
