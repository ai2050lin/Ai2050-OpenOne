# -*- coding: utf-8 -*-
"""Phase 2967: which layers/heads carry the monotone
collapse of the deep-band differential B under language
injection? (carrier anatomy of the B(s) wipe-out)

Why: 2966 found the strong signal rho(s, B) = +0.9364
(p 1e-4): injecting the language direction xdir at L17
monotonically wipes the deep-band differential
B = prof[6:13] - prof[28:36] (-1.365 -> -0.019). The
response family at L34 (h6 rho=-1.0, h31/h24/h2/h10 ~ +0.97)
was registered descriptively. Open question: which layers
and heads CAARRY the wipe-out - is the collapse localized
in the deep band itself, or distributed? Does the L34
response family replicate formally (2966 was descriptive)?

Mode: forward family IDENTICAL to 2966 (57-word language
batch verbatim from 2887, L17 xdir injection coef 1.0,
s grid = 2953 GRID17 + s=0, K=1). Same hooks; this time
the full-layer head-level contribution C[36, s, word, head]
is retained (2966 kept only L34).

Anchors (frozen):
  a1 dirs_word vs 2927 < 1e-5
  a2 baseline determinism rel < 1e-4
  a3 Vt8 vs 2939 < 1e-6
  a4 proj_func vs 2935 < 1e-4
  a5 proj_null0 vs 2935 < 1e-4
  a6 sep_func > 0
  a7 xdir self-check < 1e-9
  a8 GQA gates
  a9 A11_L17 vs 2953 npz < 1e-6
  a10 L17 recovery residual < 0.3
  a11 sep shared grid vs 2953 < 0.05
  a12 C34_curves vs 2966 npz rel < 1e-6 (same
      implementation, same batch composition -> bit-level
      replication expected; 2937->2959 precedent)
  a13 B_curve vs 2966 npz < 1e-6
  a14 sep_curve vs 2966 npz < 1e-6
  Reachability pre-check (discipline 10): a12-a14 are
  same-code same-composition replications; precedent
  a17 (2959 vs 2955) was bit 0.

Tests (frozen):
  T1 (layer level, PRIMARY): per-layer gap curve
      gap_li(s) = median_F(sum_h C[li,s,:,h])
                  - median_N(sum_h C[li,s,:,h]);
      spearman(s, gap_li) two-sided perm rng 2975 x10000,
      family = 36 layers, maxT (Westfall-Young single-step),
      q <= 0.05; collapse-carrier layer = significant AND
      |gap(2.0)| / max(|gap(0)|, 1e-30) < 0.3.
      Gate: carrier set non-empty. Non-degeneracy gate
      (discipline 12): gap curve std > 1e-12 else excluded.
  T2 (head level): per-(layer,head) curve
      curve_lih(s) = median_word C[li,s,:,h]; spearman via
      vectorized rank corr, family = 36*32 = 1152, maxT
      rng 2976 x10000, q <= 0.05; non-degeneracy: curve
      std > 1e-12. Gate: significant set non-empty.
  T3 descriptive: per-layer collapse ratios; significant
      head set by layer; intersections with 2947 D_L17
      top5 / 2964 top5 (h15@L34) / 2966 C34 family top5
      (h6/h31/h24/h2/h10 at L34) / 2953 keep sets;
      L34 formal-vs-2966-descriptive comparison.

Verdict (frozen):
  anchor fail => anchor_fail_all_void
  T1 pass & T2 pass => collapse_carrier_localized
  T1 pass & T2 fail => collapse_layer_level_only
  T1 fail => collapse_not_layer_localized
  Reachability: all branches reachable (2966 observed
  rho(s,B)=0.9364 p 1e-4 on an aggregate of these same
  layer quantities; single-layer gaps may disperse - that
  is the T1-fail branch).
"""
import hashlib
import json
import os
import sys
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
SRC_2953 = os.path.join(BASE, 'phase2953', 'a11_s_response',
                        'a11_s_response.npz')
SRC_2966 = os.path.join(BASE, 'phase2966',
                        'l34_routing_membership',
                        'l34_routing.npz')
OUT = os.path.join(BASE, 'phase2967', 'collapse_carrier_anatomy')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2967_run_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NH, HD = 32, 128
NL = 36
VOCAB = 151936
SEED = 2896
LI_INJ = 17
LI_TGT = 34
GRID17 = (0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.25,
          1.5, 2.0)
S_IDX = (0, 1, 4)
N_PERM = 10000
RNG_T1, RNG_T2 = 2975, 2976
SEP_THRESHOLD = 100.0
Q_MAX = 0.05
COLLAPSE_RATIO = 0.3
DEGEN_EPS = 1e-12
BIT_TOL = 1e-6

PREREG = {
    'mode': 'forward family identical to 2966: 57-word '
            'batch verbatim (2887), L17 xdir coef 1.0, '
            'GRID17 + s=0 (11 pts), K=1; full-layer '
            'head-level contribution C[36,s,57,32] retained',
    'question': 'which layers/heads carry the monotone '
                'collapse of B under language injection - '
                'localized in the deep band or distributed?',
    'anchors': {
        'a1': 'dirs_word vs 2927 < 1e-5',
        'a2': 'determinism rel < 1e-4',
        'a3': 'Vt8 vs 2939 < 1e-6',
        'a4': 'proj_func vs 2935 < 1e-4',
        'a5': 'proj_null0 vs 2935 < 1e-4',
        'a6': 'sep_func > 0',
        'a7': 'xdir self-check < 1e-9',
        'a8': 'GQA gates',
        'a9': 'A11_L17 vs 2953 < 1e-6',
        'a10': 'L17 recovery residual < 0.3',
        'a11': 'sep shared grid vs 2953 < 0.05',
        'a12': 'C34_curves vs 2966 npz < 1e-6',
        'a13': 'B_curve vs 2966 npz < 1e-6',
        'a14': 'sep_curve vs 2966 npz < 1e-6',
        'note': 'a12-a14 same-code same-composition '
                'replication, bit-level expected',
    },
    'T1': 'per-layer gap curve spearman, perm rng 2975 '
          'x10000, family 36 maxT q<=0.05; carrier = sig '
          'AND |gap(end)/gap(0)| < 0.3; gate: carrier set '
          'non-empty; degeneracy gate std > 1e-12',
    'T2': 'per-(layer,head) curve spearman vectorized, '
          'family 1152 maxT rng 2976 x10000 q<=0.05; gate: '
          'sig set non-empty; degeneracy std > 1e-12',
    'T3': 'descriptive: per-layer collapse ratios; sig-head '
          'layer profile; intersections with 2947/2964/2966/'
          '2953 head sets; L34 formal vs 2966 descriptive',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'T1 pass & T2 pass => '
               'collapse_carrier_localized; T1 pass & T2 '
               'fail => collapse_layer_level_only; '
               'T1 fail => collapse_not_layer_localized',
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


def rank_rows(M):
    """rank along axis=1 for a 2D array, ties averaged."""
    out = np.empty(M.shape, dtype=np.float64)
    for i in range(M.shape[0]):
        out[i] = rankdata(M[i])
    return out


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
        json.dump({'phase': 2967,
                   'name': 'collapse_carrier_anatomy',
                   'created':
                       time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2887': sha8(SRC_2887),
                               's2927': sha8(SRC_2927),
                               's2935': sha8(SRC_2935),
                               's2939': sha8(SRC_2939),
                               's2953': sha8(SRC_2953),
                               's2966': sha8(SRC_2966)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'seed': SEED, 'inj_layer': LI_INJ,
                   'grid17': list(GRID17), 'k_repeat': 1,
                   'n_perm': N_PERM,
                   'rng': {'T1': RNG_T1, 'T2': RNG_T2},
                   'sep_threshold': SEP_THRESHOLD,
                   'q_max': Q_MAX,
                   'collapse_ratio': COLLAPSE_RATIO,
                   'degen_eps': DEGEN_EPS,
                   'bit_tol': BIT_TOL,
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
    z66 = np.load(SRC_2966, allow_pickle=True)
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
    log('a10 L17 recovery residual %.2e ok=%s'
        % (res17, a10_ok), lines)

    s_grid = np.array([0.0] + [float(s) for s in GRID17])
    sep_cur = {}
    A11_cur = {}
    C34_cur = {}
    B_cur = {}
    C_all = np.zeros((len(s_grid), NL, n_words, NH))
    for si, s in enumerate(s_grid):
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
        prof = np.zeros(NL)
        for li in range(NL):
            X = xx[li]
            xm = (X * M[li]).reshape(n_words, NH, HD)
            ch = xm.sum(axis=2)
            C_all[si, li] = ch
            if li == LI_TGT:
                C34_cur[s] = ch
            prof[li] = float(ch.sum(axis=1).mean())
        B_cur[s] = float(prof[6:13].mean()
                         - prof[28:36].mean())
        log('s=%.4f sep=%.1f B=%.4f' % (s, sep_cur[
            skey(LI_INJ, s)], B_cur[s]), lines)
    a10_ok = bool(res17 < 0.3)
    log('a10 L17 recovery residual max %.2e ok=%s | L34 '
        'residual max %.2e (descriptive)'
        % (res17, a10_ok, res34), lines)

    # ---------- cross-phase anchors ----------
    a9_diff = float(np.abs(
        A11b[LI_INJ] - z53['A11b_L17']).max())
    for s in GRID17:
        a9_diff = max(a9_diff, float(np.abs(
            A11_cur[skey(LI_INJ, s)]
            - z53['A11_%s' % skey(LI_INJ, s)]).max()))
    a9_ok = bool(a9_diff < BIT_TOL)
    log('a9 A11_L17 vs 2953 npz (all 11 s) %.2e ok=%s'
        % (a9_diff, a9_ok), lines)
    sep53 = z53['sep_curves'].astype(np.float64)
    a11_diff = 0.0
    for j, s in enumerate(GRID17):
        a11_diff = max(a11_diff, abs(
            sep_cur[skey(LI_INJ, s)] - float(sep53[0, j])))
    a11_ok = bool(a11_diff < 0.05)
    log('a11 sep shared grid vs 2953 L17 row %.2e ok=%s'
        % (a11_diff, a11_ok), lines)
    C34_all = np.stack([C34_cur[s] for s in s_grid])
    a12_diff = float(np.abs(
        C34_all - z66['C34_curves']).max()
        / max(float(np.abs(z66['C34_curves']).max()), 1e-30))
    a12_ok = bool(a12_diff < BIT_TOL)
    log('a12 C34_curves vs 2966 npz %.2e ok=%s'
        % (a12_diff, a12_ok), lines)
    B_curve = np.array([B_cur[s] for s in s_grid])
    a13_diff = float(np.abs(
        B_curve - z66['B_curve']).max()
        / max(float(np.abs(z66['B_curve']).max()), 1e-30))
    a13_ok = bool(a13_diff < BIT_TOL)
    log('a13 B_curve vs 2966 npz %.2e ok=%s'
        % (a13_diff, a13_ok), lines)
    sep_curve = np.array([sep_cur[skey(LI_INJ, s)]
                          for s in s_grid])
    a14_diff = float(np.abs(
        sep_curve - z66['sep_curve']).max()
        / max(float(np.abs(z66['sep_curve']).max()), 1e-30))
    a14_ok = bool(a14_diff < BIT_TOL)
    log('a14 sep_curve vs 2966 npz %.2e ok=%s'
        % (a14_diff, a14_ok), lines)

    anchor_ok = bool(a1_ok and a2_ok and a3_ok and a4_ok
                     and a5_ok and a6_ok and a7_ok and a8_ok
                     and a9_ok and a10_ok and a11_ok
                     and a12_ok and a13_ok and a14_ok)
    verdict = None
    t1 = t2 = t3 = None
    save = {}

    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        F = lab_lang == 0
        N = lab_lang == 1

        # ---------- T1 layer-level gap curves ----------
        # layer_gap[li, si] = median_F(sum_h C) -
        #                     median_N(sum_h C)
        tot = C_all.sum(axis=3)  # (S, L, W)
        layer_gap = np.zeros((NL, len(s_grid)))
        for li in range(NL):
            for si in range(len(s_grid)):
                layer_gap[li, si] = \
                    float(np.median(tot[si, li][F])) \
                    - float(np.median(tot[si, li][N]))
        deg1 = layer_gap.std(axis=1) > DEGEN_EPS
        rng1 = np.random.default_rng(RNG_T1)
        perms = np.array([rng1.permutation(len(s_grid))
                          for _ in range(N_PERM)])
        Sr = np.stack([rankdata(s_grid[perms[p]])
                       for p in range(N_PERM)])
        Sr = Sr - Sr.mean(axis=1, keepdims=True)
        Srn = np.sqrt((Sr ** 2).sum(axis=1))
        rho1 = np.zeros(NL)
        for li in range(NL):
            rho1[li] = spearman(s_grid, layer_gap[li])
        # maxT
        cnt1 = np.zeros(NL)
        maxperm = np.zeros(N_PERM)
        for p in range(N_PERM):
            rp = np.zeros(NL)
            for li in range(NL):
                if not deg1[li]:
                    continue
                rp[li] = abs(spearman(
                    s_grid[perms[p]], layer_gap[li]))
            maxperm[p] = rp.max()
        for li in range(NL):
            if not deg1[li]:
                cnt1[li] = N_PERM
                continue
            cnt1[li] = int(np.sum(
                maxperm >= abs(rho1[li]) - 1e-12))
        q1 = (cnt1 + 1) / (N_PERM + 1)
        sig1 = deg1 & (q1 <= Q_MAX)
        ratios = np.zeros(NL)
        for li in range(NL):
            ratios[li] = abs(layer_gap[li, -1]) \
                / max(abs(layer_gap[li, 0]), 1e-30)
        carriers = sorted(int(li) for li in range(NL)
                          if sig1[li]
                          and ratios[li] < COLLAPSE_RATIO)
        t1_pass = bool(len(carriers) > 0)
        t1 = {'sig_layers': sorted(
                  int(li) for li in range(NL)
                  if sig1[li]),
              'q_max': {int(li): float('%.3e' % q1[li])
                        for li in range(NL) if sig1[li]},
              'collapse_ratios': {
                  int(li): round(float(ratios[li]), 4)
                  for li in range(NL)},
              'carrier_layers': carriers,
              'pass': t1_pass,
              'L34_in_carriers': bool(34 in carriers),
              'L34_rho': round(float(rho1[34]), 4),
              'L34_q': float('%.3e' % q1[34])}
        log('T1 sig_layers=%s carriers=%s L34(rho %.4f q %.3e '
            'ratio %.4f) pass=%s'
            % (t1['sig_layers'], carriers, rho1[34], q1[34],
               ratios[34], t1_pass), lines)

        # ---------- T2 head-level family ----------
        curves = np.zeros((NL * NH, len(s_grid)))
        for li in range(NL):
            for h in range(NH):
                for si in range(len(s_grid)):
                    curves[li * NH + h, si] = float(
                        np.median(C_all[si, li, :, h]))
        deg2 = curves.std(axis=1) > DEGEN_EPS
        rng2 = np.random.default_rng(RNG_T2)
        perms2 = np.array([rng2.permutation(len(s_grid))
                           for _ in range(N_PERM)])
        Sr2 = np.stack([rankdata(s_grid[perms2[p]])
                        for p in range(N_PERM)])
        Sr2 = Sr2 - Sr2.mean(axis=1, keepdims=True)
        Sr2n = np.sqrt((Sr2 ** 2).sum(axis=1))
        Ry = rank_rows(curves)
        Ry = Ry - Ry.mean(axis=1, keepdims=True)
        dyn = np.where(deg2,
                       np.sqrt((Ry ** 2).sum(axis=1)), 1.0)
        rho2 = (Ry / dyn[:, None]) @ (Sr2 / Sr2n[:, None]).T
        rho2 = rho2.T  # (N_PERM, F)
        obs2 = np.array([
            spearman(s_grid, curves[f]) if deg2[f] else 0.0
            for f in range(NL * NH)])
        max2 = np.abs(rho2).max(axis=1)
        cnt2 = np.array([int(np.sum(
            max2 >= abs(obs2[f]) - 1e-12))
            for f in range(NL * NH)])
        q2 = (cnt2 + 1) / (N_PERM + 1)
        sig2 = deg2 & (q2 <= Q_MAX)
        sig2_heads = sorted(
            (int(f // NH), int(f % NH))
            for f in range(NL * NH) if sig2[f])
        t2_pass = bool(len(sig2_heads) > 0)
        by_layer = {}
        for li, h in sig2_heads:
            by_layer.setdefault(li, []).append(h)
        t2 = {'n_sig': len(sig2_heads),
              'sig_heads_by_layer': {
                  int(k): v for k, v in
                  sorted(by_layer.items())},
              'L34_sig_heads': by_layer.get(34, []),
              'pass': t2_pass}
        log('T2 n_sig=%d by_layer=%s L34=%s pass=%s'
            % (len(sig2_heads),
               {k: v for k, v in sorted(
                   by_layer.items())},
               by_layer.get(34, []), t2_pass), lines)

        # ---------- T3 descriptive ----------
        top2 = [(int(f // NH), int(f % NH),
                 round(float(obs2[f]), 4))
                for f in np.argsort(-np.abs(obs2))[:12]]
        desc = {'gap_s0_by_layer': [
                    round(float(layer_gap[li, 0]), 4)
                    for li in range(NL)],
                'gap_end_by_layer': [
                    round(float(layer_gap[li, -1]), 4)
                    for li in range(NL)],
                'rho_layers': [
                    round(float(rho1[li]), 4)
                    for li in range(NL)],
                'head_top12_abs_rho': top2,
                'L34_C34_family_top5_2966': [
                    [6, -1.0], [31, 0.9818],
                    [24, 0.9818], [2, 0.9727],
                    [10, 0.9636]]}
        t3 = desc
        log('T3 head_top12=%s' % (top2,), lines)

        if t1_pass and t2_pass:
            verdict = 'collapse_carrier_localized'
        elif t1_pass:
            verdict = 'collapse_layer_level_only'
        else:
            verdict = 'collapse_not_layer_localized'

        save = {'s_grid': s_grid,
                'C_all': C_all,
                'layer_gap': layer_gap,
                'head_curves_L34': curves[
                    LI_TGT * NH:(LI_TGT + 1) * NH],
                'B_curve': B_curve,
                'sep_curve': sep_curve,
                'labels_lang': lab_lang,
                'words': np.array(
                    ['%s:%s:%s' % w for w in words],
                    dtype=object)}

    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2967, 'model': 'qwen3-4b',
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
               'a12_diff': float('%.3e' % a12_diff),
               'a12_ok': a12_ok,
               'a13_diff': float('%.3e' % a13_diff),
               'a13_ok': a13_ok,
               'a14_diff': float('%.3e' % a14_diff),
               'a14_ok': a14_ok,
               'ok': anchor_ok},
           'T1_layer_gap': t1, 'T2_head_family': t2,
           'T3_descriptive': t3,
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if verdict != 'anchor_fail_all_void':
        np.savez_compressed(os.path.join(
            OUT, 'collapse_carrier.npz'), **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2967 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
