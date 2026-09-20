# -*- coding: utf-8 -*-
"""Phase 2970: localize the cross-language peak-position delay
(fr-vs-en, 2969: 13/13 pairs fr later than en, p 2e-04) across
layers and heads.  Protocol verbatim 2968 family A (+xdir, 57-word
batch, GRID17+s0, K=1); this time the FULL layer x head
contribution tensor C[36, 11, 57, 32] is kept (2967 storage rule).

Anchors (verbatim 2968 set + a14 peak-set identity):
  a1 dirs rebuild <1e-5; a2 determinism <1e-4; a3 Vt8 vs 2939
  <1e-6; a4/a5 vs 2935 <1e-4; a6 sep_func>0; a7 xdir <1e-9;
  a8 GQA; a9 A11 vs 2953 <1e-6; a10 recovery residual <0.3;
  a11 sep vs 2953 <0.05; a12 C34_A vs 2967 npz rel <1e-6;
  a13 sep_A vs 2967 npz rel <1e-6;
  a14 per-word C15 peak set recomputed == 2968 (n_peak=40 exact).

Tests:
  T1 (head level): per (l,h) per-word curves C[s,w] -> peak_loc
      per word -> paired d[l,h] = mean(pk_L - pk_en) over valid
      pairs (both members inner-peak); validity gate n_pairs>=8
      per head; sign-flip permutation rng 2981 x10000 per head;
      maxT across valid heads (family = n_valid_heads) q<0.05.
      Invalid branch: n_valid_heads<10 -> head level registered
      descriptive only (no rerun).
  T2 (layer level): per-layer word curves = sum_h C[l,s,w,h] ->
      same pipeline, family=36, rng 2982, maxT q<0.05.
  T3 descriptive: d[34,15] must equal 2969 mean_d 0.5132
      (identity, <5.01e-4); top delayed heads table; head delay
      vs 2967 collapse response rho; layer d profile.

Verdict map (frozen):
  t1 & t2 & (L34 in sig layers) -> delay_carrier_heads_and_layers_localized
  t1 & t2 & (L34 not in sig layers) -> delay_carrier_heads_layers_split
  t1 & ~t2 -> delay_carrier_head_level_only
  ~t1 & t2 -> delay_carrier_layer_level_only
  ~t1 & ~t2 -> delay_not_localized_beyond_h15
  anchor fail => anchor_fail_all_void
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

BASE = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result' \
       r'\rdc_query_construction_20260913'
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
SRC_2967 = os.path.join(BASE, 'phase2967',
                        'collapse_carrier_anatomy',
                        'collapse_carrier.npz')
SRC_2968 = os.path.join(BASE, 'phase2968', 'h15_peak_anatomy')
OUT = os.path.join(BASE, 'phase2970', 'delay_carrier_localization')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2970_run_report.txt')
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
RNG_T1, RNG_T2 = 2981, 2982
SEP_THRESHOLD = 100.0
PAIRS_MIN_HEAD = 8
PAIRS_MIN_TOTAL = 10
NVH_MIN = 10
BIT_TOL = 1e-6
DEGEN_EPS = 1e-12

PREREG = {
    'mode': 'protocol verbatim 2968 family A only (+xdir); '
            'full C[36,11,57,32] kept; no family B',
    'question': 'where (layers, heads) does the fr-vs-en '
                'peak-position delay localize?',
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
        'a12': 'C34_A vs 2967 npz rel < 1e-6',
        'a13': 'sep_A vs 2967 npz rel < 1e-6',
        'a14': 'C15 peak set == 2968 (n_peak 40 exact)'},
    'T1': 'head level paired peak delay: per (l,h) peak_loc '
          'per word, valid pairs (both inner peaks) n>=8, '
          'd = mean(pk_L - pk_en), sign-flip perm rng 2981 '
          'x10000, maxT over valid heads q<0.05; invalid '
          'branch n_valid_heads<10 -> descriptive',
    'T2': 'layer level same pipeline on sum_h curves, '
          'family=36, rng 2982, maxT q<0.05',
    'T3': 'descriptive: d[34,15] == 2969 0.5132 identity '
          '(<5.01e-4); top delayed heads; delay vs 2967 '
          'response rho',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'see module docstring verdict map',
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


def peak_loc(s_grid, y):
    am = int(np.argmax(y))
    if am == 0 or am == len(y) - 1:
        return None
    den = y[am - 1] - 2.0 * y[am] + y[am + 1]
    if abs(den) < 1e-30:
        return float(s_grid[am])
    off = 0.5 * (y[am - 1] - y[am + 1]) / den
    return float(s_grid[am] + np.clip(off, -0.5, 0.5)
                 * (s_grid[am + 1] - s_grid[am - 1]))


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2970,
                   'name': 'delay_carrier_localization',
                   'created':
                       time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2887': sha8(SRC_2887),
                               's2927': sha8(SRC_2927),
                               's2935': sha8(SRC_2935),
                               's2939': sha8(SRC_2939),
                               's2953': sha8(SRC_2953),
                               's2967': sha8(SRC_2967),
                               's2968_result':
                                   sha8(os.path.join(
                                       SRC_2968,
                                       'result.json'))},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'seed': SEED, 'inj_layer': LI_INJ,
                   'grid17': list(GRID17), 'k_repeat': 1,
                   'n_perm': N_PERM,
                   'rng': {'T1': RNG_T1, 'T2': RNG_T2},
                   'pairs_min_head': PAIRS_MIN_HEAD,
                   'pairs_min_total': PAIRS_MIN_TOTAL,
                   'nvh_min': NVH_MIN,
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
    z67 = np.load(SRC_2967, allow_pickle=True)
    z68 = np.load(os.path.join(SRC_2968, 'h15_peak.npz'),
                  allow_pickle=True)
    r68 = json.load(open(os.path.join(SRC_2968, 'result.json'),
                         encoding='utf-8'))
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

    # ---------- pass 1: dirs_word rebuild ----------
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

    def forward_batch(toks_list, scale=0.0, inj_li=None,
                      vec_t=None):
        cap_v.clear()
        cap_x.clear()
        fin_cap.pop('x', None)
        inj['li'] = inj_li
        inj['scale'] = float(scale)
        inj['vec'] = vec_t if scale else None
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

    # ---------- baselines ----------
    fin_f1, _, _ = forward_batch(batch)
    fin_f2, _, _ = forward_batch(batch)
    a2_rel = float(np.abs(fin_f1 - fin_f2).max()
                   / max(float(np.abs(fin_f1).max()), 1e-30))
    a2_ok = bool(a2_rel < 1e-4)
    log('a2 determinism rel %.2e ok=%s' % (a2_rel, a2_ok),
        lines)
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

    # ---------- dose family A ----------
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
    res17 = 0.0
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
    a10_ok = bool(res17 < 0.3)
    log('a10 L17 recovery residual %.2e ok=%s'
        % (res17, a10_ok), lines)

    s_grid = np.array([0.0] + [float(s) for s in GRID17])
    nS = len(s_grid)

    sep_c = {}
    A11_c = {}
    C_all = np.zeros((nS, NL, n_words, NH))
    B_c = {}
    for si, s in enumerate(s_grid):
        fin, vv, xx = forward_batch(
            batch, scale=float(s),
            inj_li=(LI_INJ if s > 0 else None),
            vec_t=xdir_t)
        P = fin @ u35
        sep_c[skey(LI_INJ, s)] = float(
            P[lab_lang == 0].mean() - P[lab_lang == 1].mean())
        if s > 0:
            A, r = recover(
                xx[LI_INJ].reshape(n_words, NH, HD),
                vv[LI_INJ][0].reshape(n_words, NKV, HD),
                vv[LI_INJ][1].reshape(n_words, NKV, HD))
            A11_c[skey(LI_INJ, s)] = A
        A, r = recover(
            xx[LI_TGT].reshape(n_words, NH, HD),
            vv[LI_TGT][0].reshape(n_words, NKV, HD),
            vv[LI_TGT][1].reshape(n_words, NKV, HD))
        A11_c[skey(LI_TGT, s)] = A
        for li in range(NL):
            X = xx[li]
            xm = (X * M[li]).reshape(n_words, NH, HD)
            C_all[si, li] = xm.sum(axis=2)
        prof = C_all[si].sum(axis=2).mean(axis=1)
        B_c[s] = float(prof[6:13].mean()
                       - prof[28:36].mean())
        log('  s=%.4f sep=%.1f B=%.4f'
            % (s, sep_c[skey(LI_INJ, s)], B_c[s]), lines)

    # ---------- cross-phase anchors ----------
    a9_diff = float(np.abs(
        A11b[LI_INJ] - z53['A11b_L17']).max())
    for s in GRID17:
        a9_diff = max(a9_diff, float(np.abs(
            A11_c[skey(LI_INJ, s)]
            - z53['A11_%s' % skey(LI_INJ, s)]).max()))
    a9_ok = bool(a9_diff < BIT_TOL)
    log('a9 A11_L17 vs 2953 npz %.2e ok=%s'
        % (a9_diff, a9_ok), lines)
    sep53 = z53['sep_curves'].astype(np.float64)
    a11_diff = 0.0
    for j, s in enumerate(GRID17):
        a11_diff = max(a11_diff, abs(
            sep_c[skey(LI_INJ, s)] - float(sep53[0, j])))
    a11_ok = bool(a11_diff < 0.05)
    log('a11 sep vs 2953 %.2e ok=%s' % (a11_diff, a11_ok),
        lines)
    C34_A = C_all[:, LI_TGT, :, :]
    a12_diff = float(np.abs(
        C34_A - z67['C_all'][:, LI_TGT]).max()
        / max(float(np.abs(z67['C_all'][:, LI_TGT]).max()),
              1e-30))
    a12_ok = bool(a12_diff < BIT_TOL)
    log('a12 C34_A vs 2967 npz %.2e ok=%s'
        % (a12_diff, a12_ok), lines)
    sepA_curve = np.array([sep_c[skey(LI_INJ, s)]
                           for s in s_grid])
    a13_diff = float(np.abs(
        sepA_curve - z67['sep_curve']).max()
        / max(float(np.abs(z67['sep_curve']).max()), 1e-30))
    a13_ok = bool(a13_diff < BIT_TOL)
    log('a13 sep_A vs 2967 npz %.2e ok=%s'
        % (a13_diff, a13_ok), lines)
    # a14: C15 peak set identity vs 2968
    C15 = C_all[:, LI_TGT, :, H_TGT]
    pk68 = set()
    C15_68 = z68['C34_A'][:, :, H_TGT].astype(np.float64)
    sg68 = z68['s_grid'].astype(np.float64)
    for i in range(n_words):
        if peak_loc(sg68, C15_68[:, i]) is not None:
            pk68.add(i)
    pk_now = set()
    for i in range(n_words):
        if peak_loc(s_grid, C15[:, i]) is not None:
            pk_now.add(i)
    a14_ok = bool(pk_now == pk68
                  and len(pk_now) == int(
                      r68['T1_peaks']['n_peak_words']))
    log('a14 C15 peak set identity n=%d (2968: %d) ok=%s'
        % (len(pk_now), len(pk68), a14_ok), lines)

    anchor_ok = bool(a1_ok and a2_ok and a3_ok and a4_ok
                     and a5_ok and a6_ok and a7_ok and a8_ok
                     and a9_ok and a10_ok and a11_ok
                     and a12_ok and a13_ok and a14_ok)
    verdict = None
    t1 = t2 = t3 = None
    save = {}
    d3415 = None

    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        # ---------- paired sets ----------
        en_idx = [i for i in range(n_words)
                  if lab_lang[i] == 0]
        cidx = {}
        for i in range(n_words):
            cidx.setdefault(words[i][1], {})[lab_lang[i]] = i
        pairs_all = [(d[0], d[1]) for d in cidx.values()
                     if 0 in d and 1 in d]
        log('total en/L pairs: %d' % len(pairs_all), lines)

        def pairs_2969(pk):
            """2969 pairing口径 verbatim: filter peak words
            first, then cidx (last L per concept among peak
            words). 2887 is 4-language: cidx[cid][lang] = i
            overwrite picks the LAST matching word."""
            c = {}
            for i in range(n_words):
                if pk[i] is None:
                    continue
                c.setdefault(words[i][1], {})[lab_lang[i]] = i
            return [(d[0], d[1]) for d in c.values()
                    if 0 in d and 1 in d]

        # ---------- T2 layer level ----------
        rng2 = np.random.default_rng(RNG_T2)
        n_p2 = len(pairs_all)
        signs2 = rng2.choice([-1.0, 1.0],
                             size=(N_PERM, n_p2))
        d_layer = np.zeros(NL)
        p_layer = np.ones(NL)
        valid_l = []
        for li in range(NL):
            curve = C_all[:, li, :, :].sum(axis=2)
            pk = [peak_loc(s_grid, curve[:, i])
                  for i in range(n_words)]
            pl = pairs_2969(pk)
            d = [pk[fr] - pk[en] for en, fr in pl]
            if len(d) < PAIRS_MIN_TOTAL:
                continue
            d = np.array(d)
            d_layer[li] = float(d.mean())
            nulls = np.abs((signs2[:, :len(d)] * d).mean(1))
            p_layer[li] = (int((nulls >= abs(d_layer[li]))
                               .sum()) + 1) / (N_PERM + 1)
            valid_l.append(li)
        # maxT over valid layers (joint stream)
        q_layer = np.ones(NL)
        if valid_l:
            rngj = np.random.default_rng(RNG_T2 + 555)
            joint = np.empty(N_PERM)
            pk_cache = {}
            for li in valid_l:
                curve = C_all[:, li, :, :].sum(axis=2)
                pk_cache[li] = [peak_loc(s_grid, curve[:, i])
                                for i in range(n_words)]
            obs_max = max(abs(d_layer[li]) for li in valid_l)
            for b in range(N_PERM):
                sg = rngj.choice([-1.0, 1.0],
                                 size=(1, n_p2))[0]
                worst = 0.0
                for li in valid_l:
                    pk = pk_cache[li]
                    pl = pairs_2969(pk)
                    d = np.array([pk[fr] - pk[en]
                                  for en, fr in pl])
                    worst = max(worst, abs(
                        float((sg[:len(d)] * d).mean())))
                joint[b] = worst
            for li in valid_l:
                q_layer[li] = (int((joint >= abs(d_layer[li]))
                                   .sum()) + 1) / (N_PERM + 1)
        sig_l = [li for li in valid_l if q_layer[li] < 0.05]
        t2 = {'n_valid_layers': len(valid_l),
              'sig_layers': sig_l,
              'd_layer': {int(li): round(float(d_layer[li]), 4)
                          for li in valid_l},
              'q_layer': {int(li): float('%.4g' % q_layer[li])
                          for li in valid_l},
              'pass': bool(sig_l)}
        log('T2 sig layers: %s' % sig_l, lines)

        # ---------- T1 head level ----------
        # per (l,h) word curves; k = l*NH + h rows are (nS,
        # n_words) matrices -> transpose must put NH before
        # n_words (2969 identity: d[34,15] must reproduce
        # +0.5132; a wrong reshape folds the NH axis into the
        # element stream and yields noise)
        C_flat = C_all.transpose(1, 3, 0, 2).reshape(
            NL * NH, nS, n_words)
        valid_h = []
        d_head = np.zeros(NL * NH)
        pairs_h = {}
        pk_cache_h = {}
        for k in range(NL * NH):
            pk = [peak_loc(s_grid, C_flat[k, :, i])
                  for i in range(n_words)]
            pl = pairs_2969(pk)
            d = [pk[fr] - pk[en] for en, fr in pl]
            if len(d) >= PAIRS_MIN_HEAD:
                valid_h.append(k)
                d_head[k] = float(np.mean(d))
                pairs_h[k] = len(d)
                pk_cache_h[k] = pk
        nvh = len(valid_h)
        t1_sig = False
        t1 = {'n_valid_heads': nvh,
              'validity_ok': bool(nvh >= NVH_MIN)}
        if nvh >= NVH_MIN:
            # precompute per-head paired d arrays, group by
            # pair count for vectorized maxT
            from collections import defaultdict
            d_arr = {}
            groups = defaultdict(list)
            for k in valid_h:
                pk = pk_cache_h[k]
                pl = pairs_2969(pk)
                d = np.array([pk[fr] - pk[en]
                              for en, fr in pl])
                d_arr[k] = d
                groups[len(d)].append(k)
            rng1 = np.random.default_rng(RNG_T1)
            obs_max = max(abs(d_head[k]) for k in valid_h)
            joint = np.empty(N_PERM)
            B_SZ = 250
            done = 0
            while done < N_PERM:
                bsz = min(B_SZ, N_PERM - done)
                worst = np.zeros(bsz)
                sg = rng1.choice([-1.0, 1.0],
                                 size=(bsz, n_p2))
                for L, ks in groups.items():
                    D = np.stack([d_arr[k] for k in ks])
                    means = np.abs(
                        (sg[:, None, :L] * D[None, :, :])
                        .mean(axis=2))
                    worst = np.maximum(worst,
                                       means.max(axis=1))
                joint[done:done + bsz] = worst
                done += bsz
            q_maxt = (int((joint >= obs_max).sum()) + 1) \
                / (N_PERM + 1)
            t1_sig = bool(q_maxt < 0.05)
            t1.update({'q_maxT': float('%.4g' % q_maxt),
                       'obs_max_abs_d': round(float(obs_max),
                                              4),
                       'pass': t1_sig})
        else:
            t1.update({'pass': False,
                       'reason': 'n_valid_heads < %d'
                                 % NVH_MIN})
        # top delayed heads (descriptive)
        top_h = sorted(valid_h,
                       key=lambda k: -abs(d_head[k]))[:12]
        t1['top_heads'] = [
            {'lh': [int(k // NH), int(k % NH)],
             'd': round(float(d_head[k]), 4),
             'n_pairs': int(pairs_h[k])} for k in top_h]
        log('T1 valid heads=%d sig=%s top=%s'
            % (nvh, t1_sig, t1['top_heads'][:5]), lines)

        # ---------- T3 descriptive ----------
        d3415 = d_head[LI_TGT * NH + H_TGT]
        r69 = json.load(open(os.path.join(
            BASE, 'phase2969', 'peak_word_attributes',
            'result.json'), encoding='utf-8'))
        d3415_diff = abs(float(d3415) - float(
            r69['T2_paired_lang']['mean_d_L_minus_en']))
        a15_ok = bool(d3415_diff < 5.01e-4)
        log('a15 d[34,15] identity vs 2969 %.4f diff %.2e '
            'ok=%s' % (d3415, d3415_diff, a15_ok), lines)
        if not a15_ok:
            verdict = 'anchor_fail_all_void'
            t1 = {'invalidated': 'a15 transpose identity '
                                 'failed - head-level data '
                                 'corrupt'}
            t2 = dict(t2, invalidated=True)
        # delay vs 2967 response (head-level |rho(s,C)| family)
        # use 2967 npz C_all for response rho of L34 heads
        z67C = z67['C_all'][:, LI_TGT, :, :]
        sg67 = z67['s_grid'].astype(np.float64)
        resp = {}
        for hh in range(NH):
            ys = z67C[:, :, hh]
            # median curve rho vs s
            med = np.median(ys, axis=1)
            resp[hh] = float(np.corrcoef(
                sg67, med)[0, 1]) if np.std(med) > 0 else 0.0
        rho_d_resp = float(np.corrcoef(
            [d_head[LI_TGT * NH + hh] for hh in range(NH)],
            [resp[hh] for hh in range(NH)])[0, 1])
        t3 = {'d3415': round(float(d3415), 4),
              'd3415_vs_2969_diff': round(float(d3415_diff),
                                          6),
              'rho_delay_vs_response_L34': round(
                  rho_d_resp, 4),
              'sig_layers_d': {int(li):
                               round(float(d_layer[li]), 4)
                               for li in sig_l}}

        # ---------- verdict ----------
        l34_in = (LI_TGT in sig_l)
        if t1_sig and t2['pass'] and l34_in:
            verdict = ('delay_carrier_heads_and_layers'
                       '_localized')
        elif t1_sig and t2['pass']:
            verdict = 'delay_carrier_heads_layers_split'
        elif t1_sig and not t2['pass']:
            verdict = 'delay_carrier_head_level_only'
        elif (not t1_sig) and t2['pass']:
            verdict = 'delay_carrier_layer_level_only'
        else:
            verdict = 'delay_not_localized_beyond_h15'
        log('T3: %s' % t3, lines)

    log('verdict: %s' % verdict, lines)

    result = {
        'phase': 2970,
        'name': 'delay_carrier_localization',
        'model': 'qwen3-4b',
        'anchors': {'ok': anchor_ok,
                    'a1': a1_diff, 'a2': a2_rel, 'a3': a3_diff,
                    'a4': a4_diff, 'a5': a5_diff,
                    'a9': a9_diff, 'a10': res17,
                    'a11': a11_diff, 'a12': a12_diff,
                    'a13': a13_diff,
                    'a14_n_peak': len(pk_now)},
        'T1_heads': t1,
        'T2_layers': t2,
        'T3_descriptive': t3,
        'final_verdict': verdict,
        'runtime_s': round(time.monotonic() - t0, 1),
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    save = {'C_all': C_all.astype(np.float32),
            's_grid': s_grid,
            'words': np.array([':'.join(w) for w in words]),
            'labels_lang': lab_lang,
            'd_head': d_head, 'd_layer': d_layer,
            'valid_heads': np.array(valid_h),
            'valid_layers': np.array(valid_l)}
    np.savez_compressed(
        os.path.join(OUT, 'delay_carrier.npz'), **save)
    with open(os.path.join(OUT, 'run_log.txt'), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('DONE %s' % verdict)


if __name__ == '__main__':
    main()
