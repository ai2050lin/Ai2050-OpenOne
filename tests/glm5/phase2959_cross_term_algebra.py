# -*- coding: utf-8 -*-
"""Phase 2959: cross-term algebra - is the dominant dq.ddk cross
term of the attention logit shift (2955: X = 3.88/4.06, largest
of Q/K/X in both dose layers) predictable from the injection
direction by second-order algebra?

Why: 2955 established the source axis is qk_mixed with the CROSS
term largest, i.e. neither pure q-side nor pure k-side linearity.
If the cross term is a genuine second-order response
(xt(s) ~ s^2, response directions locked across doses), then the
whole logit shift is operationalizable from ONE small-dose probe
(xt(1.0) = 16 * xt(0.25)) - a primitive card for the atlas
(strategy v2 stage 1). If directions drift or the scaling is
anomalous, the cross term carries dose-dependent nonlinearity
beyond snapshot algebra.

Design (2955 verbatim scaffolding + s-grid):
  base x2 + L17 s in {0.25, 0.5, 0.75, 1.0, 1.5, 2.0} + L16@s2.0
  per-condition fp64 q/k recompute (q_norm/k_norm + RoPE,
  1/sqrt(HD) scale) and exact decomposition
  dz = dq.dk_b + q_b.ddk + dq.ddk   (qt / kt / xt)

Tests (frozen, L17 grid):
  T1 scaling axis: per head, log-log slope of median_w |xt(s)|
     vs s (heads with med|xt(1.0)| >= 0.05 only, count
     registered); median_h slope in [1.7, 2.3] => second_order,
     else anomalous_slope.
  T2 direction axis: per head median_w cos(dq(0.5), dq(1.0))
     and cos(ddk(0.5), ddk(1.0)) over valid words (max |.| >
     1e-6); both medians >= 0.95 => direction_locked, else
     direction_drift.
  T3 operational prediction: xt_pred(1.0) = 16 * xt(0.25);
     median_h of median_w |xt(1.0) - 16 xt(0.25)| /
     max(median_w |xt(1.0)|, 1e-30) < 0.3 AND
     spearman_h(med_w xt_obs(1.0), med_w xt_pred(1.0)) >= 0.9
     => predictable_small_dose, else not_predictable.

Anchors (frozen):
  a1 dirs rebuild vs 2927 < 1e-5
  a3 Vt8 vs 2939 < 1e-6
  a7 xdir self-check < 1e-9
  a9 structure gates
  a10 base repeat rel < 1e-6
  a11 dsc vs 2952 delta < 1e-6 (L17@s1.0 / L16@s2.0)
  a13 sep L17@0.5 vs 2945 < 0.05
  a14 A11 line-recovery residual < 0.3
  a15 A11b vs 2953 npz < 1e-6
  a16 recompute-chain dA_med max < 0.05 (base + all 7 conds)
  a17 vs 2955 npz bit-level < 1e-6: dz_full (3 saved conds),
     qt/kt/xt (L17_s1.0, L16_s2.0), zb_full (L17, L16),
     A11sm_b (L17, L16)  [2957 lesson: same batch composition +
     same session => cross-phase bit anchors]
  a18 decomposition identity |q+k+x - dz| max < 1e-6

Verdict (frozen):
  anchor fail => anchor_fail_all_void
  else f'{t1r}_{t2r}_{t3r}'
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
SRC_2939 = os.path.join(BASE, 'phase2939', 'rotation_target',
                        'rotation_target.npz')
SRC_2945 = os.path.join(BASE, 'phase2945', 'threshold_curves',
                        'threshold_curves.npz')
SRC_2952 = os.path.join(BASE, 'phase2952', 'amplification_anatomy',
                        'amplification_anatomy.npz')
SRC_2953 = os.path.join(BASE, 'phase2953', 'a11_s_response',
                        'a11_s_response.npz')
SRC_2955 = os.path.join(BASE, 'phase2955',
                        'qk_source_decomposition',
                        'qk_source_decomposition.npz')
OUT = os.path.join(BASE, 'phase2959', 'cross_term_algebra')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2959_run_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NH, HD = 32, 128
NL = 36
SEED = 2905
GRID = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
SLOPE_LO, SLOPE_HI = 1.7, 2.3
XT_MIN = 0.05
COS_TH = 0.95
PRED_REL = 0.3
PRED_RHO = 0.9

PREREG = {
    'mode': 'one forward family (base x2 + L17 s-grid '
            '{0.25,0.5,0.75,1.0,1.5,2.0} + L16@s2.0); layer-input '
            'residual pos0+pos1 capture at dose layer (self_attn '
            'pre-hook, post-injection); fp64 q/k recompute '
            '(q_norm/k_norm + RoPE + 1/sqrt(HD)) validated vs '
            'line-recovered A11 (a16); exact decomposition '
            'dz = dq.dk_b + q_b.ddk + dq.ddk',
    'question': 'is the dominant dq.ddk cross term (2955) '
                'predictable from the injection direction by '
                'second-order algebra (s^2 scaling + locked '
                'response directions)?',
    'head_set': 'ALL 32 heads (lesson 24口径 registered), '
                'medians over 57 words',
    'anchors': {
        'a1': 'dirs rebuild < 1e-5',
        'a3': 'Vt8 < 1e-6', 'a7': 'xdir self-check < 1e-9',
        'a9': 'structure gates',
        'a10': 'base repeat rel < 1e-6',
        'a11': 'dsc vs 2952 delta < 1e-6',
        'a13': 'sep L17@0.5 vs 2945 < 0.05',
        'a14': 'A11 recon residual < 0.3',
        'a15': 'A11b vs 2953 < 1e-6',
        'a16': 'max dA_med < 0.05 over base + 7 conds',
        'a17': 'vs 2955 npz bit-level < 1e-6 (dz/qt/kt/xt/zb/'
               'A11sm_b, saved keys)',
        'a18': '|q+k+x - dz| max < 1e-6',
    },
    'T1': 'per-head log-log slope of med_w|xt(s)| vs s '
          '(heads with med|xt(1.0)| >= 0.05); median_h slope '
          'in [1.7,2.3] => second_order else anomalous_slope',
    'T2': 'per-head med_w cos(dq(0.5),dq(1.0)) and '
          'cos(ddk(0.5),ddk(1.0)), valid words max|.|>1e-6; '
          'both >= 0.95 => direction_locked else direction_drift',
    'T3': 'xt_pred(1.0)=16*xt(0.25); median_h rel err < 0.3 AND '
          'spearman_h(med_w xt_obs, med_w xt_pred) >= 0.9 => '
          'predictable_small_dose else not_predictable',
    'verdict': 'anchor fail => anchor_fail_all_void; else '
               '{t1r}_{t2r}_{t3r}',
    'correction_note': 'none (first run)',
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
        json.dump({'phase': 2959,
                   'name': 'cross_term_algebra',
                   'created':
                       time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2887': sha8(SRC_2887),
                               's2927': sha8(SRC_2927),
                               's2939': sha8(SRC_2939),
                               's2945': sha8(SRC_2945),
                               's2952': sha8(SRC_2952),
                               's2953': sha8(SRC_2953),
                               's2955': sha8(SRC_2955)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'seed': SEED, 's_grid': list(GRID),
                   'slope_th': [SLOPE_LO, SLOPE_HI],
                   'xt_min': XT_MIN, 'cos_th': COS_TH,
                   'pred_rel': PRED_REL,
                   'pred_rho': PRED_RHO,
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
    z52 = np.load(SRC_2952, allow_pickle=True)
    z53 = np.load(SRC_2953, allow_pickle=True)
    z55 = np.load(SRC_2955, allow_pickle=True)

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

    cfg = model.config
    rope_theta = float(getattr(cfg, 'rope_theta', 1e6))
    eps_norm = float(getattr(cfg, 'rms_norm_eps', 1e-6))
    a9_ok = bool(
        layers[0].self_attn.v_proj.weight.shape[0] == 1024
        and layers[0].self_attn.o_proj.in_features == NH * HD
        and layers[0].self_attn.q_proj.weight.shape[0] == NH * HD
        and layers[0].self_attn.k_proj.weight.shape[0] == 1024
        and layers[0].self_attn.q_norm.weight.shape[0] == HD
        and layers[0].self_attn.k_norm.weight.shape[0] == HD)
    log('a9 structure gates ok=%s (theta=%g eps=%g)'
        % (a9_ok, rope_theta, eps_norm), lines)

    # ---------- hooks (2955 verbatim) ----------
    cap_v = {}
    cap_x = {}
    cap_in = {'on': False, 'store': {}}
    cap_xk = {'on': False, 'store': {}}
    fin_cap = {}
    state_fin = {'on': False}
    inj = {'li': None, 'scale': 0.0, 'vec': None}
    handles = []

    def pre_attn(li):
        def h(module, args, kwargs):
            x = args[0] if args else kwargs.get('hidden_states')
            if x is None or x.dim() < 2:
                return
            xuse = x
            if inj['li'] == li and inj['vec'] is not None:
                xuse = x.clone()
                xuse[:, 1, :] = xuse[:, 1, :] \
                    + inj['scale'] * inj['vec']
            if cap_xk['on'] and li in (17, 16):
                cap_xk['store'].setdefault(li, []).append(
                    (xuse[:, 0, :].detach().float()
                     .cpu().numpy(),
                     xuse[:, 1, :].detach().float()
                     .cpu().numpy()))
            if xuse is not x:
                if args:
                    return (xuse,) + tuple(args[1:]), kwargs
                nkw = dict(kwargs)
                nkw['hidden_states'] = xuse
                return args, nkw
            if cap_in['on']:
                cap_in['store'].setdefault(li, []).append(
                    x[:, 1, :].detach().float()
                    .cpu().numpy())
            return None
        return h

    def hook_v(li):
        def h(module, args, output):
            if li in (17, 16):
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
            if li in (17, 16):
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
    for li in (17, 16):
        handles.append(layers[li].self_attn.v_proj
                       .register_forward_hook(hook_v(li)))
        handles.append(layers[li].self_attn.o_proj
                       .register_forward_pre_hook(
                           hook_x(li), with_kwargs=True))
    handles.append(model.model.norm.register_forward_pre_hook(
        pre_norm, with_kwargs=True))

    # ---------- pass 1: dirs rebuild (a1/a3) ----------
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
    log('a1 dirs rebuild %.2e ok=%s' % (a1_diff, a1_ok), lines)
    _, _, Vt = np.linalg.svd(dirs_word, full_matrices=False)
    Vt8 = Vt[:8]
    a3_diff = float(np.abs(Vt8 - Vt8_39).max())
    a3_ok = bool(a3_diff < 1e-6)
    log('a3 Vt8 vs 2939 %.2e ok=%s' % (a3_diff, a3_ok), lines)
    u35 = dirs_word[NL - 1]
    xdir = dcks_39[:, [0, 1, 4]] @ Vt8[[0, 1, 4]]
    a7_diff = float(np.abs(
        xdir @ Vt8[[0, 1, 4]].T
        - dcks_39[:, [0, 1, 4]]).max())
    a7_ok = bool(a7_diff < 1e-9)
    log('a7 xdir self-check %.2e ok=%s' % (a7_diff, a7_ok),
        lines)
    xdir_t = torch.tensor(xdir, device='cuda',
                          dtype=torch.bfloat16)

    # ---------- forwards ----------
    NKV = 1024 // HD
    HPG = NH // NKV

    def forward_batch(scale=0.0, inj_li=None):
        cap_v.clear()
        cap_x.clear()
        cap_xk['store'].clear()
        fin_cap.pop('x', None)
        inj['li'] = inj_li
        inj['scale'] = float(scale)
        inj['vec'] = xdir_t if scale else None
        state_fin['on'] = True
        cap_xk['on'] = True
        with torch.no_grad():
            model(torch.tensor(batch, device='cuda'))
        inj['li'] = None
        inj['scale'] = 0.0
        inj['vec'] = None
        state_fin['on'] = False
        cap_xk['on'] = False
        fin = fin_cap['x'].astype(np.float64)
        v = {li: (np.stack([a for a, b in cap_v[li]]),
                  np.stack([b for a, b in cap_v[li]]))
             for li in cap_v}
        x = {li: np.stack(cap_x[li])[0] for li in cap_x}
        xk = {}
        for li, pairs in cap_xk['store'].items():
            a0 = np.stack([p[0] for p in pairs])[0]
            a1 = np.stack([p[1] for p in pairs])[0]
            xk[li] = (a0.astype(np.float64),
                      a1.astype(np.float64))
        return fin, v, x, xk

    fin_b1, v_base, x_base, xk_b = forward_batch()
    fin_b2, _, _, _ = forward_batch()
    a10_diff = float(np.abs(fin_b1 - fin_b2).max()
                     / max(float(np.abs(fin_b1).max()), 1e-30))
    a10_ok = bool(a10_diff < 1e-6)
    log('a10 base repeat rel %.2e ok=%s'
        % (a10_diff, a10_ok), lines)

    conds = [('L17_s%.2f' % s, 17, s) for s in GRID] \
        + [('L16_s2.0', 16, 2.0)]
    cap = {}
    for cname, li, s in conds:
        fin, vv, xx, xk = forward_batch(scale=s, inj_li=li)
        cap[cname] = {'fin': fin, 'v': vv[li], 'x': xx[li],
                      'xk': xk, 'li': li, 's': s}
    log('forward family done (base x2 + %d injections)'
        % len(conds), lines)

    # ---------- fp64 q/k recompute chain (2955 verbatim) ----------
    WQ = {}
    WK = {}
    QW = {}
    KW = {}
    BQ = {}
    for li in (17, 16):
        sa = layers[li].self_attn
        WQ[li] = sa.q_proj.weight.detach().float() \
            .cpu().numpy().astype(np.float64)
        WK[li] = sa.k_proj.weight.detach().float() \
            .cpu().numpy().astype(np.float64)
        QW[li] = sa.q_norm.weight.detach().float() \
            .cpu().numpy().astype(np.float64)
        KW[li] = sa.k_norm.weight.detach().float() \
            .cpu().numpy().astype(np.float64)
        BQ[li] = None
        if getattr(sa.q_proj, 'bias', None) is not None:
            BQ[li] = sa.q_proj.bias.detach().float() \
                .cpu().numpy().astype(np.float64)
    inv_freq = rope_theta ** (
        -np.arange(0, HD, 2, dtype=np.float64) / HD)

    def rope_rot(x, pos):
        ang = pos * inv_freq
        emb = np.concatenate([ang, ang])
        c = np.cos(emb)
        s = np.sin(emb)
        half = HD // 2
        x1h = x[..., :half]
        x2h = x[..., half:]
        return x * c \
            + np.concatenate([-x2h, x1h], axis=-1) * s

    def rmsn(x, w):
        v = (x * x).mean(-1, keepdims=True)
        return (x / np.sqrt(v + eps_norm)) * w

    def qk_of(li, x0, x1):
        q1 = x1 @ WQ[li].T
        if BQ[li] is not None:
            q1 = q1 + BQ[li]
        k1 = x1 @ WK[li].T
        k0 = x0 @ WK[li].T
        q1 = q1.reshape(n_words, NH, HD)
        k1 = k1.reshape(n_words, NKV, HD)
        k0 = k0.reshape(n_words, NKV, HD)
        q1 = rmsn(q1, QW[li])
        k1 = rmsn(k1, KW[li])
        k0 = rmsn(k0, KW[li])
        q1r = rope_rot(q1, 1.0)
        k1r = rope_rot(k1, 1.0)
        k0r = rope_rot(k0, 0.0)
        return q1r, k1r, k0r

    def sig(z):
        return 1.0 / (1.0 + np.exp(-z))

    HPIDX = np.repeat(np.arange(NKV), HPG)
    SD = float(np.sqrt(HD))

    Wo_cache = {li: layers[li].self_attn.o_proj.weight
                .detach().float().cpu().numpy()
                for li in (17, 16)}

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

    def sc_of(Xf, li):
        Wo = Wo_cache[li]
        Xh = Xf.reshape(n_words, NH, HD)
        c = np.zeros((NH, n_words))
        for hh in range(NH):
            oh = Xh[:, hh, :] \
                @ Wo[:, hh * HD:(hh + 1) * HD].T
            c[hh] = oh @ u35
        return c[:, lab_lang == 0].mean(1) \
            - c[:, lab_lang == 1].mean(1)

    A11b_rec = {}
    res_max = 0.0
    for li in (17, 16):
        v0b, v1b = v_base[li]
        A, r = recover(
            x_base[li].reshape(n_words, NH, HD),
            v0b.reshape(n_words, NKV, HD),
            v1b.reshape(n_words, NKV, HD))
        A11b_rec[li] = A
        res_max = max(res_max, r)
    a14_ok = bool(res_max < 0.3)
    a15_diff = 0.0
    for li in (17, 16):
        a15_diff = max(a15_diff, float(np.abs(
            A11b_rec[li] - z53['A11b_L%d' % li]).max()))
    a15_ok = bool(a15_diff < 1e-6)
    log('a14 recon residual %.2e ok=%s | a15 A11b vs 2953 '
        '%.2e ok=%s' % (res_max, a14_ok, a15_diff, a15_ok),
        lines)

    sc_b = {li: sc_of(x_base[li], li) for li in (17, 16)}
    sc_n = {}
    sep_new = {}
    for cname in cap:
        c = cap[cname]
        li = c['li']
        sc_n[cname] = sc_of(c['x'], li)
        p = c['fin'] @ u35
        sep_new[cname] = float(
            p[lab_lang == 0].mean() - p[lab_lang == 1].mean())

    # ---------- anchor a11 ----------
    a11_d = 0.0
    for cname, zk in (('L17_s1.00', 'L17'),
                      ('L16_s2.0', 'L16')):
        li = 17 if zk == 'L17' else 16
        d = sc_n[cname] - sc_b[li]
        a11_d = max(a11_d, float(np.abs(
            d - z52['delta_%s' % zk]).max()))
    a11_ok = bool(a11_d < 1e-6)
    i45 = layers45.index(17)
    hit = [j for j, s45 in enumerate(grid45)
           if abs(float(s45) - 0.5) < 1e-9][0]
    a13_diff = abs(sep_new['L17_s0.50']
                   - float(sep45[i45 * len(grid45) + hit]))
    a13_ok = bool(a13_diff < 0.05)
    log('a11 dsc vs 2952 %.2e ok=%s | a13 sep L17@0.5 vs 2945 '
        '%.2e ok=%s (%.1f)'
        % (a11_d, a11_ok, a13_diff, a13_ok,
           sep_new['L17_s0.50']), lines)

    # ---------- a-iso (informational) ----------
    iso_dx0_max = 0.0
    iso_dx1_L16_max = 0.0
    for cname, li, s in conds:
        x0n, x1n = cap[cname]['xk'][li]
        x0b, x1b = xk_b[li]
        iso_dx0_max = max(iso_dx0_max,
                          float(np.abs(x0n - x0b).max()))
        if li == 17:
            x0b16, x1b16 = xk_b[16]
            _, x1n16 = cap[cname]['xk'][16]
            iso_dx1_L16_max = max(iso_dx1_L16_max,
                                  float(np.abs(
                                      x1n16 - x1b16).max()))
    log('a-iso dx0 max %.2e | dx1@L16 for L17 doses %.2e'
        % (iso_dx0_max, iso_dx1_L16_max), lines)

    # ---------- decomposition + a16 + a18 ----------
    def A11_n_rec(cname, li):
        v0n, v1n = cap[cname]['v']
        A, r = recover(
            cap[cname]['x'].reshape(n_words, NH, HD),
            v0n.reshape(n_words, NKV, HD),
            v1n.reshape(n_words, NKV, HD))
        return A

    def a16_dmed(A_sm_mat, A_rec_mat):
        return float(np.median(
            np.abs(A_sm_mat - A_rec_mat)))

    a16_res = {}
    a16_dA_max = 0.0
    a18_d = 0.0
    dec = {}
    for li in (17, 16):
        x0b, x1b = xk_b[li]
        q1r, k1r, k0r = qk_of(li, x0b, x1b)
        dkh = (k1r - k0r)[:, HPIDX, :]
        zb = np.einsum('whd,whd->wh', q1r, dkh) / SD
        A_sm = sig(zb)
        d_med = a16_dmed(A_sm, A11b_rec[li])
        a16_res['base_L%d' % li] = round(d_med, 5)
        a16_dA_max = max(a16_dA_max, d_med)
        dec['base_L%d' % li] = {
            'zb': zb, 'A_sm': A_sm, 'q': q1r,
            'dk': k1r - k0r}
    for cname, li, s in conds:
        x0n, x1n = cap[cname]['xk'][li]
        q1n, k1n, k0n = qk_of(li, x0n, x1n)
        dkh_n = (k1n - k0n)[:, HPIDX, :]
        zn = np.einsum('whd,whd->wh', q1n, dkh_n) / SD
        A_smn = sig(zn)
        d_med = a16_dmed(A_smn, A11_n_rec(cname, li))
        a16_res[cname] = round(d_med, 5)
        a16_dA_max = max(a16_dA_max, d_med)
        bkey = 'base_L%d' % li
        zb = dec[bkey]['zb']
        dkh_b = dec[bkey]['dk'][:, HPIDX, :]
        dq = q1n - dec[bkey]['q']
        ddk = ((k1n - k0n) - dec[bkey]['dk'])[:, HPIDX, :]
        qt = np.einsum('whd,whd->wh', dq, dkh_b) / SD
        kt = np.einsum('whd,whd->wh',
                       dec[bkey]['q'], ddk) / SD
        xt = np.einsum('whd,whd->wh', dq, ddk) / SD
        dz = zn - zb
        a18_d = max(a18_d, float(np.abs(
            (qt + kt + xt) - dz).max()))
        dec[cname] = {'zn': zn, 'zb': zb, 'A_sm': A_smn,
                      'dz': dz, 'qt': qt, 'kt': kt, 'xt': xt,
                      'dq': dq, 'ddk': ddk}

    a16_ok = bool(a16_dA_max < 0.05)
    a18_ok = bool(a18_d < 1e-6)
    log('a16 dA_med max %.2e (<0.05 ok=%s) | a18 decomp '
        'identity %.2e ok=%s'
        % (a16_dA_max, a16_ok, a18_d, a18_ok), lines)

    # ---------- anchor a17: vs 2955 npz bit-level ----------
    a17_d = 0.0
    for key55, key59 in (
            ('dz_full_L17_s1.0', 'L17_s1.00'),
            ('dz_full_L17_s0.5', 'L17_s0.50'),
            ('dz_full_L16_s2.0', 'L16_s2.0')):
        a17_d = max(a17_d, float(np.abs(
            dec[key59]['dz'] - z55[key55]).max()))
    for key55, key59 in (
            ('qt_L17_s1.0', 'L17_s1.00'),
            ('kt_L17_s1.0', 'L17_s1.00'),
            ('xt_L17_s1.0', 'L17_s1.00'),
            ('qt_L16_s2.0', 'L16_s2.0'),
            ('kt_L16_s2.0', 'L16_s2.0'),
            ('xt_L16_s2.0', 'L16_s2.0')):
        a17_d = max(a17_d, float(np.abs(
            dec[key59][key55[:2]] - z55[key55]).max()))
    for key55, key59 in (
            ('zb_full_L17', 'base_L17'),
            ('zb_full_L16', 'base_L16')):
        a17_d = max(a17_d, float(np.abs(
            dec[key59]['zb'] - z55[key55]).max()))
    for li in (17, 16):
        a17_d = max(a17_d, float(np.abs(
            dec['base_L%d' % li]['A_sm']
            - z55['A11sm_b_L%d' % li]).max()))
    a17_ok = bool(a17_d < 1e-6)
    log('a17 vs 2955 npz %.2e ok=%s' % (a17_d, a17_ok), lines)

    anchor_prelim = bool(a1_ok and a10_ok and a3_ok and a7_ok
                         and a9_ok and a11_ok and a13_ok
                         and a14_ok and a15_ok and a16_ok
                         and a17_ok and a18_ok)

    verdict = None
    t1 = t2 = t3 = d1 = None
    save = {}
    if not anchor_prelim:
        verdict = 'anchor_fail_all_void'
    else:
        # ---------- T1: scaling axis ----------
        xt_med = {}
        for s in GRID:
            cn = 'L17_s%.2f' % s
            xt_med[s] = np.median(np.abs(dec[cn]['xt']),
                                  axis=0)
        slopes = []
        r2s = []
        used_heads = []
        for hh in range(NH):
            if xt_med[1.0][hh] < XT_MIN:
                continue
            y = np.log(np.array([max(xt_med[s][hh], 1e-12)
                                 for s in GRID]))
            xg = np.log(np.array(GRID))
            b1, b0 = np.polyfit(xg, y, 1)
            yhat = b1 * xg + b0
            ss_res = float(((y - yhat) ** 2).sum())
            ss_tot = float(((y - y.mean()) ** 2).sum())
            slopes.append(float(b1))
            r2s.append(1.0 - ss_res / max(ss_tot, 1e-30))
            used_heads.append(hh)
        slopes = np.array(slopes)
        slope_med = float(np.median(slopes))
        t1r = 'second_order' \
            if SLOPE_LO <= slope_med <= SLOPE_HI \
            else 'anomalous_slope'
        t1 = {'slope_median': round(slope_med, 4),
              'slope_q25': round(float(np.quantile(
                  slopes, 0.25)), 4),
              'slope_q75': round(float(np.quantile(
                  slopes, 0.75)), 4),
              'r2_median': round(float(np.median(r2s)), 4),
              'n_heads_used': len(used_heads),
              'used_heads': used_heads,
              'verdict': t1r}
        log('T1 slope median %.4f [%.4f, %.4f] r2med %.4f '
            'n=%d -> %s'
            % (slope_med, t1['slope_q25'],
               t1['slope_q75'], t1['r2_median'],
               len(used_heads), t1r), lines)

        # ---------- T2: direction axis ----------
        cn05 = dec['L17_s0.50']
        cn10 = dec['L17_s1.00']
        qb = dec['base_L17']['q']
        dkb = dec['base_L17']['dk'][:, HPIDX, :]

        def dir_lock(dA, dB):
            num = (dA * dB).sum(-1)
            den = np.sqrt((dA * dA).sum(-1)
                          * (dB * dB).sum(-1))
            ok = den > 1e-6
            cos = np.where(ok, num / np.where(ok, den, 1.0),
                           np.nan)
            return np.nanmedian(cos, axis=0)

        cos_dq = dir_lock(cn05['dq'], cn10['dq'])
        cos_ddk = dir_lock(cn05['ddk'], cn10['ddk'])
        cos_dq_m = float(np.nanmedian(cos_dq))
        cos_ddk_m = float(np.nanmedian(cos_ddk))
        t2r = 'direction_locked' \
            if cos_dq_m >= COS_TH and cos_ddk_m >= COS_TH \
            else 'direction_drift'
        t2 = {'cos_dq_median': round(cos_dq_m, 4),
              'cos_ddk_median': round(cos_ddk_m, 4),
              'cos_dq_min_head': round(
                  float(np.nanmin(cos_dq)), 4),
              'cos_ddk_min_head': round(
                  float(np.nanmin(cos_ddk)), 4),
              'verdict': t2r}
        log('T2 cos(dq) med %.4f min %.4f | cos(ddk) med %.4f '
            'min %.4f -> %s'
            % (cos_dq_m, t2['cos_dq_min_head'], cos_ddk_m,
               t2['cos_ddk_min_head'], t2r), lines)

        # ---------- T3: operational prediction ----------
        xt025 = dec['L17_s0.25']['xt']
        xt10 = dec['L17_s1.00']['xt']
        xt_pred = 16.0 * xt025
        rel_w = np.abs(xt10 - xt_pred) / np.maximum(
            np.abs(np.median(np.abs(xt10), axis=0,
                             keepdims=True)), 1e-30)
        rel_h = np.median(rel_w, axis=0)
        rho_pred = spearman(
            np.median(xt10, axis=0),
            np.median(xt_pred, axis=0))
        rel_med = float(np.median(rel_h))
        t3r = 'predictable_small_dose' \
            if rel_med < PRED_REL and rho_pred >= PRED_RHO \
            else 'not_predictable'
        t3 = {'rel_err_median': round(rel_med, 4),
              'rel_err_q75': round(float(np.quantile(
                  rel_h, 0.75)), 4),
              'rho_pred': round(rho_pred, 4),
              'verdict': t3r}
        log('T3 pred rel err med %.4f q75 %.4f rho %.4f -> %s'
            % (rel_med, t3['rel_err_q75'], rho_pred, t3r),
            lines)

        # ---------- D1 descriptive ----------
        dd = dec['L17_s1.00']
        Mq = np.median(np.abs(dd['qt']), axis=0)
        Mk = np.median(np.abs(dd['kt']), axis=0)
        Mx = np.median(np.abs(dd['xt']), axis=0)
        att17 = z52['ATT_L17'].astype(np.float64)
        d1 = {
            'axis_L16_s2.0': {
                'Q': round(float(np.median(np.median(
                    np.abs(dec['L16_s2.0']['qt']),
                    axis=0))), 4),
                'K': round(float(np.median(np.median(
                    np.abs(dec['L16_s2.0']['kt']),
                    axis=0))), 4),
                'X': round(float(np.median(np.median(
                    np.abs(dec['L16_s2.0']['xt']),
                    axis=0))), 4)},
            'rho_xt_att': round(spearman(Mx, att17), 4),
            'rho_xt_dz': round(spearman(
                Mx, np.median(np.abs(dd['dz']),
                              axis=0)), 4),
            'xt_med_grid_L17': {
                '%.2f' % s: [round(float(v), 4)
                             for v in xt_med[s]]
                for s in GRID}}

        save = {
            'words': np.array(['%s:%s:%s' % w
                               for w in words],
                              dtype=object),
            'labels_lang': lab_lang,
            'grid': np.array(GRID),
            'xt_L17_grid': np.stack(
                [dec['L17_s%.2f' % s]['xt'] for s in GRID]),
            'qt_L17_s1.0': dec['L17_s1.00']['qt'],
            'kt_L17_s1.0': dec['L17_s1.00']['kt'],
            'dz_L17_s1.0': dec['L17_s1.00']['dz'],
            'xt_L16_s2.0': dec['L16_s2.0']['xt'],
            'dz_L16_s2.0': dec['L16_s2.0']['dz'],
            'cos_dq': cos_dq, 'cos_ddk': cos_ddk,
            'slopes': slopes,
            'rel_h': rel_h,
        }

    if not anchor_prelim:
        verdict = 'anchor_fail_all_void'
    else:
        verdict = '%s_%s_%s' % (t1r, t2r, t3r)
    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2959, 'model': 'qwen3-4b',
           'prereg': PREREG,
           'anchors': {
               'a1_diff': float('%.3e' % a1_diff),
               'a1_ok': a1_ok,
               'a10_rel': float('%.3e' % a10_diff),
               'a10_ok': a10_ok,
               'a3_diff': float('%.3e' % a3_diff),
               'a3_ok': a3_ok,
               'a7_diff': float('%.3e' % a7_diff),
               'a7_ok': a7_ok,
               'a9_ok': a9_ok,
               'a11_dsc': float('%.3e' % a11_d),
               'a11_ok': a11_ok,
               'a13_diff': float('%.3e' % a13_diff),
               'a13_ok': a13_ok,
               'a14_diff': float('%.3e' % res_max),
               'a14_ok': a14_ok,
               'a15_diff': float('%.3e' % a15_diff),
               'a15_ok': a15_ok,
               'a16_dA_max': float('%.3e' % a16_dA_max),
               'a16_res': a16_res,
               'a16_ok': a16_ok,
               'a17_diff': float('%.3e' % a17_d),
               'a17_ok': a17_ok,
               'a18_diff': float('%.3e' % a18_d),
               'a18_ok': a18_ok,
               'a_iso_dx0': float('%.3e' % iso_dx0_max),
               'a_iso_dx1_L16': float('%.3e'
                                      % iso_dx1_L16_max),
               'ok': bool(anchor_prelim)},
           'T1_scaling': t1, 'T2_direction': t2,
           'T3_prediction': t3, 'D1_descriptive': d1,
           'sep_conditions': {k: round(v, 2)
                              for k, v in sep_new.items()},
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if save:
        np.savez_compressed(os.path.join(
            OUT, 'cross_term_algebra.npz'), **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2959 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
