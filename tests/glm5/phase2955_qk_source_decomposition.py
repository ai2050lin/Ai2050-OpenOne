# -*- coding: utf-8 -*-
"""Phase 2955: q.k source decomposition - is the A11 routing
gain driven by the injection's direct modification of q/k
logits, and how much of the A-domain x7-10 gain is softmax
steep-region amplification?

Why: 2952 attributed the beta~1.5x readout amplification to the
attention self-weight gain (A11 x7-10, ATT 90-93%), but the
ORIGIN of that gain was left open: (i) direct q/k modification
by the injected residual, vs (ii) softmax nonlinearity gain
(small logit moves in the steep region of sigma(z)).

Design (one forward family, 2954 verbatim scaffolding):
  base x2 + L17@s1.0 + L17@s0.5 + L16@s2.0
  captures per condition at the dose layer (self_attn pre-hook,
  post-injection): layer-input residual pos0+pos1; plus 2954
  captures (o_proj input pos1, v_proj pos0/pos1, final-norm).
  Recompute chain per layer/condition (fp64, validated by
  anchor a16): q1 = q_norm(Wq x1 + bq), k_p = k_norm(Wk x_p),
  RoPE at pos p, z = q~ . (k~1 - k~0), A11_sm = sigma(z).
  Exact logit decomposition per word/head:
    dz = dq~.dk~_b + q~_b.ddk~ + dq~.ddk~
    (q-term / k-term / cross-term; ddk~ = d(k~1-k~0))

Head-set口径 (lesson 24): ALL 32 heads, medians over 57 words.

Tests (frozen):
  T1 source axis (per dose layer, then both layers):
     Q = median_h median_w |q-term|; K, X analog.
     q_direct if Q >= 2*max(K,X); k_direct if K >= 2*max(Q,X);
     else qk_mixed. Verdict uses both dose layers (agree else
     qk_mixed).
  T2 regime axis (per dose layer):
     softmax_gain if median_h median_w |z_b| < 1.0 AND
     median_h median_w |dz| < 1.5; large_logit if
     median_h median_w |dz| >= 1.5; else mixed_regime.
     Verdict uses both dose layers (agree else mixed_regime).
  D1 predictive (descriptive): spearman(median_w |dz|_h at
     L17@s1.0, ATT_h from 2952) with perm null 20000 seed 2905;
     significant AND rho >= 0.6 => logit movement predicts the
     attention-gain carry.

Anchors (frozen):
  a1 dirs rebuild vs 2927 < 1e-5
  a2/a10 base repeat rel < 1e-6
  a3 Vt8 vs 2939 < 1e-6
  a7 xdir self-check < 1e-9
  a9 structure gates (v 1024 / o 4096 / q 4096 / k 1024 /
     q_norm,k_norm weight shape 128)
  a11 dsc vs 2952 delta < 1e-6 (L17@s1.0 / L16@s2.0)
  a13 sep L17@0.5 vs 2945 < 0.05
  a14 A11 line-recovery residual < 0.3
  a15 A11b (recover) vs 2953 npz < 1e-6
  a16 RoPE/q_norm chain validation: median |A11_sm - A11_rec|
      < 0.05 (base L17/L16 + dose layers)
  a18 decomposition identity |q+k+x - dz| max < 1e-6
  a-iso (informational): delta-x pos0 (all doses) and delta-x
      pos1 at L16 for L17 doses - causal isolation; NOT gating.

Verdict (frozen):
  anchor fail => anchor_fail_all_void
  else f'{source_all}_{regime_all}' with
  source_all in {q_direct, k_direct, qk_mixed} and
  regime_all in {softmax_gain, large_logit, mixed_regime}.
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
OUT = os.path.join(BASE, 'phase2955', 'qk_source_decomposition')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2955_run_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NH, HD = 32, 128
NL = 36
SEED = 2905
N_PERM = 20000
ZB_TH = 1.0
DZ_SOFT = 1.5
DOM_RATIO = 2.0

PREREG = {
    'mode': 'one forward family (base x2 + L17@s1.0 + L17@s0.5 '
            '+ L16@s2.0); layer-input residual pos0+pos1 capture '
            'at dose layer (self_attn pre-hook, post-injection); '
            'fp64 recompute of q/k (q_norm/k_norm + RoPE) '
            'validated against line-recovered A11 (a16); exact '
            'logit decomposition dz = dq.dk + q.ddk + dq.ddk',
    'question': 'is the A11 routing gain (2952, x7-10) driven '
                'by the injection direct q/k modification or by '
                'softmax steep-region gain?',
    'head_set': 'ALL 32 heads (lesson 24口径 registered), '
                'medians over 57 words',
    'anchors': {
        'a1': 'dirs rebuild < 1e-5',
        'a2/a10': 'base repeat rel < 1e-6',
        'a3': 'Vt8 < 1e-6', 'a7': 'xdir self-check < 1e-9',
        'a9': 'structure gates',
        'a11': 'dsc vs 2952 delta < 1e-6',
        'a13': 'sep L17@0.5 vs 2945 < 0.05',
        'a14': 'A11 recon residual < 0.3',
        'a15': 'A11b vs 2953 < 1e-6',
        'a16': 'recompute-chain validation: max over 5 '
               'cond-layer combos of median |A11_sm - A11_rec| '
               '< 0.05 (discriminative: wrong chain 0.033-'
               '0.062 in run2 vs correct chain ~8e-4); '
               'output-space ratio r_sm/r_rec registered '
               'descriptively',
        'a18': '|q+k+x - dz| max < 1e-6',
        'a-iso': 'causal isolation (informational, not gating)',
    },
    'T1': 'source axis: Q/K/X = median_h median_w |term|; '
          'q_direct if Q >= 2*max(K,X); k_direct if K >= '
          '2*max(Q,X); else qk_mixed; both dose layers agree '
          'else qk_mixed',
    'T2': 'regime axis: softmax_gain if med|z_b| < 1.0 and '
          'med|dz| < 1.5; large_logit if med|dz| >= 1.5; else '
          'mixed_regime; both dose layers agree else '
          'mixed_regime',
    'D1': 'spearman(med|dz|_h L17@s1.0, ATT_h 2952), perm null '
          '20000 seed 2905; rho >= 0.6 and significant => '
          'coupled (descriptive)',
    'verdict': 'anchor fail => anchor_fail_all_void; else '
               '{source_all}_{regime_all}',
    'correction_note': 'run1 crashed at base softmax A11: z '
                       'einsum emitted the KV axis (57,32,8) '
                       'instead of per-head expansion - GQA '
                       'head h uses KV head h//4, all z/qt/kt/'
                       'xt einsums must expand dk/ddk to the '
                       'head axis first (HPIDX). No results '
                       'were computed in run1 (crash before '
                       'a16); execution.json deleted per '
                       'discipline 3. run2 hit anchor a16 '
                       '(0.062 > 0.05): the criterion median '
                       '|A11_sm - A11_rec| conflates recompute-'
                       'chain error with LS-recovery noise '
                       '(recovered A11 carries noise ~ eta/'
                       '|v1-v0| from bf16 rounding; residuals '
                       'were uniform 0.033-0.062 across all 5 '
                       'cond-layer combos, no concentration). '
                       'Discipline-10-family criterion redesign, '
                       'frozen before rerun: a16 v3 = output-'
                       'space optimality ratio r_sm/r_rec < 2.0 '
                       '(a chain error would inflate the ratio '
                       'far above 2; a correct chain sits at '
                       '1-1.5). a18 identity and all other '
                       'anchors passed in run2 and are '
                       'deterministic; run3 is authoritative. '
                       'run3: a16 v3 ratio 318.7 - the anchor '
                       'caught a REAL chain error: the softmax '
                       'scale 1/sqrt(HD) was missing (logits = '
                       'q.k/sqrt(128), I used sigma(q.k)). '
                       'run4 divides z and all three '
                       'decomposition terms by sqrt(HD) (T1 '
                       'source ratios are scale-invariant; T2 '
                       'thresholds stay at their frozen values, '
                       'now on the correctly-scaled logits). '
                       'run4: ratio fell 318.7 -> 5.82 and '
                       'dA_med fell to 8e-4 - the REAL bug '
                       '(missing softmax scale) was fixed; the '
                       'v3 ratio threshold 2.0 was mis-'
                       'calibrated: per-word-per-head LS '
                       'recovery (1 param, 128-dim output) '
                       'overfits the bf16 noise floor (r_rec '
                       '3e-5), so no fixed ratio threshold is '
                       'meaningful. a16 v4 (frozen): max '
                       'dA_med < 0.05 - the original run2 '
                       'criterion, now demonstrably reachable '
                       '(8e-4, 60x margin) and discriminative '
                       '(0.062 wrong chain vs 8e-4 correct); '
                       'ratio kept descriptive. run5 is '
                       'authoritative.',
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
        json.dump({'phase': 2955,
                   'name': 'qk_source_decomposition',
                   'created':
                       time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2887': sha8(SRC_2887),
                               's2927': sha8(SRC_2927),
                               's2939': sha8(SRC_2939),
                               's2945': sha8(SRC_2945),
                               's2952': sha8(SRC_2952),
                               's2953': sha8(SRC_2953)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'seed': SEED, 'n_perm': N_PERM,
                   'zb_th': ZB_TH, 'dz_soft': DZ_SOFT,
                   'dom_ratio': DOM_RATIO,
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

    # ---------- hooks ----------
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
    log('a2/a10 base repeat rel %.2e ok=%s'
        % (a10_diff, a10_ok), lines)

    conds = [('L17_s1.0', 17, 1.0), ('L17_s0.5', 17, 0.5),
             ('L16_s2.0', 16, 2.0)]
    cap = {}
    for cname, li, s in conds:
        fin, vv, xx, xk = forward_batch(scale=s, inj_li=li)
        cap[cname] = {'fin': fin, 'v': vv[li], 'x': xx[li],
                      'xk': xk, 'li': li, 's': s}
    log('forward family done (base + 3 injections)', lines)

    # ---------- fp64 q/k recompute chain ----------
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

    # ---------- Wo cache + recovery (2954 verbatim) ----------
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
    a15_diff = 0.0
    for li in (17, 16):
        a15_diff = max(a15_diff, float(np.abs(
            A11b_rec[li] - z53['A11b_L%d' % li]).max()))
    a15_ok = bool(a15_diff < 1e-6)

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
    a14_ok = bool(res_max < 0.3)
    log('a14 recon residual %.2e ok=%s | a15 A11b vs 2953 '
        '%.2e ok=%s' % (res_max, a14_ok, a15_diff, a15_ok),
        lines)

    # ---------- anchors a11 / a13 ----------
    a11_d = 0.0
    for cname, zk in (('L17_s1.0', 'L17'), ('L16_s2.0', 'L16')):
        li = 17 if zk == 'L17' else 16
        d = sc_n[cname] - sc_b[li]
        a11_d = max(a11_d, float(np.abs(
            d - z52['delta_%s' % zk]).max()))
    a11_ok = bool(a11_d < 1e-6)
    i45 = layers45.index(17)
    hit = [j for j, s45 in enumerate(grid45)
           if abs(float(s45) - 0.5) < 1e-9][0]
    a13_diff = abs(sep_new['L17_s0.5']
                   - float(sep45[i45 * len(grid45) + hit]))
    a13_ok = bool(a13_diff < 0.05)
    log('a11 dsc vs 2952 %.2e ok=%s | a13 sep L17@0.5 vs 2945 '
        '%.2e ok=%s (%.1f)'
        % (a11_d, a11_ok, a13_diff, a13_ok,
           sep_new['L17_s0.5']), lines)

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

    # ---------- decomposition ----------
    def A11_n_rec(cname, li):
        v0n, v1n = cap[cname]['v']
        A, r = recover(
            cap[cname]['x'].reshape(n_words, NH, HD),
            v0n.reshape(n_words, NKV, HD),
            v1n.reshape(n_words, NKV, HD))
        return A

    def a16_stats(A_sm_mat, A_rec_mat, out_act, vcap):
        v0, v1 = vcap
        v0h = v0.reshape(n_words, NKV, HD)[:, HPIDX, :]
        v1h = v1.reshape(n_words, NKV, HD)[:, HPIDX, :]
        dkh = v1h - v0h
        out_sm = v0h + A_sm_mat[:, :, None] * dkh
        out_rec = v0h + A_rec_mat[:, :, None] * dkh
        r_sm = float(np.median(np.abs(out_sm - out_act)))
        r_rec = float(np.median(np.abs(out_rec - out_act)))
        d_med = float(np.median(
            np.abs(A_sm_mat - A_rec_mat)))
        return r_sm, r_rec, r_sm / max(r_rec, 1e-30), d_med

    a16_res = {}
    a16_ratio_max = 0.0
    a16_dA_max = 0.0
    a18_d = 0.0
    dec = {}
    # base softmax A11
    for li in (17, 16):
        x0b, x1b = xk_b[li]
        q1r, k1r, k0r = qk_of(li, x0b, x1b)
        dkh = (k1r - k0r)[:, HPIDX, :]
        zb = np.einsum('whd,whd->wh', q1r, dkh) / SD
        A_sm = sig(zb)
        r_sm, r_rec, ratio, d_med = a16_stats(
            A_sm, A11b_rec[li],
            x_base[li].reshape(n_words, NH, HD),
            v_base[li])
        a16_res['base_L%d' % li] = {
            'r_sm': round(r_sm, 5), 'r_rec': round(r_rec, 5),
            'ratio': round(ratio, 4),
            'dA_med': round(d_med, 5)}
        a16_ratio_max = max(a16_ratio_max, ratio)
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
        r_sm, r_rec, ratio, d_med = a16_stats(
            A_smn, A11_n_rec(cname, li),
            cap[cname]['x'].reshape(n_words, NH, HD),
            cap[cname]['v'])
        a16_res[cname] = {
            'r_sm': round(r_sm, 5), 'r_rec': round(r_rec, 5),
            'ratio': round(ratio, 4),
            'dA_med': round(d_med, 5)}
        a16_ratio_max = max(a16_ratio_max, ratio)
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
                      'dz': dz, 'qt': qt, 'kt': kt, 'xt': xt}

    a16_ok = bool(a16_dA_max < 0.05)
    a18_ok = bool(a18_d < 1e-6)
    log('a16 dA_med max %.2e (<0.05 ok=%s) | ratio max %.2f '
        '(descriptive) | a18 decomp identity %.2e ok=%s'
        % (a16_dA_max, a16_ok, a16_ratio_max, a18_d,
           a18_ok), lines)

    anchor_prelim = bool(a1_ok and a10_ok and a3_ok and a7_ok
                         and a9_ok and a11_ok and a13_ok
                         and a14_ok and a15_ok and a16_ok
                         and a18_ok)

    verdict = None
    t1 = t2 = d1 = d2 = None
    save = {}
    if not anchor_prelim:
        verdict = 'anchor_fail_all_void'
    else:
        t1 = {}
        t2 = {}
        axis_src = {}
        axis_reg = {}
        for cname, li, s in conds:
            if cname == 'L17_s0.5':
                continue
            key = 'L%d@s%.1f' % (li, s)
            dd = dec[cname]
            Mq = np.median(np.abs(dd['qt']), axis=0)
            Mk = np.median(np.abs(dd['kt']), axis=0)
            Mx = np.median(np.abs(dd['xt']), axis=0)
            Q = float(np.median(Mq))
            K = float(np.median(Mk))
            X = float(np.median(Mx))
            if Q >= DOM_RATIO * max(K, X):
                src = 'q_direct'
            elif K >= DOM_RATIO * max(Q, X):
                src = 'k_direct'
            else:
                src = 'qk_mixed'
            Zm = float(np.median(np.median(
                np.abs(dd['zb']), axis=0)))
            Dm = float(np.median(np.median(
                np.abs(dd['dz']), axis=0)))
            if Dm >= DZ_SOFT:
                reg = 'large_logit'
            elif Zm < ZB_TH:
                reg = 'softmax_gain'
            else:
                reg = 'mixed_regime'
            dz_med = np.median(np.abs(dd['dz']), axis=0)
            zb_med = np.median(np.abs(dd['zb']), axis=0)
            t1[key] = {'Q': round(Q, 4), 'K': round(K, 4),
                       'X': round(X, 4), 'src': src}
            t2[key] = {'med_abs_zb': round(Zm, 4),
                       'med_abs_dz': round(Dm, 4),
                       'regime': reg}
            axis_src[key] = src
            axis_reg[key] = reg
            log('%s T1 Q/K/X %.4f/%.4f/%.4f -> %s | T2 '
                'med|zb| %.3f med|dz| %.3f -> %s'
                % (key, Q, K, X, src, Zm, Dm, reg), lines)
        srcs = set(axis_src.values())
        src_all = axis_src['L17@s1.0'] \
            if len(srcs) == 1 else 'qk_mixed'
        regs = set(axis_reg.values())
        reg_all = axis_reg['L17@s1.0'] \
            if len(regs) == 1 else 'mixed_regime'
        verdict = '%s_%s' % (src_all, reg_all)

        # D1: |dz| vs ATT (2952)
        dd17 = dec['L17_s1.0']
        dz_med17 = np.median(np.abs(dd17['dz']), axis=0)
        att17 = z52['ATT_L17'].astype(np.float64)
        rho = spearman(dz_med17, att17)
        rng = np.random.default_rng(SEED)
        null = np.array([abs(spearman(
            rng.permutation(dz_med17), att17))
            for _ in range(N_PERM)])
        p95 = float(np.quantile(null, 0.95))
        d1 = {'rho_dz_att': round(rho, 4),
              'null_p95': round(p95, 4),
              'significant': bool(abs(rho) >= p95),
              'coupled': bool(abs(rho) >= p95
                              and abs(rho) >= 0.6)}
        log('D1 spearman(med|dz|, ATT_L17) = %.4f '
            '(p95 %.4f, coupled %s)'
            % (rho, p95, d1['coupled']), lines)

        # D2: per-head table L17@s1.0, top ATT + early flippers
        dd = dec['L17_s1.0']
        Mq = np.median(np.abs(dd['qt']), axis=0)
        Mk = np.median(np.abs(dd['kt']), axis=0)
        Mx = np.median(np.abs(dd['xt']), axis=0)
        zb_med_h = np.median(dd['zb'], axis=0)
        dz_med_h = np.median(dd['dz'], axis=0)
        A_b_med = np.median(dec['base_L17']['A_sm'], axis=0)
        A_n_med = np.median(dd['A_sm'], axis=0)
        top_att = [int(h) for h in np.argsort(
            -np.abs(att17))[:5]]
        rows = []
        for hh in sorted(set(top_att) | {20, 21}):
            rows.append({
                'head': hh,
                'ATT': round(float(att17[hh]), 3),
                'z_b': round(float(zb_med_h[hh]), 3),
                'dz': round(float(dz_med_h[hh]), 3),
                'Mq': round(float(Mq[hh]), 3),
                'Mk': round(float(Mk[hh]), 3),
                'Mx': round(float(Mx[hh]), 3),
                'A11_b': round(float(A_b_med[hh]), 4),
                'A11_n': round(float(A_n_med[hh]), 4)})
        d2 = {'table': rows}

        # s0.5 descriptive decomposition
        dd05 = dec['L17_s0.5']
        d2['L17_s0.5_axis'] = {
            'Q': round(float(np.median(np.median(
                np.abs(dd05['qt']), axis=0))), 4),
            'K': round(float(np.median(np.median(
                np.abs(dd05['kt']), axis=0))), 4),
            'X': round(float(np.median(np.median(
                np.abs(dd05['xt']), axis=0))), 4),
            'med_abs_dz': round(float(np.median(np.median(
                np.abs(dd05['dz']), axis=0))), 4)}

        save = {
            'words': np.array(['%s:%s:%s' % w
                               for w in words],
                              dtype=object),
            'labels_lang': lab_lang,
            'dz_full_L17_s1.0': dec['L17_s1.0']['dz'],
            'dz_full_L17_s0.5': dec['L17_s0.5']['dz'],
            'dz_full_L16_s2.0': dec['L16_s2.0']['dz'],
            'zb_full_L17': dec['base_L17']['zb'],
            'zb_full_L16': dec['base_L16']['zb'],
            'qt_L17_s1.0': dec['L17_s1.0']['qt'],
            'kt_L17_s1.0': dec['L17_s1.0']['kt'],
            'xt_L17_s1.0': dec['L17_s1.0']['xt'],
            'qt_L16_s2.0': dec['L16_s2.0']['qt'],
            'kt_L16_s2.0': dec['L16_s2.0']['kt'],
            'xt_L16_s2.0': dec['L16_s2.0']['xt'],
            'A11sm_b_L17': dec['base_L17']['A_sm'],
            'A11sm_b_L16': dec['base_L16']['A_sm'],
            'A11sm_L17_s1.0': dec['L17_s1.0']['A_sm'],
            'A11sm_L16_s2.0': dec['L16_s2.0']['A_sm'],
            'A11b_rec_L17': A11b_rec[17],
            'A11b_rec_L16': A11b_rec[16],
            'sc_b_L17': sc_b[17], 'sc_b_L16': sc_b[16],
        }

    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2955, 'model': 'qwen3-4b',
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
               'a16_ratio_max': float('%.4f'
                                      % a16_ratio_max),
               'a16_dA_max': float('%.3e' % a16_dA_max),
               'a16_res': a16_res,
               'a16_ok': a16_ok,
               'a18_diff': float('%.3e' % a18_d),
               'a18_ok': a18_ok,
               'a_iso_dx0': float('%.3e' % iso_dx0_max),
               'a_iso_dx1_L16': float('%.3e'
                                      % iso_dx1_L16_max),
               'ok': bool(anchor_prelim)},
           'T1_source': t1, 'T2_regime': t2,
           'D1_dz_vs_att': d1, 'D2_tables': d2,
           'sep_conditions': {k: round(v, 2)
                              for k, v in sep_new.items()},
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if save:
        np.savez_compressed(os.path.join(
            OUT, 'qk_source_decomposition.npz'), **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2955 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
