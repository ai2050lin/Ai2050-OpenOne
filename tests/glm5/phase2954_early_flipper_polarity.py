# -*- coding: utf-8 -*-
"""Phase 2954: early-flipper polarity - why the earliest routing
flippers are the 2947 resistance heads.

Why: 2953 found h20/h21 flip their routing by s=0.5 (A11 ~0.96)
yet 2947 classified them as resistance heads (ablating them
deepens collapse, D_h = -17.4/-14.0).  Candidate resolution:
their readout of the injected word value is POSITIVE in sep
terms (they resist collapse despite reading the injected word).
This phase aligns four per-head quantities on the same head set
(discipline 24: register the head-set口径):
  D_h     ablation differential (2947 npz, K=3 median,
          L17@s1.0 / L16@s2.0, o_proj-input slicing)
  dsc_h   snapshot linear change sc_h(inj) - sc_h(base)
          (this phase, 2952 sc_of verbatim)
  ATT/VAL attention-gain / value term of dsc (2952 identity)
  f_h     flip earliness = median A11_h over the early window
          s in {0.5, 0.625, 0.75} (2953 npz, all 32 heads)

Mode: zero-forward alignment + one forward family
(base x2 + L17@s1.0 + L17@s0.5 + L16@s2.0; captures o_proj
input pos1, v_proj pos0/pos1, final-norm input).

Sets (frozen rule = 2947's own |D|>2 mapping threshold):
  resist(layer)  = {h : D_h < -2}
  promote(layer) = {h : D_h > +2}

Anchors (frozen):
  a1 dirs_word rebuild vs 2927 < 1e-5
  a2 func baseline determinism rel < 1e-4
  a3 Vt8 rebuild vs 2939 < 1e-6
  a7 injection construction self-check < 1e-9
  a9 GQA gates (v_proj out 1024, o_proj in 4096)
  a10 same-session base repeat < 1e-6
  a11 dsc vs 2952 delta (s=1.0 L17 / s=2.0 L16, all heads)
      < 1e-6; identity VAL+ATT vs dsc < 1e-3 (fp bound, 2952
      a7 tolerance); ATT/VAL recompute vs 2952 < 1e-6
  a12 sc_inj vs 2950 sc_I0 (keep heads) < 1e-5
  a13 sep at L17@0.5 vs 2945 sep_curves < 0.05
  a14 A11 line-recovery residual < 0.3 (all conditions)
  a15 A11b vs 2953 npz A11b (all heads) < 1e-6

Main tests (frozen):
  T1 polarity: for EVERY h in resist(layer), dsc_h > 0 at the
     layer's standard dose (L17@s1.0, L16@s2.0) - BOTH layers.
  T2 attention carry: for EVERY h in resist(layer), ATT_h > 0
     AND |ATT|/(|VAL|+|ATT|) > 0.6 - BOTH layers.
  T3 direct share: competition share = |dsc_h + D_h| / |D_h|
     (equals |rebalance|/|D_h| under the exact identity
     D_h = -(direct + rebalance)); median over resist(layer)
     < 0.5 - BOTH layers (contrast with 2950 top5 promote
     heads, which were competition-dominated 70-74%).

Verdict (frozen):
  anchor fail            => anchor_fail_all_void
  T1 & T2 & T3           => early_flipper_resistance_direct
  T1 & T2 & ~T3          => early_flipper_resistance_att_carried
  T1 & ~T2               => early_flipper_resistance_snapshot_only
  ~T1                    => early_flipper_not_positive

Descriptive: D1 flip-earliness vs dsc spearman over all 32
heads per layer (perm null 20000, seed 2904); D2 dsc at
L17@s0.5 for resist/promote heads (polarity present at the
early-flip regime?); D3 head tables.
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
SRC_2947 = os.path.join(BASE, 'phase2947', 'head_anatomy',
                        'head_anatomy.npz')
SRC_2950 = os.path.join(BASE, 'phase2950', 'rebalance_anatomy',
                        'rebalance_anatomy.npz')
SRC_2951 = os.path.join(BASE, 'phase2951',
                        'rebalance_carrier_functional',
                        'rebalance_carrier_functional.npz')
SRC_2952 = os.path.join(BASE, 'phase2952', 'amplification_anatomy',
                        'amplification_anatomy.npz')
SRC_2953 = os.path.join(BASE, 'phase2953', 'a11_s_response',
                        'a11_s_response.npz')
OUT = os.path.join(BASE, 'phase2954', 'early_flipper_polarity')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2954_run_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NH, HD = 32, 128
NL = 36
SEED = 2904
N_PERM = 20000
S_IDX = (0, 1, 4)
D_TH = 2.0
ATT_SHARE_MIN = 0.6
DIRECT_SHARE_MAX = 0.5
EARLY_WIN = (0.5, 0.625, 0.75)

PREREG = {
    'mode': 'zero-forward alignment of D_h (2947) / dsc_h '
            '(this phase) / ATT-VAL (2952) / flip earliness '
            '(2953) on the same head set + one forward family '
            '(base x2 + L17@s1.0 + L17@s0.5 + L16@s2.0), '
            'o_proj-input snapshot sc_of 2952 verbatim',
    'question': 'why are the earliest routing flippers (h20/h21) '
                'the 2947 resistance heads - is their readout '
                'of the injected word value positive in sep '
                'terms?',
    'sets': 'resist = {D_h < -2}, promote = {D_h > 2} (2947 '
            'mapping threshold), computed per layer from 2947 '
            'npz',
    'anchors': {
        'a1': 'dirs rebuild < 1e-5', 'a2': 'baseline det < 1e-4',
        'a3': 'Vt8 < 1e-6', 'a7': 'xdir self-check < 1e-9',
        'a9': 'GQA gates', 'a10': 'base repeat < 1e-6',
        'a11': 'dsc vs 2952 delta < 1e-6; identity VAL+ATT '
               '< 1e-3; ATT/VAL recompute < 1e-6',
        'a12': 'sc_inj vs 2950 sc_I0 keep < 1e-5',
        'a13': 'sep L17@0.5 vs 2945 < 0.05',
        'a14': 'A11 recon residual < 0.3',
        'a15': 'A11b vs 2953 < 1e-6',
    },
    'T1': 'dsc_h > 0 for every resist head, both layers',
    'T2': 'ATT_h > 0 and att share > 0.6 for every resist '
          'head, both layers',
    'T3': 'median resist-head competition share |dsc+D|/|D| '
          '< 0.5, both layers',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'T1&T2&T3 => early_flipper_resistance_direct; '
               'T1&T2 => early_flipper_resistance_att_carried; '
               'T1 => early_flipper_resistance_snapshot_only; '
               'else => early_flipper_not_positive',
    'correction_note': 'run1 a11 anchor unreachable: demanded '
                       'identity VAL+ATT==dsc at 1e-6, but the '
                       'fp accumulation bound is ~7e-04 (2952 '
                       'a7 measured 1.85e-04/6.99e-04 with '
                       'tolerance 1e-3) - discipline 10 '
                       'reachability error. run2 splits a11 '
                       'into dsc<1e-6 (bit), identity<1e-3, '
                       'ATT/VAL<1e-6 (bit). Old execution/'
                       'result/npz deleted per discipline 3. '
                       'run2 crashed at D2: dsc dict was only '
                       'populated for L17_s1.0/L16_s2.0 but '
                       'read for L17_s0.5 (lesson 18 key-'
                       'spec family); run3 computes d05 '
                       'inline from sc_n/sc_b. run2 main '
                       'results were computed and are '
                       'deterministic (all anchors passed); '
                       'run3 is the authoritative rerun. '
                       'run3 crashed again at the save block '
                       "(dsc['L17_s0.5'] never written there) "
                       '- same key-spec lesson; run4 backfills '
                       'dsc[\'L17_s0.5\'] in the D2 section and '
                       'is authoritative.',
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
        json.dump({'phase': 2954,
                   'name': 'early_flipper_polarity',
                   'created':
                       time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2887': sha8(SRC_2887),
                               's2927': sha8(SRC_2927),
                               's2939': sha8(SRC_2939),
                               's2945': sha8(SRC_2945),
                               's2947': sha8(SRC_2947),
                               's2950': sha8(SRC_2950),
                               's2951': sha8(SRC_2951),
                               's2952': sha8(SRC_2952),
                               's2953': sha8(SRC_2953)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'seed': SEED, 'n_perm': N_PERM,
                   's_idx': list(S_IDX), 'd_th': D_TH,
                   'att_share_min': ATT_SHARE_MIN,
                   'direct_share_max': DIRECT_SHARE_MAX,
                   'early_win': list(EARLY_WIN),
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
    z47 = np.load(SRC_2947, allow_pickle=True)
    D47 = {17: z47['D_L17'].astype(np.float64),
           16: z47['D_L16'].astype(np.float64)}
    z50 = np.load(SRC_2950, allow_pickle=True)
    z51 = np.load(SRC_2951, allow_pickle=True)
    z52 = np.load(SRC_2952, allow_pickle=True)
    z53 = np.load(SRC_2953, allow_pickle=True)
    sets = {}
    for li in (17, 16):
        d = D47[li]
        sets[li] = {
            'resist': [int(h) for h in np.where(d < -D_TH)[0]],
            'promote': [int(h) for h in np.where(d > D_TH)[0]],
        }
    log('resist sets: L17 %s | L16 %s'
        % (sets[17]['resist'], sets[16]['resist']), lines)
    log('promote sets: L17 %s | L16 %s'
        % (sets[17]['promote'], sets[16]['promote']), lines)

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
    xdir = dcks_39[:, list(S_IDX)] @ Vt8[list(S_IDX)]
    a7_diff = float(np.abs(
        xdir @ Vt8[list(S_IDX)].T
        - dcks_39[:, list(S_IDX)]).max())
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

    fin_b1, v_base, x_base = forward_batch()
    fin_b2, _, _ = forward_batch()
    a10_diff = float(np.abs(fin_b1 - fin_b2).max()
                     / max(float(np.abs(fin_b1).max()), 1e-30))
    a10_ok = bool(a10_diff < 1e-6)
    a2_rel = a10_diff
    a2_rel_ok = a10_ok
    log('a2/a10 base repeat rel %.2e ok=%s'
        % (a10_diff, a10_ok), lines)

    conds = [('L17_s1.0', 17, 1.0), ('L17_s0.5', 17, 0.5),
             ('L16_s2.0', 16, 2.0)]
    cap = {}
    for cname, li, s in conds:
        fin, vv, xx = forward_batch(scale=s, inj_li=li)
        cap[cname] = {'fin': fin, 'v': vv[li], 'x': xx[li],
                      'li': li, 's': s}
    log('forward family done (base + 3 injections)', lines)

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

    v0b, v1b = v_base[17]
    A11b = {}
    res_max = 0.0
    for li in (17, 16):
        v0b, v1b = v_base[li]
        A, r = recover(
            x_base[li].reshape(n_words, NH, HD),
            v0b.reshape(n_words, NKV, HD),
            v1b.reshape(n_words, NKV, HD))
        A11b[li] = A
        res_max = max(res_max, r)
    a15_diff = 0.0
    for li in (17, 16):
        a15_diff = max(a15_diff, float(np.abs(
            A11b[li] - z53['A11b_L%d' % li]).max()))
    a15_ok = bool(a15_diff < 1e-6)

    sc_b = {li: sc_of(x_base[li], li) for li in (17, 16)}
    sc_n = {}
    A11_n = {}
    sep_new = {}
    for cname in cap:
        c = cap[cname]
        li = c['li']
        sc_n[cname] = sc_of(c['x'], li)
        v0n, v1n = c['v']
        A, r = recover(
            c['x'].reshape(n_words, NH, HD),
            v0n.reshape(n_words, NKV, HD),
            v1n.reshape(n_words, NKV, HD))
        A11_n[cname] = A
        res_max = max(res_max, r)
        p = c['fin'] @ u35
        sep_new[cname] = float(
            p[lab_lang == 0].mean() - p[lab_lang == 1].mean())
    a14_ok = bool(res_max < 0.3)
    log('a14 recon residual %.2e ok=%s | a15 A11b vs 2953 '
        '%.2e ok=%s' % (res_max, a14_ok, a15_diff, a15_ok),
        lines)

    # ---------- anchors a11/a12/a13 ----------
    a11_d = 0.0
    a11_att = 0.0
    a11_ident = 0.0
    dsc = {}
    att_r = {}
    val_r = {}
    for cname, zk in (('L17_s1.0', 'L17'),
                      ('L16_s2.0', 'L16')):
        li = 17 if zk == 'L17' else 16
        d = sc_n[cname] - sc_b[li]
        dsc[cname] = d
        a11_d = max(a11_d, float(np.abs(
            d - z52['delta_%s' % zk]).max()))
        Wo = Wo_cache[li]
        v0b_r = v_base[li][0].reshape(n_words, NKV, HD)
        v1b_r = v_base[li][1].reshape(n_words, NKV, HD)
        v0n_r = cap[cname]['v'][0].reshape(n_words, NKV, HD)
        v1n_r = cap[cname]['v'][1].reshape(n_words, NKV, HD)
        A11n = A11_n[cname]
        VAL = np.zeros(NH)
        ATT = np.zeros(NH)

        def sep_proj(vec_w):
            return vec_w[lab_lang == 0].mean(0) \
                - vec_w[lab_lang == 1].mean(0)

        for hh in range(NH):
            k = hh // HPG
            woh = Wo[:, hh * HD:(hh + 1) * HD]
            dA = A11n[:, hh] - A11b[li][:, hh]
            att_w = dA * ((v1n_r[:, k, :] - v0n_r[:, k, :])
                          @ woh.T @ u35)
            val_w = A11b[li][:, hh] * ((v1n_r[:, k, :]
                                        - v1b_r[:, k, :])
                                       @ woh.T @ u35)
            ATT[hh] = sep_proj(att_w)
            VAL[hh] = sep_proj(val_w)
        a11_ident = max(a11_ident, float(np.abs(
            (VAL + ATT) - d).max()))
        a11_att = max(a11_att,
                      float(np.abs(ATT - z52['ATT_%s' % zk])
                            .max()),
                      float(np.abs(VAL - z52['VAL_%s' % zk])
                            .max()))
        att_r[cname] = ATT
        val_r[cname] = VAL
    a11_ok = bool(a11_d < 1e-6 and a11_att < 1e-6
                  and a11_ident < 1e-3)

    a12_diff = 0.0
    for li, zk in ((17, 'L17'), (16, 'L16')):
        keep = z51['keep_%s' % zk]
        cname = 'L%d_s1.0' % li if li == 17 else 'L16_s2.0'
        a12_diff = max(a12_diff, float(np.abs(
            sc_n[cname][keep]
            - z50['sc_I0_%s' % zk][keep]).max()))
    a12_ok = bool(a12_diff < 1e-5)

    i45 = layers45.index(17)
    hit = [j for j, s45 in enumerate(grid45)
           if abs(float(s45) - 0.5) < 1e-9][0]
    a13_diff = abs(sep_new['L17_s0.5']
                   - float(sep45[i45 * len(grid45) + hit]))
    a13_ok = bool(a13_diff < 0.05)
    log('a11 dsc %.2e ident %.2e ATT/VAL %.2e ok=%s | a12 '
        'sc_I0 '
        'vs 2950 %.2e ok=%s | a13 sep L17@0.5 vs 2945 %.2e '
        'ok=%s (%.1f)'
        % (a11_d, a11_ident, a11_att, a11_ok, a12_diff,
           a12_ok,
           a13_diff, a13_ok, sep_new['L17_s0.5']), lines)

    anchor_prelim = bool(a1_ok and a2_rel_ok and a3_ok
                         and a7_ok and a9_ok and a10_ok
                         and a11_ok and a12_ok and a13_ok
                         and a14_ok and a15_ok)

    verdict = None
    t1 = t2 = t3 = d1 = d2 = d3 = None
    save = {}
    if not anchor_prelim:
        verdict = 'anchor_fail_all_void'
    else:
        # ---------- T1/T2/T3 ----------
        t1 = {}
        t2 = {}
        t3 = {}
        d3 = {}
        for li, zk in ((17, 'L17'), (16, 'L16')):
            key = 'L%d' % li
            cname = 'L%d_s1.0' % li if li == 17 else 'L16_s2.0'
            d = dsc[cname]
            ATT = att_r[cname]
            VAL = val_r[cname]
            Dh = D47[li]
            rows = []
            ok1 = True
            ok2 = True
            shares = []
            for h in sets[li]['resist']:
                share = float(abs(d[h] + Dh[h])
                              / max(abs(Dh[h]), 1e-30))
                shares.append(share)
                att_share = float(
                    abs(ATT[h]) / max(abs(VAL[h]) + abs(ATT[h]),
                                      1e-30))
                p1 = bool(d[h] > 0)
                p2 = bool(ATT[h] > 0
                          and att_share > ATT_SHARE_MIN)
                ok1 = ok1 and p1
                ok2 = ok2 and p2
                rows.append({
                    'head': int(h), 'D_h': round(float(Dh[h]),
                                                 2),
                    'dsc': round(float(d[h]), 3),
                    'ATT': round(float(ATT[h]), 3),
                    'VAL': round(float(VAL[h]), 3),
                    'att_share': round(att_share, 4),
                    'comp_share': round(share, 4),
                    'T1': p1, 'T2': p2})
            med_share = float(np.median(shares)) \
                if shares else 1.0
            t3_pass = bool(med_share < DIRECT_SHARE_MAX)
            t1[key] = {'pass': ok1,
                       'rows': rows}
            t2[key] = {'pass': ok2}
            t3[key] = {'median_comp_share':
                       round(med_share, 4),
                       'max': DIRECT_SHARE_MAX,
                       'pass': t3_pass}
            # promote rows (descriptive)
            prow = []
            for h in sets[li]['promote']:
                share = float(abs(d[h] + Dh[h])
                              / max(abs(Dh[h]), 1e-30))
                prow.append({
                    'head': int(h), 'D_h': round(float(Dh[h]),
                                                 2),
                    'dsc': round(float(d[h]), 3),
                    'comp_share': round(share, 4)})
            d3[key] = {'resist': rows, 'promote': prow}
            log('%s T1 %s | T2 %s | T3 med comp-share %.4f '
                '(pass %s)' % (key, ok1, ok2, med_share,
                               t3_pass), lines)
            for r in rows:
                log('  resist h%d: D %.2f dsc %.3f ATT %.3f '
                    'share %.3f comp %.3f'
                    % (r['head'], r['D_h'], r['dsc'],
                       r['ATT'], r['att_share'],
                       r['comp_share']), lines)

        # ---------- D1: flip earliness vs dsc ----------
        d1 = {}
        for li, zk in ((17, 'L17'), (16, 'L16')):
            key = 'L%d' % li
            g53 = z53['grid17' if li == 17 else 'grid16']
            widx = [j for j, s in enumerate(g53)
                    if any(abs(float(s) - w) < 1e-9
                           for w in EARLY_WIN)]
            f = np.zeros(NH)
            for hh in range(NH):
                vals = [float(np.median(
                    z53['A11_L%d|%.4f' % (li, float(g53[j]))
                        ][:, hh])) for j in widx]
                f[hh] = float(np.median(vals))
            cname = 'L%d_s1.0' % li if li == 17 else 'L16_s2.0'
            d = dsc[cname]
            rho = spearman(f, d)
            rng = np.random.default_rng(SEED)
            null = np.array([abs(spearman(
                rng.permutation(f), d))
                for _ in range(N_PERM)])
            p95 = float(np.quantile(null, 0.95))
            d1[key] = {'rho_flip_dsc': round(rho, 4),
                       'null_p95': round(p95, 4),
                       'significant':
                           bool(abs(rho) >= p95)}
            log('%s D1 spearman(flip-earliness, dsc) = %.4f '
                '(p95 %.4f, sig %s)'
                % (key, rho, p95, abs(rho) >= p95), lines)

        # ---------- D2: L17@s0.5 polarity ----------
        d2 = {}
        d05 = sc_n['L17_s0.5'] - sc_b[17]
        dsc['L17_s0.5'] = d05
        for tag, hs in (('resist', sets[17]['resist']),
                        ('promote', sets[17]['promote'])):
            d2[tag] = [{'head': int(h),
                        'dsc_s0.5': round(float(d05[h]), 3),
                        'A11_med_s0.5': round(float(
                            np.median(A11_n['L17_s0.5'][:,
                                                       h])), 4)}
                       for h in hs]
        log('D2 L17@0.5 resist dsc: %s'
            % [(r['head'], r['dsc_s0.5'])
               for r in d2['resist']], lines)
        log('D2 L17@0.5 promote dsc: %s'
            % [(r['head'], r['dsc_s0.5'])
               for r in d2['promote']], lines)

        t1_all = all(v['pass'] for v in t1.values())
        t2_all = all(v['pass'] for v in t2.values())
        t3_all = all(v['pass'] for v in t3.values())
        if not t1_all:
            verdict = 'early_flipper_not_positive'
        elif t1_all and t2_all and t3_all:
            verdict = 'early_flipper_resistance_direct'
        elif t1_all and t2_all:
            verdict = 'early_flipper_resistance_att_carried'
        else:
            verdict = 'early_flipper_resistance_snapshot_only'

        save = {
            'words': np.array(['%s:%s:%s' % w
                               for w in words],
                              dtype=object),
            'labels_lang': lab_lang,
            'dsc_L17_s1.0': dsc['L17_s1.0'],
            'dsc_L17_s0.5': dsc['L17_s0.5'],
            'dsc_L16_s2.0': dsc['L16_s2.0'],
            'ATT_L17_s1.0': att_r['L17_s1.0'],
            'VAL_L17_s1.0': val_r['L17_s1.0'],
            'ATT_L16_s2.0': att_r['L16_s2.0'],
            'VAL_L16_s2.0': val_r['L16_s2.0'],
            'A11_L17_s0.5': A11_n['L17_s0.5'],
            'sc_b_L17': sc_b[17], 'sc_b_L16': sc_b[16],
            'flip_earliness_L17': d1['L17'],
            'flip_earliness_L16': d1['L16'],
        }

    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2954, 'model': 'qwen3-4b',
           'prereg': PREREG,
           'anchors': {
               'a1_diff': float('%.3e' % a1_diff),
               'a1_ok': a1_ok,
               'a2_rel': float('%.3e' % a2_rel),
               'a2_ok': a2_rel_ok,
               'a3_diff': float('%.3e' % a3_diff),
               'a3_ok': a3_ok,
               'a7_diff': float('%.3e' % a7_diff),
               'a7_ok': a7_ok,
               'a9_ok': a9_ok,
               'a10_diff': float('%.3e' % a10_diff),
               'a10_ok': a10_ok,
               'a11_dsc': float('%.3e' % a11_d),
               'a11_ident': float('%.3e' % a11_ident),
               'a11_att': float('%.3e' % a11_att),
               'a11_ok': a11_ok,
               'a12_diff': float('%.3e' % a12_diff),
               'a12_ok': a12_ok,
               'a13_diff': float('%.3e' % a13_diff),
               'a13_ok': a13_ok,
               'a14_diff': float('%.3e' % res_max),
               'a14_ok': a14_ok,
               'a15_diff': float('%.3e' % a15_diff),
               'a15_ok': a15_ok,
               'ok': bool(anchor_prelim)},
           'sets': sets,
           'T1_polarity': t1, 'T2_att_carry': t2,
           'T3_direct_share': t3,
           'D1_flip_vs_dsc': d1, 'D2_s05_polarity': d2,
           'D3_tables': d3,
           'sep_conditions': {k: round(v, 2)
                              for k, v in sep_new.items()},
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if save:
        np.savez_compressed(os.path.join(
            OUT, 'early_flipper_polarity.npz'), **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2954 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
