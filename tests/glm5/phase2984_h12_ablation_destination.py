# -*- coding: utf-8 -*-
"""Phase 2984: destination of the h12-ablation main
effect (where does the signal go when the necessary
interaction carrier is removed).

Why: 2980 proved ablating h12 abolishes the L17 two-axis
subadditive interaction by 97% (|median I| 0.02358 ->
0.00056) with no global B-band destruction (T3 mean|dB|
balanced across conditions). Open question: is the
ablation effect ISOLATED to the interaction (main effects
of each single-axis injection preserved, interaction
merely local), or does removing h12 trigger a network-wide
REBALANCING (main effects / interaction redistribute
across layers, injection-induced band response shifts)?

Design (frozen before any observation):
  Word list / axis dirs / n17 VERBATIM from 2979 npz
  (2980 protocol identical). h_rand from rng 29804 (same
  draw as 2980 -> same control head).
  Stage0: 74 intact forwards -> n17 anchor.
  Stage1: intact dose-0.1 4 conditions x 74 (2977/2980
          protocol verbatim) -> a5/a6 anchors.
  Stage2: ablation {h12, rand} x 4 conditions x 74.
  Ablation = zero o_proj input slice at L17 (2932/2980
  verbatim); injection = self_attn pre-hook x[:,1,:] +=
  dt (2977 verbatim).
  Readout: prof[li] = sum_h u35 @ Wo_li_h @ x_li_h (2965
  verbatim); full (74, NL) profiles saved for offline
  reuse. Total 962 forwards.

Anchors (frozen):
  a1 o_proj shape L17 == (2560, 4096)
  a2 2979 _u dirs unit-norm < 1e-9 AND cos(u, 2977 raw)
     > 1-1e-9
  a3 intact n17 vs 2979 npz rel < 1e-6
  a4 single-token 74/74 + list == 2979
  a5 I(0.1,0.1) intact @L17 vs 2977 npz per-word
     max|d| < 1e-4
  a6 I(0.1,0.1) intact @L17 vs 2980 npz I_int per-word
     max|d| < 1e-4 (same-protocol bit-level identity)
  a7 ablation efficacy |C17[abl_h]| < 1e-6
  a8 non-degeneracy C17 std > 0 (32/32)

Tests (frozen):
  T1 (main): main-effect preservation under h12 ablation.
     For axis in {lang, cls}: d_l(w) =
     (profX0_abl - prof00_abl)[w,l] -
     (profX0_int - prof00_int)[w,l], X in {1}. Per-layer
     obs = median(d_l); sign-flip perm (74 paired, 10000,
     rng 29840/29841); maxT across 36 layers (family 36,
     discipline 2917); sig layers = |obs| >= maxT thr.
  T2: interaction redistribution across layers: d_l(w) =
     I_abl(w,l) - I_int(w,l) full profile; same maxT
     machinery (rng 29842).
  T3: injection-induced band response shift: e(w) =
     (bandX_abl - band00_abl) - (bandX_int - band00_int)
     for X in {10,01,11}; obs median; sign-flip perm
     (rngs 29843-29845); gate p <= 0.01.
  T4 descriptive: head-level pickup at L17 under 00:
     dC_h = C17_abl_h12 - C17_int per head; top gains +
     rand-head percentile.

Verdict (frozen):
  anchor fail => anchor_fail_all_void
  T1 lang+cls sig layers == 0 AND T2 sig layers == 0
     => ablation_local_to_interaction_only
  T2 sig layers >= 1
     => interaction_redistributes_across_layers
  T1 total sig layers >= 5
     => h12_ablation_triggers_rebalancing
  else => ablation_partial_effect_shift
"""
import hashlib
import json
import os
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC_2927 = os.path.join(BASE, 'phase2927', 'probe_relativity',
                        'probe_relativity.npz')
SRC_2977 = os.path.join(BASE, 'phase2977',
                        'two_axis_fusion_injection',
                        'two_axis_fusion_injection.npz')
SRC_2979 = os.path.join(BASE, 'phase2979', 'reversal_anatomy',
                        'reversal_anatomy.npz')
SRC_2980 = os.path.join(BASE, 'phase2980',
                        'bottleneck_head_identity',
                        'bottleneck_head_identity.npz')
OUT = os.path.join(BASE, 'phase2984',
                   'h12_ablation_destination')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2984_run_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NH, HD = 32, 128
NL = 36
L_INJ = 17
H12 = 12
DOSE = 0.1
N_PERM = 10000
RNG_T1 = {'lang': 29840, 'cls': 29841}
RNG_T2 = 29842
RNG_T3 = {'10': 29843, '01': 29844, '11': 29845}
RNG_RAND = 29804

PREREG = {
    'mode': 'stage0: 74 intact forwards; stage1: intact '
            'dose-0.1 4 conditions x 74; stage2: ablation '
            '{h12, rand} x 4 conditions x 74. total 962 '
            'forwards. ablation = zero o_proj input slice '
            'at L17 (2980 verbatim); injection = '
            'self_attn pre-hook x[:,1,:] += dt (2977 '
            'verbatim); readout = per-head contributions '
            'prof (2965 verbatim), full (74,36) profiles '
            'saved',
    'question': 'when the necessary interaction carrier '
                'h12 is ablated, is the effect isolated '
                'to the interaction (main effects '
                'preserved) or does it trigger '
                'layer/head rebalancing (main effects '
                'and interaction redistribute)?',
    'anchors': {
        'a1': 'o_proj shape L17 == (2560, 4096)',
        'a2': '2979 _u dirs unit-norm < 1e-9 AND cos(u, '
              '2977 raw) > 1-1e-9',
        'a3': 'intact n17 vs 2979 npz rel < 1e-6',
        'a4': 'single-token 74/74 + list == 2979',
        'a5': 'I(0.1,0.1) intact vs 2977 npz per-word '
              'max|d| < 1e-4',
        'a6': 'I(0.1,0.1) intact vs 2980 npz I_int '
              'per-word max|d| < 1e-4',
        'a7': 'ablation efficacy |C17[abl_h]| < 1e-6',
        'a8': 'non-degeneracy C17 std > 0 (32/32)',
    },
    'T1': 'main-effect preservation: per-layer paired '
          'median of (mainX_abl - mainX_int); sign-flip '
          'perm 10000 rngs 29840/29841; maxT family 36; '
          'sig iff |obs| >= maxT thr 0.99',
    'T2': 'interaction redistribution: per-layer paired '
          'median of (I_abl - I_int); sign-flip 10000 '
          'rng 29842; maxT family 36',
    'T3': 'injection-induced band response shift: '
          'paired median of (bandX_abl - band00_abl) - '
          '(bandX_int - band00_int), X in {10,01,11}; '
          'sign-flip 10000 rngs 29843-29845; gate p<=0.01',
    'T4': 'descriptive: head-level pickup dC_h at L17 '
          'under 00, rand percentile',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'T1 zero sig & T2 zero sig => '
               'ablation_local_to_interaction_only; '
               'T2 sig >= 1 => '
               'interaction_redistributes_across_layers; '
               'T1 total sig >= 5 => '
               'h12_ablation_triggers_rebalancing; '
               'else => ablation_partial_effect_shift',
    'correction_note': '',
}


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


def log(msg, lines):
    lines.append(msg)
    print(msg, flush=True)


def signflip_maxt(dmat, rng_seed, n_perm=N_PERM):
    """dmat: (n_words, NL) paired diffs. Returns per-layer
    obs medians, perm p per layer, and maxT sig layers."""
    n, nl = dmat.shape
    obs = np.median(dmat, axis=0)
    rng = np.random.default_rng(rng_seed)
    idx = np.arange(n)
    p_perm = np.ones(nl)
    null_max = np.zeros(n_perm)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    for k in range(n_perm):
        med = np.median(dmat * signs[k][:, None], axis=0)
        null_max[k] = float(np.abs(med).max())
        p_perm = p_perm + (np.abs(med) >= np.abs(obs)
                           - 1e-12)
    p_perm = p_perm / (n_perm + 1)
    thr = float(np.quantile(null_max, 1 - 0.01))
    sig = [li for li in range(nl)
           if abs(obs[li]) >= thr]
    return obs, p_perm, thr, sig


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2984,
                   'name': 'h12_ablation_destination',
                   'created': time.strftime(
                       '%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2927': sha8(SRC_2927),
                               's2977': sha8(SRC_2977),
                               's2979': sha8(SRC_2979),
                               's2980': sha8(SRC_2980)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'inj_layer': L_INJ, 'target_head': H12,
                   'dose_rel': DOSE,
                   'n_forwards': 962, 'n_perm': N_PERM,
                   'rng': {'T1': RNG_T1, 'T2': RNG_T2,
                           'T3': RNG_T3,
                           'rand_head': RNG_RAND},
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    # ---------- sources ----------
    z27_ = np.load(SRC_2927, allow_pickle=True)
    u35 = z27_['dirs_word'].astype(np.float64)[NL - 1]
    z77 = np.load(SRC_2977, allow_pickle=True)
    z79 = np.load(SRC_2979, allow_pickle=True)
    z80 = np.load(SRC_2980, allow_pickle=True)

    dl_u = z79['d_lang_u'].astype(np.float64)
    dc_u = z79['d_cls_u'].astype(np.float64)
    dl_raw = z77['d_lang'].astype(np.float64)
    dc_raw = z77['d_cls'].astype(np.float64)
    n17_ref = z79['n17'].astype(np.float64)
    words79 = [str(w) for w in z79['words']]
    wlist = [w.split(':')[2] for w in words79]

    a2_n1 = abs(float(np.linalg.norm(dl_u)) - 1.0)
    a2_n2 = abs(float(np.linalg.norm(dc_u)) - 1.0)
    cos_l = float(dl_u @ dl_raw / (
        np.linalg.norm(dl_u) * np.linalg.norm(dl_raw)))
    cos_c = float(dc_u @ dc_raw / (
        np.linalg.norm(dc_u) * np.linalg.norm(dc_raw)))
    a2_ok = bool(a2_n1 < 1e-9 and a2_n2 < 1e-9
                 and abs(cos_l - 1.0) < 1e-9
                 and abs(cos_c - 1.0) < 1e-9)
    log('a2 dirs unit dev %.1e/%.1e cos %.12f/%.12f '
        'ok=%s' % (a2_n1, a2_n2, cos_l, cos_c, a2_ok),
        lines)

    I77 = (z77['prof11'].astype(np.float64)
           - z77['prof10'].astype(np.float64)
           - z77['prof01'].astype(np.float64)
           + z77['prof00'].astype(np.float64))[:, L_INJ]
    I80 = z80['I_int'].astype(np.float64)

    # ---------- model ----------
    import torch
    import sys
    sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
    from phase2662_symmetric_mapping_contract import \
        load_native
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        MD, local_files_only=True, trust_remote_code=True,
        use_fast=True)
    tid_map = {}
    n_single = 0
    for w in wlist:
        ids = tok(' ' + w, add_special_tokens=False)[
            'input_ids']
        if len(ids) != 1:
            ids = tok(w, add_special_tokens=False)[
                'input_ids']
        assert len(ids) == 1, '%s -> %s' % (w, ids)
        tid_map[w] = int(ids[0])
        n_single += 1
    a4_ok = bool(n_single == 74
                 and [str(w) for w in z79['words']]
                 == words79)
    log('a4 single-token %d/74, list==2979 ok=%s'
        % (n_single, a4_ok), lines)
    ids_the = tok(' the', add_special_tokens=False)[
        'input_ids']
    assert len(ids_the) == 1
    func_tid = int(ids_the[0])

    model, _ = load_native('qwen4')
    model.eval()
    layers = model.model.layers
    log('model loaded', lines)

    Wo17 = layers[L_INJ].self_attn.o_proj.weight.detach() \
        .float().cpu().numpy()
    a1_ok = bool(Wo17.shape == (2560, NH * HD))
    log('a1 o_proj shape L17 %s ok=%s'
        % (Wo17.shape, a1_ok), lines)

    cap_op = {li: [] for li in range(NL)}
    abl = {'h': None}
    inj = {'d': None}
    handles = []

    def hs_of(args, kwargs):
        if args:
            return args[0]
        return kwargs.get('hidden_states')

    def hook_op(li):
        def h(module, args, kwargs):
            x = args[0] if args else kwargs.get('input')
            if x is None or x.dim() < 2:
                return None
            if li == L_INJ and abl['h'] is not None:
                for hh in abl['h']:
                    x[:, 1,
                      hh * HD:(hh + 1) * HD] = 0
            cap_op[li].append(
                x[:, 1, :].detach().float().cpu().numpy())
            return None
        return h

    x17_cap = []

    def hook_inj(module, args, kwargs):
        d = inj['d']
        x = hs_of(args, kwargs)
        if d is not None:
            if x is not None:
                dt = torch.as_tensor(
                    d, device=x.device, dtype=x.dtype)
                x[:, 1, :] += dt
        elif x is not None and len(x17_cap) < 200:
            x17_cap.append(
                x[:, 1, :].detach().float().cpu()
                .numpy())
        return None

    for li in range(NL):
        handles.append(
            layers[li].self_attn.o_proj
            .register_forward_pre_hook(
                hook_op(li), with_kwargs=True))
    handles.append(
        layers[L_INJ].self_attn
        .register_forward_pre_hook(
            hook_inj, with_kwargs=True))

    def clear_cap():
        for li in cap_op:
            del cap_op[li][:]

    def forward1(toks, d_inj=None, abl_head=None):
        clear_cap()
        inj['d'] = d_inj
        abl['h'] = (list(abl_head)
                    if abl_head is not None else None)
        with torch.no_grad():
            model(torch.tensor([toks], device='cuda'))
        inj['d'] = None
        abl['h'] = None
        return {li: cap_op[li][0].astype(np.float64)
                for li in range(NL)}

    M = np.zeros((NL, NH * HD))
    for li in range(NL):
        Wl = layers[li].self_attn.o_proj.weight.detach() \
            .float().cpu().numpy()
        M[li] = u35 @ Wl

    def contributions(op):
        C = np.zeros((NL, NH))
        prof = np.zeros(NL)
        for li in range(NL):
            x = op[li].reshape(-1)
            xm = (x * M[li]).reshape(NH, HD)
            C[li] = xm.sum(axis=1)
            prof[li] = float(C[li].sum())
        return C, prof

    def band_of(prof):
        return (float(prof[6:13].mean())
                - float(prof[28:36].mean()))

    # ---------- stage0: intact ----------
    n17 = np.zeros(74)
    C17_int = np.zeros((74, NH))
    del x17_cap[:]
    for i, w in enumerate(wlist):
        op = forward1([func_tid, tid_map[w]])
        C, _ = contributions(op)
        n17[i] = float(np.linalg.norm(x17_cap[i]
                                      .reshape(-1)))
        C17_int[i] = C[L_INJ]
        if (i + 1) % 20 == 0:
            log('intact [%d/74]' % (i + 1), lines)

    a3_rel = float(np.abs(n17 - n17_ref).max()
                   / max(float(np.abs(n17_ref).max()),
                         1e-30))
    a3_ok = bool(a3_rel < 1e-6)
    log('a3 intact n17 vs 2979 rel %.2e ok=%s'
        % (a3_rel, a3_ok), lines)

    a8_ok = bool(C17_int.std(axis=0).min() > 0)
    log('a8 non-degeneracy C17 std min %.3e ok=%s'
        % (C17_int.std(axis=0).min(), a8_ok), lines)

    rng_r = np.random.default_rng(RNG_RAND)
    cand = [h for h in range(NH) if h != H12]
    h_rand = int(rng_r.choice(cand))
    log('rand control head = %d (rng 29804)' % h_rand,
        lines)

    dose_scales = DOSE * n17_ref

    anchor_ok = bool(a1_ok and a2_ok and a3_ok and a4_ok
                     and a8_ok)
    verdict = None
    a5_max = a6_max = 0.0
    a5_ok = a6_ok = a7_ok = None
    a7_maxv = 0.0
    t1 = t2 = t3 = t4 = None
    save = {}

    def run_block(abl_head):
        """4 conditions x 74 -> profs dict (74, NL)."""
        pr = {k: np.zeros((74, NL))
              for k in ('00', '10', '01', '11')}
        for i, w in enumerate(wlist):
            toks = [func_tid, tid_map[w]]
            s_w = float(dose_scales[i])
            ops = {}
            for key, dvec in (('00', None),
                              ('10', s_w * dl_u),
                              ('01', s_w * dc_u),
                              ('11', s_w * (dl_u
                                            + dc_u))):
                ops[key] = forward1(
                    toks, dvec, abl_head=abl_head)
            for key, op in ops.items():
                _, prof = contributions(op)
                pr[key][i] = prof
            if abl_head is None and \
                    (i + 1) % 20 == 0:
                log('stage1 [%d/74]' % (i + 1), lines)
            if abl_head is not None and \
                    (i + 1) % 40 == 0:
                log('abl %s [%d/74]'
                    % (str(abl_head), i + 1), lines)
        return pr

    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        # ---------- stage1: intact ----------
        pi = run_block(None)
        I_int = (pi['11'] - pi['10'] - pi['01']
                 + pi['00'])[:, L_INJ]
        a5_max = float(np.abs(I_int - I77).max())
        a5_ok = bool(a5_max < 1e-4)
        log('a5 I intact vs 2977 max|d| %.2e ok=%s'
            % (a5_max, a5_ok), lines)
        a6_max = float(np.abs(I_int - I80).max())
        a6_ok = bool(a6_max < 1e-4)
        log('a6 I intact vs 2980 I_int max|d| %.2e '
            'ok=%s' % (a6_max, a6_ok), lines)

        # ---------- stage2: ablations ----------
        ABL_SETS = {'h12': [H12], 'rand': [h_rand]}
        pa = {}
        for cname, heads in ABL_SETS.items():
            pa[cname] = run_block(heads)
        # efficacy: single forward per abl cond
        a7_maxv = 0.0
        for cname, heads in ABL_SETS.items():
            op = forward1(
                [func_tid, tid_map[wlist[0]]],
                None, abl_head=heads)
            C, _ = contributions(op)
            for hh in heads:
                a7_maxv = max(
                    a7_maxv,
                    abs(float(C[L_INJ, hh])))
        a7_ok = bool(a7_maxv < 1e-6)
        log('a7 ablation efficacy max|C_abl| %.2e '
            'ok=%s' % (a7_maxv, a7_ok), lines)

        if not (a5_ok and a6_ok and a7_ok):
            verdict = 'anchor_fail_all_void'
        else:
            I_abl12 = (pa['h12']['11']
                       - pa['h12']['10']
                       - pa['h12']['01']
                       + pa['h12']['00'])
            I_abl_int_l17 = float(
                np.median(I_abl12[:, L_INJ]))
            log('h12-abl I @L17 median %.5f (intact '
                '%.5f)' % (I_abl_int_l17,
                           float(np.median(I_int))),
                lines)

            # ---------- T1 ----------
            t1 = {}
            sig_total = 0
            for axis, key in (('lang', '10'),
                              ('cls', '01')):
                main_int = (pi[key] - pi['00'])
                main_abl = (pa['h12'][key]
                            - pa['h12']['00'])
                dmat = main_abl - main_int
                obs, p_perm, thr, sig = \
                    signflip_maxt(dmat, RNG_T1[axis])
                t1[axis] = {
                    'sig_layers': sig,
                    'n_sig': len(sig),
                    'maxT_thr': round(thr, 5),
                    'top3': [(int(li),
                              round(float(obs[li]), 5))
                             for li in np.argsort(
                                 -np.abs(obs))[:3]]}
                sig_total += len(sig)
                log('T1 %s: n_sig=%d thr %.5f top3 %s'
                    % (axis, len(sig), thr,
                       t1[axis]['top3']), lines)

            # ---------- T2 ----------
            dmat2 = I_abl12 - (pi['11'] - pi['10']
                               - pi['01'] + pi['00'])
            obs2, p2, thr2, sig2 = signflip_maxt(
                dmat2, RNG_T2)
            t2 = {'sig_layers': sig2,
                  'n_sig': len(sig2),
                  'maxT_thr': round(thr2, 5),
                  'top3': [(int(li),
                            round(float(obs2[li]), 5))
                           for li in np.argsort(
                               -np.abs(obs2))[:3]]}
            log('T2 interaction redistribution: '
                'n_sig=%d thr %.5f top3 %s'
                % (len(sig2), thr2, t2['top3']), lines)

            # ---------- T3 ----------
            t3 = {}
            for key in ('10', '01', '11'):
                b_int = np.array(
                    [band_of(pi[key][i])
                     - band_of(pi['00'][i])
                     for i in range(74)])
                b_abl = np.array(
                    [band_of(pa['h12'][key][i])
                     - band_of(pa['h12']['00'][i])
                     for i in range(74)])
                e = b_abl - b_int
                obs = float(np.median(e))
                rng = np.random.default_rng(
                    RNG_T3[key])
                cnt = 0
                for _ in range(N_PERM):
                    if abs(float(np.median(
                            e * rng.choice(
                                [-1.0, 1.0],
                                size=74)))) \
                            >= abs(obs) - 1e-12:
                        cnt += 1
                p = (cnt + 1) / (N_PERM + 1)
                t3[key] = {
                    'obs_median': round(obs, 5),
                    'perm_p': float('%.3e' % p),
                    'sig': bool(p <= 0.01)}
                log('T3 %s: band-response shift med '
                    '%+.5f p %.3e sig=%s'
                    % (key, obs, p, p <= 0.01), lines)

            # ---------- T4 descriptive ----------
            C00_abl = np.zeros((74, NH))
            for i, w in enumerate(wlist):
                op = forward1(
                    [func_tid, tid_map[w]], None,
                    abl_head=[H12])
                C, _ = contributions(op)
                C00_abl[i] = C[L_INJ]
            dC = C00_abl.copy()
            dC[:, H12] = 0.0
            dC = dC - C17_int
            dC[:, H12] = 0.0
            gain = np.median(np.abs(dC), axis=0)
            order = np.argsort(-gain)
            rand_gain = float(
                gain[h_rand]) if h_rand != H12 else 0.0
            t4 = {
                'top5_gain_heads': [
                    (int(h), round(float(gain[h]), 5))
                    for h in order[:5]],
                'rand_head_gain':
                    round(rand_gain, 5),
                'dC_h12_self_median':
                    round(float(np.median(
                        np.abs(C00_abl[:, H12]))), 6),
                'note': 'descriptive only: median '
                        '|dC_h| over words, h12 column '
                        'excluded; rand head as inline '
                        'control'}
            log('T4 top5 gain heads %s'
                % t4['top5_gain_heads'], lines)

            # ---------- verdict ----------
            if len(sig2) >= 1:
                verdict = \
                    'interaction_redistributes_across' \
                    '_layers'
            elif sig_total == 0:
                verdict = \
                    'ablation_local_to_interaction_' \
                    'only'
            elif sig_total >= 5:
                verdict = \
                    'h12_ablation_triggers_rebalancing'
            else:
                verdict = 'ablation_partial_effect_' \
                          'shift'
            save = {
                'prof_int_00': pi['00'],
                'prof_int_10': pi['10'],
                'prof_int_01': pi['01'],
                'prof_int_11': pi['11'],
                'prof_h12_00': pa['h12']['00'],
                'prof_h12_10': pa['h12']['10'],
                'prof_h12_01': pa['h12']['01'],
                'prof_h12_11': pa['h12']['11'],
                'prof_rand_00': pa['rand']['00'],
                'prof_rand_10': pa['rand']['10'],
                'prof_rand_01': pa['rand']['01'],
                'prof_rand_11': pa['rand']['11'],
                'I_int': I_int,
                'C17_int': C17_int,
                'C00_abl_h12': C00_abl,
                'n17': n17,
                'h_rand': np.array([h_rand]),
                'tids': np.array([tid_map[w]
                                  for w in wlist]),
                'words': np.array(words79)}

    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2984, 'model': 'qwen3-4b',
           'prereg': PREREG,
           'anchors': {'a1_ok': a1_ok,
                       'a2_ok': a2_ok,
                       'a3_rel': float('%.3e' % a3_rel),
                       'a3_ok': a3_ok,
                       'a4_ok': a4_ok,
                       'a5_max': float('%.3e' % a5_max)
                       if anchor_ok else None,
                       'a5_ok': a5_ok
                       if anchor_ok else None,
                       'a6_max': float('%.3e' % a6_max)
                       if anchor_ok else None,
                       'a6_ok': a6_ok
                       if anchor_ok else None,
                       'a7_max': float('%.3e' % a7_maxv)
                       if a7_maxv else None,
                       'a7_ok': a7_ok,
                       'a8_ok': a8_ok,
                       'ok': bool(anchor_ok and a5_ok
                                  and a6_ok and a7_ok)
                       if anchor_ok else False},
           'T1': t1, 'T2': t2, 'T3': t3, 'T4': t4,
           'median_I_int': round(
               float(np.median(I_int)), 5)
           if anchor_ok and I_int is not None else None,
           'rand_head': h_rand,
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if verdict != 'anchor_fail_all_void':
        np.savez_compressed(os.path.join(
            OUT, 'h12_ablation_destination.npz'), **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2984 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
