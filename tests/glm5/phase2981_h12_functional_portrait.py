# -*- coding: utf-8 -*-
"""Phase 2981: functional portrait of the competitive
bottleneck head h12 (2980: ablating h12 removes 97% of
the L17 two-axis interaction at dose 0.1).

c_h = u35 @ Wo17_h @ x_h is LINEAR in x_h, so a nonzero
per-head interaction I_h requires the o_proj-INPUT
response x_h itself to be nonlinear. Question: sub/super-
linear gain (g vs 1) or redirection (cos vs 1)?

Design (frozen): word list / n17 / unit dirs VERBATIM
from 2979 npz (2980 injection protocol verbatim,
dose_scales = 0.1 * n17_ref). Stage0: 74 intact forwards.
Stage1: 74 words x 4 conditions {00,10,01,11}, capture
L17 o_proj input, per-head split x_h (32x128).
I_h(w) = dc11_h - dc10_h - dc01_h + 0; layer I = sum_h.
g_h = ||dx11|| / (||dx10||+||dx01||); cos_h = cos(dx11,
dx10+dx01); write kernel w_h = u35 @ Wo17[:, h-slice].

Anchors: a1 shape (2560,4096); a2 2979 _u unit + cos vs
2977 raw > 1-1e-9; a3 n17 vs 2979 rel < 1e-6; a4
single-token 74/74 + list == 2979; a5 layer I vs 2977
per-word max|d| < 1e-4; a6 per-head I median vs 2978
I_grid_med[17] max|d| < 1e-4; a7 non-degeneracy C17 std
> 0 (32/32) AND denominator gate >= 70/74 at h12.

Tests: T1 (designated subset, discipline 9) I_h12 / I_h9
!= 0 sign-flip perm (10000, rngs 29810/29811) gate
p <= 0.01, h12 share of layer I descriptive; T2 (main)
g_h12 vs 1, cos_h12 vs 1 (rngs 29812/29813) gate
p <= 0.01, h9 secondary (29814/29815), all-head medians
descriptive; T3 descriptive write-kernel norms rank +
axis convergence.

Verdict: anchor fail => anchor_fail_all_void; T2 g sig &
med < 1 => h12_input_sublinear_carrier; g sig & med > 1
=> h12_input_superlinear_carrier; g ns & cos sig =>
h12_input_redirect_carrier; else =>
h12_linear_response_contradiction (audit flag vs 2980).
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
SRC_2978 = os.path.join(BASE, 'phase2978',
                        'interaction_dose_headgrid',
                        'interaction_dose_headgrid.npz')
SRC_2979 = os.path.join(BASE, 'phase2979', 'reversal_anatomy',
                        'reversal_anatomy.npz')
OUT = os.path.join(BASE, 'phase2981',
                   'h12_functional_portrait')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2981_run_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NH, HD = 32, 128
NL = 36
L_INJ = 17
H9, H12 = 9, 12
DOSE = 0.1
N_PERM = 10000
RNG_T = {'I_h12': 29810, 'I_h9': 29811,
         'g_h12': 29812, 'cos_h12': 29813,
         'g_h9': 29814, 'cos_h9': 29815}

PREREG = {
    'mode': 'stage0: 74 intact forwards (2973 protocol); '
            'stage1: 74 words x 4 conditions {00,10,01,11} '
            'at dose 0.1 (2980 verbatim: 2979 _u dirs, '
            'dose_scales = 0.1 * n17_ref), capture L17 '
            'o_proj input, per-head split x_h (32x128). '
            'total 370 forwards',
    'question': 'h12 carries the L17 two-axis interaction '
                '(2980 ablation, 97% removal); c_h linear '
                'in x_h implies the nonlinearity lives in '
                'the o_proj-input response of h12: gain '
                '(g vs 1) or redirection (cos vs 1)?',
    'anchors': {
        'a1': 'o_proj shape L17 == (2560, 4096)',
        'a2': '2979 _u dirs unit-norm < 1e-9 AND cos(u, '
              '2977 raw) > 1-1e-9',
        'a3': 'intact n17 vs 2979 npz rel < 1e-6',
        'a4': 'single-token 74/74 + list == 2979',
        'a5': 'layer I(0.1,0.1) vs 2977 npz per-word '
              'max|d| < 1e-4',
        'a6': 'per-head I median vs 2978 I_grid_med[17] '
              'max|d| < 1e-4',
        'a7': 'non-degeneracy C17 std > 0 (32/32) AND '
              'denominator gate ||dx10||+||dx01|| > 1e-8 '
              'for >= 70/74 words at h12',
    },
    'T1': 'designated-subset (discipline 9): I_h12 / I_h9 '
          '!= 0 sign-flip perm (10000) gate p <= 0.01; '
          'h12 share of layer I (median ratio) '
          'descriptive',
    'T2': 'mechanism: g_h12 vs 1 and cos_h12 vs 1 '
          '(sign-flip perm 10000) gate p <= 0.01; h9 '
          'secondary; all-head medians descriptive',
    'T3': 'descriptive: write-kernel ||w_h|| (128-d '
          'head-input space) rank of h12/h9; channel '
          'overlap cos(dx10_h12, dx01_h12) and response '
          'norm medians (same space)',
    'verdict': 'anchor fail => anchor_fail_all_void; T2 g '
               'sig & med<1 => h12_input_sublinear_carrier; '
               'g sig & med>1 => '
               'h12_input_superlinear_carrier; g ns & cos '
               'sig => h12_input_redirect_carrier; else => '
               'h12_linear_response_contradiction',
    'correction_note': 'run1: crash in T3 - u35 @ Wo17[:, h-slice] is a 128-d head-input-space vector and cannot be dotted with the 2560-d layer-input axis dirs (cross-space matmul); preregistered T1/T2/verdict untouched (anchors a1-a7 all passed, verdict not yet emitted). fix: T3 replaced with same-space quantities (write-kernel norms rank + channel overlap cos(dx10,dx01) + response norm medians); artifacts deleted, rerun per discipline 3.',
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


def perm_p(d, obs, rng):
    cnt = 0
    n = len(d)
    for _ in range(N_PERM):
        if abs(float(np.median(
                d * rng.choice([-1.0, 1.0], size=n)))) \
                >= abs(obs) - 1e-12:
            cnt += 1
    return (cnt + 1) / (N_PERM + 1)


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2981,
                   'name': 'h12_functional_portrait',
                   'created': time.strftime(
                       '%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2927': sha8(SRC_2927),
                               's2977': sha8(SRC_2977),
                               's2978': sha8(SRC_2978),
                               's2979': sha8(SRC_2979)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'inj_layer': L_INJ, 'dose_rel': DOSE,
                   'target_heads': [H9, H12],
                   'n_forwards': 370, 'n_perm': N_PERM,
                   'rng': RNG_T,
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    z27_ = np.load(SRC_2927, allow_pickle=True)
    u35 = z27_['dirs_word'].astype(np.float64)[NL - 1]
    z77 = np.load(SRC_2977, allow_pickle=True)
    z78 = np.load(SRC_2978, allow_pickle=True)
    z79 = np.load(SRC_2979, allow_pickle=True)

    dl_u = z79['d_lang_u'].astype(np.float64)
    dc_u = z79['d_cls_u'].astype(np.float64)
    dl_raw = z77['d_lang'].astype(np.float64)
    dc_raw = z77['d_cls'].astype(np.float64)
    n17_ref = z79['n17'].astype(np.float64)
    words79 = [str(w) for w in z79['words']]
    wlist = [w.split(':')[2] for w in words79]
    lab_lang = np.array([w.split(':')[1]
                         for w in words79])
    lab_grp = np.array([w.split(':')[0]
                        for w in words79])

    a2_n1 = abs(float(np.linalg.norm(dl_u)) - 1.0)
    a2_n2 = abs(float(np.linalg.norm(dc_u)) - 1.0)
    cos_l = float(dl_u @ dl_raw / (
        np.linalg.norm(dl_u) * np.linalg.norm(dl_raw)))
    cos_c = float(dc_u @ dc_raw / (
        np.linalg.norm(dc_u) * np.linalg.norm(dc_raw)))
    a2_ok = bool(a2_n1 < 1e-9 and a2_n2 < 1e-9
                 and abs(cos_l - 1.0) < 1e-9
                 and abs(cos_c - 1.0) < 1e-9)
    log('a2 dirs ok=%s (2977 raw norms %.4f/%.4f)'
        % (a2_ok, float(np.linalg.norm(dl_raw)),
           float(np.linalg.norm(dc_raw))), lines)

    I77 = (z77['prof11'].astype(np.float64)
           - z77['prof10'].astype(np.float64)
           - z77['prof01'].astype(np.float64)
           + z77['prof00'].astype(np.float64))[:, L_INJ]
    Igrid78 = z78['I_grid_med'].astype(np.float64)

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

    def forward1(toks, d_inj=None):
        clear_cap()
        inj['d'] = d_inj
        with torch.no_grad():
            model(torch.tensor([toks], device='cuda'))
        inj['d'] = None
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

    a7a_ok = bool(C17_int.std(axis=0).min() > 0)
    log('a7a non-degeneracy C17 std min %.3e ok=%s'
        % (C17_int.std(axis=0).min(), a7a_ok), lines)

    dose_scales = DOSE * n17_ref

    anchor_ok = bool(a1_ok and a2_ok and a3_ok and a4_ok
                     and a7a_ok)
    verdict = None
    med_I_layer = None
    a5_max = a6_max = 0.0
    t1 = t2 = t3 = None
    save = {}
    I_h_med = None
    I_layer = None
    g_med = np.full(NH, np.nan)
    cos_med = np.full(NH, np.nan)
    n_den_ok = 0

    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        # ---------- stage1: 4 conditions ----------
        profs = {k: np.zeros((74, NL))
                 for k in ('00', '10', '01', '11')}
        xh = {k: np.zeros((74, NH, HD))
              for k in ('00', '10', '01', '11')}
        for i, w in enumerate(wlist):
            toks = [func_tid, tid_map[w]]
            s_w = float(dose_scales[i])
            ops = {'00': forward1(toks, None),
                   '10': forward1(toks, s_w * dl_u),
                   '01': forward1(toks, s_w * dc_u),
                   '11': forward1(toks,
                                  s_w * (dl_u + dc_u))}
            for key, op in ops.items():
                _, prof = contributions(op)
                profs[key][i] = prof
                xv = op[L_INJ].reshape(-1)
                xh[key][i] = xv.reshape(NH, HD)
            if (i + 1) % 20 == 0:
                log('stage1 [%d/74]' % (i + 1), lines)

        I_layer = (profs['11'] - profs['10']
                   - profs['01'] + profs['00'])[:, L_INJ]
        a5_max = float(np.abs(I_layer - I77).max())
        a5_ok = bool(a5_max < 1e-4)
        log('a5 layer I vs 2977 per-word max|d| %.2e ok=%s'
            % (a5_max, a5_ok), lines)

        Mh = M[L_INJ].reshape(NH, HD)
        Ch = {k: (xh[k] * Mh.reshape(1, NH, HD)
                  ).sum(axis=2) for k in xh}
        I_h = (Ch['11'] - Ch['10'] - Ch['01']
               + Ch['00'])  # (74, NH)
        I_h_med = np.median(I_h, axis=0)
        a6_max = float(np.abs(I_h_med - Igrid78[L_INJ])
                       .max())
        a6_ok = bool(a6_max < 1e-4)
        log('a6 per-head I median vs 2978 I_grid_med[17] '
            'max|d| %.2e ok=%s' % (a6_max, a6_ok), lines)

        dx10 = xh['10'] - xh['00']
        dx01 = xh['01'] - xh['00']
        dx11 = xh['11'] - xh['00']
        den = (np.linalg.norm(dx10, axis=2)
               + np.linalg.norm(dx01, axis=2))
        num = np.linalg.norm(dx11, axis=2)
        g_all = np.where(den > 1e-8,
                         num / np.maximum(den, 1e-30),
                         np.nan)
        ssum = dx10 + dx01
        cos_all = np.full((74, NH), np.nan)
        for h in range(NH):
            a = dx11[:, h, :]
            b = ssum[:, h, :]
            na = np.linalg.norm(a, axis=1)
            nb = np.linalg.norm(b, axis=1)
            ok = (na > 1e-8) & (nb > 1e-8)
            cos_all[:, h] = np.where(
                ok, (a * b).sum(axis=1)
                / np.maximum(na * nb, 1e-30), np.nan)
        n_den_ok = int((den[:, H12] > 1e-8).sum())
        a7_ok = bool(a7a_ok and n_den_ok >= 70)
        log('a7 denominator gate h12: %d/74 ok=%s'
            % (n_den_ok, a7_ok), lines)

        if not (a5_ok and a6_ok and a7_ok):
            verdict = 'anchor_fail_all_void'
        else:
            # ---------- T1: designated subset ----------
            med_I_layer = float(np.median(I_layer))
            t1 = {}
            for hn, h in (('h12', H12), ('h9', H9)):
                v = I_h[:, h]
                obs = float(np.median(v))
                rng = np.random.default_rng(
                    RNG_T['I_' + hn])
                p = perm_p(v, obs, rng)
                sig = bool(p <= 0.01)
                share = (obs / med_I_layer
                         if abs(med_I_layer) > 1e-12
                         else float('nan'))
                t1[hn] = {
                    'head': h,
                    'obs_median_Ih': round(obs, 6),
                    'perm_p': float('%.3e' % p),
                    'sig': sig,
                    'share_of_layer_median':
                        round(float(share), 3),
                    'n_words_used':
                        int((np.abs(v) > 0).sum())}
                log('T1 %s: Ih med %+.6f p %.3e sig=%s '
                    'share=%.3f'
                    % (hn, obs, p, sig, share), lines)

            # ---------- T2: mechanism ----------
            t2 = {}
            g12 = g_all[:, H12]
            c12 = cos_all[:, H12]
            gm12 = float(np.nanmedian(g12))
            cm12 = float(np.nanmedian(c12))
            dg = g12[np.isfinite(g12)] - 1.0
            dcsv = c12[np.isfinite(c12)] - 1.0
            pg = perm_p(dg, float(np.median(dg)),
                        np.random.default_rng(
                            RNG_T['g_h12']))
            pc = perm_p(dcsv, float(np.median(dcsv)),
                        np.random.default_rng(
                            RNG_T['cos_h12']))
            g_sig = bool(pg <= 0.01)
            c_sig = bool(pc <= 0.01)
            t2['h12'] = {
                'median_g': round(gm12, 4),
                'g_vs_1_p': float('%.3e' % pg),
                'g_sig': g_sig,
                'median_cos': round(cm12, 4),
                'cos_vs_1_p': float('%.3e' % pc),
                'cos_sig': c_sig}
            log('T2 h12: g med %.4f (p %.3e) cos med %.4f '
                '(p %.3e)'
                % (gm12, pg, cm12, pc), lines)
            g9 = g_all[:, H9]
            c9 = cos_all[:, H9]
            dg9 = g9[np.isfinite(g9)] - 1.0
            dc9 = c9[np.isfinite(c9)] - 1.0
            pg9 = perm_p(dg9, float(np.median(dg9)),
                         np.random.default_rng(
                             RNG_T['g_h9']))
            pc9 = perm_p(dc9, float(np.median(dc9)),
                         np.random.default_rng(
                             RNG_T['cos_h9']))
            t2['h9'] = {
                'median_g':
                    round(float(np.nanmedian(g9)), 4),
                'g_vs_1_p': float('%.3e' % pg9),
                'g_sig': bool(pg9 <= 0.01),
                'median_cos':
                    round(float(np.nanmedian(c9)), 4),
                'cos_vs_1_p': float('%.3e' % pc9),
                'cos_sig': bool(pc9 <= 0.01)}
            g_med = np.nanmedian(
                np.where(np.isfinite(g_all), g_all,
                         np.nan), axis=0)
            cos_med = np.nanmedian(
                np.where(np.isfinite(cos_all), cos_all,
                         np.nan), axis=0)
            t2['all_head_median_g'] = round(
                float(np.nanmedian(g_med)), 4)
            t2['all_head_median_cos'] = round(
                float(np.nanmedian(cos_med)), 4)
            log('T2 all-head: g med %.4f cos med %.4f'
                % (t2['all_head_median_g'],
                   t2['all_head_median_cos']), lines)

            # ---------- T3 descriptive ----------
            wk_norm = np.zeros(NH)
            for h in range(NH):
                wh = u35 @ Wo17[:,
                                h * HD:(h + 1) * HD]
                wk_norm[h] = float(
                    np.linalg.norm(wh))
            rank12 = int((wk_norm
                          > wk_norm[H12]).sum()) + 1
            rank9 = int((wk_norm
                         > wk_norm[H9]).sum()) + 1
            a10 = dx10[:, H12, :]
            a01 = dx01[:, H12, :]
            na10 = np.linalg.norm(a10, axis=1)
            na01 = np.linalg.norm(a01, axis=1)
            ov = np.where((na10 > 1e-8)
                          & (na01 > 1e-8),
                          (a10 * a01).sum(axis=1)
                          / np.maximum(na10 * na01,
                                       1e-30), np.nan)
            t3 = {
                'wk_norm_h12': round(
                    float(wk_norm[H12]), 4),
                'wk_norm_h9': round(
                    float(wk_norm[H9]), 4),
                'wk_norm_max': round(
                    float(wk_norm.max()), 4),
                'wk_norm_h12_rank_of32': rank12,
                'wk_norm_h9_rank_of32': rank9,
                'cos_dx10_dx01_h12_med': round(
                    float(np.nanmedian(ov)), 4),
                'norm_dx10_h12_med': round(
                    float(np.median(na10)), 6),
                'norm_dx01_h12_med': round(
                    float(np.median(na01)), 6),
                'norm_dx11_h12_med': round(
                    float(np.median(
                        np.linalg.norm(
                            dx11[:, H12, :], axis=1))),
                    6)}
            log('T3 wk_norm h12 %.4f (rank %d/32) h9 '
                '%.4f (rank %d/32) max %.4f | '
                'cos(dx10,dx01) med %.4f'
                % (wk_norm[H12], rank12, wk_norm[H9],
                   rank9, wk_norm.max(),
                   t3['cos_dx10_dx01_h12_med']), lines)

            # ---------- verdict ----------
            if g_sig and gm12 < 1.0:
                verdict = \
                    'h12_input_sublinear_carrier'
            elif g_sig and gm12 > 1.0:
                verdict = \
                    'h12_input_superlinear_carrier'
            elif (not g_sig) and c_sig:
                verdict = \
                    'h12_input_redirect_carrier'
            else:
                verdict = \
                    'h12_linear_response_' \
                    'contradiction'
            save = {'I_h': I_h,
                    'I_layer': I_layer,
                    'g_all': g_all,
                    'cos_all': cos_all,
                    'wk_norm': wk_norm,
                    'n17': n17,
                    'lab_lang': lab_lang,
                    'lab_grp': lab_grp,
                    'tids': np.array([tid_map[w]
                                      for w in wlist]),
                    'words': np.array(words79)}

    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2981, 'model': 'qwen3-4b',
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
                       'a7_ok': a7_ok
                       if anchor_ok else None,
                       'ok': bool(anchor_ok and a5_ok
                                  and a6_ok and a7_ok)
                       if anchor_ok else False},
           'T1': t1, 'T2': t2, 'T3': t3,
           'median_I_layer': round(med_I_layer, 6)
           if med_I_layer is not None else None,
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if verdict != 'anchor_fail_all_void' and save:
        np.savez_compressed(os.path.join(
            OUT, 'h12_functional_portrait.npz'), **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2981 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
