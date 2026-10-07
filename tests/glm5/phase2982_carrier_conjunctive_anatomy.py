# -*- coding: utf-8 -*-
"""Phase 2982: three-element conjunctive anatomy of the
competitive carrier heads h12 vs h9 (2981: h12 carries
98.7% of the L17 two-axis interaction with g=0.890 /
cos=0.907 / overlap 0.549, while h9 is MORE sublinear
with a LARGER write kernel (rank 2/32 vs 7/32) yet
carries only 0.8% - which element discriminates?).

Design (frozen): 2981 protocol VERBATIM (2979 _u dirs,
dose_scales = 0.1 * n17_ref, L17 o_proj input capture,
4 conditions {00,10,01,11}). New quantity: per-head
channel overlap ov_h = median_w cos(dx10_h, dx01_h)
(the channel-level basis of competition). Elements
(oriented so positive rho with |I| is predicted):
e_ov = ov_h; e_sub = -g_med_h (more sublinear = larger);
e_wk = wk_norm_h.

T1 (main, preregistered): Spearman(element, |I_h_med|)
across 32 heads x 3 elements, label-permutation null
(10000, rngs 29820-29822) gate p <= 0.01; primary
question: is overlap the best predictor and is h9
below the all-head overlap median?

Anchors: a1 shape (2560,4096); a2 2979 _u unit + cos vs
2977 raw > 1-1e-9; a3 n17 vs 2979 rel < 1e-6; a4
single-token 74/74 + list == 2981; a5 I_h vs 2981 npz
per-word max|d| < 1e-9 (bit-level); a6 g_all vs 2981
max|d| < 1e-9 (finite); a7 cos_all vs 2981 max|d| <
1e-9 (finite); a8 wk_norm vs 2981 max|d| < 1e-9;
a9 layer I vs 2977 per-word max|d| < 1e-4; a10 element
finite coverage >= 30/32 for all three.

T2 descriptive: h9 vs h12 element table + I_h component
decomposition (median c11/c10/c01/c00). T3 quasi-post-hoc
(discipline 9): S_lo-restricted Spearman overlap-vs-|I|.

Verdict: anchor fail => anchor_fail_all_void; p_ov<=0.01
& rho_ov best & h9 overlap below median =>
channel_overlap_discriminates_carrier; p_ov<=0.01 =>
channel_overlap_significant_not_discriminative; p_g<=0.01
& rho_g>rho_ov => sublinearity_predicts_interaction;
p_wk<=0.01 & rho_wk>rho_ov =>
write_gain_predicts_interaction; else =>
three_element_all_void.
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
SRC_2981 = os.path.join(BASE, 'phase2981',
                        'h12_functional_portrait',
                        'h12_functional_portrait.npz')
OUT = os.path.join(BASE, 'phase2982',
                   'carrier_conjunctive_anatomy')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2982_run_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NH, HD = 32, 128
NL = 36
L_INJ = 17
H9, H12 = 9, 12
DOSE = 0.1
N_PERM = 10000
RNG_T = {'rho_ov': 29820, 'rho_g': 29821,
         'rho_wk': 29822}

PREREG = {
    'mode': '2981 protocol verbatim: 74 intact forwards '
            '(n17/dose check) + 74 words x 4 conditions '
            '{00,10,01,11} at dose 0.1 (2979 _u dirs, '
            'dose_scales = 0.1 * n17_ref), capture L17 '
            'o_proj input, per-head split x_h (32x128). '
            'total 370 forwards',
    'question': 'h12 carries 98.7% of the L17 interaction '
                'but h9 (more sublinear, larger write '
                'kernel) carries 0.8%: which element - '
                'channel overlap cos(dx10,dx01), '
                'sublinearity 1-g, or write-kernel norm - '
                'discriminates carriers from non-carriers '
                'across all 32 heads?',
    'elements': 'oriented for positive rho with |I_h_med|: '
                'e_ov = median_w cos(dx10_h, dx01_h); '
                'e_sub = -median_w g_h; e_wk = '
                '||u35 @ Wo17_h|| (128-d head-input '
                'space)',
    'anchors': {
        'a1': 'o_proj shape L17 == (2560, 4096)',
        'a2': '2979 _u dirs unit-norm < 1e-9 AND cos(u, '
              '2977 raw) > 1-1e-9',
        'a3': 'intact n17 vs 2979 npz rel < 1e-6',
        'a4': 'single-token 74/74 + list == 2981',
        'a5': 'I_h vs 2981 npz per-word max|d| < 1e-9',
        'a6': 'g_all vs 2981 npz max|d| < 1e-9 (finite)',
        'a7': 'cos_all vs 2981 npz max|d| < 1e-9 (finite)',
        'a8': 'wk_norm vs 2981 npz max|d| < 1e-9',
        'a9': 'layer I(0.1,0.1) vs 2977 npz per-word '
              'max|d| < 1e-4',
        'a10': 'finite element coverage >= 30/32 for all '
               'three elements',
    },
    'T1': 'main: Spearman(element, |I_h_med|) across '
          'valid heads x 3 elements, label-permutation '
          'null (10000) gate p <= 0.01; primary: overlap '
          'best predictor AND h9 overlap below all-head '
          'median',
    'T2': 'descriptive: h9/h12 element table + median '
          'c11/c10/c01/c00 component decomposition',
    'T3': 'quasi-post-hoc (discipline 9): S_lo = '
          '[0,4,7,12,13,15,25] (2978) restricted '
          'Spearman overlap vs |I|, descriptive only',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'p_ov<=0.01 & rho_ov best & h9 overlap '
               'below median => '
               'channel_overlap_discriminates_carrier; '
               'p_ov<=0.01 => '
               'channel_overlap_significant_not_'
               'discriminative; p_g<=0.01 & rho_g>rho_ov '
               '=> sublinearity_predicts_interaction; '
               'p_wk<=0.01 & rho_wk>rho_ov => '
               'write_gain_predicts_interaction; else => '
               'three_element_all_void',
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


def ranks(x):
    order = np.argsort(x, kind='stable')
    r = np.empty(len(x))
    r[order] = np.arange(1, len(x) + 1)
    return r


def spearman(x, y):
    rx = ranks(x).astype(np.float64)
    ry = ranks(y).astype(np.float64)
    rx = (rx - rx.mean()) / rx.std()
    ry = (ry - ry.mean()) / ry.std()
    return float((rx * ry).mean())


def perm_rho(x, y, rng):
    obs = spearman(x, y)
    cnt = 0
    for _ in range(N_PERM):
        if abs(spearman(x, rng.permutation(y))) \
                >= abs(obs) - 1e-12:
            cnt += 1
    return obs, (cnt + 1) / (N_PERM + 1)


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2982,
                   'name': 'carrier_conjunctive_anatomy',
                   'created': time.strftime(
                       '%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2927': sha8(SRC_2927),
                               's2977': sha8(SRC_2977),
                               's2979': sha8(SRC_2979),
                               's2981': sha8(SRC_2981)},
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
    z79 = np.load(SRC_2979, allow_pickle=True)
    z81 = np.load(SRC_2981, allow_pickle=True)

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
    log('a2 dirs ok=%s' % a2_ok, lines)

    I77 = (z77['prof11'].astype(np.float64)
           - z77['prof10'].astype(np.float64)
           - z77['prof01'].astype(np.float64)
           + z77['prof00'].astype(np.float64))[:, L_INJ]
    I81 = z81['I_h'].astype(np.float64)
    g81 = z81['g_all'].astype(np.float64)
    c81 = z81['cos_all'].astype(np.float64)
    wk81 = z81['wk_norm'].astype(np.float64)
    words81 = [str(w) for w in z81['words']]

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
    a4_ok = bool(n_single == 74 and words81 == words79)
    log('a4 single-token %d/74, list==2981 ok=%s'
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
    del x17_cap[:]
    for i, w in enumerate(wlist):
        op = forward1([func_tid, tid_map[w]])
        _, _ = contributions(op)
        n17[i] = float(np.linalg.norm(x17_cap[i]
                                      .reshape(-1)))
        if (i + 1) % 20 == 0:
            log('intact [%d/74]' % (i + 1), lines)

    a3_rel = float(np.abs(n17 - n17_ref).max()
                   / max(float(np.abs(n17_ref).max()),
                         1e-30))
    a3_ok = bool(a3_rel < 1e-6)
    log('a3 intact n17 vs 2979 rel %.2e ok=%s'
        % (a3_rel, a3_ok), lines)

    dose_scales = DOSE * n17_ref

    anchor_ok = bool(a1_ok and a2_ok and a3_ok and a4_ok)
    verdict = None
    a5_max = a6_max = a7_max = a8_max = a9_max = 0.0
    a5_ok = a6_ok = a7_ok = a8_ok = a9_ok = False
    a10_ok = False
    t1 = t2 = t3 = None
    save = {}
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
        a9_max = float(np.abs(I_layer - I77).max())
        a9_ok = bool(a9_max < 1e-4)
        log('a9 layer I vs 2977 per-word max|d| %.2e '
            'ok=%s' % (a9_max, a9_ok), lines)

        Mh = M[L_INJ].reshape(NH, HD)
        Ch = {k: (xh[k] * Mh.reshape(1, NH, HD)
                  ).sum(axis=2) for k in xh}
        I_h = (Ch['11'] - Ch['10'] - Ch['01']
               + Ch['00'])  # (74, NH)
        a5_max = float(np.abs(I_h - I81).max())
        a5_ok = bool(a5_max < 1e-9)
        log('a5 I_h vs 2981 per-word max|d| %.2e ok=%s'
            % (a5_max, a5_ok), lines)

        dx10 = xh['10'] - xh['00']
        dx01 = xh['01'] - xh['00']
        dx11 = xh['11'] - xh['00']
        den = (np.linalg.norm(dx10, axis=2)
               + np.linalg.norm(dx01, axis=2))
        num = np.linalg.norm(dx11, axis=2)
        g_all = np.where(den > 1e-8,
                         num / np.maximum(den, 1e-30),
                         np.nan)
        fin = np.isfinite(g_all) & np.isfinite(g81)
        a6_max = float(np.abs(
            g_all[fin] - g81[fin]).max()) if fin.any() \
            else float('nan')
        a6_ok = bool(np.isfinite(a6_max)
                     and a6_max < 1e-9)
        log('a6 g_all vs 2981 max|d| (finite) %.2e ok=%s'
            % (a6_max, a6_ok), lines)

        ssum = dx10 + dx01
        cos_all = np.full((74, NH), np.nan)
        ov_all = np.full((74, NH), np.nan)
        for h in range(NH):
            a = dx11[:, h, :]
            b = ssum[:, h, :]
            na = np.linalg.norm(a, axis=1)
            nb = np.linalg.norm(b, axis=1)
            ok = (na > 1e-8) & (nb > 1e-8)
            cos_all[:, h] = np.where(
                ok, (a * b).sum(axis=1)
                / np.maximum(na * nb, 1e-30), np.nan)
            a10_ = dx10[:, h, :]
            a01_ = dx01[:, h, :]
            n10 = np.linalg.norm(a10_, axis=1)
            n01 = np.linalg.norm(a01_, axis=1)
            ok2 = (n10 > 1e-8) & (n01 > 1e-8)
            ov_all[:, h] = np.where(
                ok2, (a10_ * a01_).sum(axis=1)
                / np.maximum(n10 * n01, 1e-30), np.nan)
        fin2 = np.isfinite(cos_all) & np.isfinite(c81)
        a7_max = float(np.abs(
            cos_all[fin2] - c81[fin2]).max()) \
            if fin2.any() else float('nan')
        a7_ok = bool(np.isfinite(a7_max)
                     and a7_max < 1e-9)
        log('a7 cos_all vs 2981 max|d| (finite) %.2e '
            'ok=%s' % (a7_max, a7_ok), lines)

        wk_norm = np.zeros(NH)
        for h in range(NH):
            wh = u35 @ Wo17[:,
                            h * HD:(h + 1) * HD]
            wk_norm[h] = float(
                np.linalg.norm(wh))
        a8_max = float(np.abs(wk_norm - wk81).max())
        a8_ok = bool(a8_max < 1e-9)
        log('a8 wk_norm vs 2981 max|d| %.2e ok=%s'
            % (a8_max, a8_ok), lines)

        # ---------- a10: element coverage ----------
        g_med = np.nanmedian(
            np.where(np.isfinite(g_all), g_all,
                     np.nan), axis=0)
        cos_med = np.nanmedian(
            np.where(np.isfinite(cos_all), cos_all,
                     np.nan), axis=0)
        ov_med = np.nanmedian(
            np.where(np.isfinite(ov_all), ov_all,
                     np.nan), axis=0)
        e_ov = ov_med
        e_sub = -g_med
        e_wk = wk_norm
        nfin = int(np.isfinite(e_ov).sum()
                   + np.isfinite(e_sub).sum()
                   + np.isfinite(e_wk).sum())
        a10_ok = bool(np.isfinite(e_ov).sum() >= 30
                      and np.isfinite(e_sub).sum() >= 30
                      and np.isfinite(e_wk).sum() >= 30)
        log('a10 element finite coverage ov/sub/wk = '
            '%d/%d/%d ok=%s'
            % (int(np.isfinite(e_ov).sum()),
               int(np.isfinite(e_sub).sum()),
               int(np.isfinite(e_wk).sum()), a10_ok),
            lines)

        if not (a5_ok and a6_ok and a7_ok and a8_ok
                and a9_ok and a10_ok):
            verdict = 'anchor_fail_all_void'
        else:
            # ---------- T1: main ----------
            y = np.abs(np.median(I_h, axis=0))
            valid = (np.isfinite(e_ov)
                     & np.isfinite(e_sub)
                     & np.isfinite(e_wk)
                     & np.isfinite(y))
            yv = y[valid]
            t1 = {'n_valid_heads': int(valid.sum())}
            rng_ov = np.random.default_rng(
                RNG_T['rho_ov'])
            rho_ov, p_ov = perm_rho(
                e_ov[valid], yv, rng_ov)
            rng_g = np.random.default_rng(
                RNG_T['rho_g'])
            rho_g, p_g = perm_rho(
                e_sub[valid], yv, rng_g)
            rng_wk = np.random.default_rng(
                RNG_T['rho_wk'])
            rho_wk, p_wk = perm_rho(
                e_wk[valid], yv, rng_wk)
            ov_med_all = float(np.median(
                e_ov[valid]))
            h9_below = bool(e_ov[H9] < ov_med_all)
            h12_rank = int((e_ov[valid]
                            > e_ov[H12]).sum()) + 1
            h9_rank = int((e_ov[valid]
                           > e_ov[H9]).sum()) + 1
            t1.update({
                'rho_ov': round(rho_ov, 4),
                'p_ov': float('%.3e' % p_ov),
                'rho_sub': round(rho_g, 4),
                'p_sub': float('%.3e' % p_g),
                'rho_wk': round(rho_wk, 4),
                'p_wk': float('%.3e' % p_wk),
                'ov_median_all_heads':
                    round(ov_med_all, 4),
                'ov_h12': round(float(e_ov[H12]), 4),
                'ov_h9': round(float(e_ov[H9]), 4),
                'ov_rank_h12_of32': h12_rank,
                'ov_rank_h9_of32': h9_rank,
                'h9_overlap_below_median': h9_below})
            log('T1 rho_ov %.4f (p %.3e) rho_sub %.4f '
                '(p %.3e) rho_wk %.4f (p %.3e) | ov h12 '
                '%.4f (rank %d) h9 %.4f (rank %d) med '
                '%.4f h9_below=%s'
                % (rho_ov, p_ov, rho_g, p_g, rho_wk,
                   p_wk, e_ov[H12], h12_rank, e_ov[H9],
                   h9_rank, ov_med_all, h9_below), lines)

            # ---------- T2: descriptive ----------
            t2 = {}
            for hn, h in (('h12', H12), ('h9', H9)):
                t2[hn] = {
                    'median_c11': round(float(
                        np.median(Ch['11'][:, h])), 6),
                    'median_c10': round(float(
                        np.median(Ch['10'][:, h])), 6),
                    'median_c01': round(float(
                        np.median(Ch['01'][:, h])), 6),
                    'median_c00': round(float(
                        np.median(Ch['00'][:, h])), 6),
                    'median_Ih': round(float(
                        np.median(I_h[:, h])), 6),
                    'median_g': round(float(g_med[h]),
                                      4),
                    'median_cos_resp': round(
                        float(cos_med[h]), 4),
                    'median_ov': round(
                        float(e_ov[h]), 4),
                    'wk_norm': round(
                        float(wk_norm[h]), 4)}
                log('T2 %s: Ih %+.6f g %.4f cos %.4f ov '
                    '%.4f wk %.4f'
                    % (hn, t2[hn]['median_Ih'],
                       t2[hn]['median_g'],
                       t2[hn]['median_cos_resp'],
                       t2[hn]['median_ov'],
                       t2[hn]['wk_norm']), lines)

            # ---------- T3: quasi-post-hoc ----------
            S_LO = [0, 4, 7, 12, 13, 15, 25]
            msk = np.zeros(NH, dtype=bool)
            msk[S_LO] = True
            msk &= valid
            rho_lo = spearman(e_ov[msk], y[msk]) \
                if msk.sum() >= 4 else None
            t3 = {'s_lo_heads': S_LO,
                  'n_used': int(msk.sum()),
                  'rho_ov_s_lo':
                      round(float(rho_lo), 4)
                      if rho_lo is not None else None,
                  'note': 'quasi-post-hoc (S_lo from '
                          '2978); descriptive only'}
            log('T3 S_lo restricted rho_ov %s'
                % t3['rho_ov_s_lo'], lines)

            # ---------- verdict ----------
            if p_ov <= 0.01 and rho_ov >= rho_g \
                    and rho_ov >= rho_wk and h9_below:
                verdict = \
                    'channel_overlap_discriminates_' \
                    'carrier'
            elif p_ov <= 0.01:
                verdict = \
                    'channel_overlap_significant_' \
                    'not_discriminative'
            elif p_g <= 0.01 and rho_g > rho_ov:
                verdict = \
                    'sublinearity_predicts_interaction'
            elif p_wk <= 0.01 and rho_wk > rho_ov:
                verdict = \
                    'write_gain_predicts_interaction'
            else:
                verdict = 'three_element_all_void'
            save = {'ov_all': ov_all,
                    'I_h': I_h,
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
    res = {'phase': 2982, 'model': 'qwen3-4b',
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
                       'a7_max': float('%.3e' % a7_max)
                       if anchor_ok else None,
                       'a7_ok': a7_ok
                       if anchor_ok else None,
                       'a8_max': float('%.3e' % a8_max)
                       if anchor_ok else None,
                       'a8_ok': a8_ok
                       if anchor_ok else None,
                       'a9_max': float('%.3e' % a9_max)
                       if anchor_ok else None,
                       'a9_ok': a9_ok
                       if anchor_ok else None,
                       'a10_ok': a10_ok
                       if anchor_ok else None,
                       'ok': bool(anchor_ok and a5_ok
                                  and a6_ok and a7_ok
                                  and a8_ok and a9_ok
                                  and a10_ok)
                       if anchor_ok else False},
           'T1': t1, 'T2': t2, 'T3': t3,
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if verdict != 'anchor_fail_all_void' and save:
        np.savez_compressed(os.path.join(
            OUT, 'carrier_conjunctive_anatomy.npz'),
            **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2982 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
