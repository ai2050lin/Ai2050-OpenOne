# -*- coding: utf-8 -*-
"""Phase 2983: analytic decomposition of the L17 two-axis
interaction (2980-2982: h12 carries 98.7%, its input
response is sublinear g=0.890 / cos=0.907; 2982 killed
the group-level three-element law and found h9's
self-cancelling anti-parallel channels).

Design (frozen): 2982 protocol VERBATIM (370 forwards).
Readout linearity gives the exact per-word identity
I_h(w) = Mh . delta_h(w), delta = dx11 - (dx10 + dx01)
(the non-additive residual displacement in the 128-d
head input space). Decompose delta per word:
  par  = (delta . shat) shat   (compression along the
          additive sum axis - the saturation direction)
  perp = delta - par           (orthogonal redirect)
and project both through the head read vector
Mh = u35 @ Wo17_h.

T1 (main, preregistered): channel dominance
  share = sum_h |median_w Mh.par| /
          (sum_h |median_w Mh.par| +
           sum_h |median_w Mh.perp|)
  Null: word-pairing permutation - replace s(w) by
  s(pi(w)) (destroys per-word additive coupling, keeps
  marginals), 10000 perms rng 29830, one-sided
  p = P(share_null >= share_obs).
  Gates: share >= 0.7 & p <= 0.01 =>
    interaction_from_sum_axis_compression;
  share <= 0.3 & p <= 0.01 =>
    interaction_from_orthogonal_redirect.

T2 (per-word, within h12): Spearman(I_h12(w),
  deficit(w) = ||dx11|| - ||dx10|| - ||dx01||) rng 29831;
  Spearman(I_h12(w), cos(dx11, s)) rng 29832; gates
  p <= 0.01 => word_level_deficit/direction_tracks.

T3 (descriptive, quasi-post-hoc): per-head ||delta_bar||
  vs |Mh . delta_bar| ratio table; h9 self-cancellation
  audit vs h12.

Anchors: a1 shape; a2 dirs; a3 n17 vs 2979; a4 tokens
== 2982; a5 I_h vs 2982 bit-level; a6 g_all vs 2982;
a7 cos_all vs 2982; a8 wk_norm vs 2982; a9 layer I vs
2977 < 1e-4; a10 identity I_r == I_h fp < 1e-8;
a11 ov_all vs 2982 bit-level; a12 finite-head coverage
>= 30 for both channels.

Verdict: anchor fail => anchor_fail_all_void;
coverage fail => coverage_fail_all_void; then T1/T2
branches above; else => analytic_decomposition_all_void.
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
SRC_2982 = os.path.join(BASE, 'phase2982',
                        'carrier_conjunctive_anatomy',
                        'carrier_conjunctive_anatomy.npz')
OUT = os.path.join(BASE, 'phase2983',
                   'interaction_analytic_decomposition')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2983_run_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NH, HD = 32, 128
NL = 36
L_INJ = 17
H9, H12 = 9, 12
DOSE = 0.1
N_PERM = 10000
RNG_T = {'share': 29830, 'sp_deficit': 29831,
         'sp_cosdir': 29832}

PREREG = {
    'mode': '2982 protocol verbatim: 74 intact forwards '
            '(n17/dose check) + 74 words x 4 conditions '
            '{00,10,01,11} at dose 0.1 (2979 _u dirs, '
            'dose_scales = 0.1 * n17_ref), capture L17 '
            'o_proj input. total 370 forwards',
    'question': 'the L17 interaction I_h is carried 98.7% '
                'by h12: analytically, I_h(w) = Mh.delta_h(w) '
                'with delta = dx11-(dx10+dx01). Which channel '
                'carries the head-level interaction - '
                'compression along the additive sum axis '
                '(par) or orthogonal redirect (perp)? And '
                'within h12, which per-word factor (norm '
                'deficit vs direction) tracks I?',
    'identity': 'par/perp split of delta is exact algebra; '
                'WHICH channel dominates is empirical '
                '(either could) - no discipline-17 issue',
    'anchors': {
        'a1': 'o_proj shape L17 == (2560, 4096)',
        'a2': '2979 _u dirs unit-norm < 1e-9 AND cos(u, '
              '2977 raw) > 1-1e-9',
        'a3': 'intact n17 vs 2979 npz rel < 1e-6',
        'a4': 'single-token 74/74 + list == 2982',
        'a5': 'I_h vs 2982 npz per-word max|d| < 1e-9',
        'a6': 'g_all vs 2982 npz max|d| < 1e-9 (finite)',
        'a7': 'cos_all vs 2982 npz max|d| < 1e-9 (finite)',
        'a8': 'wk_norm vs 2982 npz max|d| < 1e-9',
        'a9': 'layer I(0.1,0.1) vs 2977 npz per-word '
              'max|d| < 1e-4',
        'a10': 'algebraic identity: max|I_r - (ipar_r + '
               'iperp_r)| < 1e-8 (finite)',
        'a11': 'ov_all vs 2982 npz max|d| < 1e-9 (finite)',
        'a12': 'finite-head coverage >= 30 for par and '
               'perp medians',
    },
    'T1': 'main: channel dominance share (par vs perp), '
          'word-pairing permutation null (10000, rng '
          '29830), one-sided p; gates share>=0.7 / <=0.3 '
          'with p<=0.01',
    'T2': 'per-word within h12: Spearman(I_h12, deficit) '
          'rng 29831 and Spearman(I_h12, cos(dx11,s)) rng '
          '29832, label permutation, gate p <= 0.01',
    'T3': 'descriptive (quasi-post-hoc): per-head '
          '||delta_bar|| vs |Mh.delta_bar| ratio; h9 '
          'self-cancellation audit',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'coverage fail => coverage_fail_all_void; '
               'share>=0.7 & p<=0.01 => '
               'interaction_from_sum_axis_compression; '
               'share<=0.3 & p<=0.01 => '
               'interaction_from_orthogonal_redirect; '
               'p_def<=0.01 => '
               'word_level_deficit_tracks_interaction; '
               'p_cos<=0.01 => '
               'word_level_direction_tracks_interaction; '
               'else => analytic_decomposition_all_void',
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


def perm_spearman(x, y, rng):
    obs = spearman(x, y)
    cnt = 0
    for _ in range(N_PERM):
        if abs(spearman(x, rng.permutation(y))) \
                >= abs(obs) - 1e-12:
            cnt += 1
    return obs, (cnt + 1) / (N_PERM + 1)


def channel_share(dx11_, s_sum_, Mh_):
    """share of |median read-projected par| over
    |median par| + |median perp|; returns share, nfin."""
    delta_ = dx11_ - s_sum_
    ns_ = np.linalg.norm(s_sum_, axis=2)
    ok_ = ns_ > 1e-8
    shat_ = np.where(
        ok_[..., None],
        s_sum_ / np.maximum(ns_, 1e-30)[..., None],
        0.0)
    cpar_ = (delta_ * shat_).sum(axis=2)
    ipar_ = cpar_[..., None] * shat_
    iperp_ = delta_ - ipar_
    ipar_r = (ipar_ * Mh_.reshape(1, NH, HD)).sum(axis=2)
    iperp_r = (iperp_ * Mh_.reshape(1, NH, HD)).sum(axis=2)
    m_par = np.nanmedian(
        np.where(np.isfinite(ipar_r), ipar_r, np.nan),
        axis=0)
    m_perp = np.nanmedian(
        np.where(np.isfinite(iperp_r), iperp_r, np.nan),
        axis=0)
    fin_ = np.isfinite(m_par) & np.isfinite(m_perp)
    s_par = float(np.abs(m_par[fin_]).sum())
    s_perp = float(np.abs(m_perp[fin_]).sum())
    tot = s_par + s_perp
    share = s_par / tot if tot > 1e-30 else np.nan
    return share, int(fin_.sum()), ipar_r, iperp_r


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2983,
                   'name': 'interaction_analytic_'
                           'decomposition',
                   'created': time.strftime(
                       '%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2927': sha8(SRC_2927),
                               's2977': sha8(SRC_2977),
                               's2979': sha8(SRC_2979),
                               's2982': sha8(SRC_2982)},
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
    z82 = np.load(SRC_2982, allow_pickle=True)

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
    I82 = z82['I_h'].astype(np.float64)
    g82 = z82['g_all'].astype(np.float64)
    c82 = z82['cos_all'].astype(np.float64)
    wk82 = z82['wk_norm'].astype(np.float64)
    ov82 = z82['ov_all'].astype(np.float64)
    words82 = [str(w) for w in z82['words']]

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
    a4_ok = bool(n_single == 74 and words82 == words79)
    log('a4 single-token %d/74, list==2982 ok=%s'
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
    a10_max = a11_max = 0.0
    a5_ok = a6_ok = a7_ok = a8_ok = a9_ok = False
    a10_ok = a11_ok = a12_ok = False
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
        a5_max = float(np.abs(I_h - I82).max())
        a5_ok = bool(a5_max < 1e-9)
        log('a5 I_h vs 2982 per-word max|d| %.2e ok=%s'
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
        fin = np.isfinite(g_all) & np.isfinite(g82)
        a6_max = float(np.abs(
            g_all[fin] - g82[fin]).max()) if fin.any() \
            else float('nan')
        a6_ok = bool(np.isfinite(a6_max)
                     and a6_max < 1e-9)
        log('a6 g_all vs 2982 max|d| (finite) %.2e ok=%s'
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
        fin2 = np.isfinite(cos_all) & np.isfinite(c82)
        a7_max = float(np.abs(
            cos_all[fin2] - c82[fin2]).max()) \
            if fin2.any() else float('nan')
        a7_ok = bool(np.isfinite(a7_max)
                     and a7_max < 1e-9)
        log('a7 cos_all vs 2982 max|d| (finite) %.2e '
            'ok=%s' % (a7_max, a7_ok), lines)

        wk_norm = np.zeros(NH)
        for h in range(NH):
            wh = u35 @ Wo17[:,
                            h * HD:(h + 1) * HD]
            wk_norm[h] = float(
                np.linalg.norm(wh))
        a8_max = float(np.abs(wk_norm - wk82).max())
        a8_ok = bool(a8_max < 1e-9)
        log('a8 wk_norm vs 2982 max|d| %.2e ok=%s'
            % (a8_max, a8_ok), lines)

        fin11 = np.isfinite(ov_all) & np.isfinite(ov82)
        a11_max = float(np.abs(
            ov_all[fin11] - ov82[fin11]).max()) \
            if fin11.any() else float('nan')
        a11_ok = bool(np.isfinite(a11_max)
                      and a11_max < 1e-9)
        log('a11 ov_all vs 2982 max|d| (finite) %.2e '
            'ok=%s' % (a11_max, a11_ok), lines)

        if not (a5_ok and a6_ok and a7_ok and a8_ok
                and a9_ok and a11_ok):
            verdict = 'anchor_fail_all_void'
        else:
            # ---------- a10: algebraic identity ----------
            share_obs, nfin, ipar_r, iperp_r = \
                channel_share(dx11, ssum, Mh)
            I_r = (dx11 * Mh.reshape(1, NH, HD)) \
                .sum(axis=2)
            finid = np.isfinite(ipar_r) \
                & np.isfinite(iperp_r) \
                & np.isfinite(I_r) \
                & np.isfinite(I_h)
            a10_max = float(np.abs(
                (ipar_r + iperp_r - I_h)[finid]).max()) \
                if finid.any() else float('nan')
            a10_ok = bool(np.isfinite(a10_max)
                          and a10_max < 1e-8)
            log('a10 identity |I_r-(ipar+iperp)| max %.2e '
                'ok=%s' % (a10_max, a10_ok), lines)

            # ---------- a12: coverage ----------
            a12_ok = bool(nfin >= 30)
            log('a12 finite-head coverage %d/32 ok=%s'
                % (nfin, a12_ok), lines)

            if not (a10_ok and a12_ok):
                verdict = 'coverage_fail_all_void'
            else:
                # ---------- T1: main ----------
                rng_sh = np.random.default_rng(
                    RNG_T['share'])
                cnt = 0
                for _ in range(N_PERM):
                    pi = rng_sh.permutation(74)
                    sh_n, nf_n, _, _ = channel_share(
                        dx11, ssum[pi], Mh)
                    if np.isfinite(sh_n) \
                            and sh_n >= share_obs - 1e-12:
                        cnt += 1
                p_share = (cnt + 1) / (N_PERM + 1)
                t1 = {
                    'share_par': round(
                        float(share_obs), 4),
                    'n_finite_heads': nfin,
                    'p_share_right': float(
                        '%.3e' % p_share)}
                log('T1 share_par %.4f (nfin %d) '
                    'p_right %.3e'
                    % (share_obs, nfin, p_share),
                    lines)

                # ---------- T2: within h12 ----------
                nd11 = np.linalg.norm(dx11[:, H12, :],
                                      axis=1)
                n10h = np.linalg.norm(
                    dx10[:, H12, :], axis=1)
                n01h = np.linalg.norm(
                    dx01[:, H12, :], axis=1)
                deficit = nd11 - (n10h + n01h)
                sh12 = ssum[:, H12, :]
                nsh = np.linalg.norm(sh12, axis=1)
                cosdir = np.where(
                    nsh > 1e-8,
                    (dx11[:, H12, :] * sh12).sum(axis=1)
                    / np.maximum(nd11 * nsh, 1e-30),
                    np.nan)
                y12 = I_h[:, H12]
                finD = np.isfinite(deficit) \
                    & np.isfinite(y12)
                finC = np.isfinite(cosdir) \
                    & np.isfinite(y12)
                rng_d = np.random.default_rng(
                    RNG_T['sp_deficit'])
                rho_d, p_def = perm_spearman(
                    deficit[finD], y12[finD], rng_d)
                rng_c = np.random.default_rng(
                    RNG_T['sp_cosdir'])
                rho_c, p_cos = perm_spearman(
                    cosdir[finC], y12[finC], rng_c)
                t2 = {
                    'rho_deficit': round(rho_d, 4),
                    'p_deficit': float(
                        '%.3e' % p_def),
                    'n_deficit': int(finD.sum()),
                    'rho_cosdir': round(rho_c, 4),
                    'p_cosdir': float(
                        '%.3e' % p_cos),
                    'n_cosdir': int(finC.sum())}
                log('T2 h12: rho_deficit %.4f (p %.3e) '
                    'rho_cosdir %.4f (p %.3e)'
                    % (rho_d, p_def, rho_c, p_cos),
                    lines)

                # ---------- T3: descriptive ----------
                delta = dx11 - ssum
                d_bar = np.nanmedian(
                    np.where(np.isfinite(delta),
                             delta, np.nan), axis=0)
                proj_bar = np.abs(
                    (d_bar * Mh).sum(axis=1))
                dnorm_med = np.nanmedian(
                    np.where(np.isfinite(
                        np.linalg.norm(delta, axis=2)),
                        np.linalg.norm(delta, axis=2),
                        np.nan), axis=0)
                ratio = np.where(
                    dnorm_med > 1e-8,
                    proj_bar
                    / np.maximum(dnorm_med, 1e-30),
                    np.nan)
                t3 = {
                    'h9_dnorm': round(
                        float(dnorm_med[H9]), 4),
                    'h9_proj': round(
                        float(proj_bar[H9]), 6),
                    'h9_ratio': round(
                        float(ratio[H9]), 4),
                    'h12_dnorm': round(
                        float(dnorm_med[H12]), 4),
                    'h12_proj': round(
                        float(proj_bar[H12]), 6),
                    'h12_ratio': round(
                        float(ratio[H12]), 4),
                    'ratio_median_all': round(
                        float(np.nanmedian(ratio)), 4),
                    'ratio_rank_h12': int(
                        (ratio > ratio[H12]).sum())
                    + 1,
                    'ratio_rank_h9': int(
                        (ratio > ratio[H9]).sum())
                    + 1,
                    'note': 'quasi-post-hoc; '
                            'descriptive only'}
                log('T3 h9 dnorm %.4f proj %.6f ratio %.4f '
                    '| h12 dnorm %.4f proj %.6f ratio %.4f '
                    '| med ratio %.4f'
                    % (dnorm_med[H9], proj_bar[H9],
                       ratio[H9], dnorm_med[H12],
                       proj_bar[H12], ratio[H12],
                       float(np.nanmedian(ratio))),
                    lines)

                # ---------- verdict ----------
                if share_obs >= 0.7 \
                        and p_share <= 0.01:
                    verdict = \
                        'interaction_from_sum_axis_' \
                        'compression'
                elif share_obs <= 0.3 \
                        and p_share <= 0.01:
                    verdict = \
                        'interaction_from_orthogonal_' \
                        'redirect'
                elif p_def <= 0.01:
                    verdict = \
                        'word_level_deficit_tracks_' \
                        'interaction'
                elif p_cos <= 0.01:
                    verdict = \
                        'word_level_direction_tracks_' \
                        'interaction'
                else:
                    verdict = \
                        'analytic_decomposition_' \
                        'all_void'
                save = {'ov_all': ov_all,
                        'I_h': I_h,
                        'I_layer': I_layer,
                        'g_all': g_all,
                        'cos_all': cos_all,
                        'wk_norm': wk_norm,
                        'ipar_r': ipar_r,
                        'iperp_r': iperp_r,
                        'share_obs': np.float64(
                            share_obs),
                        'deficit_h12': deficit,
                        'cosdir_h12': cosdir,
                        'd_bar': d_bar,
                        'proj_bar': proj_bar,
                        'dnorm_med': dnorm_med,
                        'ratio': ratio,
                        'n17': n17,
                        'lab_lang': lab_lang,
                        'lab_grp': lab_grp,
                        'tids': np.array(
                            [tid_map[w]
                             for w in wlist]),
                        'words': np.array(words79)}

    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2983, 'model': 'qwen3-4b',
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
                       'a10_max': float('%.3e' % a10_max)
                       if anchor_ok else None,
                       'a10_ok': a10_ok
                       if anchor_ok else None,
                       'a11_max': float('%.3e' % a11_max)
                       if anchor_ok else None,
                       'a11_ok': a11_ok
                       if anchor_ok else None,
                       'a12_ok': a12_ok
                       if anchor_ok else None,
                       'ok': bool(anchor_ok and a5_ok
                                  and a6_ok and a7_ok
                                  and a8_ok and a9_ok
                                  and a11_ok and a10_ok
                                  and a12_ok)
                       if anchor_ok else False},
           'T1': t1, 'T2': t2, 'T3': t3,
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if verdict not in ('anchor_fail_all_void',
                       'coverage_fail_all_void') \
            and save:
        np.savez_compressed(os.path.join(
            OUT, 'interaction_analytic_'
                 'decomposition.npz'),
            **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2983 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
