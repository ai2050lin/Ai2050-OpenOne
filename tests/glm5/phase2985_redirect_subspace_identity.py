# -*- coding: utf-8 -*-
"""Phase 2985: redirect subspace identity - geometry of the perp channel.

2983 protocol verbatim (370 forwards): 74 intact + 74 words x 4 conditions
{00,10,01,11} at dose 0.1 (2979 _u dirs, dose_scales = 0.1 * n17_ref).
Capture L17 o_proj input; analyze the word-level nonlinear residual
displacement delta_h12(w) = dx11-(dx10+dx01) in the 128-d head-input space:
  T1 SVD energy concentration (sign-flip null, rng 29851)
  T1b split-half subspace stability (group-relabel null, rng 29852)
  T2 axis alignment of delta-bar vs empirical response directions
     r_lang/r_cls/r_sum (sign-flip null, rng 29853)
Verdict mapping frozen below.
"""
import hashlib
import io
import json
import os
import time

import numpy as np
import torch

from transformers import AutoTokenizer

import sys
sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
from phase2662_symmetric_mapping_contract import \
    load_native

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC_2927 = os.path.join(BASE, 'phase2927', 'probe_relativity',
                        'probe_relativity.npz')
SRC_2977 = os.path.join(BASE, 'phase2977',
                        'two_axis_fusion_injection',
                        'two_axis_fusion_injection.npz')
SRC_2979 = os.path.join(BASE, 'phase2979', 'reversal_anatomy',
                        'reversal_anatomy.npz')
SRC_2983 = os.path.join(BASE, 'phase2983',
                        'interaction_analytic_decomposition',
                        'interaction_analytic_decomposition.npz')
OUT_DIR = os.path.join(BASE, 'phase2985', 'redirect_subspace_identity')

NL, NH, HD = 36, 32, 128
L_INJ = 17
H12 = 12
DOSE = 0.1
N_PERM = 10000
COV_MED = 0.05
RNG_T = {'svd_flip': 29851, 'split_half': 29852, 'axis_align': 29853}

PREREG = {
    'mode': '2983 protocol verbatim: 74 intact forwards (n17/dose check) '
            '+ 74 words x 4 conditions {00,10,01,11} at dose 0.1 '
            '(2979 _u dirs, dose_scales = 0.1 * n17_ref), capture L17 '
            'o_proj input. total 370 forwards',
    'question': '2983 showed the L17 interaction is carried ~85% by the '
                'orthogonal-redirect (perp) channel of delta_h12. What is '
                'the geometric identity of that redirect channel: a '
                'word-invariant low-dim subspace (T1 SVD concentration + '
                'T1b split-half stability), and is it locked to the '
                'injection axes (T2 alignment vs empirical response '
                'directions r_lang/r_cls/r_sum) or intrinsic?',
    'criteria': 'coverage gate: median ||delta_h12(w)|| >= 0.05. '
                'T1 concentrated: top1 SVD energy >= sign-flip null q95. '
                'T1b stable: |cos(u_even1, u_odd1)| >= group-relabel '
                'null q95. T2 aligned: max(|cos(dbar, r_lang)|, '
                '|cos(dbar, r_cls)|, |cos(dbar, r_sum)|) >= sign-flip '
                'null q95 (r directions fixed, dbar word-signs flipped).',
    'identity': 'I_h12(w) = M12 . delta_h12(w) is exact linear-algebra '
                '(readout row times displacement); par/perp split is '
                'exact algebra; no definitional identity enters the '
                'verdict branches (all three gates falsifiable)',
    'verdict': 'coverage fail => coverage_fail_all_void; '
               'T1 & T1b & T2 => redirect_subspace_axis_locked; '
               'T1 & T1b & ~T2 => redirect_intrinsic_fixed_subspace; '
               'T1 & ~T1b => redirect_concentrated_unstable; '
               '~T1 => redirect_diffuse',
    'anchors': 'a1 Wo17 shape; a2 unit dirs vs 2977 raw cos=1; '
               'a3 n17 vs 2979 rel<1e-6; a4 single-token 74/74 + '
               'wordlist==2983; a5 I_h word-level vs 2983 <1e-9; '
               'a6 g_all word-level vs 2983 <1e-9; a7 cos_all vs 2983 '
               '<1e-9; a8 d_bar[h12] vs 2983 <1e-9; a9 layer I vs 2977 '
               'prof word-level <1e-9; a10 I_h12 == M12.delta_h12 '
               'identity <1e-12',
    'correction_note': 'run1: ModuleNotFoundError native_loader - correct import is phase2662_symmetric_mapping_contract.load_native; tokenizer kwargs completed; failed at module-level import, no execution.json frozen, no artifacts; run2: a3 rel 0.78 - n17 was o_proj-input (4096d) norm, must be L17 layer-input (2560d) norm via x17_cap (2980 run1 lesson); anchor_fail path UnboundLocalError cov_ok - containers pre-initialized; artifacts deleted per discipline 3; run3: a3 ok 1.21e-07, a6/a7/a9 bit-level 0.00, but a5/a8/a10 failed - I_h was layer-profile diff (74,) instead of head-level C-matrix diff (74,32) from contributions()[0], and d_bar was mean instead of 2983 per-dim median; fixed, artifacts deleted per discipline 3'
}


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


def main():
    lines = []

    def log(msg, lines=lines):
        lines.append(msg)
        print(msg, flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    # ---------- execution.json frozen BEFORE any observation ----------
    exec_path = os.path.join(OUT_DIR, 'execution.json')
    exec_doc = {
        'phase': 2985, 'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'model': 'qwen3-4b', 'heads': NH, 'head_dim': HD,
        'n_layers': NL, 'inj_layer': L_INJ, 'target_head': H12,
        'dose_rel': DOSE, 'n_forwards': 370, 'n_perm': N_PERM,
        'rng': RNG_T, 'cov_gate': COV_MED, 'prereg': PREREG}
    with io.open(exec_path, 'w', encoding='utf-8') as f:
        json.dump(exec_doc, f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    z27_ = np.load(SRC_2927, allow_pickle=True)
    u35 = z27_['dirs_word'].astype(np.float64)[NL - 1]
    z77 = np.load(SRC_2977, allow_pickle=True)
    z79 = np.load(SRC_2979, allow_pickle=True)
    z83 = np.load(SRC_2983, allow_pickle=True)

    dl_u = z79['d_lang_u'].astype(np.float64)
    dc_u = z79['d_cls_u'].astype(np.float64)
    dl_raw = z77['d_lang'].astype(np.float64)
    dc_raw = z77['d_cls'].astype(np.float64)
    n17_ref = z79['n17'].astype(np.float64)
    words79 = [str(w) for w in z79['words']]
    wlist = [w.split(':')[2] for w in words79]
    lab_lang = np.array([w.split(':')[1] for w in words79])
    lab_grp = np.array([w.split(':')[0] for w in words79])

    a2_n1 = abs(float(np.linalg.norm(dl_u)) - 1.0)
    a2_n2 = abs(float(np.linalg.norm(dc_u)) - 1.0)
    cos_l = float(dl_u @ dl_raw / (
        np.linalg.norm(dl_u) * np.linalg.norm(dl_raw)))
    cos_c = float(dc_u @ dc_raw / (
        np.linalg.norm(dc_u) * np.linalg.norm(dc_raw)))
    a2_ok = bool(a2_n1 < 1e-9 and a2_n2 < 1e-9
                 and abs(cos_l - 1.0) < 1e-9 and abs(cos_c - 1.0) < 1e-9)
    log('a2 unit dirs + vs 2977 raw cos l=%.12f c=%.12f ok=%s'
        % (cos_l, cos_c, a2_ok), lines)

    tok = AutoTokenizer.from_pretrained(
        r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b',
        local_files_only=True, trust_remote_code=True,
        use_fast=True)
    tid_map = {}
    n_single = 0
    for w in wlist:
        ids = tok(' ' + w, add_special_tokens=False)['input_ids']
        if len(ids) != 1:
            ids = tok(w, add_special_tokens=False)['input_ids']
        assert len(ids) == 1, '%s -> %s' % (w, ids)
        tid_map[w] = int(ids[0])
        n_single += 1
    words83 = [str(w) for w in z83['words']]
    a4_ok = bool(n_single == 74 and words83 == words79)
    log('a4 single-token %d/74, list==2983 ok=%s'
        % (n_single, a4_ok), lines)
    ids_the = tok(' the', add_special_tokens=False)['input_ids']
    assert len(ids_the) == 1
    func_tid = int(ids_the[0])

    model, _ = load_native('qwen4')
    model.eval()
    layers = model.model.layers
    log('model loaded', lines)

    Wo17 = layers[L_INJ].self_attn.o_proj.weight.detach() \
        .float().cpu().numpy()
    a1_ok = bool(Wo17.shape == (2560, NH * HD))
    log('a1 o_proj shape L17 %s ok=%s' % (Wo17.shape, a1_ok), lines)

    M = np.zeros((NL, NH * HD))
    for li in range(NL):
        Wl = layers[li].self_attn.o_proj.weight.detach() \
            .float().cpu().numpy()
        M[li] = u35 @ Wl
    M12 = M[L_INJ].reshape(NH, HD)[H12]

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
                x[:, 1, :].detach().float().cpu().numpy())
        return None

    for li in range(NL):
        handles.append(layers[li].self_attn.o_proj
                       .register_forward_pre_hook(
                           hook_op(li), with_kwargs=True))
    handles.append(
        layers[L_INJ].self_attn
        .register_forward_pre_hook(hook_inj, with_kwargs=True))

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
    for i, w in enumerate(wlist):
        op = forward1([func_tid, tid_map[w]])
        _, prof = contributions(op)
        n17[i] = float(np.linalg.norm(
            x17_cap[-1].reshape(-1)))
        if (i + 1) % 20 == 0:
            log('intact [%d/74]' % (i + 1), lines)

    a3_rel = float(np.abs(n17 - n17_ref).max()
                   / max(float(np.abs(n17_ref).max()), 1e-30))
    a3_ok = bool(a3_rel < 1e-6)
    log('a3 intact n17 vs 2979 rel %.2e ok=%s' % (a3_rel, a3_ok), lines)

    dose_scales = DOSE * n17_ref

    anchor_ok = bool(a1_ok and a2_ok and a3_ok and a4_ok)
    cov_ok = False
    cov_med = 0.0
    verdict = None
    a5_max = a6_max = a7_max = a8_max = a9_max = a10_max = 0.0
    a5_ok = a6_ok = a7_ok = a8_ok = a9_ok = a10_ok = False
    t1 = t2 = t3 = None
    save = {}
    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        # ---------- stage1: 4 conditions ----------
        profs = {k: np.zeros((74, NL))
                 for k in ('00', '10', '01', '11')}
        C17 = {k: np.zeros((74, NH))
               for k in ('00', '10', '01', '11')}
        xh = {k: np.zeros((74, NH, HD))
              for k in ('00', '10', '01', '11')}
        for i, w in enumerate(wlist):
            toks = [func_tid, tid_map[w]]
            s_w = float(dose_scales[i])
            ops = {'00': forward1(toks, None),
                   '10': forward1(toks, s_w * dl_u),
                   '01': forward1(toks, s_w * dc_u),
                   '11': forward1(toks, s_w * (dl_u + dc_u))}
            for key, op in ops.items():
                C, prof = contributions(op)
                profs[key][i] = prof
                C17[key][i] = C[L_INJ]
                xv = op[L_INJ].reshape(-1)
                xh[key][i] = xv.reshape(NH, HD)
            if (i + 1) % 10 == 0:
                log('cond [%d/74]' % (i + 1), lines)

        # ---------- anchors vs upstream ----------
        I_h = (C17['11'] - C17['10'] - C17['01']
               + C17['00'])
        I83 = z83['I_h']
        a5_max = float(np.abs(I_h - I83).max())
        a5_ok = bool(a5_max < 1e-9)
        log('a5 I_h (74x32) vs 2983 word-level max|d| '
            '%.2e ok=%s' % (a5_max, a5_ok), lines)

        dx10 = xh['10'] - xh['00']
        dx01 = xh['01'] - xh['00']
        dx11 = xh['11'] - xh['00']
        den = (np.linalg.norm(dx10, axis=2)
               + np.linalg.norm(dx01, axis=2))
        num = np.linalg.norm(dx11, axis=2)
        g_all = np.where(den > 1e-8,
                         num / np.maximum(den, 1e-30), np.nan)
        g83 = z83['g_all']
        fin6 = np.isfinite(g_all) & np.isfinite(g83)
        a6_max = float(np.abs(g_all[fin6] - g83[fin6]).max())
        a6_ok = bool(a6_max < 1e-9)
        log('a6 g_all vs 2983 max|d| %.2e (n=%d) ok=%s'
            % (a6_max, int(fin6.sum()), a6_ok), lines)

        def _cos3(a, b):
            na = np.linalg.norm(a, axis=2)
            nb = np.linalg.norm(b, axis=2)
            m = (na > 1e-8) & (nb > 1e-8)
            out = np.full(a.shape[:2], np.nan)
            out[m] = np.sum(a[m] * b[m], axis=1) / (na[m] * nb[m])
            return out
        cos_all = _cos3(dx11, dx10 + dx01)
        c83 = z83['cos_all']
        fin7 = np.isfinite(cos_all) & np.isfinite(c83)
        a7_max = float(np.abs(cos_all[fin7] - c83[fin7]).max())
        a7_ok = bool(a7_max < 1e-9)
        log('a7 cos_all vs 2983 max|d| %.2e (n=%d) ok=%s'
            % (a7_max, int(fin7.sum()), a7_ok), lines)

        # ---------- T-tests ----------
        D12 = (dx11 - dx10 - dx01)[:, H12, :]          # (74,128)
        dnorm = np.linalg.norm(D12, axis=1)
        cov_med = float(np.median(dnorm))
        cov_ok = bool(cov_med >= COV_MED)
        log('T-cov median ||d_h12||=%.4f gate %.2f ok=%s'
            % (cov_med, COV_MED, cov_ok), lines)

        # a9: layer-level I vs 2977 word-level
        Il77 = (z77['prof11'][:, L_INJ] - z77['prof10'][:, L_INJ]
                - z77['prof01'][:, L_INJ] + z77['prof00'][:, L_INJ])
        a9_max = float(np.abs(
            profs['11'][:, L_INJ] - profs['10'][:, L_INJ]
            - profs['01'][:, L_INJ] + profs['00'][:, L_INJ]
            - Il77).max())
        a9_ok = bool(a9_max < 1e-9)
        log('a9 layer I vs 2977 word-level max|d| %.2e ok=%s'
            % (a9_max, a9_ok), lines)

        # a10: readout linear identity I_h12 == M12 . delta
        id_max = float(np.abs(
            I_h[:, H12] - D12 @ M12).max())
        a10_gate = 1e-9 * max(
            1.0, float(np.abs(I_h[:, H12]).max()))
        a10_ok = bool(id_max < a10_gate)
        log('a10 identity I_h12 vs M12.delta max|d| %.2e '
            'gate %.2e ok=%s'
            % (id_max, a10_gate, a10_ok), lines)

        d_bar = np.median(D12, axis=0)
        a8_max = float(np.abs(d_bar - z83['d_bar'][H12]).max())
        a8_ok = bool(a8_max < 1e-9)
        log('a8 d_bar[h12] vs 2983 max|d| %.2e ok=%s'
            % (a8_max, a8_ok), lines)

        a_ok = bool(a5_ok and a6_ok and a7_ok and a8_ok
                    and a9_ok and a10_ok)
        if not a_ok:
            verdict = 'anchor_fail_all_void'
        else:
            # ----- T1: SVD energy concentration + sign-flip null -----
            sv = np.linalg.svd(D12, compute_uv=False)
            eig = sv ** 2
            e1 = float(eig[0] / eig.sum())
            e3 = float(eig[:3].sum() / eig.sum())
            rng = np.random.default_rng(RNG_T['svd_flip'])
            null_e1 = np.zeros(N_PERM)
            for k in range(N_PERM):
                Dp = D12 * rng.choice(
                    [-1.0, 1.0], size=(74, 1))
                sp = np.linalg.svd(Dp, compute_uv=False)
                ep = sp ** 2
                null_e1[k] = ep[0] / ep.sum()
            thr_e1 = float(np.quantile(null_e1, 0.95))
            t1_conc = bool(e1 >= thr_e1)
            p_e1 = float((np.sum(null_e1 >= e1) + 1)
                         / (N_PERM + 1))
            log('T1 e1=%.4f e3=%.4f nullq95=%.4f p=%.5f conc=%s'
                % (e1, e3, thr_e1, p_e1, t1_conc), lines)

            # ----- T1b: split-half stability -----
            idx_e = np.arange(0, 74, 2)
            idx_o = np.arange(1, 74, 2)
            De = D12[idx_e] - D12[idx_e].mean(axis=0)
            Do = D12[idx_o] - D12[idx_o].mean(axis=0)
            Ve = np.linalg.svd(De, full_matrices=False)[2][0]
            Vo = np.linalg.svd(Do, full_matrices=False)[2][0]
            cos_split = float(abs(Ve @ Vo))
            rng = np.random.default_rng(RNG_T['split_half'])
            null_cs = np.zeros(N_PERM)
            for k in range(N_PERM):
                perm = rng.permutation(74)
                pe = perm[0::2]
                po = perm[1::2]
                Dpe = D12[pe] - D12[pe].mean(axis=0)
                Dpo = D12[po] - D12[po].mean(axis=0)
                v1 = np.linalg.svd(Dpe,
                                   full_matrices=False)[2][0]
                v2 = np.linalg.svd(Dpo,
                                   full_matrices=False)[2][0]
                null_cs[k] = abs(float(v1 @ v2))
            thr_cs = float(np.quantile(null_cs, 0.95))
            t1b_stable = bool(cos_split >= thr_cs)
            p_cs = float((np.sum(null_cs >= cos_split) + 1)
                         / (N_PERM + 1))
            log('T1b cos_split=%.4f nullq95=%.4f p=%.5f stable=%s'
                % (cos_split, thr_cs, p_cs, t1b_stable), lines)

            # ----- T2: axis alignment of dbar -----
            r_lang = dx10[:, H12, :].mean(axis=0)
            r_cls = dx01[:, H12, :].mean(axis=0)
            r_sum = r_lang + r_cls
            axes = {'lang': r_lang, 'cls': r_cls, 'sum': r_sum}
            rng = np.random.default_rng(RNG_T['axis_align'])
            nulls = {k: np.zeros(N_PERM) for k in axes}
            flips = np.array([
                rng.choice([-1.0, 1.0], size=(74, 1))
                for _ in range(N_PERM)])
            for k in range(N_PERM):
                db = (D12 * flips[k]).mean(axis=0)
                for name, r in axes.items():
                    nulls[name][k] = abs(
                        float(db @ r) / max(
                            np.linalg.norm(db)
                            * np.linalg.norm(r), 1e-30))
            cos_obs = {}
            p_obs = {}
            t2_flags = {}
            for name, r in axes.items():
                c = abs(float(d_bar @ r) / max(
                    np.linalg.norm(d_bar)
                    * np.linalg.norm(r), 1e-30))
                cos_obs[name] = c
                p_obs[name] = float(
                    (np.sum(nulls[name] >= c) + 1)
                    / (N_PERM + 1))
                t2_flags[name] = bool(
                    c >= float(np.quantile(nulls[name], 0.95)))
            t2_aligned = bool(any(t2_flags.values()))
            log('T2 cos lang=%.4f(p%.5f) cls=%.4f(p%.5f) '
                'sum=%.4f(p%.5f) aligned=%s'
                % (cos_obs['lang'], p_obs['lang'],
                   cos_obs['cls'], p_obs['cls'],
                   cos_obs['sum'], p_obs['sum'], t2_aligned), lines)

            t1 = {'e1': e1, 'e3': e3, 'null_q95': thr_e1,
                  'p': p_e1, 'concentrated': t1_conc}
            t2 = {'cos_lang': cos_obs['lang'],
                  'p_lang': p_obs['lang'],
                  'cos_cls': cos_obs['cls'],
                  'p_cls': p_obs['cls'],
                  'cos_sum': cos_obs['sum'],
                  'p_sum': p_obs['sum'],
                  'aligned': t2_aligned}

            # ----- T3 descriptive: e-spectrum + M12 relation -----
            cos_m12 = float(abs(d_bar @ M12) / max(
                np.linalg.norm(d_bar)
                * np.linalg.norm(M12), 1e-30))
            spec = [float(eig[i] / eig.sum())
                    for i in range(min(8, len(eig)))]
            t3 = {'e_spectrum_top8': spec,
                  'cos_dbar_M12': cos_m12,
                  'cov_med': cov_med,
                  'r_lang_norm': float(np.linalg.norm(r_lang)),
                  'r_cls_norm': float(np.linalg.norm(r_cls)),
                  'r_cross_cos': float(
                      r_lang @ r_cls / max(
                          np.linalg.norm(r_lang)
                          * np.linalg.norm(r_cls), 1e-30)),
                  'quasi_post_hoc': 'd_bar/r directions were shown '
                                    'side-by-side in 2983 outputs; '
                                    'T3 registered descriptive-only',
                  'note_T4_structural_zero': 'ablation hook sits on '
                  'o_proj input slices so same-layer other heads are '
                  'bit-identical by construction (discipline 17)'}

            # ----- verdict (frozen mapping) -----
            if not cov_ok:
                verdict = 'coverage_fail_all_void'
            elif t1_conc and t1b_stable and t2_aligned:
                verdict = 'redirect_subspace_axis_locked'
            elif t1_conc and t1b_stable:
                verdict = 'redirect_intrinsic_fixed_subspace'
            elif t1_conc:
                verdict = 'redirect_concentrated_unstable'
            else:
                verdict = 'redirect_diffuse'

            save.update({
                'e1': e1, 'e3': e3, 'thr_e1': thr_e1,
                'p_e1': p_e1, 'cos_split': cos_split,
                'thr_cs': thr_cs, 'p_cs': p_cs,
                'cos_obs': cos_obs, 'p_obs': p_obs,
                't2_flags': t2_flags, 't1_conc': t1_conc,
                't1b_stable': t1b_stable, 't2_aligned': t2_aligned,
                'null_e1_q95': thr_e1, 'null_cs_q95': thr_cs,
                'D12_norms': dnorm, 'D12': D12,
                'd_bar': d_bar, 'r_lang': r_lang, 'r_cls': r_cls,
                'e_spectrum_top8': spec,
                'cos_dbar_M12': cos_m12,
                'words': np.array(words79),
                'lab_lang': lab_lang, 'lab_grp': lab_grp,
                'I_h': I_h, 'g_all': g_all, 'cos_all': cos_all,
                'n17': n17, 'dnorm_med': cov_med})

        # word-level anchors stored regardless (only when
        # the else branch ran and defined these arrays)
    if save:
        save.update({
            'D12_norms': dnorm, 'D12': D12, 'd_bar': d_bar,
            'I_h': I_h, 'g_all': g_all, 'cos_all': cos_all,
            'n17': n17, 'words': np.array(words79),
            'lab_lang': lab_lang, 'lab_grp': lab_grp})

    # ---------- result.json ----------
    exec_sha = sha8(exec_path)
    result = {
        'phase': 2985, 'final_verdict': verdict,
        'anchors': {
            'a1': {'ok': a1_ok},
            'a2': {'ok': a2_ok, 'cos_lang_raw': cos_l,
                   'cos_cls_raw': cos_c},
            'a3': {'ok': a3_ok, 'rel': a3_rel},
            'a4': {'ok': a4_ok, 'n_single': n_single},
            'a5': {'ok': a5_ok, 'max': a5_max,
                   'note': 'I_h[h12] word-level vs 2983'},
            'a6': {'ok': a6_ok, 'max': a6_max,
                   'note': 'g_all vs 2983'},
            'a7': {'ok': a7_ok, 'max': a7_max,
                   'note': 'cos_all vs 2983'},
            'a8': {'ok': a8_ok, 'max': a8_max,
                   'note': 'd_bar[h12] vs 2983'},
            'a9': {'ok': a9_ok, 'max': a9_max,
                   'note': 'layer I vs 2977 word-level'},
            'a10': {'ok': a10_ok, 'max': a10_max,
                    'note': 'identity I_h12 == M12.delta'},
            'coverage': {'ok': cov_ok, 'median': cov_med},
        },
        'anchor_all_ok': bool(
            a1_ok and a2_ok and a3_ok and a4_ok and a5_ok
            and a6_ok and a7_ok and a8_ok and a9_ok and a10_ok
            and cov_ok),
        'T1': t1, 'T2': t2, 'T3': t3,
        'correction_note': PREREG['correction_note'],
        'elapsed_s': round(time.time() - t0, 1),
    }
    res_path = os.path.join(OUT_DIR, 'result.json')
    with io.open(res_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False,
                  default=lambda o: o.item()
                  if hasattr(o, 'item') else str(o))
    npz_path = os.path.join(OUT_DIR,
                            'redirect_subspace_identity.npz')
    np.savez_compressed(npz_path, **save)
    log('verdict=%s result+npz saved (%.1fs)'
        % (verdict, time.time() - t0), lines)

    io.open(os.path.join(OUT_DIR, 'run_log.txt'), 'w',
            encoding='utf-8').write('\n'.join(lines))
    for hd in handles:
        hd.remove()


if __name__ == '__main__':
    main()
