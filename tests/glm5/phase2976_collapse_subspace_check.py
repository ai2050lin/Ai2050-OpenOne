# -*- coding: utf-8 -*-
"""Phase 2976: fr deep-band collapse subspace check
(2938 machine, preregistered).

Why: 2973 proved the fr-cell band-signature collapse is a
DIRECTION effect (cos -> 0 in L24-34, energy share < 6%),
overlapping the 2967 collapse band and the 2970 delay band.
Open question (2938 discipline step 3): is the collapse
CONFINED to the single readout axis M[li] (subspace-internal
re-rotation, word geometry preserved) or is it GLOBAL
scrambling of the whole o_proj-input representation?

Design (frozen before any observation):
  Protocol 2973 verbatim: 74 single forwards (cells from
  2972 execution.json), capture o_proj INPUT pos1 all 36
  layers (4096-d). 22 index-aligned concept pairs
  (C_en[i] vs C_fr[i], cell order = pair order).
  Band family = 7 deep layers [24,26,27,30,31,32,34]
  (2973 T2 sig layers minus L0 early artifact; L0 reported
  descriptively only).
  Per band layer:
    T1 displacement alignment: delta_i = x_fr - x_en;
       stat = median_i |cos(delta_i, M_li)|; null = fr
       label permutation (rng 2976 x10000), maxT family 7;
       gate p<=0.01 in >=5 layers.
    T2 M-orthogonalized structure retention (CORE):
       x_perp = x - (x.Mhat)Mhat; Gram cosine matrices
       G_en, G_fr (22x22); stat = Spearman rho of upper
       triangles; null = fr label permutation
       (rng 2977 x10000, G_fr permuted by pi), maxT
       family 7; retention = p<=0.01 in >=5 layers.
    T3 descriptive: complement-energy ratio ||x_perp||
       fr/en, median cos(delta, M) sign, full-space Gram
       rho for contrast.

Anchors (frozen):
  a1 Vt8 rebuild vs 2939 npz < 1e-6
  a2 determinism < 1e-4
  a3 identity gate vs 2973 npz: norms rel < 1e-4 AND
     coss max-abs-diff < 1e-6 over 74x36 (2970 discipline)
  a4 single-token 74/74
  a5 non-degeneracy norms>0, |cos|<=1
  a6 structural: Gram diag=1, symmetric; permutation
     granularity 1/(N_PERM+1) << gate (2917)

Verdict (frozen):
  anchor fail => anchor_fail_all_void
  T1 sig and T2 retained => collapse_axis_confined_subspace_retained
  T1 sig and T2 not retained => collapse_global_scramble
  else => subspace_check_mixed_partial
"""
import hashlib
import json
import os
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC_2939 = os.path.join(BASE, 'phase2939', 'rotation_target',
                        'rotation_target.npz')
SRC_2972 = os.path.join(BASE, 'phase2972',
                        'two_factor_signature',
                        'two_factor_signature.npz')
SRC_2973 = os.path.join(BASE, 'phase2973', 'fr_scale_audit',
                        'fr_scale_audit.npz')
EXEC_2972 = os.path.join(BASE, 'phase2972',
                         'two_factor_signature',
                         'execution.json')
OUT = os.path.join(BASE, 'phase2976',
                   'collapse_subspace_check')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2976_run_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NH, HD = 32, 128
NL = 36
N_PERM = 10000
P_TH = 0.01
RNG_T1 = 2976
RNG_T2 = 2977
BAND = [24, 26, 27, 30, 31, 32, 34]
N_RET = 5

PREREG = {
    'mode': '74 single forwards (2973 protocol verbatim, '
            'NO ablation); capture o_proj input pos1 all '
            '36 layers; 22 index-aligned concept pairs; '
            'band family 7 deep layers',
    'question': 'is the fr deep-band cos collapse (2973) '
                'confined to the readout axis M (subspace '
                're-rotation) or a global scramble of the '
                'o_proj-input representation?',
    'cells_source': 'phase2972 execution.json cells '
                    'verbatim; pairing = index-aligned '
                    'C_en[i]/C_fr[i]',
    'band_family': BAND,
    'anchors': {
        'a1': 'Vt8 rebuild vs 2939 npz < 1e-6',
        'a2': 'determinism < 1e-4',
        'a3': 'identity vs 2973 npz: norms rel < 1e-4 and '
              'coss absdiff < 1e-6 (74x36)',
        'a4': 'single-token 74/74',
        'a5': 'norms > 0 and |cos| <= 1 everywhere',
        'a6': 'Gram diag/sym + perm granularity check',
    },
    'T1': 'displacement alignment: median |cos(delta,M)| '
          'per band layer; fr-label permutation null '
          'rng 2976 x10000; maxT family 7; sig >=5 layers',
    'T2': 'orthogonalized Gram retention: Spearman rho '
          'upper-tri G_en vs G_fr (x_perp); fr-label '
          'permutation null rng 2977 x10000; maxT family '
          '7; retained >=5 layers',
    'T3': 'descriptive: complement energy ratio fr/en, '
          'median signed cos(delta,M), full-space Gram '
          'rho contrast',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'T1 sig and T2 retained => '
               'collapse_axis_confined_subspace_retained; '
               'T1 sig and T2 not retained => '
               'collapse_global_scramble; else => '
               'subspace_check_mixed_partial',
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


def tri_idx(n):
    return np.triu_indices(n, 1)


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)

    # ---------- cells from 2972 execution.json ----------
    e72 = json.load(open(EXEC_2972, encoding='utf-8'))
    F_EN = e72['cells']['F_en']
    F_FR = e72['cells']['F_fr']
    C_EN = e72['cells']['C_en']
    C_FR = e72['cells']['C_fr']
    assert (len(F_EN), len(F_FR), len(C_EN), len(C_FR)) \
        == (15, 15, 22, 22), 'cell size drift'
    n_pair = len(C_EN)

    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2976,
                   'name': 'collapse_subspace_check',
                   'created': time.strftime(
                       '%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2939': sha8(SRC_2939),
                               's2972': sha8(SRC_2972),
                               's2973': sha8(SRC_2973)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'n_perm': N_PERM,
                   'rng': {'T1': RNG_T1, 'T2': RNG_T2},
                   'p_threshold': P_TH,
                   'band_family': BAND,
                   'n_ret_gate': N_RET,
                   'cells': {'F_en': F_EN, 'F_fr': F_FR,
                             'C_en': C_EN, 'C_fr': C_FR},
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen (22 pairs, band %s)'
        % BAND, lines)

    # ---------- sources ----------
    z73 = np.load(SRC_2973, allow_pickle=True)
    norms73 = z73['norms'].astype(np.float64)
    coss73 = z73['coss'].astype(np.float64)
    words73 = [str(w) for w in z73['words']]
    z39 = np.load(SRC_2939, allow_pickle=True)
    z27 = np.load(os.path.join(
        BASE, 'phase2927', 'probe_relativity',
        'probe_relativity.npz'), allow_pickle=True)
    dirs27 = z27['dirs_word'].astype(np.float64)
    _, _, Vt_loc = np.linalg.svd(dirs27,
                                 full_matrices=False)
    a1_diff = float(np.abs(Vt_loc[:8]
                           - z39['Vt8']).max())
    a1_ok = bool(a1_diff < 1e-6)
    log('a1 Vt8 rebuild diff %.2e ok=%s'
        % (a1_diff, a1_ok), lines)
    u35 = dirs27[NL - 1]

    # ---------- model ----------
    import sys
    sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
    from phase2662_symmetric_mapping_contract import \
        load_native
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        MD, local_files_only=True, trust_remote_code=True,
        use_fast=True)
    cells = [('F', 'en', w) for w in F_EN] \
        + [('F', 'fr', w) for w in F_FR] \
        + [('C', 'en', w) for w in C_EN] \
        + [('C', 'fr', w) for w in C_FR]
    words76 = ['%s:%s:%s' % c for c in cells]
    ident_ok = bool(words76 == words73)
    log('cell identity vs 2973 npz order: %s' % ident_ok,
        lines)
    assert ident_ok, 'word order drift vs 2973'

    tid_map = {}
    n_single = 0
    for _, _, w in cells:
        ids = tok(' ' + w, add_special_tokens=False)[
            'input_ids']
        if len(ids) != 1:
            ids = tok(w, add_special_tokens=False)[
                'input_ids']
        if len(ids) == 1:
            n_single += 1
        tid_map[w] = int(ids[0]) if len(ids) == 1 else -1
    n_test = len(cells)
    a4_ok = bool(n_single == n_test)
    log('a4 single-token %d/%d ok=%s'
        % (n_single, n_test, a4_ok), lines)
    ids_the = tok(' the', add_special_tokens=False)[
        'input_ids']
    assert len(ids_the) == 1
    func_tid = int(ids_the[0])

    model, _ = load_native('qwen4')
    model.eval()
    layers = model.model.layers
    log('model loaded', lines)

    cap_op = {li: [] for li in range(NL)}
    fin_cap = {}
    state_fin = {'on': False}
    handles = []

    def hook_op(li):
        def h(module, args, kwargs):
            x = args[0] if args else kwargs.get('input')
            if x is None or x.dim() < 2:
                return None
            cap_op[li].append(
                x[:, 1, :].detach().float().cpu().numpy())
            return None
        return h

    def pre_norm(module, args, kwargs):
        if state_fin['on']:
            fin_cap['x'] = args[0][:, -1, :].detach() \
                .float().cpu().numpy()

    for li in range(NL):
        handles.append(
            layers[li].self_attn.o_proj
            .register_forward_pre_hook(
                hook_op(li), with_kwargs=True))
    handles.append(model.model.norm
                   .register_forward_pre_hook(
                       pre_norm, with_kwargs=True))

    def clear_cap():
        for li in cap_op:
            del cap_op[li][:]

    def forward1(toks):
        clear_cap()
        fin_cap.pop('x', None)
        state_fin['on'] = True
        with torch.no_grad():
            model(torch.tensor([toks], device='cuda'))
        state_fin['on'] = False
        return (fin_cap['x'].astype(np.float64),
                {li: cap_op[li][0].astype(np.float64)
                 for li in range(NL)})

    M = np.zeros((NL, NH * HD))
    Mnorm = np.zeros(NL)
    for li in range(NL):
        Wo = layers[li].self_attn.o_proj.weight.detach() \
            .float().cpu().numpy()
        M[li] = u35 @ Wo
        Mnorm[li] = float(np.linalg.norm(M[li]))

    # ---------- a2 determinism ----------
    fin_a, op_a = forward1([func_tid, tid_map[C_EN[0]]])
    fin_b, op_b = forward1([func_tid, tid_map[C_EN[0]]])
    a2_rel = float(np.abs(op_a[30] - op_b[30]).max()
                   / max(float(np.abs(op_a[30]).max()),
                         1e-30))
    a2_ok = bool(a2_rel < 1e-4)
    log('a2 determinism rel %.2e ok=%s'
        % (a2_rel, a2_ok), lines)

    # ---------- main sweep: 74 words ----------
    X = np.zeros((n_test, NL, NH * HD))
    norms = np.zeros((n_test, NL))
    coss = np.zeros((n_test, NL))
    for i, (_, _, w) in enumerate(cells):
        _, op = forward1([func_tid, tid_map[w]])
        for li in range(NL):
            x = op[li].reshape(-1)
            X[i, li] = x
            norms[i, li] = float(np.linalg.norm(x))
            coss[i, li] = float(np.dot(x, M[li])) \
                / max(norms[i, li] * Mnorm[li], 1e-30)
        if (i + 1) % 20 == 0:
            log('sweep [%d/%d]' % (i + 1, n_test), lines)

    # ---------- a3 identity vs 2973 ----------
    nrm_rel = float(np.max(np.abs(norms - norms73)
                           / np.maximum(norms73, 1e-30)))
    cos_abs = float(np.max(np.abs(coss - coss73)))
    a3_ok = bool(nrm_rel < 1e-4 and cos_abs < 1e-6)
    log('a3 identity vs 2973: norms rel %.2e, coss abs '
        '%.2e ok=%s' % (nrm_rel, cos_abs, a3_ok), lines)

    # ---------- a5 non-degeneracy ----------
    a5_ok = bool(norms.min() > 0
                 and np.abs(coss).max() <= 1.0 + 1e-9)
    log('a5 norms>0 & |cos|<=1 ok=%s' % a5_ok, lines)

    anchor_ok = bool(a1_ok and a2_ok and a3_ok and a4_ok
                     and a5_ok)
    verdict = None
    t1 = t2 = t3 = None
    stat1 = p1 = stat2 = p2 = rho_full = None
    sig1 = []
    ret = []
    en_idx = np.arange(15 + 15, 15 + 15 + n_pair)
    fr_idx = np.arange(15 + 15 + n_pair, n_test)

    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        from scipy.stats import spearmanr
        tr = tri_idx(n_pair)
        Xe = X[en_idx][:, BAND, :]
        Xf = X[fr_idx][:, BAND, :]
        nL = len(BAND)

        # ---- T1 displacement alignment ----
        disp = Xf - Xe
        stat1 = np.zeros(nL)
        for j in range(nL):
            Mh = M[BAND[j]] / Mnorm[BAND[j]]
            cd = disp[:, j, :] @ Mh
            dnorm = np.linalg.norm(
                disp[:, j, :], axis=1)
            cosd = cd / np.maximum(dnorm, 1e-30)
            stat1[j] = float(np.median(np.abs(cosd)))
        rng1 = np.random.default_rng(RNG_T1)
        fam1 = np.zeros(N_PERM)
        for k in range(N_PERM):
            pi = rng1.permutation(n_pair)
            fam1[k] = float(np.max([
                np.median(np.abs(
                    (Xf[pi, j, :] - Xe[:, j, :])
                    @ (M[BAND[j]] / Mnorm[BAND[j]])
                    / np.maximum(np.linalg.norm(
                        Xf[pi, j, :] - Xe[:, j, :],
                        axis=1), 1e-30)))
                for j in range(nL)]))
        p1 = np.array([
            (np.sum(fam1 >= stat1[j] - 1e-12) + 1)
            / (N_PERM + 1) for j in range(nL)])
        sig1 = [BAND[j] for j in range(nL)
                if p1[j] <= P_TH]
        log('T1 disp-align stat %s p %s sig %s'
            % (np.round(stat1, 4).tolist(),
               ['%.1e' % v for v in p1], sig1), lines)

        # ---- T2 orthogonalized Gram retention ----
        stat2 = np.zeros(nL)
        rho_full = np.zeros(nL)
        Gf_list = []
        Ge_list = []
        for j in range(nL):
            Mh = M[BAND[j]] / Mnorm[BAND[j]]

            def perp(Xc):
                xp = Xc - np.outer(Xc @ Mh, Mh)
                return xp / np.maximum(
                    np.linalg.norm(xp, axis=1,
                                   keepdims=True), 1e-30)
            Ne = perp(Xe[:, j, :])
            Nf = perp(Xf[:, j, :])
            Ge = Ne @ Ne.T
            Gf = Nf @ Nf.T
            Ge_list.append(Ge)
            Gf_list.append(Gf)
            stat2[j] = float(spearmanr(Ge[tr],
                                       Gf[tr])[0])
            Nfe = Xf[:, j, :] / np.maximum(
                np.linalg.norm(Xf[:, j, :], axis=1,
                               keepdims=True), 1e-30)
            Nee = Xe[:, j, :] / np.maximum(
                np.linalg.norm(Xe[:, j, :], axis=1,
                               keepdims=True), 1e-30)
            rho_full[j] = float(spearmanr(
                (Nee @ Nee.T)[tr],
                (Nfe @ Nfe.T)[tr])[0])
        rng2 = np.random.default_rng(RNG_T2)
        fam2 = np.zeros(N_PERM)
        for k in range(N_PERM):
            pi = rng2.permutation(n_pair)
            fam2[k] = float(np.max([
                abs(spearmanr(
                    Ge_list[j][tr],
                    Gf_list[j][pi][:, pi][tr])[0])
                for j in range(nL)]))
        p2 = np.array([
            (np.sum(fam2 >= abs(stat2[j]) - 1e-12) + 1)
            / (N_PERM + 1) for j in range(nL)])
        ret = [BAND[j] for j in range(nL)
               if p2[j] <= P_TH]
        log('T2 ortho-Gram rho %s p %s retained %s '
            '(full-space rho %s)'
            % (np.round(stat2, 4).tolist(),
               ['%.1e' % v for v in p2], ret,
               np.round(rho_full, 4).tolist()), lines)

        # ---- a6 structural ----
        g_ok = all(
            abs(Ge_list[j].diagonal().mean() - 1) < 1e-9
            and float(np.abs(Ge_list[j]
                             - Ge_list[j].T).max())
            < 1e-9
            for j in range(nL))
        a6_ok = bool(g_ok and (N_PERM + 1) ** -1
                     * 1.0 < P_TH)
        log('a6 Gram diag/sym ok=%s, granularity 1/%d'
            % (g_ok, N_PERM + 1), lines)

        # ---- T3 descriptive ----
        t3_rows = []
        for j in range(nL):
            li_ = BAND[j]
            e_ratio = float(np.median(
                np.linalg.norm(
                    Xf[:, j, :] - np.outer(
                        Xf[:, j, :] @ (M[li_]
                                       / Mnorm[li_]),
                        M[li_] / Mnorm[li_]),
                    axis=1)
                / np.maximum(np.linalg.norm(
                    Xe[:, j, :] - np.outer(
                        Xe[:, j, :] @ (M[li_]
                                       / Mnorm[li_]),
                        M[li_] / Mnorm[li_]),
                    axis=1), 1e-30)))
            Mh = M[li_] / Mnorm[li_]
            cosd_med = float(np.median(
                (disp[:, j, :] @ Mh)
                / np.maximum(np.linalg.norm(
                    disp[:, j, :], axis=1), 1e-30)))
            t3_rows.append({
                'layer': li_,
                'complement_energy_ratio_fr_over_en':
                    round(e_ratio, 4),
                'median_signed_cos_delta_M':
                    round(cosd_med, 4),
                'gram_rho_full_space':
                    round(float(rho_full[j]), 4)})
        t3 = {'rows': t3_rows}
        log('T3 %s' % json.dumps(t3), lines)

        # ---- verdict (frozen map) ----
        n_sig1 = len(sig1)
        n_ret = len(ret)
        if n_sig1 >= N_RET and n_ret >= N_RET:
            verdict = \
                'collapse_axis_confined_subspace_retained'
        elif n_sig1 >= N_RET and n_ret < N_RET:
            verdict = 'collapse_global_scramble'
        else:
            verdict = 'subspace_check_mixed_partial'
        save = {
            'X_band': X[:, BAND, :],
            'norms': norms, 'coss': coss,
            'Mnorm': Mnorm, 'band': np.array(BAND),
            'words': np.array(words76),
            'stat1': stat1, 'p1': p1,
            'stat2': stat2, 'p2': p2,
            'rho_full': rho_full,
            'pairs_en': np.array(C_EN),
            'pairs_fr': np.array(C_FR)}

    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2976, 'model': 'qwen3-4b',
           'prereg': PREREG,
           'anchors': {'a1_diff': float('%.3e' % a1_diff),
                       'a1_ok': a1_ok,
                       'a2_rel': float('%.3e' % a2_rel),
                       'a2_ok': a2_ok,
                       'a3_nrm_rel': float('%.3e'
                                           % nrm_rel),
                       'a3_cos_abs': float('%.3e'
                                           % cos_abs),
                       'a3_ok': a3_ok,
                       'a4_ok': a4_ok,
                       'a5_ok': a5_ok,
                       'a6_ok': a6_ok,
                       'ok': anchor_ok},
           'T1': {'stat':
                      None if stat1 is None
                      else np.round(stat1, 4).tolist(),
                  'p': None if p1 is None
                       else [float('%.3e' % v)
                             for v in p1],
                  'sig_layers': sig1},
           'T2': {'rho':
                      None if stat2 is None
                      else np.round(stat2, 4).tolist(),
                  'p': None if p2 is None
                       else [float('%.3e' % v)
                             for v in p2],
                  'retained_layers': ret},
           'T3': t3,
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if verdict != 'anchor_fail_all_void':
        np.savez_compressed(os.path.join(
            OUT, 'collapse_subspace_check.npz'), **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2976 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    import torch  # noqa: E402
    main()
