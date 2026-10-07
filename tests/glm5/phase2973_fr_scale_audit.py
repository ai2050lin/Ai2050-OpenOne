# -*- coding: utf-8 -*-
"""Phase 2973: fr-cell B scale audit - norm vs cos
decomposition (preregistered).

Why: 2972 (two_factor_signature) found the within-fr class
gap collapsed (0.207 vs en 1.169) and suspected that fr
words in the English " the X" code-switching context sit in
an OOD regime where the band readout magnitude may be a NORM
artifact rather than a directional signature. Discipline
2937: a magnitude claim must be decomposed into energy
(||x||) and direction (cos) factors BEFORE naming the
mechanism.

Design (frozen before any observation):
  Protocol 2972 verbatim: 77 single forwards ([the, w]),
  capture o_proj INPUT pos1 all 36 layers. Cell lists read
  from phase2972 execution.json (identity gate).
  Per word per layer decompose the readout projection
    prof[li] = x_op[li] . M[li] = ||x|| * ||M|| * cos
  into energy factor norm[li] = ||x_op[li]|| and direction
  factor coss[li] = cos(x_op[li], M[li]). B recomputed
  identically (band 6:13 minus 28:36 of prof).

Tests (frozen):
  T1 energy factor: per-layer lang coefficient on norm
     (Freedman-Lane: reduced norm ~ 1+cls+cov, residual
     perm rng 2973 x10000, full adds lang; maxT family 36;
     gate p<=0.01 any layer).
  T2 direction factor: same on coss, rng 2974.
  T3 descriptive: band log-ratio decomposition
     log(median_fr prof / median_en prof) =
     log(norm ratio) + log(cos ratio); energy/cos shares
     at L6-12 and L28-35; cell medians of norm/coss.

Anchors (frozen):
  a1 Vt8 rebuild from 2927 vs 2939 npz < 1e-6
  a2 determinism < 1e-4
  a3 B identity: recomputed B vs 2972 npz rel < 1e-4
     for 74/74 words (identity gate, 2970 discipline)
  a4 single-token 74/74 test + 3/3 anchors
  a5 B of 3 anchor words vs 2963 npz < 1e-4 rel
  a6 non-degeneracy: norms all > 0 (74x36) and
     |coss| <= 1 everywhere

Verdict (frozen):
  anchor fail => anchor_fail_all_void
  T1 sig and T2 sig => language_effect_energy_and_direction
  T1 sig only => language_effect_energy_dominant
  T2 sig only => language_effect_direction_dominant
  else => scale_decomposition_all_void
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
SRC_2939 = os.path.join(BASE, 'phase2939', 'rotation_target',
                        'rotation_target.npz')
SRC_2963 = os.path.join(BASE, 'phase2963',
                        'frequency_controlled_band',
                        'freq_band.npz')
SRC_2972 = os.path.join(BASE, 'phase2972',
                        'two_factor_signature',
                        'two_factor_signature.npz')
EXEC_2972 = os.path.join(BASE, 'phase2972',
                         'two_factor_signature',
                         'execution.json')
OUT = os.path.join(BASE, 'phase2973', 'fr_scale_audit')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2973_run_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NH, HD = 32, 128
NL = 36
N_PERM = 10000
P_TH = 0.01
RNG_T1 = 2973
RNG_T2 = 2974
ANCHOR_WORDS = ['people', 'for', 'garden']

PREREG = {
    'mode': '77 single forwards (74 words from 2972 '
            'execution.json cells + 3 anchors), 2972 '
            'protocol verbatim, NO ablation; capture o_proj '
            'input pos1 all 36 layers; decompose prof into '
            'norm x cos',
    'question': 'is the fr-cell collapse of the band '
                'signature (2972) an energy (norm) artifact '
                'or a directional (cos) effect of the '
                'code-switching regime?',
    'cells_source': 'phase2972 execution.json cells '
                    'F_en/F_fr/C_en/C_fr verbatim (identity '
                    'gate)',
    'anchors': {
        'a1': 'Vt8 rebuild vs 2939 npz < 1e-6',
        'a2': 'determinism < 1e-4',
        'a3': 'B identity vs 2972 npz rel < 1e-4 (74/74)',
        'a4': 'single-token 74/74 + anchors 3/3',
        'a5': 'B of 3 anchors vs 2963 npz < 1e-4 rel',
        'a6': 'norms > 0 and |cos| <= 1 everywhere',
    },
    'T1': 'norm lang effect: Freedman-Lane reduced '
          'norm_li ~ 1+cls+cov, residual perm rng 2973 '
          'x10000, full + lang; maxT family 36; gate '
          'p<=0.01 any layer',
    'T2': 'cos lang effect: same, rng 2974',
    'T3': 'descriptive: band log-ratio decomposition '
          'energy/cos shares L6-12 and L28-35; cell '
          'medians',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'T1&T2 sig => '
               'language_effect_energy_and_direction; '
               'T1 only => '
               'language_effect_energy_dominant; T2 only '
               '=> language_effect_direction_dominant; '
               'else => scale_decomposition_all_void',
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


def rankdata(x):
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x), dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def lang_class_maxT(Q, lang, cls, cov, seed):
    """Q (n,36): Freedman-Lane lang coefficient per layer
    with maxT family correction. Returns p vector."""
    n = Q.shape[0]
    Xr = np.stack([np.ones(n), cls.astype(float),
                   cov], axis=1)
    Br, *_ = np.linalg.lstsq(Xr, Q, rcond=None)
    resid = Q - Xr @ Br
    Xf = np.stack([np.ones(n), cls.astype(float),
                   cov, lang.astype(float)], axis=1)
    Pinv = np.linalg.pinv(Xf)
    obs = (Pinv @ Q)[3]
    rng = np.random.default_rng(seed)
    fam = np.zeros(N_PERM)
    for k in range(N_PERM):
        ep = rng.permutation(resid)
        fam[k] = float(np.abs(
            (Pinv @ (Xr @ Br + ep))[3]).max())
    thr_ok = fam
    p = np.array([
        (np.sum(thr_ok >= abs(obs[li]) - 1e-12) + 1)
        / (N_PERM + 1) for li in range(Q.shape[1])])
    return obs, p


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

    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2973,
                   'name': 'fr_scale_audit',
                   'created': time.strftime(
                       '%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2927': sha8(SRC_2927),
                               's2939': sha8(SRC_2939),
                               's2963': sha8(SRC_2963),
                               's2972': sha8(SRC_2972)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'n_perm': N_PERM,
                   'rng': {'T1': RNG_T1, 'T2': RNG_T2},
                   'p_threshold': P_TH,
                   'cells': {'F_en': F_EN, 'F_fr': F_FR,
                             'C_en': C_EN, 'C_fr': C_FR},
                   'anchors_words': ANCHOR_WORDS,
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen (cells 15/15/22/22)',
        lines)

    # ---------- sources ----------
    z27 = np.load(SRC_2927, allow_pickle=True)
    dirs27 = z27['dirs_word'].astype(np.float64)
    z39 = np.load(SRC_2939, allow_pickle=True)
    _, _, Vt_loc = np.linalg.svd(dirs27,
                                 full_matrices=False)
    a1_diff = float(np.abs(Vt_loc[:8]
                           - z39['Vt8']).max())
    a1_ok = bool(a1_diff < 1e-6)
    log('a1 Vt8 rebuild diff %.2e ok=%s'
        % (a1_diff, a1_ok), lines)
    u35 = dirs27[NL - 1]
    z63 = np.load(SRC_2963, allow_pickle=True)
    w63 = [str(w).split(':') for w in z63['words']]
    B63 = z63['B'].astype(np.float64)
    z72 = np.load(SRC_2972, allow_pickle=True)
    B72 = z72['B'].astype(np.float64)
    words72 = [str(w) for w in z72['words']]

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
    words73 = ['%s:%s:%s' % c for c in cells]
    ident_ok = bool(words73 == words72)
    log('a3-pre cell identity vs 2972 npz order: %s'
        % ident_ok, lines)
    assert ident_ok, 'word order drift vs 2972'

    tid_map = {}
    n_single = 0
    for _, _, w in cells + [('A', 'en', w)
                            for w in ANCHOR_WORDS]:
        ids = tok(' ' + w, add_special_tokens=False)[
            'input_ids']
        if len(ids) != 1:
            ids = tok(w, add_special_tokens=False)[
                'input_ids']
        if len(ids) == 1:
            n_single += 1
        tid_map[w] = int(ids[0]) if len(ids) == 1 else -1
    n_test = len(cells)
    a4_ok = bool(n_single == n_test + 3)
    log('a4 single-token %d/%d ok=%s'
        % (n_single, n_test + 3, a4_ok), lines)
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

    # ---------- a2 determinism ----------
    fin_a, _ = forward1([func_tid, tid_map['people']])
    fin_b, _ = forward1([func_tid, tid_map['people']])
    a2_rel = float(np.abs(fin_a - fin_b).max()
                   / max(float(np.abs(fin_a).max()),
                         1e-30))
    a2_ok = bool(a2_rel < 1e-4)
    log('a2 determinism rel %.2e ok=%s'
        % (a2_rel, a2_ok), lines)

    # ---------- anchor forwards (a5) ----------
    a5_rel = 0.0
    for w in ANCHOR_WORDS:
        fin, op = forward1([func_tid, tid_map[w]])
        _, prof = contributions(op)
        i63 = [i for i, (g, ww) in enumerate(w63)
               if ww == w][0]
        a5_rel = max(a5_rel,
                     abs(band_of(prof) - B63[i63])
                     / max(abs(B63[i63]), 1e-30))
    a5_ok = bool(a5_rel < 1e-4)
    log('a5 B vs 2963 rel %.2e ok=%s'
        % (a5_rel, a5_ok), lines)

    # ---------- main sweep: 74 words ----------
    B_rec = np.zeros(n_test)
    norms = np.zeros((n_test, NL))
    coss = np.zeros((n_test, NL))
    for i, (_, _, w) in enumerate(cells):
        fin, op = forward1([func_tid, tid_map[w]])
        prof = np.zeros(NL)
        for li in range(NL):
            x = op[li].reshape(-1)
            prof[li] = float(np.dot(x, M[li]))
            nrm = float(np.linalg.norm(x))
            norms[i, li] = nrm
            coss[i, li] = prof[li] / max(
                nrm * Mnorm[li], 1e-30)
        B_rec[i] = band_of(prof)
        if (i + 1) % 20 == 0:
            log('sweep [%d/%d]' % (i + 1, n_test), lines)

    # ---------- a3 B identity vs 2972 ----------
    a3_rel = float(np.max(np.abs(B_rec - B72)
                          / np.maximum(np.abs(B72),
                                       1e-30)))
    a3_ok = bool(a3_rel < 1e-4)
    log('a3 B identity vs 2972 max rel %.2e ok=%s'
        % (a3_rel, a3_ok), lines)

    # ---------- a6 non-degeneracy ----------
    a6_ok = bool(norms.min() > 0
                 and np.abs(coss).max() <= 1.0 + 1e-9)
    log('a6 norms>0 & |cos|<=1 ok=%s (min norm %.3e, '
        'max|cos| %.6f)'
        % (a6_ok, norms.min(), np.abs(coss).max()), lines)

    anchor_ok = bool(a1_ok and a2_ok and a3_ok and a4_ok
                     and a5_ok and a6_ok)
    verdict = None
    t1 = t2 = t3 = None
    save = {'B_rec': B_rec, 'norms': norms, 'coss': coss,
            'Mnorm': Mnorm,
            'words': np.array(words73),
            'tids': np.array([tid_map[w]
                              for _, _, w in cells]),
            'anchors_words': np.array(ANCHOR_WORDS)}

    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        lang = np.array([0 if c[1] == 'en' else 1
                         for c in cells])
        cls = np.array([0 if c[0] == 'F' else 1
                        for c in cells])
        tids = np.array([tid_map[w]
                         for _, _, w in cells],
                        dtype=np.float64)
        cov = np.zeros(n_test)
        for lv in (0, 1):
            m = lang == lv
            cov[m] = rankdata(tids[m])
        cov = (cov - cov.mean()) / cov.std()

        # T1 norm / T2 cos lang effects
        obs_n, p_n = lang_class_maxT(
            norms, lang, cls, cov, RNG_T1)
        obs_c, p_c = lang_class_maxT(
            coss, lang, cls, cov, RNG_T2)
        sig_n = [li for li in range(NL)
                 if p_n[li] <= P_TH]
        sig_c = [li for li in range(NL)
                 if p_c[li] <= P_TH]

        def top3(obs, p):
            order = np.argsort(-np.abs(obs))
            return [(int(li),
                     round(float(obs[li]), 4),
                     float('%.3e' % p[li]))
                    for li in order[:3]]

        t1 = {'sig_layers': sig_n,
              'top3': top3(obs_n, p_n),
              'min_p': float('%.3e' % p_n.min())}
        t2 = {'sig_layers': sig_c,
              'top3': top3(obs_c, p_c),
              'min_p': float('%.3e' % p_c.min())}
        log('T1 norm sig %s top3 %s'
            % (sig_n, t1['top3']), lines)
        log('T2 cos sig %s top3 %s'
            % (sig_c, t2['top3']), lines)

        # T3 descriptive band decomposition
        med = lambda msk, A, li: float(
            np.median(A[msk, li]))
        decomp = {}
        for band, rng_ in [('band_L6_12', range(6, 13)),
                           ('band_L28_35',
                            range(28, 36))]:
            lr_n = float(np.mean([
                np.log(abs(med(lang == 1, norms, li)
                           / med(lang == 0, norms, li)))
                for li in rng_]))
            lr_c = float(np.mean([
                np.log(abs(med(lang == 1, coss, li)
                           / med(lang == 0, coss, li)))
                for li in rng_]))
            tot = abs(lr_n) + abs(lr_c)
            decomp[band] = {
                'log_norm_ratio': round(lr_n, 4),
                'log_cos_ratio': round(lr_c, 4),
                'energy_share': round(
                    abs(lr_n) / max(tot, 1e-30), 4)}
        cell_med = {}
        for lab_v in range(4):
            m = ((lang == lab_v % 2)
                 & (cls == lab_v // 2))
            key = '%s_%s' % (
                'F' if lab_v // 2 == 0 else 'C',
                'en' if lab_v % 2 == 0 else 'fr')
            cell_med[key] = {
                'norm_L10': round(float(
                    np.median(norms[m][:, 10])), 2),
                'norm_L30': round(float(
                    np.median(norms[m][:, 30])), 2),
                'cos_L10': round(float(
                    np.median(coss[m][:, 10])), 4),
                'cos_L30': round(float(
                    np.median(coss[m][:, 30])), 4),
                'B': round(float(
                    np.median(B_rec[m])), 4)}
        t3 = {'band_decomp': decomp,
              'cell_medians': cell_med}
        log('T3 decomp %s | cells %s'
            % (decomp, json.dumps(cell_med)), lines)

        # verdict (frozen map)
        s1 = len(sig_n) > 0
        s2 = len(sig_c) > 0
        if s1 and s2:
            verdict = \
                'language_effect_energy_and_direction'
        elif s1:
            verdict = 'language_effect_energy_dominant'
        elif s2:
            verdict = \
                'language_effect_direction_dominant'
        else:
            verdict = 'scale_decomposition_all_void'
        save.update({'p_norm': p_n, 'p_cos': p_c,
                     'obs_norm': obs_n,
                     'obs_cos': obs_c})

    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2973, 'model': 'qwen3-4b',
           'prereg': PREREG,
           'anchors': {'a1_diff': float('%.3e' % a1_diff),
                       'a1_ok': a1_ok,
                       'a2_rel': float('%.3e' % a2_rel),
                       'a2_ok': a2_ok,
                       'a3_rel': float('%.3e' % a3_rel),
                       'a3_ok': a3_ok,
                       'a4_ok': a4_ok,
                       'a5_rel': float('%.3e' % a5_rel),
                       'a5_ok': a5_ok,
                       'a6_ok': a6_ok,
                       'ok': anchor_ok},
           'T1': t1, 'T2': t2, 'T3': t3,
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if verdict != 'anchor_fail_all_void':
        np.savez_compressed(os.path.join(
            OUT, 'fr_scale_audit.npz'), **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2973 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    import torch  # noqa: E402
    main()
