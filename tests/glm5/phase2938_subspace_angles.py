# -*- coding: utf-8 -*-
"""Phase 2938: language-subspace principal-angle measurement.

Why: 2937 established scale_collapse_rewrite = direction
collapse on the SINGLE direction dirs_word[35] (cos ratio
0.05-0.24) with energy preserved. Open question: does the
final residual leave the whole LANGUAGE SUBSPACE (stacked
dirs_word span) or only that single direction (rotation
within the subspace / re-encoding)?

Mode: ONE run (qwen3-4b), NO ablation. 2937 protocol
verbatim: pass1 dirs_word rebuild (57 single forwards) +
5 conditions (func/same/null0-3) x batch57, final pre-norm
residual + per-layer pos-1 attn-input capture.

Subspace: SVD of stacked dirs_word (36 x D) -> orthonormal
components Vt. Alignment of vector x: alpha_k(x) =
||Vt[:k] x|| / ||x|| (scale-invariant, pure direction).
Grid k in {1, 4, 8, 16, 36} on the final residual, plus
per-layer profile (k=8 and single dirs_word[li]) on the
pos-1 attn-input.

Anchors (frozen):
  a1 dirs_word rebuild vs 2927 npz < 1e-5
  a2 func baseline determinism < 1e-4
  a3 proj_func(final) vs 2935 npz s_base[func] < 1e-4
  a4 proj_null0(final) vs 2935 npz s_base[null0] < 1e-4
  a5 SVD orthonormality ||Vt Vt^T - I|| < 1e-8
  a6 func separation > 0
  a7 align_dir35 this run vs 2937 npz |proj|/fin_norm
     max abs diff < 1e-6 (cross-phase bit-level)

Main tests (frozen):
  P1 final-position alignment grid: per condition median_w
     alpha_k; ratio rho_k = median_w alpha_k[null] /
     alpha_k[func] per null set + dir35 baseline;
     rho_k8 = median over 4 null sets, rho_d35 likewise.
  P2 paired permutation: func vs null0 alpha_8 difference
     median, sign-flip permutation (rng 2919, 10000),
     p = (cnt+1)/10001.
  P3 per-layer profile: ratio null/func of alpha_8 (global
     subspace) and of single-direction alignment
     |x . dirs_word[li]|/||x||, first crossing < 0.8 per
     condition.

Verdict (frozen):
  anchor fail                          => anchor_fail_all_void
  rho_k8 >= 0.6 AND rho_k8 >= rho_d35 + 0.2
                                       => subspace_rotation_retained
  rho_k8 < 0.5                         => subspace_collapse_confirmed
  else                                 => subspace_mixed

Output: phase2938/subspace_angles/.
"""
import hashlib
import json
import os
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC_2887 = os.path.join(BASE, 'phase2887', 'language_axis_mlp',
                        'language_axis_mlp.npz')
SRC_2927 = os.path.join(BASE, 'phase2927', 'probe_relativity',
                        'probe_relativity.npz')
SRC_2929 = os.path.join(BASE, 'phase2929',
                        'response_structure_atlas',
                        'response_structure_atlas.npz')
SRC_2930 = os.path.join(BASE, 'phase2930', 'direction_flip_control',
                        'direction_flip_control.npz')
SRC_2931 = os.path.join(BASE, 'phase2931', 'skeleton_overlap_null',
                        'skeleton_overlap_null.npz')
SRC_2935 = os.path.join(BASE, 'phase2935', 'null_amp_anatomy',
                        'null_amp_anatomy.npz')
SRC_2937 = os.path.join(BASE, 'phase2937', 'scale_collapse',
                        'scale_collapse.npz')
OUT = os.path.join(BASE, 'phase2938', 'subspace_angles')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2938_run_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
SEED = 2896
NH, HD = 32, 128
NL = 36
VOCAB = 151936
KS = (1, 4, 8, 16, 36)
K_MAIN = 8
RNG_P2 = 2919
N_P2 = 10000
NULL_SEEDS = (2896, 2914, 2915, 2916)

PREREG = {
    'mode': 'ONE run, NO ablation: 2937 protocol verbatim '
            '(pass1 dirs_word rebuild 57 single forwards + 5 '
            'conditions x batch57, final pre-norm residual + '
            'per-layer pos-1 attn-input capture); subspace '
            'from SVD of stacked dirs_word; alignment '
            'alpha_k = ||Vt[:k] x||/||x|| scale-invariant',
    'question': 'does the final residual leave the whole '
                'language subspace (span of stacked '
                'dirs_word) under null context, or only the '
                'single direction dirs_word[35] (rotation '
                'within the subspace)?',
    'anchors': {
        'a1': 'dirs_word rebuild vs 2927 npz < 1e-5',
        'a2': 'func baseline determinism < 1e-4',
        'a3': 'proj_func vs 2935 s_base[func] < 1e-4',
        'a4': 'proj_null0 vs 2935 s_base[null0] < 1e-4',
        'a5': 'SVD orthonormality < 1e-8',
        'a6': 'func separation > 0',
        'a7': 'align_dir35 vs 2937 npz |proj|/fin_norm '
              '< 1e-6',
    },
    'P1': 'final alignment grid k in {1,4,8,16,36} + dir35; '
          'rho_k = median_w alpha[null]/alpha[func] per null '
          'set; rho_k8 / rho_d35 = median over 4 sets',
    'P2': 'paired sign-flip permutation func vs null0 '
          'alpha_8 (rng 2919, 10000)',
    'P3': 'per-layer ratio profile of alpha_8 and single '
          'dirs_word[li] alignment, first crossing < 0.8',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'rho_k8 >= 0.6 AND rho_k8 >= rho_d35 + 0.2 => '
               'subspace_rotation_retained; rho_k8 < 0.5 => '
               'subspace_collapse_confirmed; else => '
               'subspace_mixed',
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


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2938,
                   'name': 'subspace_angles',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2887': sha8(SRC_2887),
                               's2927': sha8(SRC_2927),
                               's2929': sha8(SRC_2929),
                               's2930': sha8(SRC_2930),
                               's2931': sha8(SRC_2931),
                               's2935': sha8(SRC_2935),
                               's2937': sha8(SRC_2937)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'seed': SEED, 'ks': list(KS),
                   'k_main': K_MAIN, 'rng_p2': RNG_P2,
                   'null_seeds': list(NULL_SEEDS),
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
    z29 = np.load(SRC_2929, allow_pickle=True)
    _ = z29['rho_grid'].astype(np.float64)  # source-chain pin
    z30 = np.load(SRC_2930, allow_pickle=True)
    _ = z30['lin_r_profile'].astype(np.float64)
    z31 = np.load(SRC_2931, allow_pickle=True)
    _ = z31['S29_corrected'].astype(bool)   # source-chain pin
    z35 = np.load(SRC_2935, allow_pickle=True)
    conds35 = [str(s) for s in z35['cond_names']]
    s_base_35 = z35['s_base'].astype(np.float64)
    ifu35 = conds35.index('func')
    in035 = conds35.index('null0')
    z37 = np.load(SRC_2937, allow_pickle=True)
    conds37 = [str(s) for s in z37['cond_names']]
    proj37 = z37['proj'].astype(np.float64)
    fnorm37 = z37['fin_norm'].astype(np.float64)

    # ---------- model ----------
    import torch
    import sys
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
            assert len(ids) == 1, '%s -> %s' % (t, ids)
            tc[t] = int(ids[0])
        return tc[t]

    tid_map = {}
    for lang, ck, w in words:
        tid_map[w] = tid(w)
        if lang == 'en':
            assert tid_map[w] == int(ck), 'key mismatch %s' % w
    func_tid = tid('the')

    def same_ctx(i):
        lang = words[i][0]
        cands = [j for j in range(n_words)
                 if words[j][0] == lang and j != i]
        return min(cands, key=lambda j: tid_map[words[j][2]])

    word_tids = set(tid_map.values())

    def sample_null(seed):
        rng = np.random.default_rng(seed)
        out = []
        while len(out) < n_words:
            r = int(rng.integers(0, VOCAB))
            if r not in word_tids and r > 0:
                out.append(r)
        return out

    null_sets = {('null%d' % k): sample_null(s)
                 for k, s in enumerate(NULL_SEEDS)}
    conds_all = ['func', 'same'] + list(null_sets.keys())
    batch = {'func': [[func_tid, tid_map[words[i][2]]]
                      for i in range(n_words)],
             'same': [[tid_map[words[same_ctx(i)][2]],
                       tid_map[words[i][2]]]
                      for i in range(n_words)]}
    for k, ntids in null_sets.items():
        batch[k] = [[ntids[i], tid_map[words[i][2]]]
                    for i in range(n_words)]

    model, _ = load_native('qwen4')
    model.eval()
    layers = model.model.layers
    log('model loaded (load_native full GPU)', lines)

    cap = {'attnin': {}}
    state_fin = {'on': False}
    fin_cap = {}
    handles = []

    def pre_attn(li):
        def h(module, args, kwargs):
            x = args[0] if args else kwargs.get('hidden_states')
            if x is None or x.dim() < 2:
                return
            cap['attnin'].setdefault(li, []).append(
                x.detach().float().cpu().numpy())
        return h

    def pre_norm(module, args, kwargs):
        if state_fin['on']:
            fin_cap['x'] = args[0][:, -1, :].detach() \
                .float().cpu().numpy()

    for li in range(NL):
        handles.append(
            layers[li].self_attn.register_forward_pre_hook(
                pre_attn(li), with_kwargs=True))
    handles.append(model.model.norm.register_forward_pre_hook(
        pre_norm, with_kwargs=True))

    def clear_cap():
        for li in cap['attnin']:
            del cap['attnin'][li][:]

    def forward1(toks):
        clear_cap()
        with torch.no_grad():
            model(torch.tensor([toks], device='cuda'))
        return {li: cap['attnin'][li][0]
                for li in cap['attnin']}

    def forward_batch(toks_list):
        clear_cap()
        fin_cap.pop('x', None)
        state_fin['on'] = True
        with torch.no_grad():
            out = model(torch.tensor(toks_list, device='cuda'))
        state_fin['on'] = False
        fin = fin_cap['x'].astype(np.float64)
        attn = {li: cap['attnin'][li][0].astype(np.float64)
                for li in cap['attnin']}
        return fin, attn

    # ---------- pass 1: dirs_word rebuild ----------
    attn_store = {}
    for i, (_, _, w) in enumerate(words):
        attnin_all = forward1([func_tid, tid_map[w]])
        for li in range(NL):
            attn_store[(i, li)] = \
                attnin_all[li].astype(np.float32)
        if (i + 1) % 20 == 0:
            log('pass1 words [%d/%d]' % (i + 1, n_words), lines)
    d_dim = attn_store[(0, 0)].shape[-1]
    diffs_w = np.zeros((NL, d_dim))
    for li in range(NL):
        X = np.stack([attn_store[(i, li)][0, 1]
                      for i in range(n_words)]) \
            .astype(np.float64)
        diffs_w[li] = X[lab_lang == 0].mean(0) \
            - X[lab_lang == 1].mean(0)
    dirs_word = np.stack([unit(diffs_w[li]) for li in range(NL)])
    a1_diff = float(np.abs(dirs_word - dirs_word_27).max())
    a1_ok = bool(a1_diff < 1e-5)
    log('a1 dirs_word rebuild diff %.2e ok=%s'
        % (a1_diff, a1_ok), lines)

    # ---------- subspace: SVD of stacked dirs ----------
    U_svd, s_svd, Vt = np.linalg.svd(dirs_word,
                                     full_matrices=False)
    orth_err = float(np.abs(Vt @ Vt.T - np.eye(Vt.shape[0]))
                     .max())
    a5_ok = bool(orth_err < 1e-8)
    log('a5 SVD orthonormality err %.2e ok=%s (sing vals '
        'top5 %s)' % (orth_err, a5_ok,
                      [round(float(v), 4)
                       for v in s_svd[:5]]), lines)

    # ---------- baselines: 5 conditions ----------
    fin_f1, _ = forward_batch(batch['func'])
    fin_f2, _ = forward_batch(batch['func'])
    a2_rel = float(np.abs(fin_f1 - fin_f2).max()
                   / max(float(np.abs(fin_f1).max()), 1e-30))
    a2_ok = bool(a2_rel < 1e-4)
    log('a2 baseline determinism rel %.2e ok=%s'
        % (a2_rel, a2_ok), lines)

    u35 = dirs_word[NL - 1]
    fin_vec = {}
    attn = {}
    proj = {}
    for cn in conds_all:
        fin, at = forward_batch(batch[cn])
        fin_vec[cn] = fin
        attn[cn] = at
        proj[cn] = fin @ u35
    log('baseline sweep done (%d conds)' % len(conds_all),
        lines)

    a3_diff = float(np.abs(
        proj['func'] - s_base_35[ifu35]).max())
    a3_ok = bool(a3_diff < 1e-4)
    log('a3 proj_func vs 2935 s_base max abs diff %.2e ok=%s'
        % (a3_diff, a3_ok), lines)
    a4_diff = float(np.abs(
        proj['null0'] - s_base_35[in035]).max())
    a4_ok = bool(a4_diff < 1e-4)
    log('a4 proj_null0 vs 2935 s_base max abs diff %.2e ok=%s'
        % (a4_diff, a4_ok), lines)

    sep_f = float(proj['func'][lab_lang == 0].mean()
                  - proj['func'][lab_lang == 1].mean())
    a6_ok = bool(sep_f > 0.0)
    log('a6 func separation %.4f ok=%s' % (sep_f, a6_ok),
        lines)

    n_fin = {cn: np.linalg.norm(fin_vec[cn], axis=1)
             for cn in conds_all}
    align_d35 = {cn: np.abs(proj[cn])
                 / np.maximum(n_fin[cn], 1e-30)
                 for cn in conds_all}
    ifu37 = conds37.index('func')
    in037 = conds37.index('null0')
    a7_diff = max(
        float(np.abs(align_d35['func']
                     - np.abs(proj37[ifu37])
                     / np.maximum(fnorm37[ifu37], 1e-30)).max()),
        float(np.abs(align_d35['null0']
                     - np.abs(proj37[in037])
                     / np.maximum(fnorm37[in037], 1e-30)).max()))
    a7_ok = bool(a7_diff < 1e-6)
    log('a7 align_dir35 vs 2937 |proj|/fin_norm max diff '
        '%.2e ok=%s' % (a7_diff, a7_ok), lines)

    anchor_ok = bool(a1_ok and a2_ok and a3_ok and a4_ok
                     and a5_ok and a6_ok and a7_ok)
    verdict = None
    p1 = p2 = p3 = None
    save = {}

    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        # ---------- P1: final alignment grid ----------
        al_k = {}   # (cn, k) -> per-word alignment
        for cn in conds_all:
            X = fin_vec[cn]
            nrm = np.maximum(n_fin[cn], 1e-30)
            for k in KS:
                Uk = Vt[:k]
                al_k[(cn, k)] = np.linalg.norm(
                    X @ Uk.T, axis=1) / nrm
        p1 = {'median_align': {}, 'rho_per_null': {},
              'rho_median': {}}
        for k in list(KS) + ['dir35']:
            src = ({cn: al_k[(cn, k)] for cn in conds_all}
                   if k != 'dir35'
                   else {cn: align_d35[cn]
                         for cn in conds_all})
            med = {cn: float(np.median(src[cn]))
                   for cn in conds_all}
            p1['median_align'][str(k)] = {
                cn: round(med[cn], 6) for cn in conds_all}
            rho = {}
            for cn in conds_all:
                if cn == 'func':
                    continue
                rho[cn] = float(np.median(
                    src[cn] / np.maximum(src['func'], 1e-30)))
            p1['rho_per_null'][str(k)] = {
                cn: round(v, 4) for cn, v in rho.items()}
            nulls = [rho[cn] for cn in rho
                     if cn.startswith('null')]
            p1['rho_median'][str(k)] = round(
                float(np.median(nulls)), 4)
            log('P1 k=%s: med func %.4f null0 %.4f | rho '
                'null0 %.4f same %.4f | rho_median(null) %.4f'
                % (k, med['func'], med['null0'],
                   rho['null0'], rho['same'],
                   p1['rho_median'][str(k)]), lines)
        rho_k8 = p1['rho_median'][str(K_MAIN)]
        rho_d35 = p1['rho_median']['dir35']
        log('P1 rho_k8 %.4f vs rho_d35 %.4f (delta %+.4f)'
            % (rho_k8, rho_d35, rho_k8 - rho_d35), lines)

        # ---------- P2: paired permutation ----------
        a_f = al_k[('func', K_MAIN)]
        a_n = al_k[('null0', K_MAIN)]
        d_obs = float(np.median(a_f - a_n))
        rng2 = np.random.default_rng(RNG_P2)
        diffs = a_f - a_n
        cnt = 0
        for _ in range(N_P2):
            sg = rng2.choice(np.array([-1.0, 1.0]),
                             size=n_words)
            if float(np.median(sg * diffs)) >= d_obs - 1e-12:
                cnt += 1
        p2 = {'median_diff_func_minus_null0': round(d_obs, 6),
              'p_perm': float('%.3e'
                              % ((cnt + 1) / (N_P2 + 1))),
              'rng': RNG_P2, 'n_perm': N_P2}
        log('P2 func-null0 alpha_%d median diff %+.6f '
            'p %.3e' % (K_MAIN, d_obs, p2['p_perm']), lines)

        # ---------- P3: per-layer profile ----------
        p3 = {'ratio_alpha8': {}, 'ratio_dir': {},
              'first_crossing': {}}
        for cn in conds_all:
            if cn == 'func':
                continue
            r8 = np.zeros(NL)
            rd = np.zeros(NL)
            for li in range(NL):
                Xf = attn['func'][li][:, 1, :]
                Xc = attn[cn][li][:, 1, :]
                nf = np.maximum(np.linalg.norm(Xf, axis=1),
                                1e-30)
                nc = np.maximum(np.linalg.norm(Xc, axis=1),
                                1e-30)
                U8 = Vt[:K_MAIN]
                af = np.linalg.norm(Xf @ U8.T, axis=1) / nf
                ac = np.linalg.norm(Xc @ U8.T, axis=1) / nc
                r8[li] = float(np.median(ac
                                         / np.maximum(af,
                                                      1e-30)))
                df = np.abs(Xf @ dirs_word[li]) / nf
                dc = np.abs(Xc @ dirs_word[li]) / nc
                rd[li] = float(np.median(dc
                                         / np.maximum(df,
                                                      1e-30)))
            p3['ratio_alpha8'][cn] = [round(float(v), 4)
                                      for v in r8]
            p3['ratio_dir'][cn] = [round(float(v), 4)
                                   for v in rd]
            c8 = [int(li) for li in range(NL) if r8[li] < 0.8]
            cd = [int(li) for li in range(NL) if rd[li] < 0.8]
            p3['first_crossing'][cn] = {
                'alpha8': c8[0] if c8 else None,
                'dir': cd[0] if cd else None}
            log('P3 %s: alpha8 ratio L1 %.3f L6 %.3f L12 %.3f '
                'L24 %.3f L35 %.3f | dir ratio %.3f %.3f '
                '%.3f %.3f %.3f'
                % (cn, r8[1], r8[6], r8[12], r8[24], r8[35],
                   rd[1], rd[6], rd[12], rd[24], rd[35]),
                lines)

        # ---------- verdict ----------
        if rho_k8 >= 0.6 and rho_k8 >= rho_d35 + 0.2:
            verdict = 'subspace_rotation_retained'
        elif rho_k8 < 0.5:
            verdict = 'subspace_collapse_confirmed'
        else:
            verdict = 'subspace_mixed'

        save = {
            'words': np.array(words),
            'cond_names': np.array(conds_all),
            'align_k': np.stack(
                [np.stack([al_k[(cn, k)] for k in KS])
                 for cn in conds_all]),
            'align_dir35': np.stack(
                [align_d35[cn] for cn in conds_all]),
            'fin_norm': np.stack(
                [n_fin[cn] for cn in conds_all]),
            'sing_vals': s_svd,
            'ratio_alpha8_layers': np.stack(
                [np.array(p3['ratio_alpha8'][cn])
                 for cn in conds_all if cn != 'func']),
            'ratio_dir_layers': np.stack(
                [np.array(p3['ratio_dir'][cn])
                 for cn in conds_all if cn != 'func'])}

    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2938, 'model': 'qwen3-4b',
           'prereg': PREREG,
           'anchors': {'a1_diff': float('%.3e' % a1_diff),
                       'a1_ok': a1_ok,
                       'a2_rel': float('%.3e' % a2_rel),
                       'a2_ok': a2_ok,
                       'a3_diff': float('%.3e' % a3_diff),
                       'a3_ok': a3_ok,
                       'a4_diff': float('%.3e' % a4_diff),
                       'a4_ok': a4_ok,
                       'a5_orth_err':
                           float('%.3e' % orth_err),
                       'a5_ok': a5_ok,
                       'sep_func': round(sep_f, 4),
                       'a6_ok': a6_ok,
                       'a7_diff': float('%.3e' % a7_diff),
                       'a7_ok': a7_ok,
                       'ok': anchor_ok},
           'P1': p1, 'P2': p2, 'P3': p3,
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if save:
        np.savez_compressed(os.path.join(
            OUT, 'subspace_angles.npz'), **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2938 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
