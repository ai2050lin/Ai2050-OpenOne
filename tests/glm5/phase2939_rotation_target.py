# -*- coding: utf-8 -*-
"""Phase 2939: rotation-target localization inside the
language subspace.

Why: 2938 established subspace_rotation_retained - the
null-context collapse is strictly confined to the single
direction dirs_word[35] (rho 0.509) while subspace
alignment (k=8) is fully retained (rho 0.999, 0/57 words
leave). Open question: WHERE inside the 8-dim subspace
does the word-term structure go? Which basis directions
absorb the displaced energy (re-encoding targets)?

Mode: ONE run (qwen3-4b), NO ablation. 2938 protocol
verbatim (pass1 dirs_word rebuild + 5 conditions x
batch57, final pre-norm residual). New: per-word
coordinates in the top-8 SVD basis, c(w,k) = x_w . v_k.

Anchors (frozen):
  a1 dirs_word rebuild vs 2927 npz < 1e-5
  a2 func baseline determinism < 1e-4
  a3 proj_func(final) vs 2935 npz s_base[func] < 1e-4
  a4 proj_null0(final) vs 2935 npz s_base[null0] < 1e-4
  a5 alpha_k this run vs 2938 npz align_k max diff
     < 1e-6 (cross-phase bit-level)
  a6 func separation > 0

Main tests (frozen):
  P1 structure retention per basis: Spearman over 57
     words of coords func vs condition c, for k=1..8 and
     dir35; rho_med(k) = median over 4 null sets.
  P2 energy-share shift: share_k = mean_w c_k^2 / sum_k'
     (per condition); delta_e_med(k) = median over null
     sets of share_null - share_func; top inflow basis
     k* = argmax; permutation p (label swap per word
     pair, rng 2920, 10000) for null0 at k*.
  P3 class-mean displacement: delta_c = mean_null coords
     - mean_func coords (overall + per lab class), top
     basis components registered.

Verdict (frozen):
  anchor fail => anchor_fail_all_void
  max_k rho_med(k in U8) >= 0.7 AND
  max_k delta_e_med(k) > +0.02 => rotation_target_identified
  max_k rho_med(k in U8) < 0.5 => rotation_target_not_found
  else => rotation_target_partial

Output: phase2939/rotation_target/.
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
SRC_2938 = os.path.join(BASE, 'phase2938', 'subspace_angles',
                        'subspace_angles.npz')
OUT = os.path.join(BASE, 'phase2939', 'rotation_target')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2939_run_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
SEED = 2896
NH, HD = 32, 128
NL = 36
VOCAB = 151936
K8 = 8
RNG_P2 = 2920
N_P2 = 10000
NULL_SEEDS = (2896, 2914, 2915, 2916)
DE_MIN = 0.02
RHO_KEEP = 0.7
RHO_LOSE = 0.5

PREREG = {
    'mode': 'ONE run, NO ablation: 2938 protocol verbatim '
            '(pass1 dirs_word rebuild + 5 conditions x '
            'batch57, final pre-norm residual); NEW per-word '
            'coordinates c(w,k) = x_w . v_k in the top-8 SVD '
            'basis of the stacked dirs_word',
    'question': 'where inside the 8-dim language subspace '
                'does the word-term structure go under null '
                'context: which basis directions absorb the '
                'displaced energy (re-encoding targets)?',
    'anchors': {
        'a1': 'dirs_word rebuild vs 2927 npz < 1e-5',
        'a2': 'func baseline determinism < 1e-4',
        'a3': 'proj_func vs 2935 s_base[func] < 1e-4',
        'a4': 'proj_null0 vs 2935 s_base[null0] < 1e-4',
        'a5': 'alpha_k vs 2938 npz align_k < 1e-6',
        'a6': 'func separation > 0',
    },
    'P1': 'per-basis Spearman structure retention '
          '(k=1..8 + dir35), rho_med = median over 4 null '
          'sets',
    'P2': 'energy-share shift share_k, delta_e_med per '
          'basis, top inflow k*; label-swap permutation p '
          '(rng 2920, 10000) for null0 at k*',
    'P3': 'class-mean displacement delta_c decomposition '
          '(overall + lab0/lab1)',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'max_k rho_med(k in U8) >= 0.7 AND max_k '
               'delta_e_med(k) > +0.02 => '
               'rotation_target_identified; max_k rho_med '
               '< 0.5 => rotation_target_not_found; else '
               '=> rotation_target_partial',
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
    """Average ranks (ties safe)."""
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
        json.dump({'phase': 2939,
                   'name': 'rotation_target',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2887': sha8(SRC_2887),
                               's2927': sha8(SRC_2927),
                               's2929': sha8(SRC_2929),
                               's2930': sha8(SRC_2930),
                               's2931': sha8(SRC_2931),
                               's2935': sha8(SRC_2935),
                               's2938': sha8(SRC_2938)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'seed': SEED, 'k_basis': K8,
                   'rng_p2': RNG_P2, 'n_p2': N_P2,
                   'de_min': DE_MIN, 'rho_keep': RHO_KEEP,
                   'rho_lose': RHO_LOSE,
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
    z38 = np.load(SRC_2938, allow_pickle=True)
    conds38 = [str(s) for s in z38['cond_names']]
    align38 = z38['align_k'].astype(np.float64)

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

    U_svd, s_svd, Vt = np.linalg.svd(dirs_word,
                                     full_matrices=False)
    U8 = Vt[:K8]

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
    proj = {}
    coords = {}
    n_fin = {}
    for cn in conds_all:
        fin, _ = forward_batch(batch[cn])
        fin_vec[cn] = fin
        proj[cn] = fin @ u35
        coords[cn] = fin @ U8.T
        n_fin[cn] = np.linalg.norm(fin, axis=1)
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

    a5_diff = 0.0
    ks38 = [1, 4, 8, 16, 36]
    for ci, cn in enumerate(conds_all):
        for ki, k in enumerate(ks38):
            if k > K8:
                continue
            a = np.linalg.norm(coords[cn][:, :k], axis=1) \
                / np.maximum(n_fin[cn], 1e-30)
            b = align38[ci, ki]
            a5_diff = max(a5_diff, float(np.abs(a - b).max()))
    a5_ok = bool(a5_diff < 1e-6)
    log('a5 alpha_k vs 2938 align_k max diff %.2e ok=%s'
        % (a5_diff, a5_ok), lines)

    sep_f = float(proj['func'][lab_lang == 0].mean()
                  - proj['func'][lab_lang == 1].mean())
    a6_ok = bool(sep_f > 0.0)
    log('a6 func separation %.4f ok=%s' % (sep_f, a6_ok),
        lines)

    anchor_ok = bool(a1_ok and a2_ok and a3_ok and a4_ok
                     and a5_ok and a6_ok)
    verdict = None
    p1 = p2 = p3 = None
    save = {}

    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        # ---------- P1: per-basis structure retention ------
        p1 = {'rho_per_null': {}, 'rho_med': {}}
        bases = [('dir35', {cn: proj[cn] for cn in conds_all})]
        for k in range(K8):
            bases.append(('v%d' % (k + 1),
                          {cn: coords[cn][:, k]
                           for cn in conds_all}))
        for bname, cd in bases:
            rho = {}
            for cn in conds_all:
                if cn == 'func':
                    continue
                rho[cn] = float(spearman(cd['func'], cd[cn]))
            nulls = [rho[cn] for cn in rho
                     if cn.startswith('null')]
            p1['rho_per_null'][bname] = {
                cn: round(v, 4) for cn, v in rho.items()}
            p1['rho_med'][bname] = round(
                float(np.median(nulls)), 4)
            log('P1 %s: rho null0 %.4f same %.4f | '
                'rho_med(null) %.4f'
                % (bname, rho['null0'], rho['same'],
                   p1['rho_med'][bname]), lines)
        rho_u8 = {k: p1['rho_med']['v%d' % (k + 1)]
                  for k in range(K8)}
        max_rho = max(rho_u8.values())
        k_best = max(rho_u8, key=rho_u8.get)
        log('P1 max rho_med over U8: v%d %.4f'
            % (k_best + 1, max_rho), lines)

        # ---------- P2: energy-share shift ----------------
        sq = {cn: coords[cn] ** 2 for cn in conds_all}
        share = {}
        for cn in conds_all:
            tot = sq[cn].sum(axis=0).sum()
            share[cn] = sq[cn].sum(axis=0) / max(tot, 1e-30)
        de = np.zeros(K8)
        for cn in conds_all:
            if not cn.startswith('null'):
                continue
            de += share[cn] - share['func']
        de /= 4.0
        k_star = int(np.argmax(de))
        p2 = {'share_func': [round(float(v), 4)
                             for v in share['func']],
              'share_null0': [round(float(v), 4)
                              for v in share['null0']],
              'delta_e_med': [round(float(v), 4)
                              for v in de],
              'k_star': k_star + 1}
        log('P2 share func %s' % p2['share_func'], lines)
        log('P2 delta_e_med %s | k* = v%d (%+.4f)'
            % (p2['delta_e_med'], k_star + 1,
               de[k_star]), lines)
        # permutation: label swap per word pair, null0
        obs_de = float(share['null0'][k_star]
                       - share['func'][k_star])
        rng2 = np.random.default_rng(RNG_P2)
        cf = coords['func']
        cn0 = coords['null0']
        cnt = 0
        for _ in range(N_P2):
            sw = rng2.random(n_words) < 0.5
            a = np.where(sw[:, None], cn0, cf)
            b = np.where(sw[:, None], cf, cn0)
            sa = (a ** 2).sum(axis=0)
            sb = (b ** 2).sum(axis=0)
            d = sa[k_star] / max(float(sa.sum()), 1e-30) \
                - sb[k_star] / max(float(sb.sum()), 1e-30)
            if d >= obs_de - 1e-12:
                cnt += 1
        p2['perm_p_kstar'] = float(
            '%.3e' % ((cnt + 1) / (N_P2 + 1)))
        p2['obs_de_kstar'] = round(obs_de, 6)
        log('P2 k* v%d: obs delta_e(null0) %+.6f perm p %.3e'
            % (k_star + 1, obs_de, p2['perm_p_kstar']), lines)

        # ---------- P3: class-mean displacement -----------
        p3 = {'delta_c_overall': {}, 'delta_c_lab0': {},
              'delta_c_lab1': {}}
        m0 = lab_lang == 0
        m1 = lab_lang == 1
        for cn in conds_all:
            if cn == 'func':
                continue
            d_all = coords[cn].mean(0) - coords['func'].mean(0)
            d_0 = coords[cn][m0].mean(0) \
                - coords['func'][m0].mean(0)
            d_1 = coords[cn][m1].mean(0) \
                - coords['func'][m1].mean(0)
            p3['delta_c_overall'][cn] = [
                round(float(v), 3) for v in d_all]
            p3['delta_c_lab0'][cn] = [
                round(float(v), 3) for v in d_0]
            p3['delta_c_lab1'][cn] = [
                round(float(v), 3) for v in d_1]
            log('P3 %s delta_c overall %s'
                % (cn, p3['delta_c_overall'][cn]), lines)

        # ---------- verdict ----------
        max_de = float(de.max())
        if max_rho >= RHO_KEEP and max_de > DE_MIN:
            verdict = 'rotation_target_identified'
        elif max_rho < RHO_LOSE:
            verdict = 'rotation_target_not_found'
        else:
            verdict = 'rotation_target_partial'

        save = {
            'words': np.array(words),
            'cond_names': np.array(conds_all),
            'coords': np.stack([coords[cn]
                                for cn in conds_all]),
            'proj_dir35': np.stack([proj[cn]
                                    for cn in conds_all]),
            'fin_norm': np.stack([n_fin[cn]
                                  for cn in conds_all]),
            'sing_vals': s_svd,
            'Vt8': U8}

    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2939, 'model': 'qwen3-4b',
           'prereg': PREREG,
           'anchors': {'a1_diff': float('%.3e' % a1_diff),
                       'a1_ok': a1_ok,
                       'a2_rel': float('%.3e' % a2_rel),
                       'a2_ok': a2_ok,
                       'a3_diff': float('%.3e' % a3_diff),
                       'a3_ok': a3_ok,
                       'a4_diff': float('%.3e' % a4_diff),
                       'a4_ok': a4_ok,
                       'a5_diff': float('%.3e' % a5_diff),
                       'a5_ok': a5_ok,
                       'sep_func': round(sep_f, 4),
                       'a6_ok': a6_ok,
                       'ok': anchor_ok},
           'P1': p1, 'P2': p2, 'P3': p3,
           'max_rho_u8': (None if p1 is None
                          else [max_rho, 'v%d' % (k_best + 1)]),
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if save:
        np.savez_compressed(os.path.join(
            OUT, 'rotation_target.npz'), **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2939 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
