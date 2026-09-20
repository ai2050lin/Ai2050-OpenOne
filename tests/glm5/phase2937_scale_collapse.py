# -*- coding: utf-8 -*-
"""Phase 2937: scale-collapse mechanism anatomy.

Why: 2936 found that the 2934/2935 "null amplification" is
entirely denominator-driven: the baseline readout magnitude
scale = mean|s_base| collapses under null context (92.29 ->
37-49, ratio 0.41-0.54) while the raw ablation disturbance
slightly SHRINKS. Mechanism unresolved: why does a random
context token halve |proj of the final residual on
dirs_word[35]|? Three candidate mechanisms:
  M1 energy scaling: word-token readout structure preserved,
     residual energy scaled down (s_null ~ beta*s_func, beta
     in (0,1], high cross-word correlation);
  M2 rotation: energy preserved but direction rotates away
     from dirs_word[35];
  M3 rewrite: readout rewritten by context (beta low).

Mode: ONE run (qwen3-4b), NO ablation. Baseline forwards
only: pass1 dirs_word rebuild (57 single forwards, 2935
verbatim) + 5 conditions (func/same/null0-3) x batch57,
capturing per-layer pos-1 attn-input vectors (pre_attn hook
verbatim) and final pre-norm residual. Plus zero-forward
embedding-norm lookup (P4).

Anchors (frozen):
  a1 dirs_word rebuild vs 2927 npz max abs < 1e-5
  a2 func baseline determinism max rel < 1e-4
  a3 proj_func(final) vs 2935 npz s_base[func] < 1e-4
  a4 proj_null0(final) vs 2935 npz s_base[null0] < 1e-4
  a5 mask counts 469/382/295
  a6 func separation > 0

Main tests (frozen):
  P1 collapse-layer localization (descriptive): per layer
     ratio_c(li) = median_w ||attn_in_c[w,li,pos1]|| /
     median_w ||attn_in_func[w,li,pos1]||; first crossing
     li* = min li with ratio < 0.8, reported per null set;
     per-layer separation profile sep_c(li) = mean proj |
     lab0 - mean proj | lab1 on dirs_word[li].
  P2 rewrite vs scaling: OLS s_c(w) = beta*s_func(w) + gamma
     over 57 words per null set + same; median beta over 4
     null sets; R2 registered.
  P3 energy vs direction: per null set, ratio_norm =
     median_w ||x_n(w)||/||x_f(w)||, ratio_cos = median_w
     cos_n(w)/cos_f(w) (cos vs dirs_word[35] of final pre-
     norm residual); energy dominance iff
     |log ratio_norm| >= |log ratio_cos|.
  P4 embedding norms (zero forward): word tids vs null tids
     mean/std + permutation p (rng 2918, 10000).

Verdict (frozen):
  anchor fail                        => anchor_fail_all_void
  P2 median beta >= 0.8 AND P3 energy dominance
                                     => scale_collapse_energy_scaling
  P2 median beta >= 0.8 (else)       => scale_collapse_rotation
  P2 median beta < 0.8               => scale_collapse_rewrite

Output: phase2937/scale_collapse/.
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
OUT = os.path.join(BASE, 'phase2937', 'scale_collapse')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2937_run_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
SEED = 2896
NH, HD = 32, 128
NL = 36
VOCAB = 151936
RNG_P4 = 2918
N_GRP = 10000
EXP_N86, EXP_NWD, EXP_SHARED = 469, 382, 295
NULL_SEEDS = (2896, 2914, 2915, 2916)

PREREG = {
    'mode': 'ONE run, NO ablation: pass1 dirs_word rebuild '
            '(57 single forwards, 2935 verbatim) + 5 baseline '
            'conditions (func/same/null0-3) x batch57 with '
            'per-layer pos-1 attn-input capture (pre_attn '
            'hook verbatim) + final pre-norm residual; P4 '
            'zero-forward embedding lookup',
    'question': 'why does null context halve the baseline '
                'readout magnitude: energy scaling (M1), '
                'rotation away from dirs_word[35] (M2), or '
                'readout rewrite (M3)?',
    'anchors': {
        'a1': 'dirs_word rebuild vs 2927 npz < 1e-5',
        'a2': 'func baseline determinism < 1e-4',
        'a3': 'proj_func vs 2935 s_base[func] < 1e-4',
        'a4': 'proj_null0 vs 2935 s_base[null0] < 1e-4',
        'a5': 'mask counts 469/382/295',
        'a6': 'func separation > 0',
    },
    'P1': 'collapse-layer ratio profile per condition '
          '(descriptive, first crossing < 0.8) + per-layer '
          'separation profile',
    'P2': 'OLS s_c = beta*s_func + gamma over 57 words per '
          'condition; median beta over 4 null sets',
    'P3': 'ratio_norm vs ratio_cos decomposition of the '
          'readout collapse, energy dominance iff '
          '|log ratio_norm| >= |log ratio_cos|',
    'P4': 'embedding norm word tids vs null tids, '
          'permutation p (rng 2918, 10000)',
    'verdict': 'anchor fail => anchor_fail_all_void; P2 '
               'median beta >= 0.8 AND P3 energy dominance => '
               'scale_collapse_energy_scaling; P2 median beta '
               '>= 0.8 else => scale_collapse_rotation; P2 '
               'median beta < 0.8 => scale_collapse_rewrite',
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
        json.dump({'phase': 2937,
                   'name': 'scale_collapse',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2887': sha8(SRC_2887),
                               's2927': sha8(SRC_2927),
                               's2929': sha8(SRC_2929),
                               's2930': sha8(SRC_2930),
                               's2931': sha8(SRC_2931),
                               's2935': sha8(SRC_2935)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'seed': SEED, 'rng_p4': RNG_P4,
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
    S29c = z31['S29_corrected'].astype(bool)
    Smirc = z31['Smir_corrected'].astype(bool)
    n86, nwd, nsh = int(S29c.sum()), int(Smirc.sum()), \
        int((S29c & Smirc).sum())
    a5_ok = bool(n86 == EXP_N86 and nwd == EXP_NWD
                 and nsh == EXP_SHARED)
    log('a5 mask counts n86=%d nwd=%d shared=%d ok=%s'
        % (n86, nwd, nsh, a5_ok), lines)

    z35 = np.load(SRC_2935, allow_pickle=True)
    conds35 = [str(s) for s in z35['cond_names']]
    s_base_35 = z35['s_base'].astype(np.float64)
    scale35 = z35['scale'].astype(np.float64)
    ifu35 = conds35.index('func')
    in035 = conds35.index('null0')

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
    attn_norm = {cn: np.zeros((NL, n_words))
                 for cn in conds_all}
    attn_proj = {cn: np.zeros((NL, n_words))
                 for cn in conds_all}
    proj = {}
    for cn in conds_all:
        fin, attn = forward_batch(batch[cn])
        fin_vec[cn] = fin
        proj[cn] = fin @ u35
        for li in range(NL):
            X = attn[li][:, 1, :]
            attn_norm[cn][li] = np.linalg.norm(X, axis=1)
            attn_proj[cn][li] = X @ dirs_word[li]
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

    anchor_ok = bool(a1_ok and a2_ok and a3_ok and a4_ok
                     and a5_ok and a6_ok)
    scale_here = {cn: float(np.mean(np.abs(proj[cn])))
                  for cn in conds_all}
    verdict = None
    p1 = p2 = p3 = p4 = None
    save = {}

    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        log('scale: %s (2935 func %.4f null0 %.4f)'
            % ({k: round(v, 3) for k, v in
                scale_here.items()},
               scale35[ifu35], scale35[in035]), lines)

        # ---------- P1: collapse-layer profile ----------
        p1 = {'ratio_first_crossing': {},
              'sep_func_l35': round(sep_f, 4),
              'sep_profile': {}}
        nf = attn_norm['func']
        for cn in conds_all:
            if cn == 'func':
                continue
            r_lay = (np.median(attn_norm[cn], axis=1)
                     / np.maximum(np.median(nf, axis=1),
                                  1e-30))
            cross = [int(li) for li in range(NL)
                     if r_lay[li] < 0.8]
            p1['ratio_first_crossing'][cn] = \
                cross[0] if cross else None
            sp = attn_proj[cn]
            sep_p = []
            for li in range(NL):
                d = sp[li][lab_lang == 0].mean() \
                    - sp[li][lab_lang == 1].mean()
                sep_p.append(round(float(d), 4))
            p1['sep_profile'][cn] = sep_p
            log('P1 %s: first crossing li*=%s | ratio L1 %.3f '
                'L6 %.3f L12 %.3f L24 %.3f L35 %.3f'
                % (cn, p1['ratio_first_crossing'][cn],
                   r_lay[1], r_lay[6], r_lay[12],
                   r_lay[24], r_lay[35]), lines)

        # ---------- P2: rewrite vs scaling ----------
        betas = []
        p2 = {'fits': {}}
        for cn in conds_all:
            if cn == 'func':
                continue
            x = proj['func']
            y = proj[cn]
            xm, ym = x.mean(), y.mean()
            b = float(((x - xm) * (y - ym)).sum()
                      / max(((x - xm) ** 2).sum(), 1e-30))
            a = float(ym - b * xm)
            pred = a + b * x
            r2 = float(1.0 - ((y - pred) ** 2).sum()
                       / max(((y - ym) ** 2).sum(), 1e-30))
            p2['fits'][cn] = {'beta': round(b, 4),
                              'gamma': round(a, 4),
                              'r2': round(r2, 4)}
            if cn.startswith('null'):
                betas.append(b)
            log('P2 %s: beta %.4f gamma %.4f R2 %.4f'
                % (cn, b, a, r2), lines)
        beta_med = float(np.median(betas))
        p2['median_beta_null'] = round(beta_med, 4)
        log('P2 median beta (null) %.4f' % beta_med, lines)

        # ---------- P3: energy vs direction ----------
        p3 = {}
        n_fin = {cn: np.linalg.norm(fin_vec[cn], axis=1)
                 for cn in conds_all}
        cos_f = proj['func'] / np.maximum(n_fin['func'],
                                          1e-30)
        for cn in conds_all:
            if cn == 'func':
                continue
            cos_c = proj[cn] / np.maximum(n_fin[cn], 1e-30)
            rn = float(np.median(n_fin[cn]
                                 / np.maximum(n_fin['func'],
                                              1e-30)))
            rc = float(np.median(cos_c
                                 / np.maximum(cos_f, 1e-30)))
            ln = float(np.log(abs(rn))
                       if abs(rn) > 1e-30 else 0.0)
            lc = float(np.log(abs(rc))
                       if abs(rc) > 1e-30 else 0.0)
            p3[cn] = {'ratio_norm': round(rn, 4),
                      'ratio_cos': round(rc, 4),
                      'log_norm': round(ln, 4),
                      'log_cos': round(lc, 4),
                      'energy_dominant':
                          bool(abs(ln) >= abs(lc))}
            log('P3 %s: norm ratio %.4f cos ratio %.4f '
                'energy_dominant=%s'
                % (cn, rn, rc, p3[cn]['energy_dominant']),
                lines)
        dom_count = sum(1 for cn in p3
                        if p3[cn]['energy_dominant'])
        energy_med = bool(dom_count >= 3)
        p3['energy_dominant_median'] = energy_med

        # ---------- P4: embedding norms ----------
        emb = model.model.embed_tokens.weight.detach() \
            .float().cpu().numpy()
        w_norms = np.array(
            [float(np.linalg.norm(emb[tid_map[words[i][2]]]))
             for i in range(n_words)])
        n_norms = np.array(
            [float(np.linalg.norm(emb[t]))
             for k in null_sets for t in null_sets[k]])
        f_norm = float(np.linalg.norm(emb[func_tid]))
        rng4 = np.random.default_rng(RNG_P4)
        obs_d = float(n_norms.mean() - w_norms.mean())
        cnt = 0
        pool = np.concatenate([w_norms, n_norms])
        for _ in range(N_GRP):
            pm = rng4.permutation(len(pool))
            if abs(pool[pm[:len(w_norms)]].mean()
                   - pool[pm[len(w_norms):]].mean()) \
                    >= abs(obs_d):
                cnt += 1
        p4 = {'word_norm_mean': round(float(w_norms.mean()),
                                      4),
              'word_norm_std': round(float(w_norms.std()), 4),
              'null_norm_mean': round(float(n_norms.mean()),
                                      4),
              'null_norm_std': round(float(n_norms.std()), 4),
              'func_tid_norm': round(f_norm, 4),
              'diff': round(obs_d, 4),
              'p_perm': float('%.3e'
                              % ((cnt + 1) / (N_GRP + 1)))}
        log('P4 embed norms: words %.3f+-%.3f null %.3f+-%.3f '
            'func_tid %.3f diff %.3f p %.3e'
            % (p4['word_norm_mean'], p4['word_norm_std'],
               p4['null_norm_mean'], p4['null_norm_std'],
               f_norm, obs_d, p4['p_perm']), lines)

        # ---------- verdict ----------
        if beta_med >= 0.8 and energy_med:
            verdict = 'scale_collapse_energy_scaling'
        elif beta_med >= 0.8:
            verdict = 'scale_collapse_rotation'
        else:
            verdict = 'scale_collapse_rewrite'

        save = {
            'words': np.array(words),
            'attn_norm': np.stack(
                [attn_norm[cn] for cn in conds_all]),
            'attn_proj': np.stack(
                [attn_proj[cn] for cn in conds_all]),
            'cond_names': np.array(conds_all),
            'proj': np.stack([proj[cn]
                              for cn in conds_all]),
            'fin_norm': np.stack(
                [n_fin[cn] for cn in conds_all])}

    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2937, 'model': 'qwen3-4b',
           'prereg': PREREG,
           'anchors': {'a1_diff': float('%.3e' % a1_diff),
                       'a1_ok': a1_ok,
                       'a2_rel': float('%.3e' % a2_rel),
                       'a2_ok': a2_ok,
                       'a3_diff': float('%.3e' % a3_diff),
                       'a3_ok': a3_ok,
                       'a4_diff': float('%.3e' % a4_diff),
                       'a4_ok': a4_ok,
                       'a5_ok': a5_ok,
                       'sep_func': round(sep_f, 4),
                       'a6_ok': a6_ok,
                       'ok': anchor_ok},
           'P1': p1, 'P2': p2, 'P3': p3, 'P4': p4,
           'scale_this_run': {k: round(v, 4) for k, v in
                              scale_here.items()},
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if save:
        np.savez_compressed(os.path.join(OUT,
                                         'scale_collapse.npz'),
                            **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2937 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
