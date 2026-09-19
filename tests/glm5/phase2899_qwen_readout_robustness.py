# -*- coding: utf-8 -*-
"""Phase 2899: qwen mlp readout cells multi-draw margin-first
robustness (G_multi_draw_lesson follow-up to 2898).

2896 verdict subspace_readout_tolerant rests on SINGLE-draw
acc>p95 flags (SEED=2896).  2898 showed on GLM4 attn that such
flags are null-draw fragile while margin statistics are stable.
This phase re-quantifies the three qwen mlp cells with N_SEEDS
independent null-token draws, ONE model load, mlp channel only,
three conditions (window [26,36), words verbatim 2887, directions
zero-forward from 2886 S_last):

  stale: inject d18 (li=18), project d18
  eigen: inject d(li), project d(li)          (ANCHOR, strongest 2896)
  orth:  inject unit(d(li)-(d(li).d18)d18), project it

Per draw s in SEEDS:
  null_tids from rng(s) (2896 exclusion rules, VOCAB=151936);
  perms from rng2(s), 200 permutations;
  acc, p95, flag_acc = acc > p95; margin, p95_m,
  flag_margin = margin > p95_m.

v1 (per draw): mlp recompute rel err < 1e-6 AND hook-vs-call
noise < 1e-4 (shape-matched [1,2,2560], 2896 run-2 convention),
else that draw void.
v2 anchor gate (margin-first, G_multi_draw_lesson): eigen
margin_flag rate >= 0.8 across valid draws, else all void.

Frozen decision rules (on valid draws):
  stale:  margin_flag_rate >= 0.8 AND acc_flag_rate >= 0.5
            => qwen_stale_robust
          margin_flag_rate == 0
            => qwen_stale_margin_absent
          else qwen_stale_mixed (rates recorded)
  overall: anchor passed AND all three conditions
           margin_flag_rate >= 0.8 AND acc_flag_rate >= 0.5
            => subspace_readout_tolerant_robust
           anchor passed AND stale margin_flag_rate == 0
            => subspace_tolerant_stale_fragile
           anchor passed otherwise
            => subspace_tolerant_mixed (rates recorded)

SEEDS = [2896, 2901, 2902, 2903, 2904, 2905, 2906, 2907] (includes
the previously-run draw for within-run comparability).
Output: phase2899/qwen_readout_robustness/.
"""
import hashlib
import io
import json
import os
import sys
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC_2886 = os.path.join(BASE, 'phase2886', 'hourglass_cka',
                        'hourglass_cka.npz')
SRC_2887 = os.path.join(BASE, 'phase2887', 'language_axis_mlp',
                        'language_axis_mlp.npz')
OUT = os.path.join(BASE, 'phase2899', 'qwen_readout_robustness')
sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')

SEEDS = [2896, 2901, 2902, 2903, 2904, 2905, 2906, 2907]
N_NULL = 200
EPS = 1.0
WIN_LO, WIN_HI = 26, 36
VOCAB = 151936

PREREG = {
    'window': 'W=[26,36) frozen (qwen 2884 precedent, 2896 verbatim)',
    'sources': 'dirs zero-forward from 2886 S_last (labels i%2 '
               'asserted); words/labels verbatim 2887 npz (57 '
               'words, tid in word string); conds same/func(the)/'
               'null verbatim 2896 protocol',
    'dirs': 'd18=dir(li=18); d(li)=dir(li); d_orth(li)=unit(d(li)-'
            '(d(li).d18)d18); cos(d(li),d18) recorded',
    'design': 'N_SEEDS=%d draws; per draw: null_tids from rng(s) '
              'with word-tid exclusion (2896 rules, VOCAB=%d), '
              '200 label perms from rng2(s); mlp channel only; '
              'conditions stale(d18) + eigen(d(li), ANCHOR) + '
              'orth' % (len(SEEDS), VOCAB),
    'v1': 'mlp recompute rel err < 1e-6 and hook-vs-call noise '
          '< 1e-4 (shape-matched [1,2,2560], 2896 run-2 '
          'convention), per draw, else that draw void',
    'v2': 'anchor gate margin-first (G_multi_draw_lesson): eigen '
          'margin_flag rate >= 0.8 across valid draws, else all '
          'void (procedure instability)',
    'decision': 'stale: margin_flag_rate >= 0.8 AND acc_flag_rate '
                '>= 0.5 => qwen_stale_robust; margin_flag_rate '
                '== 0 => qwen_stale_margin_absent; else '
                'qwen_stale_mixed. overall: anchor passed AND all '
                'three margin_flag_rate >= 0.8 AND acc_flag_rate '
                '>= 0.5 => subspace_readout_tolerant_robust; '
                'anchor passed AND stale margin_flag_rate == 0 '
                '=> subspace_tolerant_stale_fragile; anchor '
                'passed otherwise => subspace_tolerant_mixed',
    'seeds': SEEDS,
}


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


def zmat(M):
    mu = M.mean(axis=0, keepdims=True)
    sd = M.std(axis=0, keepdims=True)
    Z = (M - mu) / np.maximum(sd, 1e-30)
    n = np.linalg.norm(Z, axis=1, keepdims=True)
    return Z / np.maximum(n, 1e-30)


def loo_acc(C, lab):
    S = C @ C.T
    np.fill_diagonal(S, -2.0)
    nn = S.argmax(axis=1)
    return float(np.mean(lab[nn] == lab))


def unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-30)


def log(msg):
    print(msg, flush=True)


def main():
    t0 = time.monotonic()
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2899, 'name': 'qwen_readout_robustness',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8': sha8(os.path.abspath(__file__)),
                   'sources': {'s_last_2886': sha8(SRC_2886),
                               'vocab_2887': sha8(SRC_2887)},
                   'model': 'qwen3-4b',
                   'prereg': PREREG, 'seeds': SEEDS, 'eps': EPS,
                   'window': [WIN_LO, WIN_HI]},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen')

    # ---------- zero-forward: words, labels, directions ----------
    z87 = np.load(SRC_2887, allow_pickle=True)
    words = [tuple(str(w).split(':')) for w in z87['words']]
    lab_lang = z87['labels_lang']
    lab_concept = z87['labels_concept']
    n_words = len(words)
    assert n_words == 57

    z86 = np.load(SRC_2886, allow_pickle=True)
    S_last = z86['S_last'].astype(np.float64)   # (80, 37, 2560)
    lab_sent = np.asarray(z86['labels']).astype(int)
    assert all(int(lab_sent[i]) == i % 2 for i in range(80))
    mean_en = S_last[lab_sent == 0].mean(0)
    mean_fr = S_last[lab_sent == 1].mean(0)
    diffs = mean_en - mean_fr
    d18 = unit(diffs[18])
    dirs, orth_dirs, cos18 = {}, {}, {}
    for li in range(WIN_LO, WIN_HI):
        d = unit(diffs[li])
        dirs[li] = d
        cos18[li] = float(d @ d18)
        o = d - cos18[li] * d18
        orth_dirs[li] = unit(o)
    log('cos(d(li), d18): %s' % {k: round(v, 4)
                                 for k, v in cos18.items()})

    # ---------- model + hooks (2896 run-2 verbatim) ----------
    import torch
    from rdc_atlas_census import single_token_id
    from transformers import AutoTokenizer
    from phase2662_symmetric_mapping_contract import load_native

    MODEL_DIR = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
    tok = AutoTokenizer.from_pretrained(
        MODEL_DIR, local_files_only=True, trust_remote_code=True,
        use_fast=True)
    tc = {}
    tid_map = {'the': single_token_id(tok, 'the', tc)}
    for lang, ck, w in words:
        tid_map[w] = single_token_id(tok, w, tc)
        if lang == 'en':
            assert tid_map[w] == int(ck), \
                'concept key mismatch for %s' % w

    model, _ = load_native('qwen4')
    model.eval()
    layers = model.model.layers

    cap = {'mlpin': {}, 'mlpout': {}}
    handles = []

    def pre_hook(li):
        def h(module, args, kwargs):
            x = args[0] if args else kwargs.get('hidden_states')
            if x is None or x.dim() < 2:
                return
            cap['mlpin'].setdefault(li, []).append(
                x.detach().float().cpu().numpy())   # [1, seq, d]
        return h

    def out_hook(li):
        def h(module, args, output):
            o = output[0] if isinstance(output, tuple) else output
            if o.dim() < 2:
                return
            cap['mlpout'].setdefault(li, []).append(
                o.detach().float().cpu().numpy())   # [1, seq, d]
        return h

    for li in range(WIN_LO, WIN_HI):
        handles.append(layers[li].mlp.register_forward_pre_hook(
            pre_hook(li), with_kwargs=True))
        handles.append(layers[li].mlp.register_forward_hook(
            out_hook(li)))

    def clear_cap():
        for dd in cap:
            for li in cap[dd]:
                del cap[dd][li][:]

    def forward2(toks):
        clear_cap()
        with torch.no_grad():
            model(torch.tensor([toks], device='cuda'))
        return ({li: cap['mlpin'][li][0] for li in cap['mlpin']},
                {li: cap['mlpout'][li][0] for li in cap['mlpout']})

    def mlp_call(li, x):
        out = layers[li].mlp(torch.tensor(
            x, device='cuda', dtype=torch.bfloat16))
        if isinstance(out, tuple):
            out = out[0]
        return out[0, 1].detach().float().cpu().numpy()

    def same_ctx(i):
        lang = words[i][0]
        cands = [j for j in range(n_words)
                 if words[j][0] == lang and j != i]
        return min(cands, key=lambda j: tid_map[words[j][2]])

    n_win = WIN_HI - WIN_LO
    word_tids = set(tid_map.values())
    func_tid = tid_map['the']

    def draw_null_tids(seed):
        rng = np.random.default_rng(seed)
        out = []
        while len(out) < n_words:
            r = int(rng.integers(0, VOCAB))
            if r not in word_tids and r > 0:
                out.append(r)
        return out

    def margin_of(Sm, pl):
        n = len(pl)
        sm = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(n):
                sm[i, j] = (pl[i] == pl[j]) and i != j
        df = (~np.eye(n, dtype=bool)) & (~sm)
        return float(Sm[sm].mean() - Sm[df].mean())

    def stats_of(B, seed):
        C = zmat(B)
        acc = loo_acc(C, lab_lang)
        U = B / np.maximum(
            np.linalg.norm(B, axis=1, keepdims=True), 1e-30)
        Sm = U @ U.T
        n = len(lab_lang)
        same_m = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(n):
                same_m[i, j] = (lab_lang[i] == lab_lang[j]) and \
                    i != j
        off = ~np.eye(n, dtype=bool) & ~same_m
        m_obs = float(Sm[same_m].mean() - Sm[off].mean())
        rng2 = np.random.default_rng(seed)
        perms = [rng2.permutation(lab_lang) for _ in range(N_NULL)]
        null_acc = [loo_acc(C, pl) for pl in perms]
        null_m = [margin_of(Sm, pl) for pl in perms]
        return {'acc': round(acc, 4),
                'p95': round(float(np.percentile(null_acc, 95)), 4),
                'flag_acc': bool(acc > np.percentile(null_acc, 95)),
                'margin': round(m_obs, 4),
                'margin_p95': round(
                    float(np.percentile(null_m, 95)), 4),
                'flag_margin': bool(
                    m_obs > np.percentile(null_m, 95)),
                'concept_acc': round(loo_acc(C, lab_concept), 4)}

    # capture same/func once (seed-independent)
    log('capturing same/func contexts once...')
    base_caps = {}
    for cn in ('same', 'func'):
        per_word = {}
        for i, (_, _, w) in enumerate(words):
            w_tid = tid_map[w]
            if cn == 'same':
                ctx_tid = tid_map[words[same_ctx(i)][2]]
            else:
                ctx_tid = func_tid
            per_word[i] = forward2([ctx_tid, w_tid])
        base_caps[cn] = per_word
    log('base captures done')

    RTYPE_DIRS = ('stale', 'eigen', 'orth')
    results = {rt: [] for rt in RTYPE_DIRS}
    v1_by_draw = {}
    B_store = {rt: {} for rt in RTYPE_DIRS}
    for s in SEEDS:
        td = time.monotonic()
        null_tids = draw_null_tids(s)
        g = {rt: {cn: np.zeros((n_words, n_win)) for cn in
                  ('same', 'func', 'null')} for rt in RTYPE_DIRS}
        v1_err = 0.0
        hook_noise = 0.0
        for i, (_, _, w) in enumerate(words):
            w_tid = tid_map[w]
            conds = {'same': base_caps['same'][i],
                     'func': base_caps['func'][i],
                     'null': forward2([null_tids[i], w_tid])}
            for cn, (mlpin, mlpout) in conds.items():
                for q, li in enumerate(range(WIN_LO, WIN_HI)):
                    x0 = mlpin[li]          # [1, seq, d]
                    ref = mlp_call(li, x0)  # pos-1 output
                    if cn == 'same' and q == 0:
                        ref2 = mlp_call(li, x0)
                        den = max(float(np.linalg.norm(ref)), 1e-30)
                        v1_err = max(v1_err, float(np.linalg.norm(
                            ref - ref2)) / den)
                        hr = mlpout[li][0, 1]
                        hook_noise = max(hook_noise, float(
                            np.linalg.norm(ref - hr))
                            / max(float(np.linalg.norm(hr)), 1e-30))
                    for rt in RTYPE_DIRS:
                        d_rt = d18 if rt == 'stale' else \
                            (dirs[li] if rt == 'eigen'
                             else orth_dirs[li])
                        xp = x0.copy()
                        xp[0, 1] = xp[0, 1] + EPS * d_rt
                        g[rt][cn][i, q] = float(
                            (mlp_call(li, xp) - ref) @ d_rt) / EPS
        v1_by_draw[s] = {'recompute': float('%.3e' % v1_err),
                         'hook': float('%.3e' % hook_noise)}
        if v1_err >= 1e-6 or hook_noise >= 1e-4:
            log('draw seed=%d v1=%s VOID' % (s, v1_by_draw[s]))
            continue
        for rt in RTYPE_DIRS:
            B = g[rt]['same'] - 0.5 * (g[rt]['func'] + g[rt]['null'])
            B_store[rt][s] = B.astype(np.float32)
            st = stats_of(B, s)
            results[rt].append(dict(st, seed=s))
            log('seed=%d %s: acc=%.4f p95=%.4f flag=%s margin=%.4f '
                'mp95=%.4f mflag=%s (%.0fs)'
                % (s, rt, st['acc'], st['p95'], st['flag_acc'],
                   st['margin'], st['margin_p95'],
                   st['flag_margin'], time.monotonic() - td))

    n_valid = len(results['stale'])
    rates = {}
    for rt in RTYPE_DIRS:
        rs = results[rt]
        rates[rt] = {
            'n_valid': n_valid,
            'acc_flag_rate': round(
                sum(r['flag_acc'] for r in rs) / max(n_valid, 1), 4),
            'margin_flag_rate': round(
                sum(r['flag_margin'] for r in rs) / max(n_valid, 1),
                4),
            'acc_values': [r['acc'] for r in rs],
            'margin_values': [r['margin'] for r in rs]}

    eig_mrate = rates['eigen']['margin_flag_rate']
    v2 = bool(n_valid >= 6 and eig_mrate >= 0.8)
    log('v2 anchor gate: eigen margin_flag_rate=%s n_valid=%d '
        'pass=%s' % (eig_mrate, n_valid, v2))

    def robust(rt):
        return rates[rt]['margin_flag_rate'] >= 0.8 and \
            rates[rt]['acc_flag_rate'] >= 0.5

    if not v2:
        verdict = 'all_void_procedure_unstable'
    elif all(robust(rt) for rt in RTYPE_DIRS):
        verdict = 'subspace_readout_tolerant_robust'
    elif rates['stale']['margin_flag_rate'] == 0.0:
        verdict = 'subspace_tolerant_stale_fragile'
    else:
        verdict = 'subspace_tolerant_mixed'

    res = {
        'phase': 2899, 'model': 'qwen3-4b', 'prereg': PREREG,
        'window': [WIN_LO, WIN_HI],
        'v1_by_draw': v1_by_draw, 'v2_anchor_gate': v2,
        'rates': rates, 'per_draw': results,
        'final_verdict': verdict,
        'runtime_s': round(time.monotonic() - t0, 1),
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    np.savez_compressed(
        os.path.join(OUT, 'qwen_readout_robustness.npz'),
        **{('B_%s_s%d' % (rt, s)): B_store[rt][s]
           for rt in RTYPE_DIRS for s in B_store[rt]},
        labels_lang=lab_lang, labels_concept=lab_concept,
        words=np.array(['%s:%s:%s' % w for w in words], dtype=object))
    log('==== VERDICT: %s ====' % verdict)
    log('runtime %.1fs' % (time.monotonic() - t0))


if __name__ == '__main__':
    main()
