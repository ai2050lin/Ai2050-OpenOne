# -*- coding: utf-8 -*-
"""Phase 2896: qwen3-4b channel readout discrimination (zero-forward
directions + single-window forward).

2895 (L13): rotation dynamics are model-universal; the carrier
asymmetry (qwen mlp carries language 0.7719 with the STALE li=18
direction; glm4 needs layer-matched directions) must be a CHANNEL
READOUT property.  This phase discriminates qwen's readout type
directly with three injection conditions in window W=[26,36)
(2887 verbatim protocol, 57 words/conds from 2887 npz, zero
forward reuse):

  d18        = dir(li=18)  (2887 lang_dir, the stale direction)
  d(li)      = dir(li)     (per-layer eigen-direction, 2893 analog)
  d_orth(li) = unit(d(li) - (d(li).d18) d18)  (rotation-orthogonal
               component; cos(d(li),d18) = 0.456 -> 0.068 in window,
               2895)

  For each layer li in W and word-condition (same/func/null
  verbatim 2887):
    g_stale[i,q] = [(mlp(x0+eps*d18) - ref) . d18]/eps
    g_eigen[i,q] = [(mlp(x0+eps*d(li)) - ref) . d(li)]/eps
    g_orth[i,q]  = [(mlp(x0+eps*d_orth(li)) - ref) . d_orth(li)]/eps
    B_r = g_r(same) - 0.5(g_r(func)+g_r(null)) for r in
    {stale, eigen, orth}.

  v1: mlp recompute rel err < 1e-6 + hook-vs-call noise < 1e-4
  (2887 precedent), else all void.
  A1: acc(loo-NN zmat B_stale, lab_lang) > null p95 (200 perms,
      SEED=2896) => stale_fixed_direction_signal
  A2: same for B_eigen => eigen_direction_signal
  A3: same for B_orth => orthogonal_component_signal
  Each with concept-label control (expect ~null).

  Verdict (frozen):
    A1 & A3        => subspace_readout_tolerant
    A1 only        => fixed_direction_readout_only
    A2 or A3 without A1 => direction_matched_readout_qwen
    none           => readout_absent_window (conflict with 2887,
                      per-layer investigation required)

  Cross-model bridge (descriptive): qwen A1 vs glm4 M2890 mlp
  negative with the same stale-direction logic; qwen A2/A3 vs
  glm4 M2893 positives.

SEED=2896.  Output: phase2896/qwen_readout_type/.
"""
import hashlib
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
OUT = os.path.join(BASE, 'phase2896', 'qwen_readout_type')
sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')

SEED = 2896
N_NULL = 200
EPS = 1.0
WIN_LO, WIN_HI = 26, 36
VOCAB = 151936

PREREG = {
    'window': 'W=[26,36) frozen (qwen 2884 precedent)',
    'sources': 'dirs zero-forward from 2886 S_last (labels i%2 '
               'asserted); words/labels verbatim 2887 npz (57 '
               'words, tid in word string); conds same/func(the)/'
               'null verbatim 2887 protocol',
    'dirs': 'd18=dir(li=18); d(li)=dir(li); d_orth(li)=unit(d(li)-'
            '(d(li).d18)d18); cos(d(li),d18) recorded',
    'injection': 'three per-layer injections at pos 1: stale(d18, '
                 'proj d18), eigen(d(li), proj d(li)), orth(d_orth, '
                 'proj d_orth); eps=1.0; B = g(same)-0.5(g(func)+'
                 'g(null)) per response type',
    'v1': 'mlp recompute rel err < 1e-6 and hook-vs-call noise '
          '< 1e-4, else all void',
    'amendment': 'run 1 void per frozen v1 gate (hook-vs-call '
                 '1.6e-3 > 1e-4): root cause = shape-mismatched '
                 'hook check (1-D [2560] recompute vs [1,2,2560] '
                 'real forward -> bf16 cross-shape kernel err ~2e-3, '
                 'the 2877 lesson; 2887 gate was recompute-only, '
                 'hook citation was a mis-read precedent). Fix: '
                 'shape-matched capture [1,2,2560] and recompute at '
                 'the same shape (2890-2893 convention, v1=0.0 '
                 'there); old execution.json/result.json/npz '
                 'deleted; no verdict from run 1 was used',
    'A1': 'acc(B_stale) > null p95 (200 perms, SEED=2896) => '
          'stale_fixed_direction_signal',
    'A2': 'acc(B_eigen) > null p95 => eigen_direction_signal',
    'A3': 'acc(B_orth) > null p95 => orthogonal_component_signal',
    'A4': 'concept-label control acc per response type (expect '
          '~null)',
    'verdict': 'A1&A3 => subspace_readout_tolerant; A1 only => '
               'fixed_direction_readout_only; A2|A3 without A1 => '
               'direction_matched_readout_qwen; none => '
               'readout_absent_window (conflict with 2887, '
               'per-layer investigation required)',
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
        json.dump({'phase': 2896, 'name': 'qwen_readout_type',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'sources': {'s_last_2886': sha8(SRC_2886),
                               'vocab_2887': sha8(SRC_2887)},
                   'model': 'qwen3-4b',
                   'prereg': PREREG, 'seed': SEED, 'eps': EPS,
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

    # ---------- model + hooks (2887 verbatim) ----------
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
            if x.dim() < 2:
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
    g = {rt: {cn: np.zeros((n_words, n_win)) for cn in
              ('same', 'func', 'null')}
         for rt in ('stale', 'eigen', 'orth')}
    v1_err = 0.0
    hook_noise = 0.0
    rng = np.random.default_rng(SEED)
    word_tids = set(tid_map.values())
    null_tids = []
    while len(null_tids) < n_words:
        r = int(rng.integers(0, VOCAB))
        if r not in word_tids and r > 0:
            null_tids.append(r)
    func_tid = tid_map['the']

    for i, (_, _, w) in enumerate(words):
        w_tid = tid_map[w]
        conds = {'same': [tid_map[words[same_ctx(i)][2]], w_tid],
                 'func': [func_tid, w_tid],
                 'null': [null_tids[i], w_tid]}
        for cn, toks in conds.items():
            mlpin, mlpout = forward2(toks)
            for q, li in enumerate(range(WIN_LO, WIN_HI)):
                x0 = mlpin[li]              # [1, seq, d]
                ref = mlp_call(li, x0)      # pos-1 output
                if cn == 'same' and q == 0:
                    ref2 = mlp_call(li, x0)
                    den = max(float(np.linalg.norm(ref)), 1e-30)
                    v1_err = max(v1_err, float(np.linalg.norm(
                        ref - ref2)) / den)
                    hr = mlpout[li][0, 1]
                    hook_noise = max(hook_noise, float(
                        np.linalg.norm(ref - hr))
                        / max(float(np.linalg.norm(hr)), 1e-30))
                xp = x0.copy()
                xp[0, 1] = xp[0, 1] + EPS * d18
                g['stale'][cn][i, q] = float(
                    (mlp_call(li, xp) - ref) @ d18) / EPS
                d_li = dirs[li]
                xp = x0.copy()
                xp[0, 1] = xp[0, 1] + EPS * d_li
                g['eigen'][cn][i, q] = float(
                    (mlp_call(li, xp) - ref) @ d_li) / EPS
                d_o = orth_dirs[li]
                xp = x0.copy()
                xp[0, 1] = xp[0, 1] + EPS * d_o
                g['orth'][cn][i, q] = float(
                    (mlp_call(li, xp) - ref) @ d_o) / EPS
        if (i + 1) % 10 == 0:
            log('words [%d/%d] v1=%.2e hook=%.2e'
                % (i + 1, n_words, v1_err, hook_noise))

    v1 = bool(v1_err < 1e-6 and hook_noise < 1e-4)
    log('v1: recompute=%.3e hook-vs-call=%.3e pass=%s'
        % (v1_err, hook_noise, v1))

    B = {rt: g[rt]['same'] - 0.5 * (g[rt]['func'] + g[rt]['null'])
         for rt in g}

    def margin_of(Sm, pl):
        n = len(pl)
        sm = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(n):
                sm[i, j] = (pl[i] == pl[j]) and i != j
        df = (~np.eye(n, dtype=bool)) & (~sm)
        return float(Sm[sm].mean() - Sm[df].mean())

    rng2 = np.random.default_rng(SEED)
    perms = [rng2.permutation(lab_lang) for _ in range(N_NULL)]
    out_c = {}
    for rt in ('stale', 'eigen', 'orth'):
        C = zmat(B[rt])
        acc = loo_acc(C, lab_lang)
        U = B[rt] / np.maximum(
            np.linalg.norm(B[rt], axis=1, keepdims=True), 1e-30)
        Sm = U @ U.T
        n = len(lab_lang)
        same_m = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(n):
                same_m[i, j] = (lab_lang[i] == lab_lang[j]) and i != j
        off = ~np.eye(n, dtype=bool) & ~same_m
        m_obs = float(Sm[same_m].mean() - Sm[off].mean())
        null_acc = [loo_acc(C, pl) for pl in perms]
        null_m = [margin_of(Sm, pl) for pl in perms]
        acc_p95 = float(np.percentile(null_acc, 95))
        m_p95 = float(np.percentile(null_m, 95))
        acc_c = loo_acc(C, lab_concept)
        per_layer = [round(loo_acc(zmat(B[rt][:, [q]]), lab_lang), 4)
                     for q in range(n_win)]
        out_c[rt] = {
            'acc': round(acc, 4), 'null_p95': round(acc_p95, 4),
            'null_mean': round(float(np.mean(null_acc)), 4),
            'margin': round(m_obs, 4), 'margin_p95': round(m_p95, 4),
            'concept_acc': round(acc_c, 4),
            'per_layer_acc_descriptive': per_layer,
            'layer_profile': [round(float(x), 4)
                              for x in np.abs(B[rt]).mean(axis=0)]}
        log('%s: acc=%.4f p95=%.4f margin=%.4f p95=%.4f concept=%.4f'
            % (rt, acc, acc_p95, m_obs, m_p95, acc_c))

    a1 = out_c['stale']['acc'] > out_c['stale']['null_p95']
    a2 = out_c['eigen']['acc'] > out_c['eigen']['null_p95']
    a3 = out_c['orth']['acc'] > out_c['orth']['null_p95']
    if a1 and a3:
        verdict = 'subspace_readout_tolerant'
    elif a1:
        verdict = 'fixed_direction_readout_only'
    elif a2 or a3:
        verdict = 'direction_matched_readout_qwen'
    else:
        verdict = 'readout_absent_window'

    res = {
        'phase': 2896, 'model': 'qwen3-4b', 'prereg': PREREG,
        'window': [WIN_LO, WIN_HI],
        'cos_dli_d18': {str(k): round(v, 4)
                        for k, v in cos18.items()},
        'v1': v1, 'v1_recompute_err': float('%.3e' % v1_err),
        'hook_vs_call_noise': float('%.3e' % hook_noise),
        'responses': out_c,
        'A_flags': {'A1_stale': bool(a1), 'A2_eigen': bool(a2),
                    'A3_orth': bool(a3)},
        'final_verdict': verdict,
        'runtime_s': round(time.monotonic() - t0, 1),
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    np.savez_compressed(
        os.path.join(OUT, 'qwen_readout_type.npz'),
        B_stale=B['stale'].astype(np.float32),
        B_eigen=B['eigen'].astype(np.float32),
        B_orth=B['orth'].astype(np.float32),
        cos_dli_d18=np.array([cos18[li] for li in
                              range(WIN_LO, WIN_HI)],
                             dtype=np.float32),
        labels_lang=lab_lang, labels_concept=lab_concept,
        words=np.array(['%s:%s:%s' % w for w in words], dtype=object))
    log('==== VERDICT: %s ====' % verdict)
    log('runtime %.1fs' % (time.monotonic() - t0))


if __name__ == '__main__':
    main()
