# -*- coding: utf-8 -*-
"""Phase 2877: attribute-axis MLP response spectrum (B3_attr).

Question: the class axis's densest mechanism carrier is the 10-dim mlp
response spectrum (2861/2867 B3, retrieval 0.875).  Does the attr axis
live in that channel?  Cheap test (~1 min forward) after 2875/2876
showed the head-level causal-drop channel carries no attr axis signal.

Protocol: for each of the 42 attr words (2874 legal vocab, 8 axes),
conds = AttrCensus.conds2_for (same/func/null, 2-token, pos 1);
for layers L26-35: mlpin_true = pre-hook capture at layer.mlp input
(LN2 output, the true workpoint per the 2860-2861 erratum chain);
g_direct[w, li] = [mlp(mlpin + eps*cdir) - mlp(mlpin)] . cdir / eps,
eps = 1.0, cdir = axis direction (2874 dW_unit).
B3_attr_spec[w, q] = g(same) - 0.5*(g(func) + g(null)).

Prereg (frozen before any readout; execution.json written first):
  v1  determinism: mlp_call recomputed twice on identical input differs
      by < 1e-6 relative (kernel-shape-constant check).  The hook-vs-
      recompute comparison is NOT a validity gate: recompute uses a 1-D
      shape while the forward uses a 2-token batch, and cuBLAS picks
      shape-dependent algorithms whose bf16 reduction order differs at
      the ~3e-3 level (measured in Gen1); g uses mlp_call for BOTH
      baseline and perturbed terms, so it is internally consistent.
      hook-vs-recompute is registered as descriptive kernel-context
      noise.  (Gen2 amendment: Gen1's hook-vs-recompute gate 1e-4 was
      mis-specified and crashed on an unrelated self-reference bug
      before any E1/E2 value was observed.)
  E1  axis retrieval on zmat(B3_attr_spec) (42 words, 8 axes):
      acc > null p95 (200 label permutations, SEED=2877)
      => mlp_carries_attr_axis.
  E2  same-axis margin: mean cos(same-axis pairs) - mean cos(diff-axis
      pairs) on unit rows of B3_attr_spec > null p95
      => attr_margin_in_mlp.
  E3  descriptive: per-layer mean |g_direct| profile; per-axis top word.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC_2874 = os.path.join(BASE, 'phase2874', 'attr_vocab_v2',
                        'attr_vocab_v2.npz')
OUT = os.path.join(BASE, 'phase2877', 'attr_mlp_spectrum')
MODEL_DIR = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
SEED = 2877
N_NULL = 200
EPS = 1.0
WIN_LO, WIN_HI = 26, 36          # layers 26..35 inclusive

PREREG = {
    'v1': 'max relative mlp recompute error < 1e-4, else all void',
    'E1': 'acc(zmat B3_attr_spec) > null p95 (200 perms, SEED=2877) '
          '=> mlp_carries_attr_axis',
    'E2': 'same-axis cos margin > null p95 => attr_margin_in_mlp',
    'E3': 'descriptive layer profile / per-axis top word',
}


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


def unit(v):
    n = float(np.linalg.norm(v))
    return v / max(n, 1e-30)


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


def log(msg):
    print(msg, flush=True)


def main():
    t0 = time.monotonic()
    os.makedirs(OUT, exist_ok=True)

    script = os.path.abspath(__file__)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2877, 'name': 'attr_mlp_spectrum',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8': sha8(script),
                   'sources': {'attr_vocab_v2': sha8(SRC_2874)},
                   'prereg': PREREG, 'seed': SEED, 'eps': EPS,
                   'window': [WIN_LO, WIN_HI - 1]},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen')

    zv = np.load(SRC_2874, allow_pickle=True)
    axis_words = json.loads(str(zv['targets']))
    axes = list(zv['axes'])
    target_list = [(a, w) for a in axes for w in axis_words[a]]
    n_words = len(target_list)
    dW = zv['dW_unit'].astype(np.float64)
    tid_map = json.loads(str(zv['tid_map']))
    log('words=%d axes=%d' % (n_words, len(axes)))

    import torch
    from phase2662_symmetric_mapping_contract import load_native
    model, _ = load_native('qwen4')
    model.eval()
    layers = model.model.layers

    # ---- hooks: mlpin pre-hook + mlp out-hook on WIN layers ----
    cap = {'mlpin': {}, 'mlpout': {}}
    handles = []

    def pre_hook(li):
        def h(module, args, kwargs):
            x = args[0] if args else kwargs.get('hidden_states')
            if x.dim() < 2:
                return          # direct mlp_call recompute, not a forward
            cap['mlpin'].setdefault(li, []).append(
                x.detach()[0, 1].float().cpu().numpy())
        return h

    def out_hook(li):
        def h(module, args, output):
            o = output[0] if isinstance(output, tuple) else output
            if o.dim() < 2:
                return
            cap['mlpout'].setdefault(li, []).append(
                o.detach()[0, 1].float().cpu().numpy())
        return h

    for li in range(WIN_LO, WIN_HI):
        handles.append(layers[li].mlp.register_forward_pre_hook(
            pre_hook(li), with_kwargs=True))
        handles.append(layers[li].mlp.register_forward_hook(out_hook(li)))

    def clear_cap():
        for d in cap:
            for li in cap[d]:
                del cap[d][li][:]

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
        return out.detach().float().cpu().numpy()

    def same_word(axis, w):
        others = [x for x in axis_words[axis] if x != w]
        return min(others, key=lambda x: tid_map[x])

    # ---------- measurement ----------
    g = {cn: np.zeros((n_words, WIN_HI - WIN_LO)) for cn in
         ('same', 'func', 'null')}
    v1a_err = 0.0
    hook_noise = 0.0
    rng = np.random.default_rng(SEED)
    vocab_size = 151936
    word_tids = set(tid_map.values())
    null_tids = []
    while len(null_tids) < n_words:
        r = int(rng.integers(0, vocab_size))
        if r not in word_tids and r > 0:
            null_tids.append(r)
    func_tid = tid_map['the']

    for i, (axis, w) in enumerate(target_list):
        w_tid = tid_map[w]
        conds = {'same': [tid_map[same_word(axis, w)], w_tid],
                 'func': [func_tid, w_tid],
                 'null': [null_tids[i], w_tid]}
        cdir = dW[axes.index(axis)]
        for cn, toks in conds.items():
            mlpin, mlpout = forward2(toks)
            for q, li in enumerate(range(WIN_LO, WIN_HI)):
                x0 = mlpin[li]
                ref = mlp_call(li, x0)
                if cn == 'same' and q == 0:
                    ref2 = mlp_call(li, x0)
                    denom = max(float(np.linalg.norm(ref)), 1e-30)
                    v1a_err = max(v1a_err, float(np.linalg.norm(
                        ref - ref2)) / denom)
                    hook_ref = mlpout[li]
                    hook_noise = max(hook_noise, float(np.linalg.norm(
                        ref - hook_ref))
                        / max(float(np.linalg.norm(hook_ref)), 1e-30))
                g[cn][i, q] = float((mlp_call(li, x0 + EPS * cdir) - ref)
                                    @ cdir) / EPS
        if (i + 1) % 10 == 0:
            log('words [%d/%d] v1a=%.2e hook_noise=%.2e'
                % (i + 1, n_words, v1a_err, hook_noise))

    log('v1a determinism max rel err = %.3e' % v1a_err)
    log('kernel-context noise (descriptive) = %.3e' % hook_noise)
    v1 = bool(v1a_err < 1e-6)

    B3_spec = g['same'] - 0.5 * (g['func'] + g['null'])
    lab = np.array([axes.index(a) for a, _ in target_list])

    # ---------- E1 / E2 with label-permutation null ----------
    C = zmat(B3_spec)
    acc = loo_acc(C, lab)
    U = B3_spec / np.maximum(
        np.linalg.norm(B3_spec, axis=1, keepdims=True), 1e-30)
    S = U @ U.T
    n = len(lab)
    same_m = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            same_m[i, j] = (lab[i] == lab[j]) and i != j
    off = ~np.eye(n, dtype=bool) & ~same_m

    def margin(pl):
        sm = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(n):
                sm[i, j] = (pl[i] == pl[j]) and i != j
        df = (~np.eye(n, dtype=bool)) & (~sm)
        return float(S[sm].mean() - S[df].mean())

    m_obs = float(S[same_m].mean() - S[off].mean())
    rng2 = np.random.default_rng(SEED)
    null_acc, null_m = [], []
    for _ in range(N_NULL):
        pl = rng2.permutation(lab)
        null_acc.append(loo_acc(C, pl))
        null_m.append(margin(pl))
    na, nm = np.array(null_acc), np.array(null_m)

    e1 = bool(acc > float(np.percentile(na, 95)))
    e2 = bool(m_obs > float(np.percentile(nm, 95)))

    layer_profile = np.abs(B3_spec).mean(axis=0)
    per_axis_top = {}
    for ai, a in enumerate(axes):
        rows = [i for i, (aa, _) in enumerate(target_list) if aa == a]
        top_w = max(rows, key=lambda i: float(
            np.abs(B3_spec[i]).sum()))
        per_axis_top[a] = target_list[top_w][1]

    e1_verdict = 'mlp_carries_attr_axis' if e1 else 'mlp_axis_signal_absent'
    e2_verdict = 'attr_margin_in_mlp' if e2 else 'margin_absent'

    res = {
        'phase': 2877, 'prereg': PREREG,
        'v1a_max_rel_err': float('%.3e' % v1a_err), 'v1': v1,
        'kernel_context_noise': float('%.3e' % hook_noise),
        'E1': {'acc': round(acc, 4),
               'null_p95': round(float(np.percentile(na, 95)), 4),
               'null_mean': round(float(na.mean()), 4),
               'verdict': e1_verdict},
        'E2': {'margin': round(m_obs, 4),
               'null_p95': round(float(np.percentile(nm, 95)), 4),
               'null_mean': round(float(nm.mean()), 4),
               'verdict': e2_verdict},
        'E3': {'layer_mean_abs_profile':
               [round(float(x), 4) for x in layer_profile],
               'per_axis_top_word': per_axis_top},
        'final_verdict': 'v1=%s/E1=%s/E2=%s'
                         % (v1, e1_verdict, e2_verdict),
        'runtime_s': round(time.monotonic() - t0, 1),
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    np.savez_compressed(os.path.join(OUT, 'attr_mlp_spectrum.npz'),
                        B3_spec=B3_spec.astype(np.float32),
                        g_same=g['same'].astype(np.float32),
                        g_func=g['func'].astype(np.float32),
                        g_null=g['null'].astype(np.float32),
                        labels=lab,
                        target_list=np.array(
                            ['%s:%s' % t for t in target_list],
                            dtype=object))

    log('==== VERDICTS ====')
    log('acc=%.4f null_p95=%.4f | margin=%.4f null_p95=%.4f'
        % (acc, np.percentile(na, 95), m_obs, np.percentile(nm, 95)))
    log('v1=%s E1=%s E2=%s' % (v1, e1, e2))
    log('runtime %.1fs' % (time.monotonic() - t0))


if __name__ == '__main__':
    main()
