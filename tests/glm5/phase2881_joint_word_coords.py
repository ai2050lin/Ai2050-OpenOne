# -*- coding: utf-8 -*-
"""Phase 2881: three-family joint word coordinates (B3_joint).

Question (closure of the family-integration line): the mlp channel is
the densest mechanism carrier for all three measured axis families
(class 0.875 / attr 0.500 / syntax 0.7083, phases 2867/2877/2879; zero
new head components in every step).  Do the three families live in ONE
joint response space, or in three separate channel partitions?

Key semantics (2861/2867 confirmed): in the single-family phases each
word was projected on its OWN class/axis direction, 10 layers L26-35 =>
the 10-dim B3.  The joint spectrum here measures EVERY word under ALL
21 family directions (10 class centroid-difference directions built
exactly as 2861/2867 from the 80-word CATS vocab + 8 attr directions
from 2874 + 3 syntax directions from 2878), 10 layers each =>
B3_joint (190, 210) with column blocks class 100 / attr 80 / syntax 30
(direction-major, layer-minor).

Vocabulary: class = 2867 CATS verbatim (80 words, matches the 0.875
result); attr = 2874 SURVIVING axes (42 words); syntax = 2878
surviving axes (48 words).  Cross-family duplicate words are measured
once per family they belong to (same tid, different conds).

Prereg (frozen before any readout; execution.json written first):
  v1  determinism: mlp_call recomputed twice on identical input
      differs by < 1e-6 relative (2877 Gen2 recompute-determinism).
  J1  joint fine-label retrieval: LOO-NN acc on zmat(B3_joint) over 21
      fine labels (10 class + 8 attr + 3 syntax) > null p95
      (200 label permutations, SEED=2881)
      => joint_spectrum_carries_all_families.
  J2  density-gated fusion: block-zscored; own = the word's family
      block, cross = the other two blocks.  acc(alpha) for alpha in
      {0,...,1.0} on [own, alpha*cross] rows-renormalized;
      delta* = max acc - acc(0); verdict gated_fusion_generalizes iff
      delta* > perm-null p95 (200, same SEED) and alpha* > 0.
  J3  cross-family transfer: nearest-centroid classifier for family
      fb's fine labels built from family fa-direction block centroids
      over fb words, tested on fb words => 3x3 matrix (diagonal =
      own-family self acc reference).
  J4  descriptive: family centroid cosines; per-family own acc.
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
SRC_2874 = os.path.join(BASE, 'phase2874', 'attr_vocab_v2',
                        'attr_vocab_v2.npz')
SRC_2878 = os.path.join(BASE, 'phase2878', 'syntax_trans_vocab',
                        'syntax_trans_vocab.npz')
OUT = os.path.join(BASE, 'phase2881', 'joint_word_coords')
MODEL_DIR = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
SEED = 2881
N_NULL = 200
EPS = 1.0
WIN_LO, WIN_HI = 26, 36

PREREG = {
    'v1': 'max relative mlp recompute error < 1e-6, else all void',
    'J1': 'acc(zmat B3_joint, 21 fine labels) > null p95 (200 perms, '
          'SEED=2881) => joint_spectrum_carries_all_families',
    'J2': 'delta* = max_a acc(a) - acc(0) on [own, alpha*cross] vs '
          'perm null p95; gated_fusion_generalizes iff delta*>p95 and '
          'alpha*>0; alpha_zero_optimal iff argmax=0',
    'J3': 'cross-family centroid transfer 3x3 (descriptive, no gate)',
    'J4': 'descriptive family centroid cos / per-family own acc',
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


def load_embed(model_dir):
    idx_path = os.path.join(model_dir, 'model.safetensors.index.json')
    if os.path.exists(idx_path):
        idx = json.load(io.open(idx_path, encoding='utf-8'))
        shard = idx['weight_map']['model.embed_tokens.weight']
    else:
        shard = 'model.safetensors'
    from safetensors import safe_open
    with safe_open(os.path.join(model_dir, shard), framework='pt') as f:
        emb = f.get_tensor('model.embed_tokens.weight').float().numpy()
    return emb


def main():
    t0 = time.monotonic()
    os.makedirs(OUT, exist_ok=True)

    script = os.path.abspath(__file__)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2881, 'name': 'joint_word_coords',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8': sha8(script),
                   'sources': {'attr_vocab_v2_2874': sha8(SRC_2874),
                               'syntax_trans_vocab_2878':
                                   sha8(SRC_2878)},
                   'prereg': PREREG, 'seed': SEED, 'eps': EPS,
                   'window': [WIN_LO, WIN_HI - 1]},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen')

    # ---------- class vocab + directions (2867 verbatim logic) ----------
    exec2806 = json.load(io.open(
        os.path.join(BASE, 'phase2806', 'qwen4_hierarchy',
                     'execution.json'), encoding='utf-8'))
    CATS = exec2806['cats']
    CAT_WORDS = list(CATS.keys())
    MAX_WORDS = 8

    # tokenizer first, then vocab/directions
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        MODEL_DIR, local_files_only=True, trust_remote_code=True,
        use_fast=True)
    tc = {}

    def tid(t):
        if t not in tc:
            ids = tok(' ' + t, add_special_tokens=False)['input_ids']
            if len(ids) != 1:
                ids = tok(t, add_special_tokens=False)['input_ids']
            assert len(ids) == 1, '%s -> %s' % (t, ids)
            tc[t] = int(ids[0])
        return tc[t]

    import torch
    from phase2662_symmetric_mapping_contract import load_native
    model, _ = load_native('qwen4')
    model.eval()
    W_U = model.lm_head.weight.detach().float().cpu().numpy()

    all_words = [w for v in CATS.values() for w in v]
    single_tok = []
    for w in all_words:
        try:
            tid(w)
            single_tok.append(w)
        except AssertionError:
            pass
    class_targets = {}
    for cat in CAT_WORDS:
        class_targets[cat] = [w for w in CATS[cat]
                              if w in single_tok][:MAX_WORDS]
    assert sum(len(v) for v in class_targets.values()) == 80

    Erows = {w: W_U[tid(w)].astype(np.float64) for w in single_tok}
    cents = []
    for cat in CAT_WORDS:
        ws = [w for w in CATS[cat] if w in single_tok]
        cents.append(np.stack([Erows[w] for w in ws]).mean(0))
    Cm = np.stack(cents)
    dW_c = Cm - (Cm.sum(0, keepdims=True) - Cm) / 9.0
    dW_class = np.stack([unit(dW_c[i]) for i in range(10)])

    # ---------- attr + syntax vocab/directions ----------
    zv74 = np.load(SRC_2874, allow_pickle=True)
    aw74 = json.loads(str(zv74['targets']))
    axes74 = list(zv74['axes'])           # 8 surviving attr axes
    dW_attr = zv74['dW_unit'].astype(np.float64)
    tid74 = json.loads(str(zv74['tid_map']))
    zv78 = np.load(SRC_2878, allow_pickle=True)
    aw78 = json.loads(str(zv78['targets']))
    dW_syn = zv78['dW_unit'].astype(np.float64)
    tid78 = json.loads(str(zv78['tid_map']))

    tc.update(tid74)
    tc.update(tid78)

    words = []          # (family, fine_label, word)
    for c in CAT_WORDS:
        for w in class_targets[c]:
            words.append(('class', c, w))
    for a in axes74:
        for w in aw74[a]:
            words.append(('attr', a, w))
    for a in aw78:
        for w in aw78[a]:
            words.append(('syntax', a, w))
    n_words = len(words)
    n_dir = 10 + len(axes74) + len(aw78)
    fam_dirs = {'class': (0, 10), 'attr': (10, 10 + len(axes74)),
                'syntax': (10 + len(axes74), n_dir)}
    dW = np.zeros((n_dir, dW_class.shape[1]))
    dW[0:10] = dW_class
    dW[10:10 + dW_attr.shape[0]] = dW_attr
    dW[10 + dW_attr.shape[0]:] = dW_syn
    fam_of = {'class': 0, 'attr': 1, 'syntax': 2}
    log('words=%d (class %d / attr %d / syntax %d) dirs=%d'
        % (n_words, sum(1 for x in words if x[0] == 'class'),
           sum(1 for x in words if x[0] == 'attr'),
           sum(1 for x in words if x[0] == 'syntax'), n_dir))

    w_tids = [tid(w) for _, _, w in words]
    tid_map = dict(tc)

    rng = np.random.default_rng(SEED)
    word_tid_set = set(w_tids)
    null_tids = []
    vocab_size = int(W_U.shape[0])
    while len(null_tids) < n_words:
        r = int(rng.integers(0, vocab_size))
        if r not in word_tid_set and r > 0:
            null_tids.append(r)
    func_tid = tid('the')

    fine = sorted(set((f, l) for f, l, _ in words))
    fine_idx = {fl: i for i, fl in enumerate(fine)}
    lab_fine = np.array([fine_idx[(f, l)] for f, l, _ in words])
    lab_fam = np.array([fam_of[f] for f, _, _ in words])

    layers = model.model.layers

    cap = {'mlpin': {}}
    handles = []

    def pre_hook(li):
        def h(module, args, kwargs):
            x = args[0] if args else kwargs.get('hidden_states')
            if x.dim() < 2:
                return
            cap['mlpin'].setdefault(li, []).append(
                x.detach()[0, 1].float().cpu().numpy())
        return h

    for li in range(WIN_LO, WIN_HI):
        handles.append(layers[li].mlp.register_forward_pre_hook(
            pre_hook(li), with_kwargs=True))

    def clear_cap():
        for li in cap['mlpin']:
            del cap['mlpin'][li][:]

    def forward2(toks):
        clear_cap()
        with torch.no_grad():
            model(torch.tensor([toks], device='cuda'))
        return {li: cap['mlpin'][li][0] for li in cap['mlpin']}

    def mlp_call(li, x):
        out = layers[li].mlp(torch.tensor(
            x, device='cuda', dtype=torch.bfloat16))
        if isinstance(out, tuple):
            out = out[0]
        return out.detach().float().cpu().numpy()

    same_of = {}
    for i, (f, l, w) in enumerate(words):
        others = [words[j][2] for j in range(n_words)
                  if j != i and words[j][0] == f
                  and words[j][1] == l]
        assert others, 'fine label with single word: %s' % ((f, l),)
        same_of[i] = min(others, key=lambda x: tid_map[x])

    NL = WIN_HI - WIN_LO
    g = {cn: np.zeros((n_words, NL, n_dir)) for cn in
         ('same', 'func', 'null')}
    v1a_err = 0.0

    for i, (f, l, w) in enumerate(words):
        w_tid = w_tids[i]
        conds = {'same': [tid_map[same_of[i]], w_tid],
                 'func': [func_tid, w_tid],
                 'null': [null_tids[i], w_tid]}
        for cn, toks in conds.items():
            mlpin = forward2(toks)
            for q, li in enumerate(range(WIN_LO, WIN_HI)):
                x0 = mlpin[li]
                ref = mlp_call(li, x0)
                if cn == 'same' and q == 0:
                    ref2 = mlp_call(li, x0)
                    denom = max(float(np.linalg.norm(ref)), 1e-30)
                    v1a_err = max(v1a_err, float(np.linalg.norm(
                        ref - ref2)) / denom)
                for d in range(n_dir):
                    gp = mlp_call(li, x0 + EPS * dW[d])
                    g[cn][i, q, d] = float((gp - ref) @ dW[d]) / EPS
        if (i + 1) % 10 == 0:
            log('words [%d/%d] v1a=%.2e' % (i + 1, n_words, v1a_err))

    log('v1a determinism max rel err = %.3e' % v1a_err)
    v1 = bool(v1a_err < 1e-6)

    spec = g['same'] - 0.5 * (g['func'] + g['null'])
    # column layout: direction-major, layer-minor =>
    # col = d * NL + layer
    B3 = spec.reshape(n_words, NL * n_dir)     # (n, 210)
    blk_cols = {}
    for fam, (d0, d1) in fam_dirs.items():
        cols = [d * NL + q for d in range(d0, d1)
                for q in range(NL)]
        blk_cols[fam] = cols

    # ---------- J1 ----------
    C = zmat(B3)
    acc_joint = loo_acc(C, lab_fine)
    rng2 = np.random.default_rng(SEED)
    null_acc = []
    for _ in range(N_NULL):
        null_acc.append(loo_acc(C, rng2.permutation(lab_fine)))
    na = np.array(null_acc)
    j1 = bool(acc_joint > float(np.percentile(na, 95)))
    j1_label = 'joint_spectrum_carries_all_families' if j1 \
        else 'joint_spectrum_insufficient'

    # ---------- J2 ----------
    Zblk = np.zeros_like(B3)
    for fam, cols in blk_cols.items():
        Zblk[:, cols] = zmat(B3[:, cols])
    Own = np.zeros_like(B3)
    Cross = np.zeros_like(B3)
    for f in range(3):
        fam = ['class', 'attr', 'syntax'][f]
        cols = blk_cols[fam]
        m = lab_fam == f
        Own[np.ix_(m, cols)] = Zblk[np.ix_(m, cols)]
        Cross[np.ix_(~m, cols)] = Zblk[np.ix_(~m, cols)]
    # zero out own columns for other families (Own rows already only
    # have own cols); Cross rows only have foreign cols
    rn0 = np.maximum(np.linalg.norm(Own, axis=1, keepdims=True), 1e-30)
    acc0 = loo_acc(Own / rn0, lab_fine)
    alphas = [round(0.1 * k, 1) for k in range(11)]
    acc_curve = []
    for a in alphas:
        M = Own + a * Cross
        rn = np.maximum(np.linalg.norm(M, axis=1, keepdims=True),
                        1e-30)
        acc_curve.append(loo_acc(M / rn, lab_fine))
    best_i = int(np.argmax(acc_curve))
    alpha_star = alphas[best_i]
    delta_star = acc_curve[best_i] - acc0

    rng3 = np.random.default_rng(SEED)
    null_delta = []
    for _ in range(N_NULL):
        pl = rng3.permutation(lab_fine)
        a0 = loo_acc(Own / rn0, pl)
        best_d = -9.9
        for a in alphas:
            M = Own + a * Cross
            rn = np.maximum(np.linalg.norm(M, axis=1, keepdims=True),
                            1e-30)
            d = loo_acc(M / rn, pl) - a0
            best_d = max(best_d, d)
        null_delta.append(best_d)
    nd = np.array(null_delta)
    j2_gated = bool(delta_star > float(np.percentile(nd, 95))
                    and alpha_star > 0)
    j2_label = 'gated_fusion_generalizes' if j2_gated else (
        'alpha_zero_optimal' if best_i == 0 else 'fusion_no_gain')

    # ---------- J3 ----------
    fam_names = ['class', 'attr', 'syntax']
    trans2 = np.zeros((3, 3))
    for fa in range(3):
        for fb in range(3):
            cands = sorted(set(lab_fine[lab_fam == fb].tolist()))
            if len(cands) < 2:
                trans2[fa, fb] = float('nan')
                continue
            cent_fb_fa = {}
            for fl in cands:
                rows = lab_fine == fl
                # centroid over fb words in fa block, per fine label
                cfa = Zblk[rows][:, blk_cols[fam_names[fa]]]
                cent_fb_fa[fl] = cfa.mean(0) / max(
                    float(np.linalg.norm(cfa.mean(0))), 1e-30)
            ok = tot = 0
            for i in range(n_words):
                if lab_fam[i] != fb:
                    continue
                x = Zblk[i, blk_cols[fam_names[fa]]]
                sims = {fl: float(x @ c) for fl, c in
                        cent_fb_fa.items()}
                if max(sims, key=sims.get) == lab_fine[i]:
                    ok += 1
                tot += 1
            trans2[fa, fb] = ok / max(tot, 1)

    # ---------- J4 ----------
    Cfull = zmat(B3)
    fam_cent = []
    for f in range(3):
        c = Cfull[lab_fam == f].mean(axis=0)
        fam_cent.append(c / max(float(np.linalg.norm(c)), 1e-30))
    fam_cos = [[round(float(fam_cent[a] @ fam_cent[b]), 4)
                for b in range(3)] for a in range(3)]
    per_fam_acc = {}
    for f in range(3):
        fam = fam_names[f]
        rows = lab_fam == f
        per_fam_acc[fam] = round(loo_acc(
            zmat(B3[rows][:, blk_cols[fam]]), lab_fine[rows]), 4)

    res = {
        'phase': 2881, 'prereg': PREREG,
        'v1a_max_rel_err': float('%.3e' % v1a_err), 'v1': v1,
        'n_words': n_words, 'n_dirs': n_dir,
        'family_word_counts': {fam: int((lab_fam == f).sum())
                               for f, fam in enumerate(fam_names)},
        'J1': {'acc_joint': round(acc_joint, 4),
               'null_p95': round(float(np.percentile(na, 95)), 4),
               'null_mean': round(float(na.mean()), 4),
               'verdict': j1_label},
        'J2': {'own_only_acc': round(acc0, 4),
               'acc_curve': dict(zip(map(str, alphas),
                                     [round(a, 4) for a in
                                      acc_curve])),
               'alpha_star': alpha_star,
               'delta_star': round(delta_star, 4),
               'delta_null_p95': round(float(np.percentile(nd, 95)),
                                       4),
               'verdict': j2_label},
        'J3': {'transfer_matrix_rows=fa_dirs_cols=fb_words':
               {fam_names[a]: {fam_names[b]:
                               (None if np.isnan(trans2[a, b])
                                else round(float(trans2[a, b]), 4))
                               for b in range(3)} for a in range(3)}},
        'J4': {'family_centroid_cos': fam_cos,
               'per_family_own_acc': per_fam_acc},
        'final_verdict': 'v1=%s/J1=%s(%s)/J2=%s(alpha*=%s,d*=%.3f)'
                         % (v1, j1, j1_label, j2_label, alpha_star,
                            delta_star),
        'runtime_s': round(time.monotonic() - t0, 1),
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    np.savez_compressed(os.path.join(OUT, 'joint_word_coords.npz'),
                        B3_joint=B3.astype(np.float32),
                        labels_fine=lab_fine,
                        labels_fam=lab_fam,
                        target_list=np.array(
                            ['%s:%s:%s' % t for t in words],
                            dtype=object))

    log('==== VERDICTS ====')
    log('J1 acc=%.4f null_p95=%.4f | J2 own=%.4f alpha*=%s d*=%.4f '
        'null_p95=%.4f' % (acc_joint, np.percentile(na, 95), acc0,
                           alpha_star, delta_star,
                           np.percentile(nd, 95)))
    log('J3 transfer=%s' % json.dumps(res['J3']))
    log('J4 fam_cos=%s per_fam=%s' % (fam_cos, per_fam_acc))
    log('v1=%s J1=%s J2=%s' % (v1, j1, j2_label))
    log('runtime %.1fs' % (time.monotonic() - t0))


if __name__ == '__main__':
    main()
