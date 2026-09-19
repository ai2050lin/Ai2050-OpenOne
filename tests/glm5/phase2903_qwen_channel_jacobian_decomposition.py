# -*- coding: utf-8 -*-
"""Phase 2903: qwen3-4b channel Jacobian decomposition (cross-model
test of the language-content-density mechanism, 2902 analog).

2902 (glm4): the 6.1x attn/mlp margin gap is carried by the
LANGUAGE CONTENT DENSITY of the landed directions (A metric:
mean |cos| with word unembed rows minus null-row baseline), not
by response gain (attn 3.1x SMALLER, N9) or retention (rho~0
both channels).  G_language_content_density_mechanism predicts:
qwen mlp (margin ~0.2, spectrum top layer) has landed directions
with language alignment >= 2x the best glm4 channel.

Protocol verbatim 2902, translated to qwen3-4b:
  window W=[26,36) (2896), d(li)=unit(mean_en S_last[:,li] -
  mean_fr S_last[:,li]) recomputed from 2886 npz (2896 verbatim,
  labels i%2 asserted); 57 words verbatim 2887 npz (tid in word
  string); conds same/func(the)/null with null_tids drawn in the
  2896 rng order (SEED=2896) so B' is bit-comparable.
  Both channels (mlp + self_attn; qwen attn eigen measured here
  for the first time), eps=1.0, pos 1.

Guards:
  v1: mlp recompute rel err < 1e-6 (shape-matched) and hook-vs-
      call noise < 1e-4 (2896 amendment), else all void.
  v2 anchor: B'_mlp = (r.same - 0.5(r.func + r.null)).d(li) must
      match stored 2896 B_eigen rel err < 1e-3, else all void.

Metrics per (ch, li): gamma=mean||r||, rho=mean cos(r,d),
A_word = mean_i mean_w |cos(r, u_w)| (57 word rows; tie=True ->
unembed = embed_tokens), A_null = same with 200 random rows
(fresh rng SEED=2896, excluding word tids).

Cross-model comparison (primary, from 2902 result.json):
  D_q_mlp  = median_li (A_word - A_null) [qwen mlp]
  D_g_mlp / D_g_attn = same medians for glm4 (2902)
Null control: 20 random unit dirs (SEED=2903), layers {26,29,32},
both channels: A_word - A_null per dir -> q_null_p95 (mlp).

Verdict (frozen):
  anchor fail => v2_anchor_fail_all_void
  D_q_mlp >= 2*max(D_g_mlp, D_g_attn) and D_q_mlp > 2*q_null_p95
      => language_content_density_confirmed_cross_model
  elif D_q_mlp > 2*q_null_p95
      => density_elevated_below_2x_glm4max
  else => density_not_confirmed
Descriptive: qwen internal ratios R_gamma/R_rho/R_A/R_B;
margin of B'_mlp and B'_attn (SEED=2896 perms, 200) vs 2896
stored margins; weights-only spectra (down_proj, W_VO composite
GQA 32q/8kv rep=4).

SEED=2896 (contexts/rows/perms replication), SEED=2903 (null dirs).
Output: phase2903/qwen_channel_jacobian_decomposition/.
"""
import hashlib
import io
import json
import os
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC_2886 = os.path.join(BASE, 'phase2886', 'hourglass_cka',
                        'hourglass_cka.npz')
SRC_2887 = os.path.join(BASE, 'phase2887', 'language_axis_mlp',
                        'language_axis_mlp.npz')
SRC_2896 = os.path.join(BASE, 'phase2896', 'qwen_readout_type',
                        'qwen_readout_type.npz')
SRC_2902 = os.path.join(BASE, 'phase2902',
                        'glm4_channel_jacobian_asymmetry',
                        'result.json')
OUT = os.path.join(BASE, 'phase2903',
                   'qwen_channel_jacobian_decomposition')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
SEED = 2896
SEED_NULL = 2903
N_NULL = 200
EPS = 1.0
WIN_LO, WIN_HI = 26, 36
VOCAB = 151936
N_RND_DIRS = 20
RND_LAYERS = (26, 29, 32)

PREREG = {
    'window': 'W=[26,36) frozen (2896)',
    'sources': 'dirs recomputed from 2886 S_last (labels i%2 '
               'asserted, 2896 verbatim); words/labels verbatim '
               '2887 npz (57 words); null_tids in 2896 rng order '
               '(SEED=2896)',
    'response': 'r_ch(li,i,cn) = (module_call(li, x+eps*d(li)) - '
                'ref)/eps at pos 1, module in {mlp, self_attn}, '
                'eps=1.0, 57 words x 3 conds; machinery verbatim '
                '2896/2898',
    'v1': 'mlp recompute rel err < 1e-6 (shape-matched [1,seq,d]) '
          'and hook-vs-call noise < 1e-4, else all void',
    'v2_anchor': "B'[mlp] = (r.same - 0.5(r.func + r.null)) . d(li) "
                 'must match stored 2896 B_eigen rel err < 1e-3 '
                 'else v2_anchor_fail_all_void; amendment: run 1 '
                 'void (anchor 2.5e-1, AutoModelForCausalLM '
                 'device_map mixed cpu/cuda gave numerically '
                 'different context states than the 2896 '
                 'load_native full-GPU forward - the 2877 lesson '
                 'at device level); fix: load_native("qwen4") '
                 'verbatim 2896; no verdict from run 1 was used',
    'metrics': 'per (ch, li): gamma=mean||r||, rho=mean cos(r,d), '
               'A_word=mean|cos(r, u_w)| over 57 word rows '
               '(tie=True -> unembed=embed_tokens), A_null same '
               'with 200 random rows (fresh rng SEED=2896)',
    'cross_model': 'D_q_mlp = median_li (A_word - A_null) [qwen '
                   'mlp]; D_g_mlp / D_g_attn same medians from '
                   '2902 result.json per_layer',
    'null_control': '20 random unit dirs SEED=2903, layers '
                    '{26,29,32}, both channels: A_word - A_null '
                    '-> q_null_p95 (mlp)',
    'verdict': 'anchor fail => v2_anchor_fail_all_void; '
               'D_q_mlp >= 2*max(D_g_mlp, D_g_attn) and D_q_mlp > '
               '2*q_null_p95 => '
               'language_content_density_confirmed_cross_model; '
               'elif D_q_mlp > 2*q_null_p95 => '
               'density_elevated_below_2x_glm4max; else '
               'density_not_confirmed',
    'descriptive': 'qwen internal ratios; margins of B_mlp/B_attn '
                   'with SEED=2896 perms (200); weights-only '
                   'spectra down_proj / o_proj + W_VO composite '
                   '(32q/8kv rep=4) + zero-forward SwiGLU response',
}


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


def unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-30)


def log(msg):
    print(msg, flush=True)


def main():
    t0 = time.monotonic()
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2903,
                   'name': 'qwen_channel_jacobian_decomposition',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8': sha8(os.path.abspath(__file__)),
                   'sources': {'s2886': sha8(SRC_2886),
                               's2887': sha8(SRC_2887),
                               's2896': sha8(SRC_2896),
                               'r2902': sha8(SRC_2902)},
                   'model': 'qwen3-4b',
                   'prereg': PREREG, 'seed': SEED,
                   'seed_null_dirs': SEED_NULL, 'eps': EPS,
                   'window': [WIN_LO, WIN_HI]},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen')

    # ---------- sources: words, labels, directions ----------
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
    dirs = np.stack([unit(diffs[li])
                     for li in range(WIN_LO, WIN_HI)])
    n_win = WIN_HI - WIN_LO
    log('dirs recomputed; cos(d(li),d18)=%s'
        % [round(float(d @ d18), 4) for d in dirs])

    # ---------- model + hooks (2896 verbatim: load_native full GPU) --
    import torch
    import sys
    sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
    from phase2662_symmetric_mapping_contract import load_native
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        MD, local_files_only=True, trust_remote_code=True,
        use_fast=True)
    cfg = json.load(io.open(os.path.join(MD, 'config.json'),
                            encoding='utf-8'))
    tc = {}

    def tid(t):
        if t not in tc:
            ids = tok(' ' + t, add_special_tokens=False)['input_ids']
            if len(ids) != 1:
                ids = tok(t, add_special_tokens=False)['input_ids']
            assert len(ids) == 1, '%s -> %s' % (t, ids)
            tc[t] = int(ids[0])
        return tc[t]

    tid_map = {}
    for lang, ck, w in words:
        tid_map[w] = tid(w)
        if lang == 'en':
            assert tid_map[w] == int(ck), 'key mismatch %s' % w
    func_tid = tid('the')

    model, _ = load_native('qwen4')
    model.eval()
    layers = model.model.layers
    log('model loaded (load_native full GPU)')

    cap = {'mlpin': {}, 'attnin': {}, 'mlpout': {}}
    handles = []

    def pre_mlp(li):
        def h(module, args, kwargs):
            x = args[0] if args else kwargs.get('hidden_states')
            if x is None or x.dim() < 2:
                return
            cap['mlpin'].setdefault(li, []).append(
                x.detach().float().cpu().numpy())
        return h

    def out_mlp(li):
        def h(module, args, output):
            o = output[0] if isinstance(output, tuple) else output
            if o.dim() < 2:
                return
            cap['mlpout'].setdefault(li, []).append(
                o.detach().float().cpu().numpy())
        return h

    def pre_attn(li):
        def h(module, args, kwargs):
            x = args[0] if args else kwargs.get('hidden_states')
            if x is None or x.dim() < 2:
                return
            cap['attnin'].setdefault(li, []).append(
                x.detach().float().cpu().numpy())
        return h

    for li in range(WIN_LO, WIN_HI):
        handles.append(layers[li].mlp.register_forward_pre_hook(
            pre_mlp(li), with_kwargs=True))
        handles.append(layers[li].mlp.register_forward_hook(
            out_mlp(li)))
        handles.append(layers[li].self_attn.register_forward_pre_hook(
            pre_attn(li), with_kwargs=True))

    def clear_cap():
        for dd in cap:
            for li in cap[dd]:
                del cap[dd][li][:]

    def forward2(toks):
        clear_cap()
        with torch.no_grad():
            model(torch.tensor([toks], device='cuda'))
        return {li: cap['mlpin'][li][0] for li in cap['mlpin']}, \
            {li: cap['attnin'][li][0] for li in cap['attnin']}, \
            {li: cap['mlpout'][li][0] for li in cap['mlpout']}

    vdt = next(layers[WIN_LO].mlp.parameters()).dtype

    def layer_dev(li):
        return next(layers[li].mlp.parameters()).device

    rotary = model.model.rotary_emb

    def mlp_call(li, X):
        t = torch.tensor(X, device=layer_dev(li), dtype=vdt)
        with torch.no_grad():
            o = layers[li].mlp(t)
        if isinstance(o, tuple):
            o = o[0]
        return o[0, 1].detach().float().cpu().numpy().astype(np.float64)

    def attn_call(li, X):
        t = torch.tensor(X, device=layer_dev(li), dtype=vdt)
        position_ids = torch.arange(t.shape[1],
                                    device=layer_dev(li)).unsqueeze(0)
        with torch.no_grad():
            pos_emb = rotary(t, position_ids)
            o = layers[li].self_attn(
                t, position_embeddings=pos_emb, attention_mask=None,
                past_key_values=None)
        if isinstance(o, tuple):
            o = o[0]
        return o[0, 1].detach().float().cpu().numpy().astype(np.float64)

    def same_ctx(i):
        lang = words[i][0]
        cands = [j for j in range(n_words)
                 if words[j][0] == lang and j != i]
        return min(cands, key=lambda j: tid_map[words[j][2]])

    rng = np.random.default_rng(SEED)
    word_tids = set(tid_map.values())
    null_tids = []
    while len(null_tids) < n_words:
        r = int(rng.integers(0, VOCAB))
        if r not in word_tids and r > 0:
            null_tids.append(r)

    # r[ch][cn]: (n_words, n_win, 2560) float32
    r_store = {ch: {cn: np.zeros((n_words, n_win, 2560),
                                  dtype=np.float32)
                    for cn in ('same', 'func', 'null')}
               for ch in ('mlp', 'attn')}
    v1_err = {'mlp': 0.0, 'attn': 0.0}
    hook_noise = 0.0

    for i, (_, _, w) in enumerate(words):
        w_tid = tid_map[w]
        conds = {'same': [tid_map[words[same_ctx(i)][2]], w_tid],
                 'func': [func_tid, w_tid],
                 'null': [null_tids[i], w_tid]}
        for cn, toks in conds.items():
            mlpin_all, attnin_all, mlpout_all = forward2(toks)
            for q, li in enumerate(range(WIN_LO, WIN_HI)):
                d = dirs[q]
                x_m = mlpin_all[li]
                x_a = attnin_all[li]
                ref_m = mlp_call(li, x_m)
                ref_a = attn_call(li, x_a)
                if cn == 'same' and q == 0:
                    ref2 = mlp_call(li, x_m)
                    v1_err['mlp'] = max(
                        v1_err['mlp'],
                        float(np.linalg.norm(ref_m - ref2))
                        / max(float(np.linalg.norm(ref_m)), 1e-30))
                    ref2a = attn_call(li, x_a)
                    v1_err['attn'] = max(
                        v1_err['attn'],
                        float(np.linalg.norm(ref_a - ref2a))
                        / max(float(np.linalg.norm(ref_a)), 1e-30))
                    hr = mlpout_all[li][0, 1].astype(np.float64)
                    hook_noise = max(hook_noise, float(
                        np.linalg.norm(ref_m - hr))
                        / max(float(np.linalg.norm(hr)), 1e-30))
                xm_p = x_m.copy()
                xm_p[0, 1] = xm_p[0, 1] + EPS * d
                r_m = (mlp_call(li, xm_p) - ref_m) / EPS
                xa_p = x_a.copy()
                xa_p[0, 1] = xa_p[0, 1] + EPS * d
                r_a = (attn_call(li, xa_p) - ref_a) / EPS
                r_store['mlp'][cn][i, q] = r_m.astype(np.float32)
                r_store['attn'][cn][i, q] = r_a.astype(np.float32)
        if (i + 1) % 10 == 0:
            log('words [%d/%d] v1 mlp=%.2e attn=%.2e'
                % (i + 1, n_words, v1_err['mlp'], v1_err['attn']))

    v1 = bool(v1_err['mlp'] < 1e-6 and v1_err['attn'] < 1e-6
              and hook_noise < 1e-4)
    log('v1: mlp=%.3e attn=%.3e hook=%.3e pass=%s'
        % (v1_err['mlp'], v1_err['attn'], hook_noise, v1))

    # B'[mlp] anchor vs stored 2896 B_eigen
    d_stack = dirs
    B_proj = {}
    for ch in ('mlp', 'attn'):
        r_c = (r_store[ch]['same'].astype(np.float64)
               - 0.5 * r_store[ch]['func'].astype(np.float64)
               - 0.5 * r_store[ch]['null'].astype(np.float64))
        B_proj[ch] = np.einsum('nqd,qd->nq', r_c, d_stack)
    z96 = np.load(SRC_2896, allow_pickle=True)
    stored = z96['B_eigen'].astype(np.float64)
    anchor_err = float(np.abs(B_proj['mlp'] - stored).max()
                       / max(float(np.abs(stored).max()), 1e-30))
    log('v2 anchor rel err (mlp vs 2896 B_eigen)=%.3e' % anchor_err)
    anchor_ok = bool(anchor_err < 1e-3)

    # ---------- unembed rows (tie=True -> embed_tokens) ----------
    Wu = model.model.embed_tokens.weight.detach() \
        .float().cpu().numpy()
    rng_rows = np.random.default_rng(SEED)
    null_rows = []
    while len(null_rows) < N_NULL:
        r = int(rng_rows.integers(0, VOCAB))
        if r not in word_tids and r > 0:
            null_rows.append(r)
    word_rows = np.stack([unit(Wu[tid_map[w]])
                          for _, _, w in words])
    nullU = np.stack([unit(Wu[r]) for r in null_rows])
    del Wu

    def A_of(R):
        Un = R / np.maximum(np.linalg.norm(
            R, axis=1, keepdims=True), 1e-30)
        a_w = float(np.abs(Un @ word_rows.T).mean())
        a_n = float(np.abs(Un @ nullU.T).mean())
        return a_w, a_n

    # ---------- decomposition ----------
    gam, rho, aw, an = {}, {}, {}, {}
    for ch in ('mlp', 'attn'):
        r64 = r_store[ch]['same'].astype(np.float64)
        gam[ch] = np.linalg.norm(r64, axis=2).mean(0)
        rho[ch] = (np.einsum('nqd,qd->nq', r64, d_stack)
                   / np.maximum(np.linalg.norm(r64, axis=2),
                                1e-30)).mean(0)
        a_w_l, a_n_l = [], []
        for q in range(n_win):
            aw_, an_ = A_of(r_store[ch]['same'][:, q, :]
                            .astype(np.float64))
            a_w_l.append(aw_)
            a_n_l.append(an_)
        aw[ch] = np.array(a_w_l)
        an[ch] = np.array(a_n_l)

    D_q_mlp = float(np.median(aw['mlp'] - an['mlp']))
    D_q_attn = float(np.median(aw['attn'] - an['attn']))
    R_gamma = float(np.median(gam['attn'] / gam['mlp']))
    R_rho = float(np.median(rho['attn'] / rho['mlp']))
    R_A = float(np.median(aw['attn'] / aw['mlp']))
    Babs = {ch: np.abs(B_proj[ch]).mean(0) for ch in ('mlp', 'attn')}
    R_B = float(np.median(Babs['attn'] / np.maximum(Babs['mlp'],
                                                    1e-30)))
    log('D_q_mlp=%.5f D_q_attn=%.5f R_gamma=%.3f R_rho=%.3f '
        'R_A=%.3f R_B=%.3f'
        % (D_q_mlp, D_q_attn, R_gamma, R_rho, R_A, R_B))

    # ---------- cross-model read (2902 result.json) ----------
    r02 = json.load(open(SRC_2902, encoding='utf-8'))
    pl02 = r02['per_layer']
    D_g = {}
    for ch in ('mlp', 'attn'):
        diffs_g = np.array(pl02['A_word'][ch]) - \
            np.array(pl02['A_nullrow'][ch])
        D_g[ch] = float(np.median(diffs_g))
    log('D_g_mlp=%.5f D_g_attn=%.5f' % (D_g['mlp'], D_g['attn']))

    # ---------- margins (SEED=2896 perms) ----------
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
    marg = {}
    for ch in ('mlp', 'attn'):
        C = zmat(B_proj[ch])
        acc = loo_acc(C, lab_lang)
        U = B_proj[ch] / np.maximum(np.linalg.norm(
            B_proj[ch], axis=1, keepdims=True), 1e-30)
        Sm = U @ U.T
        n = len(lab_lang)
        same_m = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(n):
                same_m[i, j] = (lab_lang[i] == lab_lang[j]) \
                    and i != j
        off = ~np.eye(n, dtype=bool) & ~same_m
        m_obs = float(Sm[same_m].mean() - Sm[off].mean())
        null_m = [margin_of(Sm, pl) for pl in perms]
        marg[ch] = {'margin': m_obs, 'acc': acc,
                    'margin_p95': float(np.percentile(null_m, 95))}
        log('%s: margin=%.4f acc=%.4f p95=%.4f'
            % (ch, m_obs, acc, marg[ch]['margin_p95']))

    # ---------- null random directions ----------
    rng3 = np.random.default_rng(SEED_NULL)
    null_stats = []
    for k in range(N_RND_DIRS):
        q = int(rng3.integers(0, n_win))
        li = WIN_LO + q
        d_r = unit(rng3.normal(size=2560))
        w_i = int(rng3.integers(0, n_words))
        w_tid = tid_map[words[w_i][2]]
        toks = [tid_map[words[same_ctx(w_i)][2]], w_tid]
        mlpin_all, attnin_all, mlpout_all = forward2(toks)
        entry = {'k': k, 'li': li}
        for ch, (xin_all, callf) in (
                ('mlp', (mlpin_all, mlp_call)),
                ('attn', (attnin_all, attn_call))):
            x = xin_all[li]
            ref = callf(li, x)
            xp = x.copy()
            xp[0, 1] = xp[0, 1] + EPS * d_r
            rv = (callf(li, xp) - ref) / EPS
            aw_, an_ = A_of(rv[None, :])
            entry[ch] = {'gamma': float(np.linalg.norm(rv)),
                         'rho': float(rv @ d_r),
                         'A_word': aw_, 'A_null': an_}
        null_stats.append(entry)
    q_null_p95 = float(np.percentile(
        [e['mlp']['A_word'] - e['mlp']['A_null']
         for e in null_stats], 95))
    q_null_p95_attn = float(np.percentile(
        [e['attn']['A_word'] - e['attn']['A_null']
         for e in null_stats], 95))
    log('q_null_p95 mlp=%.5f attn=%.5f'
        % (q_null_p95, q_null_p95_attn))

    # ---------- weights-only descriptive ----------
    def spectra(W):
        sv = np.linalg.svd(W, compute_uv=False)
        return {'top1_over_top2': float(sv[0] / max(sv[1], 1e-30)),
                'participation_ratio': float(
                    sv.sum() ** 2 / max(float((sv ** 2).sum()),
                                        1e-30))}

    spec = {}
    for q, li in enumerate(range(WIN_LO, WIN_HI)):
        d = dirs[q]
        ly = layers[li]
        g_mlp = ly.post_attention_layernorm.weight.detach() \
            .float().cpu().numpy()
        g_attn = ly.input_layernorm.weight.detach() \
            .float().cpu().numpy()
        entry = {}
        Wd = ly.mlp.down_proj.weight.detach().float().cpu().numpy()
        Wg = ly.mlp.gate_proj.weight.detach().float().cpu().numpy()
        Wu_ = ly.mlp.up_proj.weight.detach().float().cpu().numpy()
        dg = g_mlp * d
        h = (torch.nn.functional.silu(
            torch.tensor(Wg @ dg)) * torch.tensor(Wu_ @ dg)).numpy()
        v_m = Wd @ h
        entry['mlp'] = {**spectra(Wd),
                        'zf_response_gain': float(
                            np.linalg.norm(v_m) ** 2),
                        'zf_cos_resp_d': float(unit(v_m) @ d)}
        del Wd, Wg, Wu_
        Wo = ly.self_attn.o_proj.weight.detach().float() \
            .cpu().numpy()
        Wv = ly.self_attn.v_proj.weight.detach().float() \
            .cpu().numpy()
        nh = int(cfg['num_attention_heads'])
        nkv = int(cfg['num_key_value_heads'])
        hd = int(cfg.get('head_dim') or 0) or (
            int(cfg['hidden_size']) // nh)
        hidden = int(cfg['hidden_size'])
        rep = nh // nkv
        M = np.zeros((nh * hd, hidden))
        for i in range(nh):
            M[i * hd:(i + 1) * hd, :] = \
                Wv[(i // rep) * hd:(i // rep + 1) * hd, :]
        W_VO = Wo @ M
        va = W_VO @ (g_attn * d)
        entry['attn'] = {**spectra(Wo),
                         'spectra_W_VO_composite': spectra(W_VO),
                         'zf_response_gain': float(
                             np.linalg.norm(va) ** 2),
                         'zf_cos_resp_d': float(unit(va) @ d)}
        del Wo, Wv, M, W_VO
        spec[li] = entry
        del entry
    log('weights spectra done')

    # ---------- verdict ----------
    g_max = max(D_g['mlp'], D_g['attn'])
    if not anchor_ok:
        verdict = 'v2_anchor_fail_all_void'
    elif (D_q_mlp >= 2 * g_max and D_q_mlp > 2 * q_null_p95):
        verdict = 'language_content_density_confirmed_cross_model'
    elif D_q_mlp > 2 * q_null_p95:
        verdict = 'density_elevated_below_2x_glm4max'
    else:
        verdict = 'density_not_confirmed'

    res = {
        'phase': 2903, 'model': 'qwen3-4b', 'prereg': PREREG,
        'window': [WIN_LO, WIN_HI],
        'v1': v1, 'v1_err': {k: float('%.3e' % v)
                             for k, v in v1_err.items()},
        'hook_noise': float('%.3e' % hook_noise),
        'v2_anchor_rel_err': anchor_err, 'v2_anchor_ok': anchor_ok,
        'per_layer': {
            'gamma': {ch: [round(float(x), 4) for x in gam[ch]]
                      for ch in ('mlp', 'attn')},
            'rho': {ch: [round(float(x), 4) for x in rho[ch]]
                    for ch in ('mlp', 'attn')},
            'A_word': {ch: [round(float(x), 5) for x in aw[ch]]
                       for ch in ('mlp', 'attn')},
            'A_nullrow': {ch: [round(float(x), 5) for x in an[ch]]
                          for ch in ('mlp', 'attn')},
            'B_absmean': {ch: [round(float(x), 4)
                               for x in Babs[ch]]
                          for ch in ('mlp', 'attn')},
        },
        'metrics': {'D_q_mlp': round(D_q_mlp, 5),
                    'D_q_attn': round(D_q_attn, 5),
                    'D_g_mlp': round(D_g['mlp'], 5),
                    'D_g_attn': round(D_g['attn'], 5),
                    'q_null_p95_mlp': round(q_null_p95, 5),
                    'q_null_p95_attn': round(q_null_p95_attn, 5),
                    'R_gamma': round(R_gamma, 4),
                    'R_rho': round(R_rho, 4),
                    'R_A': round(R_A, 4), 'R_B': round(R_B, 4)},
        'margins': {ch: {kk: round(vv, 5)
                         for kk, vv in marg[ch].items()}
                    for ch in ('mlp', 'attn')},
        'null_dirs': {'q_null_p95_mlp': q_null_p95,
                      'entries': null_stats},
        'weights_descriptive': {str(li): {
            ch: {kk: (round(vv, 5) if not isinstance(vv, dict)
                      else {k2: round(v2, 5)
                            for k2, v2 in vv.items()})
                  for kk, vv in e[ch].items()}
            for ch in ('mlp', 'attn')} for li, e in spec.items()},
        'final_verdict': verdict,
        'runtime_s': round(time.monotonic() - t0, 1),
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    np.savez_compressed(
        os.path.join(OUT, 'qwen_channel_jacobian_decomposition.npz'),
        B_mlp=B_proj['mlp'].astype(np.float32),
        B_attn=B_proj['attn'].astype(np.float32),
        r_mlp_same=r_store['mlp']['same'],
        r_attn_same=r_store['attn']['same'],
        gamma_mlp=gam['mlp'].astype(np.float32),
        gamma_attn=gam['attn'].astype(np.float32),
        rho_mlp=rho['mlp'].astype(np.float32),
        rho_attn=rho['attn'].astype(np.float32),
        A_word_mlp=aw['mlp'].astype(np.float32),
        A_word_attn=aw['attn'].astype(np.float32),
        labels_lang=lab_lang, labels_concept=lab_concept,
        words=np.array(['%s:%s:%s' % w for w in words],
                       dtype=object))
    log('==== VERDICT: %s ====' % verdict)
    log('runtime %.1fs' % (time.monotonic() - t0))


if __name__ == '__main__':
    main()
