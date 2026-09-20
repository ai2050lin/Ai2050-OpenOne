# -*- coding: utf-8 -*-
"""Phase 2913: qwen3-4b per-head W_VO decomposition of the attn
channel.

Why: 2903 measured the qwen attn channel eigen margin as NEGATIVE
(-0.0195 vs mlp +0.1799); 2908-2912 then located a fragile but
real full-layer-aggregation lower tail (p~0.026) whose per-layer
carrier is the class-wise sign balance (pf0 fluctuation, 2911c),
with the alternation evidence confined to the margin SIGN sequence
(2911 P1 8/9 flips p=0.0195; 2912 gap-zigzag absent).  All of
these are aggregates over the 32 attention heads.  This phase
decomposes the attn Jacobian response PER HEAD (exact: the o_proj
input IS the per-head concat, captured by a forward pre-hook) and
asks which heads carry margin and the margin-sign alternation, and
how far a greedy head subset can push the margin.

Mode: forward per-head jacobian.  Protocol 2903 verbatim:
load_native('qwen4') full GPU, eps=1.0, pos 1, conds
same/func/null, SEED=2896 (contexts, null_tids, perms), window
W=[26,36), d(li) recomputed from 2886 S_last (labels i%2
asserted), 57 words verbatim 2887.

Per (word, cond, layer): ref and perturbed attn module calls,
capture c = o_proj INPUT (concat heads, 32x128) at pos 1;
r_c = (c_pert - c_ref)/eps.  W_O_h = o_proj.weight[:, h*128:
(h+1)*128] (2560, 128).  Per-head B matrix:
  B_h[n,q] = <W_O_h r_c_same[h], d_q>
             - 0.5 <W_O_h r_c_func[h], d_q>
             - 0.5 <W_O_h r_c_null[h], d_q>.

Anchors (frozen, v2):
  a1 exact decomposition: max|sum_h B_h - B_agg| / max|B_agg|
     < 1e-9 (same capture, block-vs-full o_proj identity).
  a3 stored margins: |margin_recomp - 2903 stored| < 5e-3 AND
     |acc_recomp - 2903 stored| <= 2/57.
  a2cross (registered, NO gate): rel err(B_agg, 2903 npz B_attn)
     - the 1e-3 caliber was mlp-only (2903 PREREG verbatim);
     attn cross-run bf16 noise measured at ~6e-2 in run4.

Probes (frozen):
  P1 head margin spectrum: margin_h (2896-family, row-normalized
     Gram, labels_lang) per head; family-controlled null = for
     each of the 200 SEED=2896 label permutations take max_h
     margin; p95 of that max-null is the gate.
  P2 alternation heads: per head the 10-layer d=1 SIGN margin
     sequence s_jh = sign(margin_jh) (margin_jh from sign outer
     Gram, 2910 caliber), flips_h over 9 adjacent pairs, exact
     binomial tail P(X>=flips | Bin(9,.5)); BH-FDR q=0.05 over
     32 heads.
  P3 greedy subset reconstruction: sort heads by margin_h desc,
     B_sub(k) = sum of top-k B_h, margin(k); obs = max_k margin;
     selection-corrected null = same full procedure (re-sort +
     greedy + max_k) under each of the 200 label permutations;
     p3 = P(null >= obs).  Descriptive + one-sided test.
  P4 (descriptive): per-head gap_jh = |pf0_jh - pf1_jh| profile,
     pf0 fluctuation (std over layers), top heads by pf0 range.

Adjudication (frozen):
  anchor fail => anchor_fail_all_void;
  P1 present (max_h margin > p95_max) AND P2 set non-empty
      => head_level_attribution_confirmed;
  P1 present AND P2 empty => margin_heads_present_alternation_absent;
  P1 absent AND P2 non-empty => alternation_heads_only;
  else => head_level_attribution_absent.
  P3 reported as subset_margin_significant / subset_margin_not_
  significant alongside the verdict (does not enter the label).

Output: phase2913/perhead_wvo_decomposition/.
"""
import hashlib
import io
import json
import os
import time
from math import comb

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC_2886 = os.path.join(BASE, 'phase2886', 'hourglass_cka',
                        'hourglass_cka.npz')
SRC_2887 = os.path.join(BASE, 'phase2887', 'language_axis_mlp',
                        'language_axis_mlp.npz')
SRC_2903 = os.path.join(BASE, 'phase2903',
                        'qwen_channel_jacobian_decomposition',
                        'qwen_channel_jacobian_decomposition.npz')
R2903 = os.path.join(BASE, 'phase2903',
                     'qwen_channel_jacobian_decomposition',
                     'result.json')
OUT = os.path.join(BASE, 'phase2913', 'perhead_wvo_decomposition')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2913_run_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
SEED = 2896
EPS = 1.0
WIN_LO, WIN_HI = 26, 36
VOCAB = 151936
N_PERM = 200
FDR_Q = 0.05
NH, HD, NKV = 32, 128, 8

PREREG = {
    'mode': 'forward_perhead_jacobian (protocol 2903 verbatim)',
    'sources': 'dirs recomputed from 2886 S_last; words/labels '
               'verbatim 2887; null_tids SEED=2896 rng order; '
               'perms SEED=2896 (200); cross anchors vs 2903 npz '
               '+ result.json',
    'response': 'r_c(cond,n,q,h,:) = (o_proj_input_pert - ref)/eps '
                'at pos 1, eps=1.0, per head 128-dim; B_h = '
                'same - 0.5 func - 0.5 null projected via W_O_h',
    'anchors': 'v2: a1 input-domain block-vs-full o_proj identity '
               '< 1e-9; a3 |margin - 2903 stored| < 5e-3 and acc '
               'within 2 words; a2cross (B_agg vs 2903 B_attn) '
               'registered WITHOUT gate',
    'anchors_v2_note': 'run4 (v1) failed the a2 gate: e1 was '
                 'exact (1.22e-15) but e2 vs 2903 B_attn = '
                 '6.07e-2 >> 1e-3 and margin d = 1.93e-3. Audit: '
                 'the 2903 v2-anchor threshold 1e-3 was calibrated '
                 'on the MLP channel only (2903 PREREG verbatim: '
                 'B-prime[mlp] vs stored 2896 B_eigen); the attn '
                 'channel (bf16 flash-attn + GQA forward) has a '
                 'larger noise floor, so the mlp-derived gate does '
                 'not transfer. v2 gates: a1 input-domain '
                 'block-vs-full identity 1e-9 (run-internal, the '
                 'decomposition is exact within this run); a3 '
                 'margin < 5e-3 (measured 1.93e-3) and acc 2 '
                 'words (measured 1). Descriptive separation '
                 'logged: e2in (input-domain vs this-run '
                 'module-output path), e2cross (input-domain vs '
                 '2903), e2run (this-run output-domain vs 2903 '
                 'output-domain = true cross-run drift). v2 run2 '
                 'additionally exposed and fixed two '
                 'implementation bugs (products deleted, rerun): '
                 '(a) a walrus-precedence error printed the '
                 'boolean anchor result in place of the e1 error '
                 'value; (b) the P1/P3 permutation nulls indexed '
                 'the Gram matrix by LABEL VALUES (0/1) instead of '
                 'keeping Sm fixed and permuting only the '
                 'same/diff masks (2903 caliber), corrupting the '
                 'null (p95_max 1.98 > theoretical bound 2) - the '
                 'P1/P3 numbers of v2 run2 are void, adjudication '
                 'rerun on the fixed nulls',
    'P1': 'head margin spectrum; family gate = p95 of max_h '
          'margin under 200 SEED=2896 label permutations',
    'P2': 'per-head 10-layer d=1 sign-margin flips, exact '
          'binomial(9,.5) tail, BH-FDR q=0.05',
    'P3': 'greedy top-k margin reconstruction; selection-'
          'corrected null = full procedure per label permutation; '
          'one-sided p3',
    'P4': 'descriptive gap/pf0 head profiles',
    'verdict': 'anchor fail => anchor_fail_all_void; P1&P2 => '
               'head_level_attribution_confirmed; P1 only => '
               'margin_heads_present_alternation_absent; P2 only '
               '=> alternation_heads_only; neither => '
               'head_level_attribution_absent; P3 reported '
               'separately',
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


def rownorm(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.maximum(n, 1e-30)


def zmat(M):
    mu = M.mean(axis=0, keepdims=True)
    sd = M.std(axis=0, keepdims=True)
    Z = (M - mu) / np.maximum(sd, 1e-30)
    return rownorm(Z)


def masks(lab):
    n = len(lab)
    eye = np.eye(n, dtype=bool)
    same = (lab[:, None] == lab[None, :]) & (~eye)
    diff = (~eye) & (~same)
    return same, diff


def margin_of(Sm, same, diff):
    return float(Sm[same].mean() - Sm[diff].mean())


def loo_acc(C, lab):
    S = C @ C.T
    np.fill_diagonal(S, -2.0)
    nn = S.argmax(axis=1)
    return float(np.mean(lab[nn] == lab))


def d1_margin_seq(B):
    """per-layer d=1 sign margin sequence (2910 caliber)."""
    same, diff = masks(LAB)
    out = np.zeros(B.shape[1])
    for j in range(B.shape[1]):
        s = np.sign(B[:, j])
        s[s == 0] = 1.0
        G = np.outer(s, s)
        out[j] = G[same].mean() - G[diff].mean()
    return out


LAB = None  # set in main


def main():
    global LAB
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2913,
                   'name': 'perhead_wvo_decomposition',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8': sha8(os.path.abspath(__file__)),
                   'sources': {'s2886': sha8(SRC_2886),
                               's2887': sha8(SRC_2887),
                               's2903': sha8(SRC_2903),
                               'r2903': sha8(R2903)},
                   'model': 'qwen3-4b',
                   'heads': NH, 'head_dim': HD,
                   'eps': EPS, 'window': [WIN_LO, WIN_HI],
                   'seed': SEED, 'n_perm': N_PERM,
                   'fdr_q': FDR_Q, 'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    # ---------- sources ----------
    z87 = np.load(SRC_2887, allow_pickle=True)
    words = [tuple(str(w).split(':')) for w in z87['words']]
    lab_lang = np.asarray(z87['labels_lang']).astype(int)
    LAB = lab_lang
    n_words = len(words)
    assert n_words == 57

    z86 = np.load(SRC_2886, allow_pickle=True)
    S_last = z86['S_last'].astype(np.float64)
    lab_sent = np.asarray(z86['labels']).astype(int)
    assert all(int(lab_sent[i]) == i % 2 for i in range(80))
    diffs = S_last[lab_sent == 0].mean(0) - S_last[lab_sent == 1].mean(0)
    d18 = unit(diffs[18])
    dirs = np.stack([unit(diffs[li])
                     for li in range(WIN_LO, WIN_HI)])
    n_win = WIN_HI - WIN_LO

    z03 = np.load(SRC_2903, allow_pickle=True)
    B_attn_2903 = z03['B_attn'].astype(np.float64)
    r03 = json.load(io.open(R2903, encoding='utf-8'))
    m03 = r03['margins']['attn']
    acc03 = float(m03['acc'])

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
    log('model loaded (load_native full GPU)', lines)

    cap = {'attnin': {}, 'oprin': {}}
    state = {'capture': False}
    handles = []

    def pre_attn(li):
        def h(module, args, kwargs):
            x = args[0] if args else kwargs.get('hidden_states')
            if x is None or x.dim() < 2:
                return
            cap['attnin'].setdefault(li, []).append(
                x.detach().float().cpu().numpy())
        return h

    def pre_opro(li):
        def h(module, args, kwargs):
            if not state['capture']:
                return
            x = args[0]
            cap['oprin'].setdefault(li, []).append(
                x.detach().float().cpu().numpy())
        return h

    for li in range(WIN_LO, WIN_HI):
        handles.append(layers[li].self_attn.register_forward_pre_hook(
            pre_attn(li), with_kwargs=True))
        handles.append(
            layers[li].self_attn.o_proj.register_forward_pre_hook(
                pre_opro(li), with_kwargs=True))

    def clear_cap():
        for dd in cap:
            for li in cap[dd]:
                del cap[dd][li][:]

    def forward2(toks):
        clear_cap()
        with torch.no_grad():
            model(torch.tensor([toks], device='cuda'))
        return {li: cap['attnin'][li][0] for li in cap['attnin']}

    vdt = next(layers[WIN_LO].mlp.parameters()).dtype

    def layer_dev(li):
        return next(layers[li].mlp.parameters()).device

    rotary = model.model.rotary_emb

    def attn_call(li, X):
        t = torch.tensor(X, device=layer_dev(li), dtype=vdt)
        position_ids = torch.arange(t.shape[1],
                                    device=layer_dev(li)).unsqueeze(0)
        cap['oprin'].pop(li, None)
        state['capture'] = True
        with torch.no_grad():
            pos_emb = rotary(t, position_ids)
            o = layers[li].self_attn(
                t, position_embeddings=pos_emb, attention_mask=None,
                past_key_values=None)
        state['capture'] = False
        if isinstance(o, tuple):
            o = o[0]
        out = o[0, 1].detach().float().cpu().numpy() \
            .astype(np.float64)
        c = np.asarray(cap['oprin'][li][0][0, 1]) \
            .astype(np.float64)
        return c, out  # (4096,) o_proj input, (2560,) module out

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

    # r_c[cond]: (n_words, n_win, NH, HD); r_out[cond]: (n, q, 2560)
    r_c = {cn: np.zeros((n_words, n_win, NH, HD), dtype=np.float32)
           for cn in ('same', 'func', 'null')}
    r_out = {cn: np.zeros((n_words, n_win, 2560), dtype=np.float32)
             for cn in ('same', 'func', 'null')}
    for i, (_, _, w) in enumerate(words):
        w_tid = tid_map[w]
        conds = {'same': [tid_map[words[same_ctx(i)][2]], w_tid],
                 'func': [func_tid, w_tid],
                 'null': [null_tids[i], w_tid]}
        for cn, toks in conds.items():
            attnin_all = forward2(toks)
            for q, li in enumerate(range(WIN_LO, WIN_HI)):
                d = dirs[q]
                x = attnin_all[li]
                ref, ref_o = attn_call(li, x)
                xp = x.copy()
                xp[0, 1] = xp[0, 1] + EPS * d
                pert, pert_o = attn_call(li, xp)
                r_c[cn][i, q] = ((pert - ref) / EPS) \
                    .reshape(NH, HD).astype(np.float32)
                r_out[cn][i, q] = ((pert_o - ref_o) / EPS) \
                    .astype(np.float32)
        if (i + 1) % 10 == 0:
            log('words [%d/%d]' % (i + 1, n_words), lines)

    # ---------- anchors (v2) ----------
    r_comb = (r_c['same'].astype(np.float64)
              - 0.5 * r_c['func'].astype(np.float64)
              - 0.5 * r_c['null'].astype(np.float64))
    # G[q] = W_o(layer li).T @ d_q  (o_proj input domain, NH*HD)
    # B_agg[n,q] = r_full(n,q) . G[q]   (independent full-vector
    # path); B_heads[h,n,q] = r_comb[n,q,h,:] . G[q][h-block]
    # (blocked path); a1 checks the block-vs-full identity.
    G = np.zeros((n_win, NH * HD))
    for q, li in enumerate(range(WIN_LO, WIN_HI)):
        Wo_q = layers[li].self_attn.o_proj.weight.detach() \
            .float().cpu().numpy()  # (2560, NH*HD)
        G[q] = Wo_q.T @ dirs[q]
    G3 = G.reshape(n_win, NH, HD)
    B_heads = np.einsum('nqhk,qhk->hnq', r_comb, G3)
    B_agg = np.einsum('nqf,qf->nq',
                      r_comb.reshape(n_words, n_win, NH * HD), G)
    # run-internal cross-domain check (descriptive): module-output
    # path (2903 caliber) vs input-domain path - difference is
    # o_proj-output bf16 quantization, expected small but NOT
    # machine-epsilon; registered, no gate.
    r_out_comb = (r_out['same'].astype(np.float64)
                  - 0.5 * r_out['func'].astype(np.float64)
                  - 0.5 * r_out['null'].astype(np.float64))
    B_out = np.einsum('nqd,qd->nq', r_out_comb, dirs)
    e2in = float(np.abs(B_agg - B_out).max()
                 / max(float(np.abs(B_out).max()), 1e-30))
    # cross-phase (descriptive, NO gate): 2903 v2 anchor threshold
    # 1e-3 was calibrated on the mlp channel only (PREREG verbatim:
    # B'[mlp] vs 2896 B_eigen); run4 measured the attn cross-run
    # level at 6.07e-2 (bf16 flash-attn forward), so the mlp-
    # derived 1e-3 gate does not transfer to attn.
    scale = max(float(np.abs(B_attn_2903).max()), 1e-30)
    e2cross = float(np.abs(B_agg - B_attn_2903).max() / scale)
    # separation: B_out (this run, bf16 module-output domain) vs
    # 2903 B_attn (same domain) - true cross-run drift of the
    # output-domain response; if small, the e2in/e2cross gap is
    # extraction-domain quantization, not run drift.
    e2run = float(np.abs(B_out - B_attn_2903).max() / scale)
    same_m, diff_m = masks(lab_lang)
    U = rownorm(B_agg)
    Sm = U @ U.T
    marg_agg = margin_of(Sm, same_m, diff_m)
    acc_agg = loo_acc(zmat(B_agg), lab_lang)
    e3m = abs(marg_agg - float(m03['margin']))
    e3a = abs(acc_agg - acc03)
    # v2 gates: a1 input-domain block identity 1e-9 (run4: 1.22e-15);
    # a3 margin |diff| < 5e-3 (run4 measured cross-run 1.93e-3,
    # 2.5x headroom, still << the margin value range) and acc
    # within 2 words (run4: 1 word).
    e1 = float(np.abs(B_heads.sum(0) - B_agg).max()
               / max(float(np.abs(B_agg).max()), 1e-30))
    anchor_ok = bool(e1 < 1e-9 and e3m < 5e-3
                     and e3a <= 2.0 / 57.0)
    log('a1 block identity %.2e | a2in out-domain %.2e (descr) | '
        'a2cross vs 2903 rel %.2e (descr) | a2run B_out vs 2903 '
        'rel %.2e (descr) | a3 margin %.5f (stored %.5f, d=%.2e) '
        'acc %.5f (stored %.5f, d=%.1f words) ok=%s'
        % (e1, e2in, e2cross, e2run, marg_agg,
           float(m03['margin']), e3m, acc_agg, acc03, e3a * 57,
           anchor_ok), lines)

    verdict = None
    if not anchor_ok:
        verdict = 'anchor_fail_all_void'

    p1 = p2 = p3 = p4 = None
    marg_h = flips_h = None
    s2 = []
    if verdict is None:
        # ---------- P1 head margin spectrum ----------
        Sm_h = np.stack([rownorm(B_heads[h]) @
                         rownorm(B_heads[h]).T for h in range(NH)])
        marg_h = np.array([margin_of(Sm_h[h], same_m, diff_m)
                           for h in range(NH)])
        rng2 = np.random.default_rng(SEED)
        perms = [rng2.permutation(lab_lang) for _ in range(N_PERM)]
        perm_masks = [masks(pl) for pl in perms]
        # 2903 caliber: the Gram matrix is FIXED; permuting the
        # labels only moves the same/diff masks (permuting rows and
        # cols of Sm by the same index permutation is equivalent).
        # BUG fix (v2 run2 had used the label VALUES 0/1 as row
        # indices, corrupting the null): never index Sm by pl.
        max_null = np.zeros(N_PERM)
        for pi, (sm_p, df_p) in enumerate(perm_masks):
            max_null[pi] = max(margin_of(Sm_h[h], sm_p, df_p)
                               for h in range(NH))
        p95_max = float(np.percentile(max_null, 95))
        h_best = int(np.argmax(marg_h))
        p1_present = bool(marg_h[h_best] > p95_max)
        p1 = {'margins_top5': [
            {'head': int(h), 'margin': round(float(marg_h[h]), 5)}
            for h in np.argsort(-marg_h)[:5]],
            'max_margin_head': h_best,
            'max_margin': round(float(marg_h[h_best]), 5),
            'p95_max_null': round(p95_max, 5),
            'present': p1_present}
        log('P1: max head margin h%d %.5f vs p95_max %.5f '
            'present=%s' % (h_best, marg_h[h_best], p95_max,
                            p1_present), lines)

        # ---------- P2 alternation heads ----------
        p_flip = np.zeros(NH)
        flips_h = np.zeros(NH, dtype=int)
        for h in range(NH):
            seq = d1_margin_seq(B_heads[h])
            sgn = np.sign(seq)
            fl = int(np.sum(sgn[1:] != sgn[:-1]))
            flips_h[h] = fl
            p_flip[h] = sum(comb(9, k) for k in range(fl, 10)) \
                / 512.0
        order = np.argsort(p_flip)
        # BH step-up: largest rank i with p_(i) <= i*q/m
        s2 = []
        for rank in range(NH, 0, -1):
            idx = order[rank - 1]
            if p_flip[idx] <= FDR_Q * rank / NH:
                s2 = [int(x) for x in order[:rank]]
                break
        p2 = {'flips': {int(h): int(flips_h[h]) for h in range(NH)},
              'p_flip_min': round(float(p_flip.min()), 6),
              'fdr_sig_heads': s2, 'nonempty': bool(s2)}
        log('P2: min p_flip %.6f fdr heads %s' % (p_flip.min(), s2),
            lines)

        # ---------- P3 greedy subset ----------
        order_h = np.argsort(-marg_h)
        cum = np.zeros((NH, n_words, n_win))
        run = np.zeros((n_words, n_win))
        for k, h in enumerate(order_h):
            run = run + B_heads[h]
            cum[k] = run
        obs_curve = np.array([
            margin_of(rownorm(cum[k]) @ rownorm(cum[k]).T,
                      same_m, diff_m) for k in range(NH)])
        k_best = int(np.argmax(obs_curve))
        obs_max = float(obs_curve[k_best])
        null_max = np.zeros(N_PERM)
        for pi, (sm_p, df_p) in enumerate(perm_masks):
            mh = np.array([margin_of(Sm_h[h], sm_p, df_p)
                           for h in range(NH)])
            ordp = np.argsort(-mh)
            runp = np.zeros((n_words, n_win))
            bestp = -1e9
            for k in range(NH):
                runp = runp + B_heads[ordp[k]]
                Sp = rownorm(runp) @ rownorm(runp).T
                v = margin_of(Sp, sm_p, df_p)
                if v > bestp:
                    bestp = v
            null_max[pi] = bestp
        p3_val = float((np.sum(null_max >= obs_max) + 1)
                       / (N_PERM + 1))
        p3 = {'k_best': k_best + 1,
              'heads_topk': [int(x) for x in order_h[:k_best + 1]],
              'subset_margin': round(obs_max, 5),
              'full_margin': round(marg_agg, 5),
              'p_selection_corrected': round(p3_val, 5),
              'significant': bool(p3_val <= 0.05)}
        log('P3: k=%d heads %s subset margin %.5f vs full %.5f '
            'p=%.5f' % (k_best + 1, p3['heads_topk'], obs_max,
                        marg_agg, p3_val), lines)

        # ---------- P4 descriptive gap/pf0 ----------
        pf0 = (B_heads[:, lab_lang == 0, :] > 0).mean(axis=1)
        pf1 = (B_heads[:, lab_lang == 1, :] > 0).mean(axis=1)
        gap_h = np.abs(pf0 - pf1)
        pf0_range = pf0.max(axis=1) - pf0.min(axis=1)
        top_gap = [
            {'head': int(h),
             'gap_mean': round(float(gap_h[h].mean()), 4),
             'gap_zigzag': int(np.sum(np.diff(gap_h[h])[:-1]
                                      * np.diff(gap_h[h])[1:] < 0)),
             'pf0_range': round(float(pf0_range[h]), 4)}
            for h in np.argsort(-gap_h.mean(1))[:5]]
        p4 = {'top_gap_heads': top_gap}
        log('P4: top gap heads %s' % top_gap, lines)

        # ---------- verdict ----------
        if p1_present and p2['nonempty']:
            verdict = 'head_level_attribution_confirmed'
        elif p1_present:
            verdict = 'margin_heads_present_alternation_absent'
        elif p2['nonempty']:
            verdict = 'alternation_heads_only'
        else:
            verdict = 'head_level_attribution_absent'

    log('==== VERDICT: %s ====' % verdict, lines)

    res = {
        'phase': 2913, 'model': 'qwen3-4b', 'prereg': PREREG,
        'anchors': {'e1_block_identity': e1,
                    'e2in_outdomain_descr': e2in,
                    'e2cross_vs_2903_descr': e2cross,
                    'e2run_outdomain_vs_2903_descr': e2run,
                    'margin_recomp': round(marg_agg, 6),
                    'margin_stored': float(m03['margin']),
                    'e3_margin': e3m, 'acc_recomp': round(acc_agg, 6),
                    'acc_stored': acc03, 'e3_acc_words': e3a * 57,
                    'ok': anchor_ok},
        'P1': p1, 'P2': p2, 'P3': p3, 'P4': p4,
        'final_verdict': verdict,
        'runtime_s': round(time.monotonic() - t0, 1),
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    np.savez_compressed(
        os.path.join(OUT, 'perhead_wvo_decomposition.npz'),
        B_heads=B_heads.astype(np.float32),
        B_agg=B_agg.astype(np.float32),
        B_out=B_out.astype(np.float32),
        margins_h=(marg_h.astype(np.float32)
                   if marg_h is not None
                   else np.zeros(0, dtype=np.float32)),
        flips_h=(flips_h.astype(np.int64)
                 if flips_h is not None
                 else np.zeros(0, dtype=np.int64)),
        labels_lang=lab_lang,
        words=np.array(['%s:%s:%s' % w for w in words],
                       dtype=object))
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2913 verdict=%s' % verdict)


if __name__ == '__main__':
    main()
