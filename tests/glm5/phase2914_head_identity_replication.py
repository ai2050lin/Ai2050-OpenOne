# -*- coding: utf-8 -*-
"""Phase 2914: head identity replication + per-head W_VO spectra.

Why: 2913 decomposed the qwen attn channel margin PER HEAD (exact
o_proj-input capture) and found h7/h8 as the margin carriers
(top5 h7/h8/h6/h22/h21 = 0.1771/0.1343/0.0789/0.0601/0.0554;
{7,8} greedy reconstruction 0.28036 vs full -0.02139,
selection-corrected p=0.01493; alternation absent per head).
The 2913 hard-constraint clause requires cross-run confirmation
of the h7/h8 identity before it enters formal conclusions.  This
phase (1) re-runs the 2913 protocol verbatim (Run B) and tests
identity replication, and (2) - zero forward, weight domain only -
computes per-head W_VO spectra (thin-SVD trick) and per-head zf
language responses, asking whether the carrier heads are
spectrally special and whether composite spectra replicate the
2903 registered weights_descriptive.

Mode: Run B replication (2913 v2 protocol verbatim: SEED=2896,
eps=1.0, pos 1, window [26,36), 57 words verbatim 2887, dirs
recomputed from 2886 S_last, conds same/func/null, null_tids rng
order, o_proj-input capture) + weights-only per-head spectra.

Anchors (frozen):
  a1 block-vs-full o_proj input identity < 1e-9 (2913 caliber).
  a2 |margin_runB - 2903 stored| < 5e-3 AND acc within 2/57.
  a3 composite W_VO spectra replicate 2903
     weights_descriptive[li][attn] for ALL 10 layers:
     |t12 diff| < 1e-4 (abs), |PR diff|/PR_stored < 1e-4 (rel),
     |zf_gain diff|/gain_stored < 1e-4 (rel), |zf_cos diff| < 1e-4.
  a4a per-head zf identity: max|sum_h va_h - composite va| /
     max|va| < 1e-9 (exact column-block algebra, fp64).
  a4b thin-SVD self-check: sv_mid(A,B) vs svd(A@B) on random
     matrices, rel < 1e-10.

Probes (frozen):
  P1 identity replication: relB = rel err(B_heads_runB vs
     B_heads_2913) registered WITHOUT gate (bf16 forward noise
     floor, 2913 e2run family); Spearman/Kendall(margins_runB,
     margins_2913) handwritten (average ranks / tau-b); top2 set
     equality vs {7,8}; P1 family gate re-run on Run B (200
     SEED=2896 label permutations, Sm fixed + mask permutation,
     p95 of max_h margin); {7,8} subset margin on Run B, |diff
     vs 0.28036| < 5e-3.
  P2 flips_h_runB vs 2913 flips_h equality (registered, no gate;
     2913 P2 was absent - alternation has no single-head carrier).
  P3 greedy + selection-corrected null re-run on Run B (same 200
     perms): p3, k_best, heads (descriptive, one-sided).
  P4 per-head spectra (zero forward): per layer per head
     W_VO_h = W_O_h @ Wv_kv[kv(h)] (2560x2560 implicit), thin-SVD
     singular values via middle 128x128 matrix
     M_c = S_A (V_A^T U_B) S_B; t12_h, PR_h, zf_gain_h,
     zf_cos_h; layer-aggregated head rankings; Spearman of each
     spectral statistic vs margins_h_2913; h7/h8 profile.

Adjudication (frozen):
  anchor fail => anchor_fail_all_void;
  spearman >= 0.999 AND top2 == {7,8} AND p1_gate_reproduced AND
      recon_ok => head_identity_reproduced;
  elif spearman >= 0.9 => margin_spectrum_reproduced_identity_
      shifted;
  elif spearman >= 0.5 => head_ordering_partially_reproduced;
  else => head_ordering_not_reproduced.
  P3/P4 descriptive alongside (do not enter the label).

Output: phase2914/head_identity_replication/.
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
SRC_2913 = os.path.join(BASE, 'phase2913',
                        'perhead_wvo_decomposition',
                        'perhead_wvo_decomposition.npz')
R2913 = os.path.join(BASE, 'phase2913',
                     'perhead_wvo_decomposition', 'result.json')
OUT = os.path.join(BASE, 'phase2914', 'head_identity_replication')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2914_run_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
SEED = 2896
EPS = 1.0
WIN_LO, WIN_HI = 26, 36
VOCAB = 151936
N_PERM = 200
NH, HD, NKV = 32, 128, 8
RECON_REF = 0.28036   # 2913 P3 {7,8} subset margin (frozen ref)
TOP2_REF = [7, 8]     # 2913 top2 (frozen ref)

PREREG = {
    'mode': 'runB_replication (2913 v2 protocol verbatim) + '
            'weights-only per-head W_VO spectra',
    'question': '2913 hard-constraint clause: h7/h8 head identity '
                'must be cross-run confirmed before formal '
                'conclusions; per-head W_VO spectral structure of '
                'the margin carrier heads (zero-forward weight '
                'domain)',
    'runB': 'SEED=2896, eps=1.0, pos 1, window [26,36), 57 words '
            'verbatim 2887, dirs from 2886 S_last, conds '
            'same/func/null, null_tids rng order, perms rng2 reset '
            'SEED, o_proj-input capture hooks - all verbatim 2913 '
            'v2; margins/flips/greedy recomputed on Run B',
    'anchors': {
        'a1': 'block-vs-full o_proj input identity < 1e-9',
        'a2': '|margin_runB - 2903| < 5e-3 AND acc within 2/57',
        'a3': 'composite W_VO spectra vs 2903 '
              'weights_descriptive all 10 layers: t12 abs 1e-4, '
              'PR rel 1e-4, zf_gain rel 1e-4, zf_cos abs 1e-4',
        'a4a': 'sum_h va_h == composite va rel < 1e-9',
        'a4b': 'thin-SVD self-check rel < 1e-10',
    },
    'probes': {
        'P1': 'relB registered (no gate); handwritten Spearman/'
              'Kendall(margins_runB, margins_2913); top2 == {7,8}; '
              'family gate re-run (200 perms, Sm fixed + mask '
              'perm); {7,8} subset margin |diff vs 0.28036| < 5e-3',
        'P2': 'flips_h_runB == flips_h_2913 equality (no gate)',
        'P3': 'greedy + selection-corrected null on Run B (200 '
              'perms), one-sided p3',
        'P4': 'per-head thin-SVD spectra + zf responses, layer '
              'aggregation, Spearman vs margins_h, h7/h8 profile',
    },
    'verdict': 'anchor fail => anchor_fail_all_void; spearman>=0.999 '
               'AND top2=={7,8} AND p1_gate_reproduced AND recon_ok '
               '=> head_identity_reproduced; elif spearman>=0.9 => '
               'margin_spectrum_reproduced_identity_shifted; elif '
               'spearman>=0.5 => head_ordering_partially_reproduced; '
               'else => head_ordering_not_reproduced; P3/P4 '
               'descriptive',
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


def d1_margin_seq(B, lab):
    """per-layer d=1 sign margin sequence (2910 caliber)."""
    same, diff = masks(lab)
    out = np.zeros(B.shape[1])
    for j in range(B.shape[1]):
        s = np.sign(B[:, j])
        s[s == 0] = 1.0
        G = np.outer(s, s)
        out[j] = G[same].mean() - G[diff].mean()
    return out


def rankdata(x):
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x), dtype=np.float64)
    sx = x[order].astype(np.float64)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(a, b):
    ra = rankdata(a) - rankdata(a).mean()
    rb = rankdata(b) - rankdata(b).mean()
    return float((ra @ rb)
                 / max(float(np.sqrt((ra @ ra) * (rb @ rb))), 1e-30))


def kendall_tau(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = len(a)
    num = 0.0
    nties_a = nties_b = 0
    for i in range(n):
        da = a[i] - a[i + 1:]
        db = b[i] - b[i + 1:]
        sa = np.sign(da)
        sb = np.sign(db)
        num += float(np.sum(sa * sb))
        nties_a += int(np.sum(sa == 0))
        nties_b += int(np.sum(sb == 0))
    n0 = n * (n - 1) / 2.0
    return float(num / max(np.sqrt((n0 - nties_a) * (n0 - nties_b)),
                           1e-30))


def sv_mid(A, Bm):
    """singular values of A @ Bm via the middle k x k matrix
    (A: (m,k), Bm: (k,n), k small) - thin-SVD trick."""
    UA, SA, VAt = np.linalg.svd(A, full_matrices=False)
    UB, SB, VBt = np.linalg.svd(Bm, full_matrices=False)
    Mc = (SA[:, None] * (VAt @ UB)) * SB[None, :]
    return np.linalg.svd(Mc, compute_uv=False)


def spectra_sv(sv):
    sv = np.asarray(sv, dtype=np.float64)
    return (float(sv[0] / max(float(sv[1]), 1e-30)),
            float(sv.sum() ** 2
                  / max(float((sv ** 2).sum()), 1e-30)))


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2914,
                   'name': 'head_identity_replication',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8': sha8(os.path.abspath(__file__)),
                   'sources': {'s2886': sha8(SRC_2886),
                               's2887': sha8(SRC_2887),
                               's2903': sha8(SRC_2903),
                               'r2903': sha8(R2903),
                               's2913': sha8(SRC_2913),
                               'r2913': sha8(R2913)},
                   'model': 'qwen3-4b',
                   'heads': NH, 'head_dim': HD,
                   'eps': EPS, 'window': [WIN_LO, WIN_HI],
                   'seed': SEED, 'n_perm': N_PERM,
                   'recon_ref': RECON_REF, 'top2_ref': TOP2_REF,
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    # ---------- sources ----------
    z87 = np.load(SRC_2887, allow_pickle=True)
    words = [tuple(str(w).split(':')) for w in z87['words']]
    lab_lang = np.asarray(z87['labels_lang']).astype(int)
    n_words = len(words)
    assert n_words == 57

    z86 = np.load(SRC_2886, allow_pickle=True)
    S_last = z86['S_last'].astype(np.float64)
    lab_sent = np.asarray(z86['labels']).astype(int)
    assert all(int(lab_sent[i]) == i % 2 for i in range(80))
    diffs = S_last[lab_sent == 0].mean(0) - S_last[lab_sent == 1].mean(0)
    dirs = np.stack([unit(diffs[li])
                     for li in range(WIN_LO, WIN_HI)])
    n_win = WIN_HI - WIN_LO

    z03 = np.load(SRC_2903, allow_pickle=True)
    B_attn_2903 = z03['B_attn'].astype(np.float64)
    r03 = json.load(io.open(R2903, encoding='utf-8'))
    m03 = r03['margins']['attn']
    acc03 = float(m03['acc'])
    wd03 = r03['weights_descriptive']

    z13 = np.load(SRC_2913, allow_pickle=True)
    B_heads_2913 = z13['B_heads'].astype(np.float64)
    B_agg_2913 = z13['B_agg'].astype(np.float64)
    margins_h_2913 = z13['margins_h'].astype(np.float64)
    flips_h_2913 = z13['flips_h'].astype(int)
    r13 = json.load(io.open(R2913, encoding='utf-8'))
    log('sources loaded: 2913 B_heads %s margins top5 %s'
        % (B_heads_2913.shape,
           [(int(h), round(float(margins_h_2913[h]), 5))
            for h in np.argsort(-margins_h_2913)[:5]]), lines)

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
        return c, out

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

    # ---------- Run B forward (2913 verbatim) ----------
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
            log('Run B words [%d/%d]' % (i + 1, n_words), lines)

    # ---------- anchors a1/a2 (+ e2 family registered) ----------
    r_comb = (r_c['same'].astype(np.float64)
              - 0.5 * r_c['func'].astype(np.float64)
              - 0.5 * r_c['null'].astype(np.float64))
    G = np.zeros((n_win, NH * HD))
    for q, li in enumerate(range(WIN_LO, WIN_HI)):
        Wo_q = layers[li].self_attn.o_proj.weight.detach() \
            .float().cpu().numpy()
        G[q] = Wo_q.T @ dirs[q]
    G3 = G.reshape(n_win, NH, HD)
    B_heads = np.einsum('nqhk,qhk->hnq', r_comb, G3)
    B_agg = np.einsum('nqf,qf->nq',
                      r_comb.reshape(n_words, n_win, NH * HD), G)
    r_out_comb = (r_out['same'].astype(np.float64)
                  - 0.5 * r_out['func'].astype(np.float64)
                  - 0.5 * r_out['null'].astype(np.float64))
    B_out = np.einsum('nqd,qd->nq', r_out_comb, dirs)
    e1 = float(np.abs(B_heads.sum(0) - B_agg).max()
               / max(float(np.abs(B_agg).max()), 1e-30))
    e2in = float(np.abs(B_agg - B_out).max()
                 / max(float(np.abs(B_out).max()), 1e-30))
    scale03 = max(float(np.abs(B_attn_2903).max()), 1e-30)
    e2cross = float(np.abs(B_agg - B_attn_2903).max() / scale03)
    e2run = float(np.abs(B_out - B_attn_2903).max() / scale03)
    same_m, diff_m = masks(lab_lang)
    U = rownorm(B_agg)
    Sm = U @ U.T
    marg_agg = margin_of(Sm, same_m, diff_m)
    acc_agg = loo_acc(zmat(B_agg), lab_lang)
    e3m = abs(marg_agg - float(m03['margin']))
    e3a = abs(acc_agg - acc03)
    relB = float(np.abs(B_heads - B_heads_2913).max()
                 / max(float(np.abs(B_heads_2913).max()), 1e-30))
    rel_agg_run = float(np.abs(B_agg - B_agg_2913).max()
                        / max(float(np.abs(B_agg_2913).max()), 1e-30))
    a1_ok = bool(e1 < 1e-9)
    a2_ok = bool(e3m < 5e-3 and e3a <= 2.0 / 57.0)
    log('a1 %.2e | e2in %.2e e2cross %.2e e2run %.2e (descr) | '
        'a2 margin %.5f (stored %.5f) acc %.5f (%.1f words) ok=%s | '
        'relB vs 2913 %.2e rel_agg %.2e'
        % (e1, e2in, e2cross, e2run, marg_agg,
           float(m03['margin']), acc_agg, e3a * 57, a2_ok,
           relB, rel_agg_run), lines)

    # ---------- a3/a4: weight-domain spectra ----------
    # a4b thin-SVD self-check (random, fp64)
    rngc = np.random.default_rng(2914)
    A_r = rngc.standard_normal((300, 128))
    B_r = rngc.standard_normal((128, 200))
    sv_direct = np.linalg.svd(A_r @ B_r, compute_uv=False)
    sv_thin = sv_mid(A_r, B_r)
    # A@Bm is 300x200: min(m,n)=200 singular values, of which
    # 72 are exactly zero (rank 128); sv_mid returns the 128
    # nonzero spectrum - compare the leading 128.
    e4b = float(np.abs(sv_thin
                       - sv_direct[:sv_thin.shape[0]]).max()
                / max(float(sv_direct[0]), 1e-30))
    a3_layers = []
    spec_h = np.zeros((n_win, NH, 4))   # t12, PR, zf_gain, zf_cos
    e4a = 0.0
    a3_ok = True
    for q, li in enumerate(range(WIN_LO, WIN_HI)):
        ly = layers[li]
        g_attn = ly.input_layernorm.weight.detach() \
            .float().cpu().numpy()
        Wo = ly.self_attn.o_proj.weight.detach() \
            .float().cpu().numpy()          # (2560, 4096) fp32
        Wv = ly.self_attn.v_proj.weight.detach() \
            .float().cpu().numpy()          # (1024, 2560) fp32
        rep = NH // NKV
        # 2903 caliber verbatim: M defaults to fp64, Wo @ M is a
        # fp32@fp64 -> fp64 matmul, svd is fp64.
        M = np.zeros((NH * HD, 2560))
        for i in range(NH):
            M[i * HD:(i + 1) * HD, :] = \
                Wv[(i // rep) * HD:(i // rep + 1) * HD, :]
        W_VO = Wo @ M
        sv = np.linalg.svd(W_VO, compute_uv=False)
        t12_c, pr_c = spectra_sv(sv)
        x_zf = g_attn * dirs[q]              # fp32*fp64 -> fp64
        va_c = W_VO @ x_zf
        gain_c = float(np.linalg.norm(va_c) ** 2)
        cos_c = float(unit(va_c) @ dirs[q])
        st = wd03[str(li)]['attn']
        d_t12 = abs(t12_c - float(st['spectra_W_VO_composite']
                                  ['top1_over_top2']))
        pr_st = float(st['spectra_W_VO_composite']
                      ['participation_ratio'])
        d_pr = abs(pr_c - pr_st) / max(abs(pr_st), 1e-30)
        gain_st = float(st['zf_response_gain'])
        d_gain = abs(gain_c - gain_st) / max(abs(gain_st), 1e-30)
        d_cos = abs(cos_c - float(st['zf_cos_resp_d']))
        ok_l = bool(d_t12 < 1e-4 and d_pr < 1e-4
                    and d_gain < 1e-4 and d_cos < 1e-4)
        a3_ok = a3_ok and ok_l
        a3_layers.append({'layer': li,
                          't12': round(t12_c, 6),
                          't12_stored': float(st[
                              'spectra_W_VO_composite'][
                              'top1_over_top2']),
                          'd_t12': float('%.3e' % d_t12),
                          'PR': round(pr_c, 5), 'PR_stored': pr_st,
                          'd_PR_rel': float('%.3e' % d_pr),
                          'zf_gain': round(gain_c, 6),
                          'zf_gain_stored': gain_st,
                          'd_gain_rel': float('%.3e' % d_gain),
                          'zf_cos': round(cos_c, 6),
                          'zf_cos_stored': float(st[
                              'zf_cos_resp_d']),
                          'd_cos': float('%.3e' % d_cos),
                          'ok': ok_l})
        # per-head spectra (fp64, thin-SVD) + zf identity a4a
        Wo64 = Wo.astype(np.float64)
        Wv64 = Wv.astype(np.float64)
        x_zf64 = x_zf.astype(np.float64)
        v_kv = Wv64 @ x_zf64                 # (1024,)
        va_sum = np.zeros(2560)
        for h in range(NH):
            Wo_h = Wo64[:, h * HD:(h + 1) * HD]
            Wv_h = Wv64[(h // rep) * HD:(h // rep + 1) * HD, :]
            va_h = Wo_h @ (Wv_h @ x_zf64)
            va_sum += va_h
            sv_h = sv_mid(Wo_h, Wv_h)
            t12_h, pr_h = spectra_sv(sv_h)
            spec_h[q, h] = (t12_h, pr_h,
                            float(np.linalg.norm(va_h) ** 2),
                            float(unit(va_h) @ dirs[q]))
        va_ref = W_VO.astype(np.float64) @ x_zf64
        e4a = max(e4a, float(np.abs(va_sum - va_ref).max()
                             / max(float(np.abs(va_ref).max()),
                                   1e-30)))
        del Wo, Wv, M, W_VO, Wo64, Wv64
    a4_ok = bool(e4a < 1e-9 and e4b < 1e-10)
    a3_worst = max(max(l['d_t12'], l['d_PR_rel'], l['d_gain_rel'],
                       l['d_cos']) for l in a3_layers)
    log('a3 composite spectra worst diff %.3e ok=%s | a4a %.2e '
        'a4b %.2e ok=%s' % (a3_worst, a3_ok, e4a, e4b, a4_ok), lines)

    anchor_ok = bool(a1_ok and a2_ok and a3_ok and a4_ok)

    # ---------- probes ----------
    verdict = None
    p1 = p2 = p3 = p4 = None
    marg_h = None
    flips_h = None
    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        Sm_h = np.stack([rownorm(B_heads[h])
                         @ rownorm(B_heads[h]).T
                         for h in range(NH)])
        marg_h = np.array([margin_of(Sm_h[h], same_m, diff_m)
                           for h in range(NH)])
        # P1 family gate re-run (2903 caliber: Sm fixed, mask perm)
        rng2 = np.random.default_rng(SEED)
        perms = [rng2.permutation(lab_lang) for _ in range(N_PERM)]
        perm_masks = [masks(pl) for pl in perms]
        max_null = np.zeros(N_PERM)
        for pi, (sm_p, df_p) in enumerate(perm_masks):
            max_null[pi] = max(margin_of(Sm_h[h], sm_p, df_p)
                               for h in range(NH))
        p95_max = float(np.percentile(max_null, 95))
        h_best = int(np.argmax(marg_h))
        p1_gate = bool(marg_h[h_best] > p95_max)
        rho = spearman(marg_h, margins_h_2913)
        tau = kendall_tau(marg_h, margins_h_2913)
        top2 = [int(h) for h in np.argsort(-marg_h)[:2]]
        top2_equal = bool(sorted(top2) == sorted(TOP2_REF))
        run78 = B_heads[7] + B_heads[8]
        m78 = margin_of(rownorm(run78) @ rownorm(run78).T,
                        same_m, diff_m)
        recon_ok = bool(abs(m78 - RECON_REF) < 5e-3)
        p1 = {'relB': float('%.3e' % relB),
              'rel_agg_vs_2913': float('%.3e' % rel_agg_run),
              'spearman': round(rho, 6),
              'kendall_tau_b': round(tau, 6),
              'top2_runB': top2, 'top2_equal': top2_equal,
              'margins_top5_runB': [
                  {'head': int(h),
                   'margin': round(float(marg_h[h]), 5)}
                  for h in np.argsort(-marg_h)[:5]],
              'max_margin_head': h_best,
              'max_margin': round(float(marg_h[h_best]), 5),
              'p95_max_null_runB': round(p95_max, 5),
              'p1_gate_reproduced': p1_gate,
              'margin_78_runB': round(m78, 6),
              'recon_ref': RECON_REF,
              'recon_absdiff': round(abs(m78 - RECON_REF), 6),
              'recon_ok': recon_ok}
        log('P1: spearman %.6f tau %.6f top2 %s (equal %s) gate '
            'h%d %.5f vs p95 %.5f (%s) | {7,8} %.6f vs ref %.5f '
            '(d=%.6f ok=%s) | relB %.2e'
            % (rho, tau, top2, top2_equal, h_best,
               marg_h[h_best], p95_max, p1_gate, m78, RECON_REF,
               abs(m78 - RECON_REF), recon_ok, relB), lines)

        # P2 flips equality (registered, no gate)
        flips_h = np.zeros(NH, dtype=int)
        p_flip = np.zeros(NH)
        for h in range(NH):
            seq = d1_margin_seq(B_heads[h], lab_lang)
            sgn = np.sign(seq)
            sgn[sgn == 0] = 1.0
            fl = int(np.sum(sgn[1:] != sgn[:-1]))
            flips_h[h] = fl
            p_flip[h] = sum(comb(9, k) for k in range(fl, 10)) \
                / 512.0
        flips_equal = bool(np.array_equal(flips_h, flips_h_2913))
        p2 = {'flips_equal': flips_equal,
              'flips_runB': {int(h): int(flips_h[h])
                             for h in range(NH)},
              'p_flip_min': round(float(p_flip.min()), 6)}
        log('P2: flips_equal=%s p_flip_min %.6f'
            % (flips_equal, p_flip.min()), lines)

        # P3 greedy + selection-corrected null (Run B)
        order_h = np.argsort(-marg_h)
        run = np.zeros((n_words, n_win))
        obs_curve = np.zeros(NH)
        for k, h in enumerate(order_h):
            run = run + B_heads[h]
            Sk = rownorm(run) @ rownorm(run).T
            obs_curve[k] = margin_of(Sk, same_m, diff_m)
        k_best = int(np.argmax(obs_curve))
        obs_max = float(obs_curve[k_best])
        null_max = np.zeros(N_PERM)
        for pi, (sm_p, df_p) in enumerate(perm_masks):
            mh_p = np.array([margin_of(Sm_h[h], sm_p, df_p)
                             for h in range(NH)])
            ordp = np.argsort(-mh_p)
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
        log('P3: k=%d heads %s margin %.5f p=%.5f'
            % (k_best + 1, p3['heads_topk'], obs_max, p3_val), lines)

        # P4 per-head spectra aggregation (descriptive)
        spec_mean = spec_h.mean(axis=0)   # (NH, 4)
        names = ['t12', 'PR', 'zf_gain', 'zf_cos']
        rank_table = {}
        for si, nm in enumerate(names):
            ordh = np.argsort(-spec_mean[:, si])
            rank_table[nm] = {
                'top5': [{'head': int(h),
                          'val': round(float(spec_mean[h, si]), 5)}
                         for h in ordh[:5]],
                'rank_h7': int(np.where(ordh == 7)[0][0]) + 1,
                'rank_h8': int(np.where(ordh == 8)[0][0]) + 1,
                'spearman_vs_margins': round(
                    spearman(spec_mean[:, si], margins_h_2913), 4)}
        p4 = {'per_head_layer_mean': {
                  nm: [round(float(spec_mean[h, si]), 5)
                       for h in range(NH)]
                  for si, nm in enumerate(names)},
              'rankings': rank_table,
              'h7_h8_profile': {
                  nm: {'h7': round(float(spec_mean[7, si]), 5),
                       'h8': round(float(spec_mean[8, si]), 5),
                       'mean32': round(float(spec_mean[:, si]
                                            .mean()), 5)}
                  for si, nm in enumerate(names)}}
        log('P4: h7/h8 ranks %s' % json.dumps(
            {nm: {'h7': rank_table[nm]['rank_h7'],
                  'h8': rank_table[nm]['rank_h8'],
                  'rho_margins': rank_table[nm][
                      'spearman_vs_margins']}
             for nm in names}), lines)

        # ---------- verdict ----------
        if (rho >= 0.999 and top2_equal and p1_gate and recon_ok):
            verdict = 'head_identity_reproduced'
        elif rho >= 0.9:
            verdict = 'margin_spectrum_reproduced_identity_shifted'
        elif rho >= 0.5:
            verdict = 'head_ordering_partially_reproduced'
        else:
            verdict = 'head_ordering_not_reproduced'

    log('==== VERDICT: %s ====' % verdict, lines)

    res = {
        'phase': 2914, 'model': 'qwen3-4b', 'prereg': PREREG,
        'anchors': {'e1_block_identity': e1,
                    'e2in_outdomain_descr': e2in,
                    'e2cross_vs_2903_descr': e2cross,
                    'e2run_outdomain_vs_2903_descr': e2run,
                    'margin_recomp_runB': round(marg_agg, 6),
                    'margin_stored_2903': float(m03['margin']),
                    'e3_margin': e3m,
                    'acc_recomp_runB': round(acc_agg, 6),
                    'acc_stored': acc03,
                    'e3_acc_words': e3a * 57,
                    'a1_ok': a1_ok, 'a2_ok': a2_ok,
                    'a3_worst_diff': float('%.3e' % a3_worst),
                    'a3_layers': a3_layers, 'a3_ok': a3_ok,
                    'e4a_zf_identity': float('%.3e' % e4a),
                    'e4b_thin_svd_selfcheck': float('%.3e' % e4b),
                    'a4_ok': a4_ok,
                    'relB_vs_2913': float('%.3e' % relB),
                    'rel_agg_vs_2913': float('%.3e' % rel_agg_run),
                    'ok': anchor_ok},
        'P1': p1, 'P2': p2, 'P3': p3, 'P4': p4,
        'final_verdict': verdict,
        'runtime_s': round(time.monotonic() - t0, 1),
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    save = {'B_heads_runB': B_heads.astype(np.float32),
            'B_agg_runB': B_agg.astype(np.float32),
            'B_out_runB': B_out.astype(np.float32),
            'margins_h_runB': (marg_h.astype(np.float32)
                               if marg_h is not None
                               else np.zeros(0, np.float32)),
            'margins_h_2913': margins_h_2913.astype(np.float32),
            'flips_h_runB': (flips_h if flips_h is not None
                             else np.zeros(0, np.int64)),
            'flips_h_2913': flips_h_2913,
            'spec_h': spec_h.astype(np.float32),
            'labels_lang': lab_lang,
            'words': np.array(['%s:%s:%s' % w for w in words],
                              dtype=object)}
    np.savez_compressed(
        os.path.join(OUT, 'head_identity_replication.npz'), **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2914 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
