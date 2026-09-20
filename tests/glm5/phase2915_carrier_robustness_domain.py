# -*- coding: utf-8 -*-
"""Phase 2915: carrier identity robustness domain (windows x
direction family).

Why: 2913/2914 established the qwen attn margin carriers h7/h8
with EXACT cross-run reproduction (Spearman/Kendall 1.0, {7,8}
reconstruction absdiff 0.0, relB 3.16e-08) and showed the carrier
identity has NO static spectral signature (P4: spectra-vs-margins
rho ~ 0) - identity is runtime-response-determined.  Open
question (2914 MEMO, next-phase candidate A): is the {7,8}
identity stable across protocol parameters (window, direction
family) or a window-local phenomenon?  Since identity lives in
the data-dependent response, changing the window / direction
family changes the response and re-asks carrier selection.

Mode: forward per-head jacobian (2913 v2 protocol) across 5
variants:
  V1 [26,36) + 2886 layer-matched class-diff dirs (2913 REF)
  V2 [20,26) + 2886 dirs
  V3 [30,36) + 2886 dirs
  V4 [16,26) + 2886 dirs
  V5 [26,36) + FIXED 2887 global lang_dir (stale-direction
     family control; descriptive only, NOT in the verdict axis)

Protocol (per variant): SEED=2896, eps=1.0, pos 1, 57 words
verbatim 2887, conds same/func/null, null_tids rng order,
o_proj-input pre-hook capture, B_h = same - 0.5 func - 0.5 null
projected via W_O_h, family gate = p95 of max_h margin over 200
SEED=2896 label permutations (Sm fixed + mask permutation,
2903 caliber).

Anchors (frozen):
  a1 per-variant block-vs-full o_proj input identity < 1e-9
     (max over variants).
  a2 V1 margin vs 2903 stored < 5e-3 AND acc within 2/57
     (reference caliber).
  a3 V1 margins_h vs 2913 npz margins_h: Spearman >= 0.9999 AND
     max rel < 1e-5 (2914 measured 1.0 / 3.16e-08).

Probes (frozen):
  P1 per-variant: top2/top5 head sets, h7/h8 margins and ranks,
     per-variant family gate (max_h > p95).
  P2 cross-variant Spearman matrix of margins_h (5x5 incl V5)
     + vs-2913 column.
  P3 V5 stale-direction control: top2, h7/h8 ranks (descriptive;
     the 2903 readout spectrum predicts stale directions may
     weaken/shift carriers).
  P4 descriptive tables (margins per variant for h7/h8/top5).

Adjudication (frozen):
  anchor fail => anchor_fail_all_void;
  n_top2 = #{V in V1..V4 : {7,8} subset of top2(V)}:
    n_top2 == 4 => head_identity_protocol_general;
    n_top2 >= 2 => head_identity_broadly_stable;
    n_top2 == 1 => head_identity_partially_local;
    n_top2 == 0 => head_identity_window_local.
  n_top5 = #{V in V1..V4 : {7,8} subset of top5(V)} registered
  alongside (does not enter the label).

Output: phase2915/carrier_robustness_domain/.
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
OUT = os.path.join(BASE, 'phase2915', 'carrier_robustness_domain')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2915_run_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
SEED = 2896
EPS = 1.0
VOCAB = 151936
N_PERM = 200
NH, HD, NKV = 32, 128, 8
WINDOWS = {'V1': (26, 36), 'V2': (20, 26),
           'V3': (30, 36), 'V4': (16, 26)}
VARIANTS = ['V1', 'V2', 'V3', 'V4', 'V5']
TOP2_REF = [7, 8]

PREREG = {
    'mode': 'forward per-head jacobian across protocol variants '
            '(windows x direction family)',
    'question': '2914 candidate A: is the {7,8} carrier identity '
                'protocol-parameter-stable or window-local; '
                'identity is runtime-response-determined (2914 '
                'P4), so the response is re-probed across windows '
                'and a stale direction family',
    'variants': 'V1 [26,36) 2886 layer-matched class-diff dirs '
                '(2913 REF); V2 [20,26); V3 [30,36); V4 [16,26); '
                'V5 [26,36) with FIXED 2887 global lang_dir '
                '(stale-direction control, descriptive)',
    'protocol': 'SEED=2896, eps=1.0, pos 1, 57 words verbatim '
                '2887, conds same/func/null, null_tids rng order, '
                'o_proj-input capture, 200 SEED=2896 label perms '
                '(Sm fixed + mask perm) per variant',
    'anchors': {
        'a1': 'per-variant block-vs-full identity < 1e-9 (max)',
        'a2': 'V1 margin vs 2903 stored < 5e-3 AND acc <= 2/57',
        'a3': 'V1 margins vs 2913 npz: Spearman >= 0.9999 AND '
              'max rel < 1e-5',
    },
    'probes': {
        'P1': 'per-variant top2/top5, h7/h8 margins+ranks, '
              'per-variant family gate',
        'P2': 'cross-variant Spearman matrix (incl vs-2913)',
        'P3': 'V5 stale-direction control (descriptive)',
        'P4': 'descriptive margin tables',
    },
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'n_top2({7,8} both in top2, V1-V4) == 4 => '
               'head_identity_protocol_general; >= 2 => '
               'head_identity_broadly_stable; == 1 => '
               'head_identity_partially_local; 0 => '
               'head_identity_window_local; n_top5 registered '
               'alongside',
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
    ra = rankdata(a)
    rb = rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    return float((ra @ rb)
                 / max(float(np.sqrt((ra @ ra) * (rb @ rb))), 1e-30))


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2915,
                   'name': 'carrier_robustness_domain',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8': sha8(os.path.abspath(__file__)),
                   'sources': {'s2886': sha8(SRC_2886),
                               's2887': sha8(SRC_2887),
                               's2903': sha8(SRC_2903),
                               'r2903': sha8(R2903),
                               's2913': sha8(SRC_2913)},
                   'model': 'qwen3-4b',
                   'heads': NH, 'head_dim': HD,
                   'eps': EPS, 'windows': WINDOWS,
                   'seed': SEED, 'n_perm': N_PERM,
                   'top2_ref': TOP2_REF, 'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    # ---------- sources ----------
    z87 = np.load(SRC_2887, allow_pickle=True)
    words = [tuple(str(w).split(':')) for w in z87['words']]
    lab_lang = np.asarray(z87['labels_lang']).astype(int)
    lang_dir = z87['lang_dir'].astype(np.float64)
    n_words = len(words)
    assert n_words == 57

    z86 = np.load(SRC_2886, allow_pickle=True)
    S_last = z86['S_last'].astype(np.float64)
    lab_sent = np.asarray(z86['labels']).astype(int)
    assert all(int(lab_sent[i]) == i % 2 for i in range(80))
    diffs = S_last[lab_sent == 0].mean(0) \
        - S_last[lab_sent == 1].mean(0)

    dirs_var = {}
    for vn, (lo, hi) in WINDOWS.items():
        dirs_var[vn] = np.stack([unit(diffs[li])
                                 for li in range(lo, hi)])
    dirs_var['V5'] = np.stack([unit(lang_dir)] * 10)
    n_win_v = {vn: dirs_var[vn].shape[0] for vn in VARIANTS}
    lay_of = {}
    for vn in VARIANTS:
        lo, hi = WINDOWS.get(vn, (26, 36))
        lay_of[vn] = list(range(lo, hi))

    z03 = np.load(SRC_2903, allow_pickle=True)
    r03 = json.load(io.open(R2903, encoding='utf-8'))
    m03 = r03['margins']['attn']
    acc03 = float(m03['acc'])

    z13 = np.load(SRC_2913, allow_pickle=True)
    margins_h_2913 = z13['margins_h'].astype(np.float64)
    log('sources ok; 2913 margins top5 %s'
        % [(int(h), round(float(margins_h_2913[h]), 5))
           for h in np.argsort(-margins_h_2913)[:5]], lines)

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

    all_layers = sorted({li for vn in VARIANTS
                         for li in lay_of[vn]})
    for li in all_layers:
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

    vdt = next(layers[all_layers[0]].mlp.parameters()).dtype

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
        c = np.asarray(cap['oprin'][li][0][0, 1]) \
            .astype(np.float64)
        return c

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

    # ---------- forward capture across variants ----------
    r_c = {vn: {cn: np.zeros((n_words, n_win_v[vn], NH, HD),
                             dtype=np.float32)
                for cn in ('same', 'func', 'null')}
           for vn in VARIANTS}
    for i, (_, _, w) in enumerate(words):
        w_tid = tid_map[w]
        conds = {'same': [tid_map[words[same_ctx(i)][2]], w_tid],
                 'func': [func_tid, w_tid],
                 'null': [null_tids[i], w_tid]}
        for cn, toks in conds.items():
            attnin_all = forward2(toks)
            for vn in VARIANTS:
                for q, li in enumerate(lay_of[vn]):
                    d = dirs_var[vn][q]
                    x = attnin_all[li]
                    ref = attn_call(li, x)
                    xp = x.copy()
                    xp[0, 1] = xp[0, 1] + EPS * d
                    pert = attn_call(li, xp)
                    r_c[vn][cn][i, q] = \
                        ((pert - ref) / EPS) \
                        .reshape(NH, HD).astype(np.float32)
        if (i + 1) % 10 == 0:
            log('words [%d/%d] (all variants)' % (i + 1, n_words),
                lines)

    # ---------- per-variant decomposition + anchors ----------
    Wo_cache = {}

    def Wo_of(li):
        if li not in Wo_cache:
            Wo_cache[li] = layers[li].self_attn.o_proj.weight \
                .detach().float().cpu().numpy().astype(np.float64)
        return Wo_cache[li]

    same_m, diff_m = masks(lab_lang)
    B_heads_v = {}
    B_agg_v = {}
    e1_v = {}
    marg_v = {}
    acc_v = {}
    for vn in VARIANTS:
        r_comb = (r_c[vn]['same'].astype(np.float64)
                  - 0.5 * r_c[vn]['func'].astype(np.float64)
                  - 0.5 * r_c[vn]['null'].astype(np.float64))
        nwin = n_win_v[vn]
        G = np.zeros((nwin, NH * HD))
        for q, li in enumerate(lay_of[vn]):
            G[q] = Wo_of(li).T @ dirs_var[vn][q]
        G3 = G.reshape(nwin, NH, HD)
        Bh = np.einsum('nqhk,qhk->hnq', r_comb, G3)
        Ba = np.einsum('nqf,qf->nq',
                       r_comb.reshape(n_words, nwin, NH * HD), G)
        B_heads_v[vn] = Bh
        B_agg_v[vn] = Ba
        e1_v[vn] = float(np.abs(Bh.sum(0) - Ba).max()
                         / max(float(np.abs(Ba).max()), 1e-30))
        U = rownorm(Ba)
        Sm = U @ U.T
        marg_v[vn] = margin_of(Sm, same_m, diff_m)
        acc_v[vn] = loo_acc(zmat(Ba), lab_lang)
    a1 = max(e1_v.values())
    e3m = abs(marg_v['V1'] - float(m03['margin']))
    e3a = abs(acc_v['V1'] - acc03)
    a2_ok = bool(e3m < 5e-3 and e3a <= 2.0 / 57.0)
    # a3: V1 vs 2913 npz
    U1 = rownorm(B_heads_v['V1'][7] + B_heads_v['V1'][8])
    m78_V1 = margin_of(U1 @ U1.T, same_m, diff_m)
    Sm_h_V1 = np.stack([rownorm(B_heads_v['V1'][h])
                        @ rownorm(B_heads_v['V1'][h]).T
                        for h in range(NH)])
    marg_h_V1 = np.array([margin_of(Sm_h_V1[h], same_m, diff_m)
                          for h in range(NH)])
    rho_2913 = spearman(marg_h_V1, margins_h_2913)
    rel_2913 = float(np.abs(marg_h_V1 - margins_h_2913).max()
                     / max(float(np.abs(margins_h_2913).max()),
                           1e-30))
    a3_ok = bool(rho_2913 >= 0.9999 and rel_2913 < 1e-5)
    anchor_ok = bool(a1 < 1e-9 and a2_ok and a3_ok)
    log('a1 max %.2e | a2 V1 margin %.5f (stored %.5f, %.1f words) '
        'ok=%s | a3 spearman %.6f rel %.2e ok=%s | m78_V1 %.6f '
        '(2914 ref 0.28036)'
        % (a1, marg_v['V1'], float(m03['margin']), e3a * 57, a2_ok,
           rho_2913, rel_2913, a3_ok, m78_V1), lines)

    # ---------- probes ----------
    verdict = None
    p1 = p2 = p3 = p4 = None
    n_top2 = n_top5 = None
    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        # shared permutation masks (labels only; Sm fixed per head)
        rng2 = np.random.default_rng(SEED)
        perms = [rng2.permutation(lab_lang) for _ in range(N_PERM)]
        perm_masks = [masks(pl) for pl in perms]

        p1 = {}
        gates = {}
        p95s = {}
        top2s = {}
        top5s = {}
        ranks78 = {}
        for vn in VARIANTS:
            Sm_h = np.stack([rownorm(B_heads_v[vn][h])
                             @ rownorm(B_heads_v[vn][h]).T
                             for h in range(NH)])
            mh = np.array([margin_of(Sm_h[h], same_m, diff_m)
                           for h in range(NH)])
            max_null = np.zeros(N_PERM)
            for pi, (sm_p, df_p) in enumerate(perm_masks):
                max_null[pi] = max(margin_of(Sm_h[h], sm_p, df_p)
                                   for h in range(NH))
            p95 = float(np.percentile(max_null, 95))
            p95s[vn] = p95
            order = np.argsort(-mh)
            top2s[vn] = [int(x) for x in order[:2]]
            top5s[vn] = [int(x) for x in order[:5]]
            ranks78[vn] = {'h7': int(np.where(order == 7)[0][0]) + 1,
                           'h8': int(np.where(order == 8)[0][0]) + 1}
            gates[vn] = bool(mh.max() > p95)
            p1[vn] = {
                'top2': top2s[vn], 'top5': top5s[vn],
                'margins_top5': [{'head': int(h),
                                  'margin': round(float(mh[h]), 5)}
                                 for h in order[:5]],
                'h7_margin': round(float(mh[7]), 5),
                'h8_margin': round(float(mh[8]), 5),
                'h7_rank': ranks78[vn]['h7'],
                'h8_rank': ranks78[vn]['h8'],
                'p95_max_null': round(p95, 5),
                'gate_present': gates[vn],
                'channel_margin': round(marg_v[vn], 5),
                'channel_acc': round(acc_v[vn], 5)}
            log('P1 %s: top2 %s top5 %s | h7 %.5f (rank %d) h8 '
                '%.5f (rank %d) | gate %s (p95 %.5f) | chan '
                'margin %.5f acc %.5f'
                % (vn, top2s[vn], top5s[vn], mh[7],
                   ranks78[vn]['h7'], mh[8], ranks78[vn]['h8'],
                   gates[vn], p95, marg_v[vn], acc_v[vn]), lines)

        # P2 cross-variant Spearman matrix
        mat = {}
        for a in VARIANTS:
            ma = np.array([margin_of(
                rownorm(B_heads_v[a][h])
                @ rownorm(B_heads_v[a][h]).T, same_m, diff_m)
                for h in range(NH)])
            row = {}
            for b in VARIANTS:
                mb = np.array([margin_of(
                    rownorm(B_heads_v[b][h])
                    @ rownorm(B_heads_v[b][h]).T, same_m, diff_m)
                    for h in range(NH)])
                row[b] = round(spearman(ma, mb), 4)
            row['s2913'] = round(spearman(ma, margins_h_2913), 4)
            mat[a] = row
        p2 = {'spearman_matrix': mat}
        log('P2 spearman vs V1: %s | V2-vs-V4 %s'
            % (mat['V1'], mat['V2']['V4']), lines)

        # P3 V5 control highlights (already in p1 tables)
        p3 = {'v5_top2': top2s['V5'],
              'v5_h7_rank': ranks78['V5']['h7'],
              'v5_h8_rank': ranks78['V5']['h8'],
              'v5_gate': gates['V5'],
              'note': 'stale fixed lang_dir family, descriptive'}

        # verdict axis on V1-V4
        n_top2 = sum(1 for vn in ('V1', 'V2', 'V3', 'V4')
                     if set(TOP2_REF).issubset(set(top2s[vn])))
        n_top5 = sum(1 for vn in ('V1', 'V2', 'V3', 'V4')
                     if set(TOP2_REF).issubset(set(top5s[vn])))
        p4 = {'n_top2_of4': n_top2, 'n_top5_of4': n_top5,
              'gates': {vn: gates[vn] for vn in VARIANTS},
              'h7h8_margins': {vn: [p1[vn]['h7_margin'],
                                    p1[vn]['h8_margin']]
                               for vn in VARIANTS}}
        log('P4: n_top2=%d/4 n_top5=%d/4 gates=%s'
            % (n_top2, n_top5, gates), lines)

        if n_top2 == 4:
            verdict = 'head_identity_protocol_general'
        elif n_top2 >= 2:
            verdict = 'head_identity_broadly_stable'
        elif n_top2 == 1:
            verdict = 'head_identity_partially_local'
        else:
            verdict = 'head_identity_window_local'

    log('==== VERDICT: %s ====' % verdict, lines)

    res = {
        'phase': 2915, 'model': 'qwen3-4b', 'prereg': PREREG,
        'anchors': {'a1_max_block_identity': a1,
                    'e1_per_variant': {vn: float('%.3e'
                                               % e1_v[vn])
                                       for vn in VARIANTS},
                    'a2_v1_margin': round(marg_v['V1'], 6),
                    'margin_stored_2903': float(m03['margin']),
                    'e3_margin': e3m,
                    'acc_v1': round(acc_v['V1'], 6),
                    'e3_acc_words': e3a * 57, 'a2_ok': a2_ok,
                    'a3_spearman_vs_2913': round(rho_2913, 6),
                    'a3_rel_vs_2913': float('%.3e' % rel_2913),
                    'a3_ok': a3_ok,
                    'm78_V1': round(m78_V1, 6),
                    'ok': anchor_ok},
        'P1': p1, 'P2': p2, 'P3': p3, 'P4': p4,
        'n_top2_of4': n_top2, 'n_top5_of4': n_top5,
        'final_verdict': verdict,
        'runtime_s': round(time.monotonic() - t0, 1),
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    save = {'labels_lang': lab_lang,
            'words': np.array(['%s:%s:%s' % w for w in words],
                              dtype=object),
            'margins_h_2913': margins_h_2913.astype(np.float32)}
    for vn in VARIANTS:
        save['B_heads_' + vn] = B_heads_v[vn].astype(np.float32)
    save['margins_V'] = np.stack([
        np.array([margin_of(rownorm(B_heads_v[vn][h])
                            @ rownorm(B_heads_v[vn][h]).T,
                            same_m, diff_m)
                  for h in range(NH)])
        for vn in VARIANTS]).astype(np.float32)
    save['variant_order'] = np.array(VARIANTS, dtype=object)
    np.savez_compressed(
        os.path.join(OUT, 'carrier_robustness_domain.npz'), **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2915 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
