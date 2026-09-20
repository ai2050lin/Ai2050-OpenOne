# -*- coding: utf-8 -*-
"""Phase 2930: direction-flip control (mirrored word probe).

Why: 2928 found L+ class 0/4 survivor wipe-out and flagged a
possible word-probe DIRECTION BIAS (dirs_word = lab0 - lab1).
2927 declared sign convention irrelevant (sign-Gram margin is
sign-flip invariant), but that argument is about the margin
STATISTIC; the deeper fact is that B = einsum(r(d), Wo.T @ d)
is a QUADRATIC form in the probe direction - under exact
linearity a full mirror (inject -d, readout Wo.T @ (-d)) leaves
B unchanged (double sign flip cancels). At finite eps = 1.0 the
second-order term H[d,d] breaks this: B_mirror - Bwd measures
the direction-convention sensitivity of the ENTIRE event/skeleton
machinery.

Question (2929 candidate A): is the 49-event atlas and the
2929 probe-invariant skeleton robust to the word-probe direction
sign convention, or convention-relative?

Mode: one forward pass (qwen3-4b), 2917/2927 protocol verbatim
(SEED=2896, eps=1.0, pos 1, 57 words verbatim 2887, conds
same/func/null, null_tids rng order, o_proj-input capture, all
36 layers), pass 2 with THREE directions:
  +dirs86   - 2886 sentence caliber (anchor chain a1-a4)
  +dirs_word- word probe rebuilt from THIS run's func captures
              (anchors a5-a7 vs 2927 npz)
  -dirs_word- MIRROR probe (new data; dirs_neg == -dirs_word
              asserted bit-exact, G_neg == -G asserted)

Anchors (frozen):
  a1 B86[:,:,26:36] vs 2913 npz max rel < 1e-5
  a2 |margin({7,8} on [26,36)) - 0.28036| < 5e-3
  a3 sign_M86 vs 2917 npz max abs diff < 1e-4
  a4 maxT event set (p_maxT86 <= 0.05) == 2917 24-event set
  a5 dirs_word(this run) vs 2927 npz dirs_word max abs < 1e-5
  a6 Bwd(this run) vs 2927 npz B_word max rel < 1e-4
  a7 Ewd(this run) == Ewd(2927 npz p_maxT_word <= 0.05) AND
     p_maxTwd vs npz max abs diff < 1e-6
  a8 dirs_neg == -dirs_word AND G_neg == -Gwd (bit-exact,
     construction assertion)

P1 (main, frozen verdict map):
  maxT (200 perms rng2 2896 shared) on B_mirror -> E_mirror.
  anchor fail                            => anchor_fail_all_void
  E_mirror == Ewd(2927)                  => direction_flip_margin_invariant
  E_mirror != Ewd(2927)                  => direction_flip_margin_shifts
  (sign_M_mirror vs sign_Mwd max abs diff registered
   descriptive; margin is noise-limited at ~1e-6 run level)

P2 mirror deviation (descriptive):
  r-level linearity: ||r_comb_neg + r_comb_w||_F / ||r_comb_w||_F
  per layer; B-level: mir_err[h,l] = ||B_mirror-Bwd||_2 /
  max(||Bwd||_2, 1e-30) grid; rho_mirror grid vs 2929 rho_grid
  dev distribution.

P3 skeleton convention-invariance (zero forward, frozen):
  layer-wise head-permutation null replayed with rng base 2904
  (2000/layer) on Bwd ranks AND on B_mirror ranks;
  skeleton_mirror = rho_mirror >= null p95(mirror);
  jaccard(skeleton_mirror, skeleton_2929) + L0 forensics.

P4 survivor/lost annotation (descriptive): survivor 7 / lost 17
(2928 seal lists verbatim) rho_2929 vs rho_mirror; L+ class
{(26,6),(25,3),(24,23),(27,24)} highlighted.

Output: phase2930/direction_flip_control/.
"""
import hashlib
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
SRC_2913 = os.path.join(BASE, 'phase2913',
                        'perhead_wvo_decomposition',
                        'perhead_wvo_decomposition.npz')
SRC_2917 = os.path.join(BASE, 'phase2917', 'event_atlas',
                        'event_atlas.npz')
SRC_2927 = os.path.join(BASE, 'phase2927', 'probe_relativity',
                        'probe_relativity.npz')
SRC_2929 = os.path.join(BASE, 'phase2929',
                        'response_structure_atlas',
                        'response_structure_atlas.npz')
OUT = os.path.join(BASE, 'phase2930', 'direction_flip_control')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2930_run_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
SEED = 2896
EPS = 1.0
VOCAB = 151936
N_PERM = 200
NH, HD = 32, 128
NL = 36
FDR_Q = 0.05
TOP1 = (7, 19)
RECON_REF = 0.28036
CONDS = ('same', 'func', 'null')
N_PERM_L = 2000
RNG_PERM = 2904
SURV = [(1, 6), (5, 6), (7, 19), (8, 2), (14, 9), (20, 8),
        (21, 6)]
LOST = [(1, 4), (4, 1), (4, 19), (4, 22), (6, 19), (13, 22),
        (15, 13), (17, 28), (21, 16), (22, 12), (24, 9),
        (24, 23), (25, 3), (26, 5), (26, 6), (27, 24), (28, 1)]
L_POS = [(24, 23), (25, 3), (26, 6), (27, 24)]

PREREG = {
    'mode': 'forward per-head jacobian full-layer scan with '
            'THREE directions (+dirs86 anchor, +dirs_word '
            'repro anchor, -dirs_word mirror), 2917/2927 '
            'protocol verbatim otherwise',
    'question': '2929 candidate A: is the 49-event atlas and '
                'the 2929 probe-invariant skeleton robust to '
                'the word-probe DIRECTION SIGN convention, or '
                'convention-relative? B = einsum(r(d), Wo.T@d) '
                'is quadratic in d: exact linearity predicts '
                'B_mirror == Bwd (double flip cancels); '
                'finite-eps H[d,d] term breaks this and the '
                'break size is the convention sensitivity.',
    'mirror': 'dirs_neg = -dirs_word (bit-exact assertion); '
              'inject -dirs_word, readout G_neg = Wo.T @ '
              'dirs_neg; r_comb_neg = same - 0.5 func - 0.5 '
              'null; B_mirror = einsum(r_comb_neg, G_neg)',
    'anchors': {
        'a1': 'B86[:,:,26:36] vs 2913 npz max rel < 1e-5',
        'a2': '|m78(2886 caliber) - 0.28036| < 5e-3',
        'a3': 'sign_M86 vs 2917 npz max abs diff < 1e-4',
        'a4': 'maxT event set (p<=0.05) == 2917 24-event set',
        'a5': 'dirs_word(this run) vs 2927 npz max abs < 1e-5',
        'a6': 'Bwd(this run) vs 2927 npz B_word max rel < 1e-4',
        'a7': 'Ewd(this run) == Ewd(2927) AND p_maxTwd diff '
              '< 1e-6',
        'a8': 'dirs_neg == -dirs_word AND G_neg == -Gwd '
              '(bit-exact)',
    },
    'P1': 'maxT (200 perms rng2 2896 shared) on B_mirror -> '
          'E_mirror; E_mirror == Ewd(2927) => '
          'direction_flip_margin_invariant; else '
          'direction_flip_margin_shifts',
    'P2': 'r-level ||r_neg + r_w||_F/||r_w||_F per layer; '
          'B-level mir_err grid; rho_mirror vs 2929 rho dev',
    'P3': 'layer-wise head-permutation null rng 2904 replayed '
          'on Bwd and B_mirror ranks; skeleton_mirror vs '
          'skeleton_2929 jaccard; L0 forensics',
    'P4': 'survivor 7 / lost 17 rho_2929 vs rho_mirror; L+ '
          'class {(26,6),(25,3),(24,23),(27,24)} highlighted',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'E_mirror == Ewd(2927) => '
               'direction_flip_margin_invariant; else => '
               'direction_flip_margin_shifts',
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


def masks(lab):
    n = len(lab)
    eye = np.eye(n, dtype=bool)
    same = (lab[:, None] == lab[None, :]) & (~eye)
    diff = (~eye) & (~same)
    return same, diff


def gram_margin(B, same, diff):
    U = rownorm(B)
    Sm = U @ U.T
    return float(Sm[same].mean() - Sm[diff].mean())


def avg_ranks(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x))
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(x, y):
    rx = avg_ranks(x)
    ry = avg_ranks(y)
    if rx.std() < 1e-30 or ry.std() < 1e-30:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def centered(M):
    return M - M.mean(axis=1, keepdims=True)


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2930,
                   'name': 'direction_flip_control',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2886': sha8(SRC_2886),
                               's2887': sha8(SRC_2887),
                               's2913': sha8(SRC_2913),
                               's2917': sha8(SRC_2917),
                               's2927': sha8(SRC_2927),
                               's2929': sha8(SRC_2929)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'eps': EPS, 'seed': SEED, 'n_perm': N_PERM,
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
    diffs = S_last[lab_sent == 0].mean(0) \
        - S_last[lab_sent == 1].mean(0)
    dirs86 = np.stack([unit(diffs[li]) for li in range(NL)])

    z13 = np.load(SRC_2913, allow_pickle=True)
    B_heads_2913 = z13['B_heads'].astype(np.float64)
    z17 = np.load(SRC_2917, allow_pickle=True)
    sign_M_17 = z17['sign_M'].astype(np.float64)
    p_maxT_17 = z17['p_maxT'].astype(np.float64)
    E17 = set((h, li) for h in range(NH) for li in range(NL)
              if p_maxT_17[h, li] <= FDR_Q)
    z27 = np.load(SRC_2927, allow_pickle=True)
    dirs_word_27 = z27['dirs_word'].astype(np.float64)
    Bwd_27 = z27['B_word'].astype(np.float64)
    p_maxTwd_27 = z27['p_maxT_word'].astype(np.float64)
    Ewd_27 = set((h, li) for h in range(NH) for li in range(NL)
                 if p_maxTwd_27[h, li] <= FDR_Q)
    z29 = np.load(SRC_2929, allow_pickle=True)
    rho_grid_29 = z29['rho_grid'].astype(np.float64)
    skel_29 = z29['skel_mask'].astype(bool)
    log('sources ok (E17 n=%d, Ewd27 n=%d)'
        % (len(E17), len(Ewd_27)), lines)

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

    for li in range(NL):
        handles.append(
            layers[li].self_attn.register_forward_pre_hook(
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

    vdt = next(layers[0].mlp.parameters()).dtype

    def layer_dev(li):
        return next(layers[li].mlp.parameters()).device

    rotary = model.model.rotary_emb

    def attn_call(li, X):
        t = torch.tensor(X, device=layer_dev(li), dtype=vdt)
        position_ids = torch.arange(
            t.shape[1], device=layer_dev(li)).unsqueeze(0)
        cap['oprin'].pop(li, None)
        state['capture'] = True
        with torch.no_grad():
            pos_emb = rotary(t, position_ids)
            o = layers[li].self_attn(
                t, position_embeddings=pos_emb,
                attention_mask=None, past_key_values=None)
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

    # ---------- pass 1 ----------
    attn_store = {}
    for i, (_, _, w) in enumerate(words):
        w_tid = tid_map[w]
        conds = {'same': [tid_map[words[same_ctx(i)][2]], w_tid],
                 'func': [func_tid, w_tid],
                 'null': [null_tids[i], w_tid]}
        for cn, toks in conds.items():
            attnin_all = forward2(toks)
            for li in range(NL):
                attn_store[(i, cn, li)] = \
                    attnin_all[li].astype(np.float32)
        if (i + 1) % 20 == 0:
            log('pass1 words [%d/%d]' % (i + 1, n_words), lines)
    log('pass1 done (%d sequences)' % (3 * n_words), lines)

    # ---------- word probe rebuilt ----------
    d_dim = attn_store[(0, 'func', 0)].shape[-1]
    diffs_w = np.zeros((NL, d_dim))
    for li in range(NL):
        X = np.stack([attn_store[(i, 'func', li)][0, 1]
                      for i in range(n_words)]).astype(np.float64)
        diffs_w[li] = X[lab_lang == 0].mean(0) \
            - X[lab_lang == 1].mean(0)
    dirs_word = np.stack([unit(diffs_w[li]) for li in range(NL)])
    a5_diff = float(np.abs(dirs_word - dirs_word_27).max())
    a5_ok = bool(a5_diff < 1e-5)
    dirs_neg = -dirs_word
    cos_dw = [float(dirs_word[li] @ dirs86[li])
              for li in range(NL)]
    log('dirs_word rebuilt | a5 diff %.2e ok=%s | '
        'cos(dirs_word, dirs86) median %.4f'
        % (a5_diff, a5_ok, float(np.median(cos_dw))), lines)

    # ---------- pass 2: three directions ----------
    r_c86 = {cn: np.zeros((n_words, NL, NH, HD), np.float32)
             for cn in CONDS}
    r_cw = {cn: np.zeros((n_words, NL, NH, HD), np.float32)
            for cn in CONDS}
    r_cn = {cn: np.zeros((n_words, NL, NH, HD), np.float32)
            for cn in CONDS}
    for i in range(n_words):
        for cn in CONDS:
            for li in range(NL):
                x = attn_store[(i, cn, li)].astype(np.float64)
                ref = attn_call(li, x)
                xp = x.copy()
                xp[0, 1] = xp[0, 1] + EPS * dirs86[li]
                pert = attn_call(li, xp)
                r_c86[cn][i, li] = ((pert - ref) / EPS) \
                    .reshape(NH, HD).astype(np.float32)
                xp2 = x.copy()
                xp2[0, 1] = xp2[0, 1] + EPS * dirs_word[li]
                pert2 = attn_call(li, xp2)
                r_cw[cn][i, li] = ((pert2 - ref) / EPS) \
                    .reshape(NH, HD).astype(np.float32)
                xp3 = x.copy()
                xp3[0, 1] = xp3[0, 1] + EPS * dirs_neg[li]
                pert3 = attn_call(li, xp3)
                r_cn[cn][i, li] = ((pert3 - ref) / EPS) \
                    .reshape(NH, HD).astype(np.float32)
        if (i + 1) % 10 == 0:
            log('pass2 words [%d/%d]' % (i + 1, n_words), lines)
    log('pass2 done (3 directions)', lines)

    # ---------- B matrices ----------
    G86 = np.zeros((NL, NH * HD))
    Gwd = np.zeros((NL, NH * HD))
    Gneg = np.zeros((NL, NH * HD))
    for li in range(NL):
        Wo_li = layers[li].self_attn.o_proj.weight.detach() \
            .float().cpu().numpy().astype(np.float64)
        G86[li] = Wo_li.T @ dirs86[li]
        Gwd[li] = Wo_li.T @ dirs_word[li]
        Gneg[li] = Wo_li.T @ dirs_neg[li]
    a8_gdiff = float(np.abs(Gneg + Gwd).max())
    a8_ok = bool(a8_gdiff == 0.0)
    comb_w = (r_cw['same'].astype(np.float64)
              - 0.5 * r_cw['func'].astype(np.float64)
              - 0.5 * r_cw['null'].astype(np.float64))
    comb_n = (r_cn['same'].astype(np.float64)
              - 0.5 * r_cn['func'].astype(np.float64)
              - 0.5 * r_cn['null'].astype(np.float64))
    B86 = np.einsum('nlhk,lhk->hnl',
                    (r_c86['same'].astype(np.float64)
                     - 0.5 * r_c86['func'].astype(np.float64)
                     - 0.5 * r_c86['null'].astype(np.float64)),
                    G86.reshape(NL, NH, HD))
    Bwd = np.einsum('nlhk,lhk->hnl', comb_w,
                    Gwd.reshape(NL, NH, HD))
    Bmir = np.einsum('nlhk,lhk->hnl', comb_n,
                     Gneg.reshape(NL, NH, HD))
    log('B86 %s | Bwd %s | Bmir %s | a8 G_neg==-Gwd: %s'
        % (B86.shape, Bwd.shape, Bmir.shape, a8_ok), lines)

    # ---------- anchors ----------
    blk = B86[:, :, 26:36]
    rel_a1 = float(np.abs(blk - B_heads_2913).max()
                   / max(float(np.abs(B_heads_2913).max()),
                         1e-30))
    a1_ok = bool(rel_a1 < 1e-5)
    same_m, diff_m = masks(lab_lang)
    m78 = gram_margin(blk[7] + blk[8], same_m, diff_m)
    a2_ok = bool(abs(m78 - RECON_REF) < 5e-3)

    def sign_matrix(B):
        out = np.zeros((NH, NL))
        for h in range(NH):
            for li in range(NL):
                s = np.sign(B[h, :, li])
                s[s == 0] = 1.0
                Gm = np.outer(s, s)
                out[h, li] = Gm[same_m].mean() \
                    - Gm[diff_m].mean()
        return out

    sign_M86 = sign_matrix(B86)
    a3_diff = float(np.abs(sign_M86 - sign_M_17).max())
    a3_ok = bool(a3_diff < 1e-4)
    log('a1 rel %.2e ok=%s | a2 m78 %.6f ok=%s | a3 %.2e ok=%s'
        % (rel_a1, a1_ok, m78, a2_ok, a3_diff, a3_ok), lines)

    # maxT machinery (2917 verbatim, shared perms)
    rng2 = np.random.default_rng(SEED)
    perms = [rng2.permutation(lab_lang) for _ in range(N_PERM)]
    perm_masks = [masks(pl) for pl in perms]

    def maxT_run(B):
        sign_M = sign_matrix(B)
        Gs = {}
        for h in range(NH):
            for li in range(NL):
                s = np.sign(B[h, :, li])
                s[s == 0] = 1.0
                Gs[(h, li)] = np.outer(s, s)
        max_perm = np.zeros(N_PERM)
        p_M = np.zeros((NH, NL))
        p_maxT = np.zeros((NH, NL))
        for h in range(NH):
            for li in range(NL):
                Gm = Gs[(h, li)]
                obs = sign_M[h, li]
                cnt = 0
                for pi, (sm_p, df_p) in enumerate(perm_masks):
                    v = Gm[sm_p].mean() - Gm[df_p].mean()
                    if v >= obs:
                        cnt += 1
                    if v > max_perm[pi]:
                        max_perm[pi] = v
                p_M[h, li] = float(cnt + 1) / (N_PERM + 1)
        for h in range(NH):
            for li in range(NL):
                p_maxT[h, li] = float(
                    np.sum(max_perm >= sign_M[h, li]) + 1) \
                    / (N_PERM + 1)
        return sign_M, p_M, p_maxT

    sign_M86b, p_M86, p_maxT86 = maxT_run(B86)
    E86 = set((h, li) for h in range(NH) for li in range(NL)
              if p_maxT86[h, li] <= FDR_Q)
    a4_ok = bool(E86 == E17)
    log('a4 maxT event set == 2917: %s (E86 n=%d)'
        % (a4_ok, len(E86)), lines)

    a6_diff = float(np.abs(Bwd - Bwd_27).max())
    a6_rel = float(a6_diff / max(float(np.abs(Bwd_27).max()),
                                 1e-30))
    a6_ok = bool(a6_rel < 1e-4)
    sign_Mwd, p_Mwd, p_maxTwd = maxT_run(Bwd)
    Ewd = set((h, li) for h in range(NH) for li in range(NL)
              if p_maxTwd[h, li] <= FDR_Q)
    a7_pdiff = float(np.abs(p_maxTwd - p_maxTwd_27).max())
    a7_ok = bool(Ewd == Ewd_27 and a7_pdiff < 1e-6)
    log('a6 Bwd vs 2927 rel %.2e ok=%s | a7 Ewd==2927: %s '
        'pdiff %.2e ok=%s'
        % (a6_rel, a6_ok, Ewd == Ewd_27, a7_pdiff, a7_ok),
        lines)
    anchor_ok = bool(a1_ok and a2_ok and a3_ok and a4_ok
                     and a5_ok and a6_ok and a7_ok and a8_ok)

    verdict = None
    p1 = p2 = p3 = p4 = None
    save = {}
    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        # ---------- P1: mirror maxT (main) -----------------
        sign_Mmir, p_Mmir, p_maxTmir = maxT_run(Bmir)
        Emir = set((h, li) for h in range(NH)
                   for li in range(NL)
                   if p_maxTmir[h, li] <= FDR_Q)
        smir_diff = float(np.abs(sign_Mmir - sign_Mwd).max())
        n_ov_mir = len(Emir & Ewd_27)
        if Emir == Ewd_27:
            verdict = 'direction_flip_margin_invariant'
        else:
            verdict = 'direction_flip_margin_shifts'
        lost_mir = sorted(Ewd_27 - Emir)
        new_mir = sorted(Emir - Ewd_27)
        p1 = {'n_events_mirror': len(Emir),
              'n_overlap_with_Ewd27': n_ov_mir,
              'jaccard': round(n_ov_mir
                               / max(len(Emir | Ewd_27), 1), 4),
              'sign_diff_max': float('%.3e' % smir_diff),
              'top1_in_Emir': bool(TOP1 in Emir),
              'lost_in_mirror': [{'cell': list(e)} for e
                                 in lost_mir],
              'new_in_mirror': [{'cell': list(e)} for e
                                in new_mir]}
        log('P1: E_mirror n=%d overlap %d/49 jacc %.3f | '
            'sign diff %.2e | top1 in: %s | VERDICT-BASIS: %s'
            % (len(Emir), n_ov_mir,
               n_ov_mir / max(len(Emir | Ewd_27), 1),
               smir_diff, TOP1 in Emir, verdict), lines)
        log('P1 lost_in_mirror: %s' % lost_mir, lines)
        log('P1 new_in_mirror: %s' % new_mir, lines)

        # ---------- P2: mirror deviation -------------------
        lin_r = np.zeros(NL)
        for li in range(NL):
            lin_r[li] = float(
                np.linalg.norm(comb_n[:, li] + comb_w[:, li])
                / max(np.linalg.norm(comb_w[:, li]), 1e-30))
        mir_err = np.linalg.norm(Bmir - Bwd, axis=1) \
            / np.maximum(np.linalg.norm(Bwd, axis=1), 1e-30)
        rho_mir = np.zeros((NH, NL))
        for h in range(NH):
            for li in range(NL):
                rho_mir[h, li] = spearman(B86[h, :, li],
                                          Bmir[h, :, li])
        dev_rho = np.abs(rho_mir - rho_grid_29)
        p2 = {'lin_r_median': round(float(np.median(lin_r)), 4),
              'lin_r_max': round(float(lin_r.max()), 4),
              'mir_err_median':
                  round(float(np.median(mir_err)), 4),
              'mir_err_p90':
                  round(float(np.percentile(mir_err, 90)), 4),
              'mir_err_max': round(float(mir_err.max()), 4),
              'rho_dev_median':
                  round(float(np.median(dev_rho)), 4),
              'rho_dev_max': round(float(dev_rho.max()), 4),
              'lin_r_profile':
                  [round(float(v), 4) for v in lin_r]}
        log('P2: lin_r median %.4f max %.4f | mir_err median '
            '%.4f p90 %.4f max %.4f | rho dev median %.4f max '
            '%.4f'
            % (np.median(lin_r), lin_r.max(),
               np.median(mir_err),
               np.percentile(mir_err, 90), mir_err.max(),
               np.median(dev_rho), dev_rho.max()), lines)

        # ---------- P3: skeleton convention-invariance -----
        rng3 = np.random.default_rng(RNG_PERM)
        Rwd_rank = {}
        for li in range(NL):
            Rwd_rank[li] = np.stack(
                [avg_ranks(Bwd[h, :, li]) for h in range(NH)])
        Rmir_rank = {}
        for li in range(NL):
            Rmir_rank[li] = np.stack(
                [avg_ranks(Bmir[h, :, li]) for h in range(NH)])
        p95_w = np.zeros(NL)
        p95_m = np.zeros(NL)
        for li in range(NL):
            perms_h = np.stack(
                [rng3.permutation(NH)
                 for _ in range(N_PERM_L)])
            R86c = centered(np.stack(
                [avg_ranks(B86[h, :, li]) for h in range(NH)]))
            for tag, Rrc, out in (('w', centered(Rwd_rank[li]),
                                   p95_w),
                                  ('m', centered(
                                      Rmir_rank[li]), p95_m)):
                Rw_p = Rrc[perms_h]
                num = np.einsum('hd,khd->kh', R86c, Rw_p)
                den = np.sqrt(
                    (R86c ** 2).sum(1)[None, :]
                    * (Rw_p ** 2).sum(2))
                rho_perm = num / np.maximum(den, 1e-30)
                out[li] = float(np.percentile(rho_perm, 95))
        skel_mir = rho_mir >= p95_m[None, :]
        both = int((skel_mir & skel_29).sum())
        jacc_s = both / max(int((skel_mir | skel_29).sum()), 1)
        p3 = {'n_skel_mirror': int(skel_mir.sum()),
              'n_skel_2929': int(skel_29.sum()),
              'intersection': both,
              'jaccard': round(jacc_s, 4),
              'L0_mirror_all32': bool(skel_mir[:, 0].all()),
              'L0_rho_mir_max':
                  round(float(rho_mir[:, 0].max()), 6),
              'mirror_p95_median':
                  round(float(np.median(p95_m)), 4),
              'bwd_p95_median':
                  round(float(np.median(p95_w)), 4)}
        log('P3: skel_mirror %d vs skel_2929 %d | inter %d '
            'jacc %.4f | L0 mirror all32 %s (rho max %.6f)'
            % (skel_mir.sum(), skel_29.sum(), both, jacc_s,
               skel_mir[:, 0].all(), rho_mir[:, 0].max()),
            lines)

        # ---------- P4: survivor/lost annotation -----------
        rows = []
        for name, cells in (('survivor', SURV),
                            ('lost', LOST), ('L+', L_POS)):
            rr = []
            for (h, li) in cells:
                rr.append({'cell': [h, li],
                           'rho_2929': round(
                               float(rho_grid_29[h, li]), 4),
                           'rho_mirror': round(
                               float(rho_mir[h, li]), 4)})
            rows.append({name: rr})
        p4 = {'groups': rows}
        for g in rows:
            k = list(g.keys())[0]
            log('P4 %s: %s' % (k, [(d['cell'],
                                    d['rho_2929'],
                                    d['rho_mirror'])
                                   for d in g[k]]), lines)

        save = {'B_mirror': Bmir.astype(np.float32),
                'sign_M_mirror': sign_Mmir.astype(np.float32),
                'p_maxT_mirror': p_maxTmir.astype(np.float32),
                'rho_mirror_grid': rho_mir,
                'mir_err_grid': mir_err,
                'lin_r_profile': lin_r,
                'skel_mask_mirror': skel_mir,
                'p95_mirror': p95_m,
                'p95_bwd': p95_w}

    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2930, 'model': 'qwen3-4b', 'prereg': PREREG,
           'anchors': {'a1_rel': float('%.3e' % rel_a1),
                       'a1_ok': a1_ok,
                       'a2_m78': round(m78, 6),
                       'a2_ok': a2_ok,
                       'a3_diff': float('%.3e' % a3_diff),
                       'a3_ok': a3_ok,
                       'a4_ok': a4_ok,
                       'a5_diff': float('%.3e' % a5_diff),
                       'a5_ok': a5_ok,
                       'a6_rel': float('%.3e' % a6_rel),
                       'a6_ok': a6_ok,
                       'a7_pdiff': float('%.3e' % a7_pdiff),
                       'a7_ok': a7_ok,
                       'a8_ok': a8_ok,
                       'ok': anchor_ok},
           'P1': p1, 'P2': p2, 'P3': p3, 'P4': p4,
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if save:
        np.savez_compressed(
            os.path.join(OUT, 'direction_flip_control.npz'),
            **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2930 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
