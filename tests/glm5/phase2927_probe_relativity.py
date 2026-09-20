# -*- coding: utf-8 -*-
"""Phase 2927: probe relativity of the lang event atlas.

Why: 2920 established "functional transfer != geometric identity"
(sentence dirs do NOT match word-level encoding directions,
median |cos| 0.166). Open question (2926 candidate A): is the
24-event atlas itself probe-INVARIANT (events are properties of
the head x layer circuitry, robust to the choice of readout
direction) or probe-RELATIVE (artifacts of the 2886 sentence
probe)?

Mode: one forward pass (qwen3-4b), 2917 protocol verbatim
(SEED=2896, eps=1.0, pos 1, 57 words verbatim 2887, conds
same/func/null, null_tids rng order, o_proj-input capture, all
36 layers), but TWO readout directions per perturbation:
  dirs86    - 2886 sentence class-diff unit per layer (2917
              verbatim; anchor caliber)
  dirs_word - WORD-LEVEL probe constructed from this run's own
              func-condition captures: pos-1 attention-input
              residual stream of [the, word], lab_lang==0 group
              mean minus lab_lang==1 group mean per layer, unit.
              Direction-sign convention is irrelevant: sign-Gram
              margin is invariant to a global sign flip of B
              (outer(s,s)).

Two-pass structure: pass 1 forwards all 171 sequences and
stores attn inputs; dirs_word is then constructed; pass 2 runs
the per-layer attn_call perturbations for BOTH directions
(shared ref), stored per (condition, word). Expected ~90 s.

Anchors (frozen; all on the 2886-caliber B recomputed from THIS
run - they certify the forward; the word probe is then judged
against a certified replica of 2917):
  a1 B86[:,:,26:36] vs 2913 npz max rel < 1e-5
  a2 |margin({7,8} on [26,36) block) - 0.28036| < 5e-3
  a3 sign_M86 vs 2917 npz sign_M max abs diff < 1e-4
  a4 maxT event set (p_maxT86 <= 0.05) == 2917 24-event set

Main test (P1, frozen): same maxT machinery (200 perms rng2
2896 verbatim) on sign_M_word -> event set E'. Overlap with E
(24 events):
  n_overlap >= 12 AND (7,19) in E' => events_probe_invariant
  n_overlap >= 6                    => events_probe_partially_invariant
  else                              => events_probe_relative

P2 (descriptive): full-grid Spearman(sign_M_word, sign_M86).
P3 (descriptive): per-layer cos(dirs_word, dirs86) profile.
P4 (descriptive): E' layer/head distribution vs E.

Output: phase2927/probe_relativity/.
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
OUT = os.path.join(BASE, 'phase2927', 'probe_relativity')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2927_run_report.txt')
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

PREREG = {
    'mode': 'forward per-head jacobian full-layer scan with TWO '
            'readout directions (2886 sentence dirs anchor '
            'caliber + word-level dirs_word), 2917 protocol '
            'verbatim otherwise',
    'question': '2926 candidate A: is the 24-event lang atlas '
                'probe-invariant (property of head x layer '
                'circuitry) or probe-relative (artifact of the '
                '2886 sentence probe)?',
    'word_probe': 'dirs_word[li] = unit(mean over words of pos-1 '
                  'attn-input residual stream under [the, word] '
                  'func condition, lab_lang==0 group minus '
                  'lab_lang==1 group); sign convention '
                  'irrelevant (sign-Gram margin is sign-flip '
                  'invariant)',
    'anchors': {
        'a1': 'B86[:,:,26:36] vs 2913 npz max rel < 1e-5',
        'a2': '|m78(2886 caliber) - 0.28036| < 5e-3',
        'a3': 'sign_M86 vs 2917 npz max abs diff < 1e-4',
        'a4': 'maxT event set (p<=0.05) == 2917 24-event set',
    },
    'P1': 'maxT (200 perms, rng2 2896 verbatim) on sign_M_word '
          "-> E'; n_overlap = |E' cap E|, E = 2917 24 events",
    'P2': 'full-grid Spearman(sign_M_word, sign_M86), '
          'descriptive',
    'P3': 'per-layer cos(dirs_word, dirs86), descriptive',
    'P4': "E' layer/head distribution, descriptive",
    'verdict': 'anchor fail => anchor_fail_all_void; '
               "n_overlap >= 12 AND top1 in E' => "
               'events_probe_invariant; n_overlap >= 6 => '
               'events_probe_partially_invariant; else => '
               'events_probe_relative',
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


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2927,
                   'name': 'probe_relativity',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8': sha8(os.path.abspath(__file__)),
                   'sources': {'s2886': sha8(SRC_2886),
                               's2887': sha8(SRC_2887),
                               's2913': sha8(SRC_2913),
                               's2917': sha8(SRC_2917)},
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
    log('sources ok (E17 n=%d)' % len(E17), lines)

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

    for li in range(NL):
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

    vdt = next(layers[0].mlp.parameters()).dtype

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

    # ---------- pass 1: forward capture, store attn inputs ------
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

    # ---------- word-level probe from func captures -------------
    d_dim = attn_store[(0, 'func', 0)].shape[-1]
    diffs_w = np.zeros((NL, d_dim))
    for li in range(NL):
        X = np.stack([attn_store[(i, 'func', li)][0, 1]
                      for i in range(n_words)]).astype(np.float64)
        diffs_w[li] = X[lab_lang == 0].mean(0) \
            - X[lab_lang == 1].mean(0)
    dirs_word = np.stack([unit(diffs_w[li]) for li in range(NL)])
    cos_dw = [float(dirs_word[li] @ dirs86[li])
              for li in range(NL)]
    log('dirs_word built | cos(dirs_word, dirs86): min %.4f '
        'median %.4f max %.4f'
        % (min(cos_dw), float(np.median(cos_dw)), max(cos_dw)),
        lines)

    # ---------- pass 2: dual-direction perturbations ------------
    r_c86 = {cn: np.zeros((n_words, NL, NH, HD), np.float32)
             for cn in CONDS}
    r_cw = {cn: np.zeros((n_words, NL, NH, HD), np.float32)
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
        if (i + 1) % 10 == 0:
            log('pass2 words [%d/%d]' % (i + 1, n_words), lines)
    log('pass2 done', lines)

    # ---------- B matrices (2917 verbatim combination) ----------
    G86 = np.zeros((NL, NH * HD))
    Gwd = np.zeros((NL, NH * HD))
    for li in range(NL):
        Wo_li = layers[li].self_attn.o_proj.weight.detach() \
            .float().cpu().numpy().astype(np.float64)
        G86[li] = Wo_li.T @ dirs86[li]
        Gwd[li] = Wo_li.T @ dirs_word[li]
    B86 = np.einsum('nlhk,lhk->hnl',
                    (r_c86['same'].astype(np.float64)
                     - 0.5 * r_c86['func'].astype(np.float64)
                     - 0.5 * r_c86['null'].astype(np.float64)),
                    G86.reshape(NL, NH, HD))
    Bwd = np.einsum('nlhk,lhk->hnl',
                    (r_cw['same'].astype(np.float64)
                     - 0.5 * r_cw['func'].astype(np.float64)
                     - 0.5 * r_cw['null'].astype(np.float64)),
                    Gwd.reshape(NL, NH, HD))
    log('B86 %s | Bwd %s' % (B86.shape, Bwd.shape), lines)

    # ---------- anchors (2886 caliber certifies the forward) ----
    blk = B86[:, :, 26:36]
    rel_a1 = float(np.abs(blk - B_heads_2913).max()
                   / max(float(np.abs(B_heads_2913).max()), 1e-30))
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
                out[h, li] = Gm[same_m].mean() - Gm[diff_m].mean()
        return out

    sign_M86 = sign_matrix(B86)
    a3_diff = float(np.abs(sign_M86 - sign_M_17).max())
    a3_ok = bool(a3_diff < 1e-4)
    anchor_ok = bool(a1_ok and a2_ok and a3_ok)
    log('a1 rel %.2e ok=%s | a2 m78 %.6f ok=%s | a3 sign diff '
        '%.2e ok=%s'
        % (rel_a1, a1_ok, m78, a2_ok, a3_diff, a3_ok), lines)

    verdict = None
    p1 = p2 = p4 = None
    a4_ok = False
    E86 = set()
    sign_M86b = p_maxT86 = sign_Mwd = p_maxTwd = None
    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        # ---------- maxT machinery (2917 verbatim) --------------
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
            p_maxT = np.zeros((NH, NL))
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
        anchor_ok = anchor_ok and a4_ok
        if not anchor_ok:
            verdict = 'anchor_fail_all_void'
        else:
            # ---------- P1: word-probe event set ----------------
            sign_Mwd, p_Mwd, p_maxTwd = maxT_run(Bwd)
            Ewd = set((h, li) for h in range(NH)
                      for li in range(NL)
                      if p_maxTwd[h, li] <= FDR_Q)
            n_overlap = len(Ewd & E17)
            jacc = n_overlap / len(Ewd | E17)
            top1_in = TOP1 in Ewd
            log('P1: E_word n=%d | n_overlap %d/24 | jaccard %.3f '
                '| top1 (7,19) in E\': %s'
                % (len(Ewd), n_overlap, jacc, top1_in), lines)
            log('P1 E_word: %s'
                % sorted([(h, li, round(float(sign_Mwd[h, li]), 4))
                          for (h, li) in Ewd],
                         key=lambda t: -t[2])[:26], lines)
            p1 = {'n_events_word': len(Ewd),
                  'n_overlap': n_overlap,
                  'jaccard': round(jacc, 4),
                  'top1_in_Eword': top1_in,
                  'events_word': sorted(
                      [{'head': h, 'layer': li,
                        'sign_margin': round(float(
                            sign_Mwd[h, li]), 5),
                        'p_maxT': round(float(p_maxTwd[h, li]), 6),
                        'in_E17': bool((h, li) in E17)}
                       for (h, li) in Ewd],
                      key=lambda e: -e['sign_margin'])}
            # ---------- verdict --------------------------------
            if n_overlap >= 12 and top1_in:
                verdict = 'events_probe_invariant'
            elif n_overlap >= 6:
                verdict = 'events_probe_partially_invariant'
            else:
                verdict = 'events_probe_relative'

            # ---------- P2: grid-level rank agreement ----------
            rho_grid = spearman(sign_Mwd.ravel(), sign_M86b.ravel())
            p2 = {'spearman_grid': round(rho_grid, 4)}
            log('P2 full-grid Spearman(sign_M_word, sign_M86): '
                '%.4f' % rho_grid, lines)

            # ---------- P4: E' structure ------------------------
            layers_wd = sorted(set(li for (h, li) in Ewd))
            heads_wd = {}
            for (h, li) in Ewd:
                heads_wd[h] = heads_wd.get(h, 0) + 1
            layers_e = sorted(set(li for (h, li) in E17))
            p4 = {'layers_word': layers_wd, 'layers_e17': layers_e,
                  'head_counts_word': {str(k): v for k, v in
                                       sorted(heads_wd.items(),
                                              key=lambda t: -t[1])}}
            log('P4: E_word layers %s | E17 layers %s'
                % (layers_wd, layers_e), lines)

    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)

    res = {
        'phase': 2927, 'model': 'qwen3-4b', 'prereg': PREREG,
        'anchors': {'a1_rel_vs_2913': float('%.3e' % rel_a1),
                    'a1_ok': a1_ok,
                    'a2_m78': round(m78, 6),
                    'a2_ok': a2_ok,
                    'a3_sign_diff': float('%.3e' % a3_diff),
                    'a3_ok': a3_ok,
                    'a4_ok': a4_ok,
                    'ok': anchor_ok},
        'P1': p1, 'P2': p2, 'P3': {'cos_profile': [
            round(c, 4) for c in cos_dw],
            'cos_median': round(float(np.median(cos_dw)), 4)},
        'P4': p4,
        'final_verdict': verdict,
        'runtime_s': round(time.monotonic() - t0, 1),
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    save = {'B86': B86.astype(np.float32),
            'B_word': Bwd.astype(np.float32),
            'sign_M86': sign_M86.astype(np.float32),
            'dirs_word': dirs_word.astype(np.float32),
            'cos_profile': np.array(cos_dw),
            'labels_lang': lab_lang,
            'words': np.array(['%s:%s:%s' % w for w in words],
                              dtype=object)}
    if sign_M86b is not None:
        save['sign_M86_maxT'] = sign_M86b.astype(np.float32)
        save['p_maxT86'] = p_maxT86.astype(np.float32)
        save['sign_M_word'] = sign_Mwd.astype(np.float32)
        save['p_maxT_word'] = p_maxTwd.astype(np.float32)
    np.savez_compressed(
        os.path.join(OUT, 'probe_relativity.npz'), **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2927 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
