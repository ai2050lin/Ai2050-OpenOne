# -*- coding: utf-8 -*-
"""Phase 2917: full-layer (head,layer) event atlas.

Why: 2916 established that carrier margins are driven by SINGLE
(head, layer) events (h7@L19 early segment, h7@L34 + h8@L34 late,
h27@L24, h31@L22), but only layers 16-35 within frozen windows
were scanned, and per-event significance was window-family only.
Open question (2916 candidate A): what does the FULL 32x36 atlas
look like - do the 6 known events survive family-wise FDR over
all 1152 (head,layer) cells, and are there events outside the
scanned windows?

Mode: forward per-head jacobian over ALL 36 layers (qwen3-4b,
layer 0..35): per layer li the injected direction is the 2886
class-diff direction diffs[li] (unit); protocol otherwise 2913
verbatim (SEED=2896, eps=1.0, pos 1, 57 words verbatim 2887,
conds same/func/null, null_tids rng order, o_proj-input capture).
Event statistic: per (head, layer) sign-margin (2910 caliber,
sign outer Gram same/diff difference).  Null: 200 SEED=2896 label
permutations - the sign Gram G is FIXED per event, permutations
only move the same/diff masks (2903 caliber).  BH-FDR q=0.05
over the 1152-event family (2913 P2 step-up caliber).

Known events (frozen, from 2916 P2 leave-one-layer-out table):
  (7,19), (27,24), (31,22), (8,23), (7,34), (8,34)
Classification of each FDR-significant event:
  known     - exact (h, li) in KNOWN_EVENTS
  satellite - same head, |li - li_known| <= 1 for some known event
  novel     - everything else

Anchors (frozen):
  a1 B_heads[:, :, 26:36] vs 2913 npz B_heads: max rel < 1e-5
     (2914/2915 measured 3.16e-08 / 2.73e-08).
  a2 |margin({7,8} on the [26,36) block) - 0.28036| < 5e-3.

Probes (frozen):
  P1 FDR-significant event set with known/satellite/novel
     classification; n_known_sig (of 6), n_novel (verdict axis).
  P2 known-event replication table (obs sign-margin, rank in
     atlas, p_fdr).
  P3 atlas summaries: largest event per layer; per-head event
     counts; top-20 events overall.
  P4 descriptive.

Adjudication (frozen):
  anchor fail => anchor_fail_all_void;
  n_known_sig == 6 AND n_novel == 0 => event_atlas_closed;
  n_known_sig == 6 AND n_novel > 0 => event_atlas_extended;
  n_known_sig >= 3 => event_atlas_partially_replicated;
  else => event_atlas_not_replicated.

Output: phase2917/event_atlas/.
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
SRC_2913 = os.path.join(BASE, 'phase2913',
                        'perhead_wvo_decomposition',
                        'perhead_wvo_decomposition.npz')
OUT = os.path.join(BASE, 'phase2917', 'event_atlas')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2917_run_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
SEED = 2896
EPS = 1.0
VOCAB = 151936
N_PERM = 200
NH, HD, NKV = 32, 128, 8
NL = 36                      # attn layers 0..35
FDR_Q = 0.05
KNOWN_EVENTS = [(7, 19), (27, 24), (31, 22),
                (8, 23), (7, 34), (8, 34)]
RECON_REF = 0.28036

PREREG = {
    'mode': 'forward per-head jacobian full-layer scan (36 layers '
            'x 32 heads) - (head,layer) event atlas',
    'question': '2916 candidate A: do the 6 known events survive '
                'family-wise BH-FDR over the full 1152-cell atlas, '
                'and are there events outside the 2915/2916 '
                'windows (novel or satellite)',
    'known_events': KNOWN_EVENTS,
    'protocol': 'SEED=2896, eps=1.0, pos 1, 57 words verbatim '
                '2887, conds same/func/null, dirs = 2886 class-diff '
                'unit per layer [0,36), o_proj-input capture all '
                '36 layers; event statistic = sign-margin (2910 '
                'caliber, sign outer Gram fixed, mask permutation '
                'null, 200 SEED=2896 perms); BH-FDR q=0.05 step-up '
                'over 1152 events (2913 P2 caliber)',
    'anchors': {
        'a1': 'B_heads[:,:,26:36] vs 2913 npz max rel < 1e-5',
        'a2': '|margin({7,8} on [26,36) block) - 0.28036| < 5e-3',
    },
    'probes': {
        'P1': 'FDR-significant set, known/satellite/novel '
              'classification, n_known_sig/n_novel',
        'P2': 'known-event replication table',
        'P3': 'atlas summaries (per-layer max, per-head counts, '
              'top-20)',
        'P4': 'descriptive',
    },
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'n_known_sig==6 AND n_novel==0 => event_atlas_closed; '
               'n_known_sig==6 AND n_novel>0 => event_atlas_extended; '
               'n_known_sig>=3 => event_atlas_partially_replicated; '
               'else => event_atlas_not_replicated',
    'verdict_v2_note': 'run1 (v1) adjudicated '
                 'event_atlas_not_replicated under per-event BH-FDR '
                 '(0/1152 significant) - AUDIT: the v1 criterion has '
                 'STRUCTURALLY ZERO POWER, a design error of mine: '
                 'the smallest attainable per-event permutation p is '
                 '1/201 = 0.004975 (N_PERM=200), while the BH first '
                 'threshold is q/m = 0.05/1152 = 4.34e-5; BH can '
                 'only fire if >= 115 events sit exactly at the '
                 'granularity floor, which did not happen. This is '
                 'the same discrete-p calibration failure class as '
                 '2912 (E[p] = 0.5 + tau/2 family). v2 (rerun on '
                 'fresh forward, products deleted) freezes the '
                 'Westfall-Young single-step maxT family-wise '
                 'correction instead: p_maxT(event) = (1 + #{perms: '
                 'max_{h,l} sign_M_perm >= sign_M(event)}) / (1 + '
                 'N_PERM); the v1 BH p-values stay registered as '
                 'descriptive. Verdict axis unchanged in form, '
                 'computed on maxT significance',
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


def margin_of(Sm, same, diff):
    return float(Sm[same].mean() - Sm[diff].mean())


def gram_margin(B, same, diff):
    U = rownorm(B)
    return margin_of(U @ U.T, same, diff)


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2917,
                   'name': 'event_atlas',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8': sha8(os.path.abspath(__file__)),
                   'sources': {'s2886': sha8(SRC_2886),
                               's2887': sha8(SRC_2887),
                               's2913': sha8(SRC_2913)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'eps': EPS, 'seed': SEED, 'n_perm': N_PERM,
                   'fdr_q': FDR_Q, 'known_events': KNOWN_EVENTS,
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
    assert diffs.shape[0] >= NL, diffs.shape
    dirs = np.stack([unit(diffs[li]) for li in range(NL)])

    z13 = np.load(SRC_2913, allow_pickle=True)
    B_heads_2913 = z13['B_heads'].astype(np.float64)
    log('sources ok (dirs %s)' % (dirs.shape,), lines)

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

    # ---------- forward capture (all 36 layers) ----------
    r_c = {cn: np.zeros((n_words, NL, NH, HD), dtype=np.float32)
           for cn in ('same', 'func', 'null')}
    for i, (_, _, w) in enumerate(words):
        w_tid = tid_map[w]
        conds = {'same': [tid_map[words[same_ctx(i)][2]], w_tid],
                 'func': [func_tid, w_tid],
                 'null': [null_tids[i], w_tid]}
        for cn, toks in conds.items():
            attnin_all = forward2(toks)
            for li in range(NL):
                d = dirs[li]
                x = attnin_all[li]
                ref = attn_call(li, x)
                xp = x.copy()
                xp[0, 1] = xp[0, 1] + EPS * d
                pert = attn_call(li, xp)
                r_c[cn][i, li] = ((pert - ref) / EPS) \
                    .reshape(NH, HD).astype(np.float32)
        if (i + 1) % 10 == 0:
            log('words [%d/%d]' % (i + 1, n_words), lines)

    # ---------- B_heads (32, 57, 36) ----------
    r_comb = (r_c['same'].astype(np.float64)
              - 0.5 * r_c['func'].astype(np.float64)
              - 0.5 * r_c['null'].astype(np.float64))
    G = np.zeros((NL, NH * HD))
    for li in range(NL):
        Wo_li = layers[li].self_attn.o_proj.weight.detach() \
            .float().cpu().numpy().astype(np.float64)
        G[li] = Wo_li.T @ dirs[li]
    G3 = G.reshape(NL, NH, HD)
    B_heads = np.einsum('nlhk,lhk->hnl', r_comb, G3)
    del r_comb
    log('B_heads %s' % (B_heads.shape,), lines)

    # ---------- anchors ----------
    blk = B_heads[:, :, 26:36]
    rel_a1 = float(np.abs(blk - B_heads_2913).max()
                   / max(float(np.abs(B_heads_2913).max()), 1e-30))
    a1_ok = bool(rel_a1 < 1e-5)
    same_m, diff_m = masks(lab_lang)
    m78 = gram_margin(blk[7] + blk[8], same_m, diff_m)
    a2_ok = bool(abs(m78 - RECON_REF) < 5e-3)
    anchor_ok = bool(a1_ok and a2_ok)
    log('a1 rel vs 2913 %.2e ok=%s | a2 m78 %.6f ok=%s'
        % (rel_a1, a1_ok, m78, a2_ok), lines)

    # ---------- event atlas ----------
    verdict = None
    p1 = p2 = p3 = p4 = None
    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        sign_M = np.zeros((NH, NL))
        Gs = {}
        for h in range(NH):
            for li in range(NL):
                s = np.sign(B_heads[h, :, li])
                s[s == 0] = 1.0
                Gm = np.outer(s, s)
                Gs[(h, li)] = Gm
                sign_M[h, li] = Gm[same_m].mean() \
                    - Gm[diff_m].mean()
        rng2 = np.random.default_rng(SEED)
        perms = [rng2.permutation(lab_lang) for _ in range(N_PERM)]
        perm_masks = [masks(pl) for pl in perms]
        p_M = np.zeros((NH, NL))       # per-event (descriptive)
        max_perm = np.zeros(N_PERM)    # maxT family-wise
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
        del Gs
        # maxT family-wise (v2 criterion, Westfall-Young single
        # step): the per-permutation MAX over the full 1152-event
        # family is the corrected null statistic.
        p_maxT = np.zeros((NH, NL))
        for h in range(NH):
            for li in range(NL):
                p_maxT[h, li] = float(
                    np.sum(max_perm >= sign_M[h, li]) + 1) \
                    / (N_PERM + 1)
        # BH-FDR on per-event p (v1 caliber, descriptive only)
        flat = [(float(p_M[h, li]), h, li)
                for h in range(NH) for li in range(NL)]
        flat.sort()
        m = len(flat)
        k_crit = 0
        for rank in range(1, m + 1):
            if flat[rank - 1][0] <= FDR_Q * rank / m:
                k_crit = rank
        sig_bh = set((h, li) for _, h, li in flat[:k_crit])
        log('atlas v1 BH (descriptive): %d/%d | v2 maxT: %d/%d '
            'events significant (q=0.05)'
            % (len(sig_bh), m,
               int(np.sum(p_maxT <= FDR_Q)), m), lines)
        sig_set = set((h, li) for h in range(NH)
                      for li in range(NL)
                      if p_maxT[h, li] <= FDR_Q)

        # classification
        def classify(h, li):
            if (h, li) in KNOWN_EVENTS:
                return 'known'
            for (hk, lk) in KNOWN_EVENTS:
                if h == hk and abs(li - lk) <= 1:
                    return 'satellite'
            return 'novel'

        sig_sorted = sorted(
            [(h, li, float(sign_M[h, li]), float(p_maxT[h, li]))
             for (h, li) in sig_set],
            key=lambda t: -t[2])
        n_known_sig = sum(1 for (h, li, _, _)
                          in sig_sorted
                          if classify(h, li) == 'known')
        n_novel = sum(1 for (h, li, _, _)
                      in sig_sorted
                      if classify(h, li) == 'novel')
        n_satellite = len(sig_sorted) - n_known_sig - n_novel
        p1 = {'n_sig': len(sig_set), 'n_known_sig': n_known_sig,
              'n_satellite': n_satellite, 'n_novel': n_novel,
              'events': [{'head': h, 'layer': li,
                          'sign_margin': round(v, 5),
                          'p_maxT': round(p, 6),
                          'class': classify(h, li)}
                         for (h, li, v, p) in sig_sorted]}
        log('P1: known %d/6 satellite %d novel %d'
            % (n_known_sig, n_satellite, n_novel), lines)
        log('P1 events: %s'
            % [(e['head'], e['layer'], e['sign_margin'],
                e['class']) for e in p1['events'][:20]], lines)

        # P2 known-event replication table
        p2 = []
        for (h, li) in KNOWN_EVENTS:
            rank = int(np.sum(
                sign_M[:, li] > sign_M[h, li])) + 1
            p2.append({'head': h, 'layer': li,
                       'sign_margin': round(float(
                           sign_M[h, li]), 5),
                       'p_maxT': round(float(p_maxT[h, li]), 6),
                       'p_per_event': round(float(p_M[h, li]), 6),
                       'in_layer_rank': rank,
                       'maxT_sig': bool((h, li) in sig_set)})
        log('P2 known: %s' % [(e['head'], e['layer'],
                               e['sign_margin'], e['maxT_sig'])
                              for e in p2], lines)

        # P3 atlas summaries
        per_layer_max = []
        for li in range(NL):
            h = int(np.argmax(sign_M[:, li]))
            per_layer_max.append({'layer': li, 'head': h,
                                  'sign_margin': round(
                                      float(sign_M[h, li]), 5)})
        head_counts = {}
        for (h, li) in sig_set:
            head_counts[h] = head_counts.get(h, 0) + 1
        p3 = {'per_layer_max': per_layer_max,
              'sig_head_counts': {str(k): v for k, v
                                  in sorted(head_counts.items(),
                                            key=lambda t: -t[1])}}
        log('P3 per-layer max: %s'
            % [(e['layer'], e['head'], e['sign_margin'])
               for e in per_layer_max], lines)

        p4 = {'note': 'full sign_M and p_M stored in npz'}

        # ---------- verdict ----------
        if n_known_sig == 6 and n_novel == 0:
            verdict = 'event_atlas_closed'
        elif n_known_sig == 6 and n_novel > 0:
            verdict = 'event_atlas_extended'
        elif n_known_sig >= 3:
            verdict = 'event_atlas_partially_replicated'
        else:
            verdict = 'event_atlas_not_replicated'

    log('==== VERDICT: %s ====' % verdict, lines)

    res = {
        'phase': 2917, 'model': 'qwen3-4b', 'prereg': PREREG,
        'anchors': {'a1_rel_vs_2913': float('%.3e' % rel_a1),
                    'a1_ok': a1_ok,
                    'a2_m78': round(m78, 6),
                    'a2_ref': RECON_REF, 'a2_ok': a2_ok,
                    'ok': anchor_ok},
        'P1': p1, 'P2': p2, 'P3': p3, 'P4': p4,
        'final_verdict': verdict,
        'runtime_s': round(time.monotonic() - t0, 1),
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    np.savez_compressed(
        os.path.join(OUT, 'event_atlas.npz'),
        B_heads=B_heads.astype(np.float32),
        sign_M=sign_M.astype(np.float32),
        p_M=p_M.astype(np.float32),
        p_maxT=p_maxT.astype(np.float32),
        max_perm=max_perm.astype(np.float32),
        dirs=dirs.astype(np.float32),
        labels_lang=lab_lang,
        words=np.array(['%s:%s:%s' % w for w in words],
                       dtype=object))
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2917 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
