# -*- coding: utf-8 -*-
"""Phase 3001: Omega-G1 GLM4 robustness source localization.

Why: 2998/2999 established that GLM4's language separation is
nearly immune to xdir injection (max ratio 0.1285 @L4, s_c>6
everywhere) and that its null0 baseline already carries 72% of
the functional separation (sep_null0 61.4 vs sep_f 85.7).
Open question (next-step A): is the robustness because the
class signal is CARRIED BY THE WORD TOKEN itself (context
irrelevant, nothing for an injection to move), or because an
ACTIVE eraser selectively removes the injected direction from
the residual stream?

Design (2999 machine verbatim; 98 cells, dirs_g bit-anchored
to the 2996 npz, Vt8/u39, in-session dcks, xdir single-layer
attn_in pos-1 injection, final-norm readout):
  T1 context-swap arms (word-position readout):
     A [func, w]  baseline   (anchor: 2999 sep_f bit-level)
     B [null0, w] random ctx (anchor: 2999 sep_null0 61.43)
     C [w, w]     same-word ctx (descriptive)
     D [w_p, w]   seeded random partner word ctx
     additive 2x2 decomposition m(word, ctx):
     word_eff = col-mean diff, ctx_eff = row-mean diff;
     carry := word_eff >= 0.7 * sep_f.
  T2 direction selectivity at L17/L19, s=2, K=2:
     2 random unit vectors in span(Vt8) orthogonal to xdir
     (subspace-matched, primary) + 1 random full-space unit
     vector (descriptive).  selective := median sub-random
     ratio(L19) >= 0.2 AND >= 2 * ratio_xdir(L19).
  T3 (DESCRIPTIVE, no verdict branch) tracking: capture
     pos-1 attn_in at every layer during xdir s=2 injection
     at L4 and L19 (modified x captured at the injection
     layer); per-layer trk_ratio(l) = med||Delta_l||/s and
     xdir projection; eraser layer = first l > L_inj with
     trk_ratio < 0.5.

Verdict (frozen):
  anchor fail                     => anchor_fail_all_void
  carry AND selective             => word_carry_selective_erasure_glm4
  carry AND NOT selective         => word_carry_general_erasure_glm4
  NOT carry                       => context_entangled_glm4

Anchors (frozen):
  a0 words_g == 2996 glm src re-export
  a1 dirs_g[17/18/19] vs 2996 dirs_attn_glm < 1e-6
  a2 baseline determinism < 1e-4
  a3/a4 xdir identity < 1e-9
  a5 same-session determinism < 1e-6 (all K-repeat arms)
  a6 |sep_f - 2999.a6_sep_f| < 1e-4 (cross-run bit-level)
  a7 u39 unit < 1e-12
  a8 null0 collision-free
  a9 cross-card 2999: |ratio_L19 - 2999.a9_ref_ratio| < 1e-6
     AND |sep_n0 - 61.43| <= 0.02 AND
     |sep_mirror_L19 - 85.26| <= 0.05

Tags: Omega-G1 / lang axis / len-2 / en classes / snapshot
machine (no ablation) / dimensionless gates / descriptive T3.
"""
import hashlib
import json
import os
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC_2972 = os.path.join(BASE, 'phase2972',
                        'two_factor_signature',
                        'execution.json')
SRC_2996 = os.path.join(BASE, 'phase2996',
                        'omega_f2a_registry_caliber_audit',
                        'omega_f2a_registry_caliber_audit.npz')
OUT = os.path.join(BASE, 'phase3001',
                   'omega_g1_robustness_source_glm4')
MD_G = r'D:\AI2050\Ai2050-OpenOne\models\hf\glm4-9b-chat-hf'
L_CAND = ["because", "therefore", "although", "unless",
          "however", "thus", "moreover", "since", "whereas",
          "despite", "hence", "nevertheless", "consequently",
          "furthermore", "otherwise", "instead", "while",
          "accordingly", "likewise", "meanwhile", "nonetheless",
          "thereafter", "whereby", "albeit"]
NL, HID, VOCAB = 40, 4096, 151552
SEL_LAYERS = (17, 19)
REF_LAYER = 19
TRK_LAYERS = (4, 19)
S_SCAN = 2.0
K_SCAN = 2
S_IDX = (0, 1, 4)
SEED_NULL = 2889
SEED_RND = 3001
SEED_PARTNER = 3002
CARRY_FRAC = 0.7
SEL_GATE = 0.2
SEL_MULT = 2.0
TRK_DROP = 0.5
REF_2999 = {'sep_f': 85.73204044720937,
            'ratio19': 0.01664657989077104,
            'sep_n0': 61.43,
            'mirror19_sep': 85.26,
            'mirror4_sep': 85.35}
N_SUB_RND = 2
N_FULL_RND = 1

PREREG = {
    'mode': 'glm4-9b only; 2999 machine verbatim (98 '
            'cells, dirs_g anchored to 2996 npz, Vt8/u39, '
            'in-session dcks, xdir single-layer attn_in '
            'pos-1 injection); + tracking capture mode '
            '(pos-1 attn_in at every layer incl. modified '
            'x at the injection layer)',
    'question': 'is GLM4 robustness to xdir injection '
                'explained by word-token carry of class '
                'separation (context irrelevant) or by '
                'active direction-selective erasure in '
                'the residual stream?',
    'T1': 'context-swap arms at word-position readout: '
          'A [func,w] anchor 2999 sep_f; B [null0,w] '
          'anchor 2999 sep_null0; C [w,w] descriptive; '
          'D [w_p,w] seeded random partner word ctx '
          '(SEED_PARTNER, partner != self); additive '
          '2x2 decomposition m(word,ctx): word_eff = '
          'col-mean diff, ctx_eff = row-mean diff; '
          'carry = word_eff >= 0.7*sep_f; ctx_eff '
          'descriptive (NOT strictly opposite pairing - '
          'avoids the vacuous ctx-label identity)',
    'T2': 'direction selectivity at L17/L19 s=2 K=2: '
          '2 per-cell random unit vecs in span(Vt8) '
          'orthogonal to that cell xdir (primary) + 1 '
          'full-space random unit vec (descriptive); '
          'selective = median sub-random ratio(L19) >= '
          '0.2 AND >= 2*ratio_xdir(L19)',
    'T3': 'DESCRIPTIVE tracking: xdir s=2 at L4 and L19; '
          'per-layer trk_ratio(l)=med||Delta_l||/s and '
          'xdir projection; eraser layer = first l > '
          'L_inj with trk_ratio < 0.5; no verdict branch',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'carry AND selective => '
               'word_carry_selective_erasure_glm4; '
               'carry AND NOT selective => '
               'word_carry_general_erasure_glm4; '
               'NOT carry => context_entangled_glm4',
    'tags': 'Omega-G1 / lang axis / len-2 / en classes / '
            'snapshot machine (no ablation) / '
            'dimensionless gates / descriptive T3',
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
    lines.append('[%s] %s' % (time.strftime('%H:%M:%S'),
                              msg))
    with open(os.path.join(OUT, 'run_log.txt'), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 3001,
                   'name': 'omega_g1_robustness_source_glm4',
                   'created':
                       time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2972': sha8(SRC_2972),
                               's2996npz': sha8(SRC_2996)},
                   'model': 'glm4-9b', 'n_layers': NL,
                   'hidden': HID, 'vocab': VOCAB,
                   'sel_layers': list(SEL_LAYERS),
                   'ref_layer': REF_LAYER,
                   'trk_layers': list(TRK_LAYERS),
                   's_scan': S_SCAN, 'k_scan': K_SCAN,
                   's_idx': list(S_IDX),
                   'seed_null': SEED_NULL,
                   'seed_rnd': SEED_RND,
                   'carry_frac': CARRY_FRAC,
                   'sel_gate': SEL_GATE,
                   'sel_mult': SEL_MULT,
                   'trk_drop': TRK_DROP,
                   'ref_2999': REF_2999,
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    # ---------- sources ----------
    z96 = np.load(SRC_2996, allow_pickle=True)
    words_g = [str(w) for w in z96['words_g']]
    dirs96 = z96['dirs_attn_glm'].astype(np.float64)
    assert dirs96.shape == (NL, HID)
    e72 = json.load(open(SRC_2972, encoding='utf-8'))
    src = ([('F', 'en', w) for w in e72['cells']['F_en']]
           + [('C', 'en', w) for w in e72['cells']['C_en']]
           + [('L', 'en', w) for w in L_CAND]
           + [('F', 'fr', w) for w in e72['cells']['F_fr']]
           + [('C', 'fr', w) for w in e72['cells']['C_fr']])
    lab_src = ['%s:%s:%s' % c for c in src]
    a0_ok = bool(words_g == lab_src)
    log('a0 words_g == 2996 glm src re-export: %s'
        % a0_ok, lines)
    log('sources ok (n=%d)' % len(words_g), lines)

    # ---------- model ----------
    import torch as _t
    from transformers import AutoTokenizer, \
        AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(
        MD_G, local_files_only=True, use_fast=True)
    tc = {}

    def tid_of(w):
        if w not in tc:
            ids = tok(' ' + w, add_special_tokens=False)[
                'input_ids']
            if len(ids) != 1:
                ids = tok(w, add_special_tokens=False)[
                    'input_ids']
            tc[w] = int(ids[0]) if len(ids) == 1 else -1
        return tc[w]

    cells = []
    for s in words_g:
        cat, lng, w = s.split(':')
        t = tid_of(w)
        if t != -1:
            cells.append((0 if lng == 'en' else 1,
                          cat, lng, w))
    n = len(cells)
    lang = np.array([c[0] for c in cells])
    i_en = [i for i in range(n) if lang[i] == 0]
    i_non = [i for i in range(n) if lang[i] == 1]
    assert n == 98, n
    log('cells %d (en %d non-en %d)'
        % (n, len(i_en), len(i_non)), lines)

    word_tids = set(tid_of(c[3]) for c in cells)
    func_tid = tid_of('the')
    assert func_tid > 0
    rng0 = np.random.default_rng(SEED_NULL)
    null0_tids = []
    while len(null0_tids) < n:
        r = int(rng0.integers(0, VOCAB))
        if r not in word_tids and r > 0:
            null0_tids.append(r)
    a8_ok = bool(len(null0_tids) == n
                 and not (set(null0_tids) & word_tids))
    log('a8 null0 collision-free: %s' % a8_ok, lines)

    model = AutoModelForCausalLM.from_pretrained(
        MD_G, torch_dtype=_t.bfloat16).cuda().eval()
    layers = model.model.layers
    assert len(layers) == NL
    log('model loaded', lines)

    cap = {'ai': {}, 'trk': {}}
    state_fin = {'on': False}
    state_trk = {'on': False}
    fin_cap = {}
    inj = {'on': False, 'scale': 0.0, 'vec': None,
           'layer': None}
    handles = []

    def pre_attn(li):
        def h(module, args, kwargs):
            x = kwargs.get('hidden_states')
            if x is None:
                x = args[0] if args else None
            if x is None or x.dim() < 2:
                return None
            ret = None
            if inj['on'] and inj['layer'] is not None \
                    and li == inj['layer']:
                x = x.clone()
                x[:, 1, :] = x[:, 1, :] \
                    + inj['scale'] * inj['vec']
                nkw = dict(kwargs)
                nkw['hidden_states'] = x
                ret = (args, nkw)
            if state_trk['on']:
                cap['trk'].setdefault(li, []).append(
                    x[:, 1, :].detach().float().cpu()
                    .numpy().copy())
            elif not inj['on']:
                cap['ai'].setdefault(li, []).append(
                    x.detach().float().cpu().numpy()
                    .copy())
            return ret
        return h

    def pre_norm(module, args, kwargs):
        if state_fin['on']:
            fin_cap['x'] = args[0][:, -1, :].detach() \
                .float().cpu().numpy().copy()
        return None

    for li in range(NL):
        handles.append(layers[li].self_attn
                       .register_forward_pre_hook(
                           pre_attn(li), with_kwargs=True))
    handles.append(model.model.norm
                   .register_forward_pre_hook(
                       pre_norm, with_kwargs=True))

    def clear_cap():
        for li in cap['ai']:
            del cap['ai'][li][:]
        for li in cap['trk']:
            del cap['trk'][li][:]

    def forward_batch(toks_list, scale=0.0, layer=None):
        clear_cap()
        fin_cap.pop('x', None)
        state_fin['on'] = True
        inj['on'] = scale != 0.0
        inj['scale'] = float(scale)
        inj['layer'] = layer
        with _t.no_grad():
            model(_t.tensor(toks_list, device='cuda'))
        inj['on'] = False
        state_fin['on'] = False
        return fin_cap['x'].astype(np.float64)

    def forward_trk(toks_list, scale=0.0, layer=None):
        clear_cap()
        inj['on'] = scale != 0.0
        inj['scale'] = float(scale)
        inj['layer'] = layer
        state_trk['on'] = True
        with _t.no_grad():
            model(_t.tensor(toks_list, device='cuda'))
        inj['on'] = False
        state_trk['on'] = False
        return {li: np.stack(cap['trk'][li]).astype(
            np.float64) for li in cap['trk']}

    seqs = [[func_tid, tid_of(c[3])] for c in cells]

    # ---------- dirs_g rebuild (2999 verbatim) ----------
    store = {}
    for i, c in enumerate(cells):
        clear_cap()
        with _t.no_grad():
            model(_t.tensor([seqs[i]], device='cuda'))
        for li in range(NL):
            store[(i, li)] = cap['ai'][li][0] \
                .astype(np.float32)
        if (i + 1) % 25 == 0:
            log('pass1 [%d/%d]' % (i + 1, n), lines)
    d_w = np.zeros((NL, HID))
    for li in range(NL):
        X = np.stack([store[(i, li)][0, 1]
                      for i in range(n)]).astype(np.float64)
        d_w[li] = X[i_en].mean(0) - X[i_non].mean(0)
    dirs_g = np.stack([unit(d_w[li]) for li in range(NL)])
    a1_diffs = {li: float(np.abs(dirs_g[li] - dirs96[li])
                          .max()) for li in (17, 18, 19)}
    a1_ok = bool(all(v < 1e-6
                     for v in a1_diffs.values()))
    log('a1 dirs_g vs 2996 rows %s ok=%s'
        % ({k: ('%.2e' % v) for k, v
            in a1_diffs.items()}, a1_ok), lines)

    _, _, Vt = np.linalg.svd(dirs_g, full_matrices=False)
    Vt8 = Vt[:8]
    u39 = dirs_g[NL - 1]
    a7_diff = abs(float(np.linalg.norm(u39)) - 1.0)
    a7_ok = bool(a7_diff < 1e-12)
    log('a7 u39 unit |1-n|=%.2e ok=%s'
        % (a7_diff, a7_ok), lines)

    # ---------- baselines ----------
    fin_f1 = forward_batch(seqs)
    fin_f2 = forward_batch(seqs)
    a2_rel = float(np.abs(fin_f1 - fin_f2).max()
                   / max(float(np.abs(fin_f1).max()),
                         1e-30))
    a2_ok = bool(a2_rel < 1e-4)
    log('a2 baseline determinism rel %.2e ok=%s'
        % (a2_rel, a2_ok), lines)
    fin_n0 = forward_batch(
        [[null0_tids[i], tid_of(c[3])]
         for i, c in enumerate(cells)])

    def reads(fin):
        return fin @ u39, fin @ Vt8.T

    proj_f0, c8_f0 = reads(fin_f1)
    proj_n0, c8_n0 = reads(fin_n0)
    sep_f = float(proj_f0[i_en].mean()
                  - proj_f0[i_non].mean())
    a6_diff = abs(sep_f - REF_2999['sep_f'])
    a6_ok = bool(a6_diff < 1e-4)
    sep_n = float(proj_n0[i_en].mean()
                  - proj_n0[i_non].mean())
    log('a6 sep_f=%.10f vs 2999 diff=%.2e ok=%s '
        '(null0 %.3f)'
        % (sep_f, a6_diff, a6_ok, sep_n), lines)

    dcks = c8_n0 - c8_f0
    dcks_S = dcks[:, list(S_IDX)]
    Vt8_S = Vt8[list(S_IDX)]
    xdir = dcks_S @ Vt8_S
    a3_diff = float(np.abs(xdir @ Vt8_S.T - dcks_S).max())
    a3_ok = bool(a3_diff < 1e-9)
    a4_ok = a3_ok
    log('a3/a4 xdir identity %.2e ok=%s'
        % (a3_diff, a3_ok), lines)
    med_dS = float(np.median(np.linalg.norm(dcks_S, axis=1)))
    xdir_t = _t.tensor(xdir, device='cuda',
                       dtype=_t.bfloat16)
    inj['vec'] = xdir_t
    log('inj vec armed n=%d' % xdir.shape[0], lines)

    def arm(vec_t, scale, layer, k, tag, spreads):
        projs = []
        ratios = []
        inj['vec'] = vec_t
        for _ in range(k):
            fin = forward_batch(seqs, scale=scale,
                                layer=layer)
            p, c8 = reads(fin)
            projs.append(p)
            cs = c8[:, list(S_IDX)] \
                - c8_f0[:, list(S_IDX)]
            ratios.append(float(np.median(
                np.linalg.norm(cs, axis=1))))
        inj['vec'] = xdir_t
        P = np.stack(projs)
        spreads[tag] = float('%.2e' % float(
            np.abs(P - P.mean(0)).max()))
        p_med = np.median(P, axis=0)
        sep_m = float(p_med[i_en].mean()
                      - p_med[i_non].mean())
        ratio_m = float(np.median(ratios)) \
            / max(med_dS, 1e-30)
        return p_med, sep_m, ratio_m

    anchor_prelim = bool(a0_ok and a8_ok and a1_ok
                         and a7_ok and a2_ok and a6_ok
                         and a3_ok and a4_ok)
    verdict = None
    T1 = T2 = T3 = None
    spreads = {}
    a5_diff = None
    save = {}
    carry_bool = None
    selective = None
    a9_parts = {}
    a9_ok = False

    if anchor_prelim:
        # ---------- T1 context-swap arms ----------
        seqs_null = [[null0_tids[i], tid_of(c[3])]
                     for i, c in enumerate(cells)]
        pB = reads(forward_batch(seqs_null))[0]
        sep_B = float(pB[i_en].mean() - pB[i_non].mean())
        save['proj_null'] = pB
        seqs_same = [[tid_of(c[3]), tid_of(c[3])]
                     for c in cells]
        pC = reads(forward_batch(seqs_same))[0]
        sep_C = float(pC[i_en].mean() - pC[i_non].mean())
        save['proj_sameword'] = pC
        # seeded random partner word context
        # (NOT strictly opposite: keeps the 2x2
        # decomposition non-degenerate)
        rngp = np.random.default_rng(SEED_PARTNER)
        partner = []
        for i in range(n):
            j = int(rngp.integers(0, n))
            while j == i:
                j = int(rngp.integers(0, n))
            partner.append(cells[j])
        seqs_swap = [[tid_of(partner[i][3]),
                      tid_of(cells[i][3])]
                     for i in range(n)]
        pD = reads(forward_batch(seqs_swap))[0]
        lab_w = lang.copy()
        lab_c = np.array([0 if partner[i][2] == 'en'
                          else 1 for i in range(n)])
        m00 = float(pD[(lab_w == 0)
                       & (lab_c == 0)].mean())
        m01 = float(pD[(lab_w == 0)
                       & (lab_c == 1)].mean())
        m10 = float(pD[(lab_w == 1)
                       & (lab_c == 0)].mean())
        m11 = float(pD[(lab_w == 1)
                       & (lab_c == 1)].mean())
        word_eff = (m00 + m01) / 2.0 \
            - (m10 + m11) / 2.0
        ctx_eff = (m00 + m10) / 2.0 \
            - (m01 + m11) / 2.0
        sep_D = float(pD[i_en].mean()
                      - pD[i_non].mean())
        save['proj_swapword'] = pD
        carry_index = word_eff / sep_f
        carry_bool = bool(carry_index >= CARRY_FRAC)
        T1 = {'sep_f': round(sep_f, 2),
              'sep_null0': round(sep_B, 2),
              'sep_sameword': round(sep_C, 2),
              'sep_swapword': round(sep_D, 2),
              'm_word0_ctx0': round(m00, 2),
              'm_word0_ctx1': round(m01, 2),
              'm_word1_ctx0': round(m10, 2),
              'm_word1_ctx1': round(m11, 2),
              'word_eff': round(word_eff, 2),
              'ctx_eff': round(ctx_eff, 2),
              'carry_index': round(carry_index, 4),
              'carry_bool': carry_bool,
              'carry_frac_gate': CARRY_FRAC}
        log('T1 sep: f=%.1f null0=%.1f same=%.1f '
            'swap=%.1f; m00=%.1f m01=%.1f m10=%.1f '
            'm11=%.1f word_eff=%.1f ctx_eff=%.1f '
            'carry_idx=%.3f carry=%s'
            % (sep_f, sep_B, sep_C, sep_D,
               m00, m01, m10, m11, word_eff, ctx_eff,
               carry_index, carry_bool), lines)

        # ---------- T2 direction selectivity ----------
        rng = np.random.default_rng(SEED_RND)
        xc = (Vt8 @ xdir.T).T
        xh = xc / np.linalg.norm(
            xc, axis=1, keepdims=True)
        rnd_sub = []
        for _ in range(N_SUB_RND):
            g = rng.standard_normal((n, 8))
            g = g - np.sum(g * xh, axis=1,
                           keepdims=True) * xh
            g = g / np.linalg.norm(g, axis=1,
                                   keepdims=True)
            rnd_sub.append(g @ Vt8)
        rnd_full = [unit(rng.standard_normal(HID))]
        t2 = {}
        raw_x = {}
        for li in SEL_LAYERS:
            _, sep_x, ratio_x = arm(
                xdir_t, S_SCAN, li, K_SCAN,
                'xdir|%d' % li, spreads)
            raw_x[li] = ratio_x
            t2['L%d' % li] = {'xdir': {
                'sep': round(sep_x, 2),
                'ratio': round(ratio_x, 4)}}
            sub_ratios = []
            for j, v in enumerate(rnd_sub):
                vt = _t.tensor(v, device='cuda',
                               dtype=_t.bfloat16)
                _, sep_r, ratio_r = arm(
                    vt, S_SCAN, li, K_SCAN,
                    'subrnd%d|%d' % (j, li), spreads)
                t2['L%d' % li]['sub_rnd%d' % j] = {
                    'ratio': round(ratio_r, 4)}
                sub_ratios.append(ratio_r)
                save['proj_subrnd%d_%d' % (j, li)] = \
                    np.zeros(1)
            full_ratios = []
            for j, v in enumerate(rnd_full):
                vt = _t.tensor(v, device='cuda',
                               dtype=_t.bfloat16)
                _, sep_r, ratio_r = arm(
                    vt, S_SCAN, li, K_SCAN,
                    'fullrnd%d|%d' % (j, li), spreads)
                t2['L%d' % li]['full_rnd%d' % j] = {
                    'ratio': round(ratio_r, 4)}
                full_ratios.append(ratio_r)
            t2['L%d' % li]['sub_rnd_median'] = round(
                float(np.median(sub_ratios)), 4)
            t2['L%d' % li]['full_rnd_median'] = round(
                float(np.median(full_ratios)), 4)
            log('T2 L%d xdir ratio=%.4f sub_rnd med=%.4f '
                'full_rnd med=%.4f'
                % (li, ratio_x,
                   float(np.median(sub_ratios)),
                   float(np.median(full_ratios))), lines)
        sel19_sub = t2['L%d' % REF_LAYER][
            'sub_rnd_median']
        xdir19 = t2['L%d' % REF_LAYER]['xdir']['ratio']
        selective = bool(sel19_sub >= SEL_GATE
                         and sel19_sub >= SEL_MULT
                         * xdir19)
        T2 = dict(t2)
        T2['selective'] = selective
        T2['sel_gate'] = SEL_GATE
        T2['sel_mult'] = SEL_MULT
        log('T2 selective=%s (sub19=%.4f vs xdir19=%.4f)'
            % (selective, sel19_sub, xdir19), lines)

        # ---------- T3 tracking (descriptive) ----------
        trk = {}
        base_trk = forward_trk(seqs)
        for li_inj in TRK_LAYERS:
            inj_trk = forward_trk(seqs, scale=S_SCAN,
                                  layer=li_inj)
            prof = {}
            for li in range(li_inj, NL):
                Dl = inj_trk[li] - base_trk[li]
                nr = float(np.median(
                    np.linalg.norm(Dl, axis=1))) / S_SCAN
                pj = float(np.median(
                    np.sum(Dl * xdir, axis=1)
                    / S_SCAN))
                prof[li] = {'trk_ratio': round(nr, 4),
                            'xdir_proj': round(pj, 4)}
            eraser = None
            for li in range(li_inj + 1, NL):
                if prof[li]['trk_ratio'] < TRK_DROP:
                    eraser = li
                    break
            trk['L%d' % li_inj] = {
                'profile': prof,
                'eraser_layer': eraser}
            log('T3 L%d inj: trk_ratio l=%d..%d '
                'eraser=%s'
                % (li_inj, li_inj, NL - 1, eraser), lines)
            save['trk_L%d' % li_inj] = np.array(
                [prof[li]['trk_ratio']
                 for li in range(li_inj, NL)])
        T3 = {'layers': list(TRK_LAYERS), 'trk': trk,
              'trk_drop': TRK_DROP,
              'note': 'descriptive; no verdict branch'}

        # ---------- mirror anchors (2999 cross-card) ----------
        _, sep_m19, _ = arm(xdir_t, -S_SCAN, REF_LAYER,
                            K_SCAN, 'mirror|19', spreads)
        _, sep_m4, _ = arm(xdir_t, -S_SCAN, 4, K_SCAN,
                           'mirror|4', spreads)
        a9_parts = {
            'ratio19': abs(
                raw_x[REF_LAYER]
                - REF_2999['ratio19']),
            'sep_n0': abs(sep_B - REF_2999['sep_n0']),
            'mirror19': abs(
                sep_m19 - REF_2999['mirror19_sep']),
            'mirror4': abs(
                sep_m4 - REF_2999['mirror4_sep'])}
        a9_ok = bool(
            a9_parts['ratio19'] < 1e-6
            and a9_parts['sep_n0'] <= 0.02
            and a9_parts['mirror19'] <= 0.05
            and a9_parts['mirror4'] <= 0.05)
        log('a9 cross-card 2999 %s ok=%s'
            % ({k: ('%.2e' % v) for k, v
                in a9_parts.items()}, a9_ok), lines)

        a5_diff = max(spreads.values())
        a5_ok = bool(a5_diff < 1e-6)
        log('a5 same-session determinism %.2e ok=%s'
            % (a5_diff, a5_ok), lines)

        if not (a9_ok and a5_ok):
            verdict = 'anchor_fail_all_void'
        elif carry_bool and selective:
            verdict = 'word_carry_selective_erasure_glm4'
        elif carry_bool:
            verdict = 'word_carry_general_erasure_glm4'
        else:
            verdict = 'context_entangled_glm4'
    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('VERDICT %s' % verdict, lines)

    elapsed = time.monotonic() - t0

    anchors = {
        'a0_words_order': a0_ok, 'a8_collision': a8_ok,
        'a1_diffs': {str(k): v
                     for k, v in a1_diffs.items()},
        'a1_ok': a1_ok, 'a2_rel': a2_rel,
        'a3_diff': a3_diff, 'a5_diff': a5_diff,
        'a6_diff': a6_diff, 'a6_ok': a6_ok,
        'a6_sep_f': sep_f, 'a7_diff': a7_diff,
        'a9_parts': {k: v for k, v in
                     a9_parts.items()},
        'a9_ok': a9_ok if anchor_prelim else None,
    }
    res = {
        'phase': 3001,
        'final_verdict': verdict,
        'anchor_all_ok': bool(
            a0_ok and a8_ok and a1_ok and a7_ok
            and a2_ok and a6_ok and a3_ok and a4_ok
            and a5_ok and a9_ok),
        'anchors': anchors,
        'scale': {'sep_f': round(sep_f, 2),
                  'sep_null0': round(sep_n, 2),
                  'med_dS': round(med_dS, 4)},
        'T1': T1, 'T2': T2, 'T3': T3,
        'tags': PREREG['tags'],
        'elapsed_s': round(elapsed, 1),
        'correction_note':
            'run1: crashed at xdir_coords matmul '
            '(xdir is per-cell (98,4096)); prereg T1 '
            'redesigned BEFORE any verdict: strictly-'
            'opposite pairing made the ctx-label sep '
            'vacuously -sep_D (always-true identity, '
            'banned); replaced with seeded random '
            'partner + additive 2x2 decomposition; '
            'run2: SyntaxError paren mismatch, never '
            'executed; run3: stale xdir_coords left '
            'on disk by a phantom edit, crashed '
            'post-T1; run4: full pass but a9 '
            'ratio19 compared the 4dp-rounded scan '
            'value against the full-precision 2999 '
            'reference (diff 4.66e-5 = pure '
            'rounding vs gate 1e-6) - anchor '
            'comparison bug, verdict void by '
            'protocol; run5: raw ratio retained '
            'for a9; authoritative',
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    save.pop('proj_xdir_17', None)
    save.pop('proj_subrnd0_17', None)
    for kk in [k for k in list(save.keys())
               if save[k].shape == (1,)]:
        save.pop(kk)
    save['dirs_g'] = dirs_g
    save['Vt8_g'] = Vt
    save['u39'] = u39
    save['xdir'] = xdir
    save['words_g'] = np.array(words_g)
    save['lang_g'] = lang
    save['null0_tids'] = np.array(null0_tids)
    save['partner_idx'] = np.array(
        [words_g.index('%s:%s:%s'
                       % (partner[i][1], partner[i][2],
                          partner[i][3]))
         for i in range(n)])
    save['proj_base'] = proj_f0
    save['proj_null'] = save.get('proj_null',
                                 np.zeros(1)) \
        if save.get('proj_null') is not None \
        else np.zeros(1)
    save['proj_sameword'] = save.get('proj_sameword',
                                     np.zeros(1))
    save['proj_swapword'] = save.get('proj_swapword',
                                     np.zeros(1))
    npz_path = os.path.join(
        OUT, 'omega_g1_robustness_source_glm4.npz')
    np.savez_compressed(npz_path, **save)

    seal = {
        'npz_sha256_8': sha8(npz_path),
        'result_sha256_8': sha8(
            os.path.join(OUT, 'result.json')),
        'exec_sha256_8': sha8(
            os.path.join(OUT, 'execution.json')),
    }
    with open(os.path.join(OUT, 'seal.json'), 'w',
              encoding='utf-8') as f:
        json.dump(seal, f, indent=2)
    log('sealed %s' % json.dumps(seal), lines)
    log('elapsed %.1fs' % elapsed, lines)


if __name__ == '__main__':
    main()
