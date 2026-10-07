# -*- coding: utf-8 -*-
"""Phase 3002: Omega-G2 qwen robustness source (GLM4 3001
mirror).

Why: 3001 (GLM4) localized the GLM4 robustness source:
word-token carry (word_eff 74.1 = 86% of sep_f) + general
(non-direction-selective) mid-band erasure (T2 sub-random
ratio 0.009 < xdir 0.017), with T3 giving the mechanism
split: L19 injection killed next layer (trk 1.96->0.32)
while L4 injection survived in amplitude but was
orthogonalized (trk->1.99, proj->0.12).  Open question
(plan v4 Omega-G): is the qwen signal-carry structure the
same?  qwen propagation is strong (3000/2945 ratio 1.2-1.5
@L15/17) - if qwen is ALSO word-carry, the cross-model gap
lives entirely in the propagation/erasure band, not in the
signal source; if qwen is context-entangled, the two models
differ at the signal source itself.

Design (qwen 3000 machine verbatim: 57 words 2887,
dirs_word rebuild a1 vs 2927, Vt8 a3 vs 2939, u35 readout,
dcks from 2939 coords, xdir = dcks_S @ Vt8_S, attn_in pos-1
single-layer coef injection, bf16):
  T1 context-swap arms at word-position readout
     (labels by lang):
     A [func, w]  anchor: 2935 proj_func (bit-level a4)
     B [null0, w] anchor: 2935 proj_null0 (bit-level a5)
     C [w, w]     descriptive
     D [w_p, w]   seeded random partner word ctx
     (SEED_PARTNER, partner != self; NOT strictly-opposite
     pairing - avoids the vacuous ctx-label identity,
     3001 lesson)
     additive 2x2 decomposition m(word, ctx):
     word_eff = col-mean diff, ctx_eff = row-mean diff;
     carry := word_eff >= 0.7 * sep_f.
  T2 direction selectivity at L15/L17, s=2, K=2:
     2 per-cell random unit vecs in span(Vt8) orthogonal
     to that cell xdir (primary) + 1 full-space random
     unit vec (descriptive).  selective := median
     sub-random ratio(L17) >= 0.2 AND >= 2*ratio_xdir(L17).
  T3 (DESCRIPTIVE, no verdict branch) tracking: capture
     pos-1 attn_in at every layer during xdir s=2
     injection at L4 and L17 (modified x captured at the
     injection layer); per-layer trk_ratio(l) =
     med||Delta_l||/s and xdir projection; eraser layer =
     first l > L_inj with trk_ratio < 0.5.

Verdict (frozen):
  anchor fail                     => anchor_fail_all_void
  carry AND selective             => word_carry_selective_
                                     amplification_qwen
  carry AND NOT selective         => word_carry_general_
                                     amplification_qwen
  NOT carry                       => context_entangled_qwen

Anchors (frozen):
  a0 words == 2887 re-export (57)
  a1 dirs_word vs 2927 < 1e-5
  a2 baseline determinism < 1e-4
  a3 Vt8 vs 2939 < 1e-6
  a4 proj_func vs 2935 s_base[func] < 1e-4 (bit-level)
  a5 proj_null0 vs 2935 s_base[null0] < 1e-4 (bit-level)
  a6 sep_f > 0
  a7 xdir identity < 1e-9
  a8 null0 collision-free
  a9 2945 repro (raw ratio, NOT rounded - 3001 run4
     lesson): |ratio_L15 - 1.2149| < 0.1 AND
     |ratio_L17 - 1.5004| < 0.1
  a10 3001 source integrity: result hash == seal AND
      verdict == word_carry_general_erasure_glm4
  a11 same-session determinism < 1e-6 (all K-repeat arms)

Tags: Omega-G2 / lang axis / len-2 / en classes / snapshot
machine (no ablation) / dimensionless gates / descriptive
T3 / cross-model mirror of 3001.
"""
import hashlib
import json
import os
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC_2887 = os.path.join(BASE, 'phase2887',
                        'language_axis_mlp',
                        'language_axis_mlp.npz')
SRC_2927 = os.path.join(BASE, 'phase2927',
                        'probe_relativity',
                        'probe_relativity.npz')
SRC_2935 = os.path.join(BASE, 'phase2935',
                        'null_amp_anatomy',
                        'null_amp_anatomy.npz')
SRC_2939 = os.path.join(BASE, 'phase2939',
                        'rotation_target',
                        'rotation_target.npz')
SRC_2945 = os.path.join(BASE, 'phase2945',
                        'threshold_curves',
                        'result.json')
D_3001 = os.path.join(BASE, 'phase3001',
                      'omega_g1_robustness_source_glm4')
OUT = os.path.join(BASE, 'phase3002',
                   'omega_g2_robustness_source_qwen')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NL, HID, VOCAB = 36, 2560, 151936
SEL_LAYERS = (15, 17)
REF_LAYER = 17
TRK_LAYERS = (4, 17)
S_SCAN = 2.0
K_SCAN = 2
SEED_NULL = 2896          # 3000 verbatim
SEED_RND = 3004
SEED_PARTNER = 3005
CARRY_FRAC = 0.7
SEL_GATE = 0.2
SEL_MULT = 2.0
TRK_DROP = 0.5
S_IDX = (0, 1, 4)
REF_2945 = {'15': 1.2149, '17': 1.5004}
A9_TOL = 0.1
N_SUB_RND = 2
N_FULL_RND = 1

PREREG = {
    'mode': 'qwen3-4b only; 3000 machine verbatim (57 '
            'words 2887, dirs_word a1 vs 2927, Vt8 a3 vs '
            '2939, u35 readout, dcks from 2939 coords, '
            'xdir = dcks_S@Vt8_S, attn_in pos-1 '
            'single-layer coef injection, bf16); '
            'cross-model mirror of 3001 (GLM4)',
    'question': 'is the qwen class-separation signal '
                'carried by the word token (context '
                'irrelevant) as in GLM4, or '
                'context-entangled?  Determines whether '
                'the cross-model robustness gap lives in '
                'the propagation band or at the signal '
                'source.',
    'T1': 'context-swap arms at word-position readout: '
          'A [func,w] anchor 2935 proj_func bit-level; '
          'B [null0,w] anchor 2935 proj_null0 bit-level; '
          'C [w,w] descriptive; D [w_p,w] seeded random '
          'partner word ctx (SEED_PARTNER, partner != '
          'self; NOT strictly-opposite - avoids the '
          'vacuous ctx-label identity); additive 2x2 '
          'decomposition m(word,ctx): word_eff = col-mean '
          'diff, ctx_eff = row-mean diff; carry = '
          'word_eff >= 0.7*sep_f',
    'T2': 'direction selectivity at L15/L17 s=2 K=2: 2 '
          'per-cell random unit vecs in span(Vt8) '
          'orthogonal to that cell xdir (primary) + 1 '
          'full-space random unit vec (descriptive); '
          'selective = median sub-random ratio(L17) >= '
          '0.2 AND >= 2*ratio_xdir(L17)',
    'T3': 'DESCRIPTIVE tracking: xdir s=2 at L4 and L17; '
          'per-layer trk_ratio(l)=med||Delta_l||/s and '
          'xdir projection; eraser layer = first l > '
          'L_inj with trk_ratio < 0.5; no verdict branch',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'carry AND selective => '
               'word_carry_selective_amplification_qwen; '
               'carry AND NOT selective => '
               'word_carry_general_amplification_qwen; '
               'NOT carry => context_entangled_qwen',
    'tags': 'Omega-G2 / lang axis / len-2 / en classes / '
            'snapshot machine (no ablation) / '
            'dimensionless gates / descriptive T3 / '
            'cross-model mirror of 3001',
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
        json.dump({'phase': 3002,
                   'name':
                       'omega_g2_robustness_source_qwen',
                   'created':
                       time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {
                       's2887': sha8(SRC_2887),
                       's2927': sha8(SRC_2927),
                       's2935': sha8(SRC_2935),
                       's2939': sha8(SRC_2939),
                       's2945': sha8(SRC_2945),
                       's3001result': sha8(
                           D_3001 + r'\result.json'),
                       's3001seal': sha8(
                           D_3001 + r'\seal.json')},
                   'model': 'qwen3-4b',
                   'n_layers': NL, 'hidden': HID,
                   'vocab': VOCAB,
                   'sel_layers': list(SEL_LAYERS),
                   'ref_layer': REF_LAYER,
                   'trk_layers': list(TRK_LAYERS),
                   's_scan': S_SCAN, 'k_scan': K_SCAN,
                   's_idx': list(S_IDX),
                   'seed_null': SEED_NULL,
                   'seed_rnd': SEED_RND,
                   'seed_partner': SEED_PARTNER,
                   'carry_frac': CARRY_FRAC,
                   'sel_gate': SEL_GATE,
                   'sel_mult': SEL_MULT,
                   'trk_drop': TRK_DROP,
                   'ref_2945': REF_2945,
                   'a9_tol': A9_TOL,
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    # ---------- sources ----------
    z87 = np.load(SRC_2887, allow_pickle=True)
    words = [tuple(str(w).split(':'))
             for w in z87['words']]
    lab_lang = np.asarray(z87['labels_lang']).astype(int)
    n_words = len(words)
    a0_ok = bool(n_words == 57)
    log('a0 words == 2887 re-export: %s (n=%d)'
        % (a0_ok, n_words), lines)

    z27 = np.load(SRC_2927, allow_pickle=True)
    dirs27 = z27['dirs_word'].astype(np.float64)
    z35 = np.load(SRC_2935, allow_pickle=True)
    conds35 = [str(s) for s in z35['cond_names']]
    s_base_35 = z35['s_base'].astype(np.float64)
    ifu35 = conds35.index('func')
    in035 = conds35.index('null0')
    z39 = np.load(SRC_2939, allow_pickle=True)
    Vt8_39 = z39['Vt8'].astype(np.float64)
    coords_39 = z39['coords'].astype(np.float64)
    conds39 = [str(s) for s in z39['cond_names']]
    dcks_39 = coords_39[conds39.index('null0')] \
        - coords_39[conds39.index('func')]

    r45 = json.load(open(SRC_2945, encoding='utf-8'))
    d3 = r45['D3_ratio']

    # a10 3001 source integrity
    seal01 = json.load(open(D_3001 + r'\seal.json',
                            encoding='utf-8'))
    a10_ok = bool(seal01['result_sha256_8']
                  == sha8(D_3001 + r'\result.json'))
    r01 = json.load(open(D_3001 + r'\result.json',
                         encoding='utf-8'))
    a10_ok = a10_ok and bool(
        r01['final_verdict']
        == 'word_carry_general_erasure_glm4'
        and r01['anchor_all_ok'] is True)
    t1_g = r01['T1']
    log('a10 3001 integrity %s (glm word_eff=%s '
        'ctx_eff=%s carry_idx=%s)'
        % (a10_ok, t1_g['word_eff'], t1_g['ctx_eff'],
           t1_g['carry_index']), lines)

    # ---------- model ----------
    import torch
    import sys
    sys.path.insert(
        0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
    from phase2662_symmetric_mapping_contract \
        import load_native
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
            assert tid_map[w] == int(ck), \
                'key mismatch %s' % w
    func_tid = tid('the')
    word_tids = set(tid_map.values())

    rng0 = np.random.default_rng(SEED_NULL)
    null0_tids = []
    while len(null0_tids) < n_words:
        r = int(rng0.integers(0, VOCAB))
        if r not in word_tids and r > 0:
            null0_tids.append(r)
    a8_ok = bool(len(null0_tids) == n_words
                 and not (set(null0_tids) & word_tids))
    log('a8 null0 collision-free: %s' % a8_ok, lines)

    batch = {'func': [[func_tid, tid_map[words[i][2]]]
                      for i in range(n_words)],
             'null0': [[null0_tids[i],
                        tid_map[words[i][2]]]
                       for i in range(n_words)]}

    model, _ = load_native('qwen4')
    model.eval()
    layers = model.model.layers
    assert len(layers) == NL
    log('model loaded', lines)

    cap = {'ai': {}, 'trk': {}}
    state_fin = {'on': False}
    state_trk = {'on': False}
    fin_cap = {}
    inj = {'coef': None, 'scale': 0.0, 'vec': None,
           'layer': None}
    handles = []

    def pre_attn(li):
        def h(module, args, kwargs):
            x = args[0] if args \
                else kwargs.get('hidden_states')
            if x is None or x.dim() < 2:
                return None
            ret = None
            if inj['coef'] is not None \
                    and li in inj['coef']:
                c = inj['coef'][li]
                if c != 0.0:
                    x = x.clone()
                    x[:, 1, :] = x[:, 1, :] \
                        + c * inj['scale'] * inj['vec']
                if args:
                    ret = ((x,) + tuple(args[1:]),
                           kwargs)
                else:
                    nkw = dict(kwargs)
                    nkw['hidden_states'] = x
                    ret = (args, nkw)
            if state_trk['on']:
                cap['trk'].setdefault(li, []).append(
                    x[:, 1, :].detach().float().cpu()
                    .numpy().copy())
            elif inj['coef'] is None:
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

    def forward1(toks):
        clear_cap()
        with torch.no_grad():
            model(torch.tensor([toks], device='cuda'))
        return {li: cap['ai'][li][0]
                for li in cap['ai']}

    def forward_batch(toks_list, coef=None, scale=0.0):
        clear_cap()
        fin_cap.pop('x', None)
        state_fin['on'] = True
        inj['coef'] = coef
        inj['scale'] = float(scale)
        with torch.no_grad():
            model(torch.tensor(toks_list, device='cuda'))
        inj['coef'] = None
        state_fin['on'] = False
        return fin_cap['x'].astype(np.float64)

    def forward_trk(toks_list, coef=None, scale=0.0):
        clear_cap()
        inj['coef'] = coef
        inj['scale'] = float(scale)
        state_trk['on'] = True
        with torch.no_grad():
            model(torch.tensor(toks_list, device='cuda'))
        inj['coef'] = None
        state_trk['on'] = False
        return {li: np.stack(cap['trk'][li]).astype(
            np.float64) for li in cap['trk']}

    # ---------- pass 1: dirs_word rebuild ----------
    attn_store = {}
    for i, (_, _, w) in enumerate(words):
        attnin_all = forward1(
            [func_tid, tid_map[w]])
        for li in range(NL):
            attn_store[(i, li)] = \
                attnin_all[li].astype(np.float32)
        if (i + 1) % 20 == 0:
            log('pass1 [%d/%d]' % (i + 1, n_words), lines)
    d_w = np.zeros((NL, HID))
    for li in range(NL):
        X = np.stack([attn_store[(i, li)][0, 1]
                      for i in range(n_words)]) \
            .astype(np.float64)
        d_w[li] = X[lab_lang == 0].mean(0) \
            - X[lab_lang == 1].mean(0)
    dirs_word = np.stack([unit(d_w[li])
                          for li in range(NL)])
    a1_diff = float(np.abs(dirs_word - dirs27).max())
    a1_ok = bool(a1_diff < 1e-5)
    log('a1 dirs_word vs 2927 %.2e ok=%s'
        % (a1_diff, a1_ok), lines)

    _, _, Vt = np.linalg.svd(dirs_word,
                             full_matrices=False)
    Vt8 = Vt[:8]
    a3_diff = float(np.abs(Vt8 - Vt8_39).max())
    a3_ok = bool(a3_diff < 1e-6)
    log('a3 Vt8 vs 2939 %.2e ok=%s'
        % (a3_diff, a3_ok), lines)
    u35 = dirs_word[NL - 1]

    dcks_S = dcks_39[:, list(S_IDX)]
    Vt8_S = Vt8[list(S_IDX)]
    xdir = dcks_S @ Vt8_S
    a7_diff = float(np.abs(xdir @ Vt8_S.T - dcks_S).max())
    a7_ok = bool(a7_diff < 1e-9)
    log('a7 xdir identity %.2e ok=%s'
        % (a7_diff, a7_ok), lines)
    med_dS = float(np.median(np.linalg.norm(dcks_S,
                                            axis=1)))
    xdir_t = torch.tensor(xdir, device='cuda',
                          dtype=torch.bfloat16)
    inj['vec'] = xdir_t
    log('inj vec armed n=%d' % xdir.shape[0], lines)

    # ---------- baselines ----------
    fin_f1 = forward_batch(batch['func'])
    fin_f2 = forward_batch(batch['func'])
    a2_rel = float(np.abs(fin_f1 - fin_f2).max()
                   / max(float(np.abs(fin_f1).max()),
                         1e-30))
    a2_ok = bool(a2_rel < 1e-4)
    log('a2 baseline determinism rel %.2e ok=%s'
        % (a2_rel, a2_ok), lines)

    def reads(fin):
        return fin @ u35, fin @ Vt8.T

    proj_f0, c8_f0 = reads(fin_f1)
    a4_diff = float(np.abs(proj_f0 - s_base_35[ifu35])
                    .max())
    a4_ok = bool(a4_diff < 1e-4)
    fin_n0 = forward_batch(batch['null0'])
    proj_n0, c8_n0 = reads(fin_n0)
    a5_diff = float(np.abs(proj_n0 - s_base_35[in035])
                    .max())
    a5_ok = bool(a5_diff < 1e-4)
    sep_f = float(proj_f0[lab_lang == 0].mean()
                  - proj_f0[lab_lang == 1].mean())
    a6_ok = bool(sep_f > 0.0)
    sep_n = float(proj_n0[lab_lang == 0].mean()
                  - proj_n0[lab_lang == 1].mean())
    log('a4 %.2e a5 %.2e ok=%s/%s; a6 sep_f=%.2f '
        '(null0 %.2f, ratio null0/f=%.3f) ok=%s'
        % (a4_diff, a5_diff, a4_ok, a5_ok, sep_f,
           sep_n, sep_n / max(sep_f, 1e-30), a6_ok),
        lines)

    def arm(vec_t, layer, k, tag, spreads):
        projs = []
        ratios = []
        coef = {layer: 1.0}
        inj['vec'] = vec_t
        for _ in range(k):
            fin = forward_batch(batch['func'],
                                coef=coef, scale=S_SCAN)
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
        sep_m = float(p_med[lab_lang == 0].mean()
                      - p_med[lab_lang == 1].mean())
        ratio_m = float(np.median(ratios)) \
            / max(med_dS, 1e-30)
        return p_med, sep_m, ratio_m

    anchor_prelim = bool(a0_ok and a8_ok and a1_ok
                         and a3_ok and a2_ok and a4_ok
                         and a5_ok and a6_ok and a7_ok
                         and a10_ok)
    verdict = None
    T1 = T2 = T3 = None
    spreads = {}
    a11_diff = None
    save = {}
    carry_bool = None
    selective = None
    a9_diffs = None
    a9_ok = False
    i_en = lab_lang == 0
    i_non = lab_lang == 1

    if anchor_prelim:
        # ---------- T1 context-swap arms ----------
        pB = reads(forward_batch(batch['null0']))[0]
        sep_B = float(pB[i_en].mean()
                      - pB[i_non].mean())
        save['proj_null'] = pB
        seqs_same = [[tid_map[words[i][2]],
                      tid_map[words[i][2]]]
                     for i in range(n_words)]
        pC = reads(forward_batch(seqs_same))[0]
        sep_C = float(pC[i_en].mean()
                      - pC[i_non].mean())
        save['proj_sameword'] = pC
        # seeded random partner word context
        # (NOT strictly opposite: keeps the 2x2
        # decomposition non-degenerate)
        rngp = np.random.default_rng(SEED_PARTNER)
        partner = []
        for i in range(n_words):
            j = int(rngp.integers(0, n_words))
            while j == i:
                j = int(rngp.integers(0, n_words))
            partner.append(j)
        seqs_swap = [[tid_map[words[partner[i]][2]],
                      tid_map[words[i][2]]]
                     for i in range(n_words)]
        pD = reads(forward_batch(seqs_swap))[0]
        lab_c = lab_lang[partner].copy()
        m00 = float(pD[i_en & (lab_c == 0)].mean())
        m01 = float(pD[i_en & (lab_c == 1)].mean())
        m10 = float(pD[i_non & (lab_c == 0)].mean())
        m11 = float(pD[i_non & (lab_c == 1)].mean())
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
              'carry_frac_gate': CARRY_FRAC,
              'glm4_3001_mirror': {
                  'word_eff': t1_g['word_eff'],
                  'ctx_eff': t1_g['ctx_eff'],
                  'carry_index':
                      t1_g['carry_index']}}
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
            g = rng.standard_normal((n_words, 8))
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
                xdir_t, li, K_SCAN,
                'xdir|%d' % li, spreads)
            raw_x[li] = ratio_x
            t2['L%d' % li] = {'xdir': {
                'sep': round(sep_x, 2),
                'ratio': round(ratio_x, 4)}}
            sub_ratios = []
            for j, v in enumerate(rnd_sub):
                vt = torch.tensor(
                    v, device='cuda',
                    dtype=torch.bfloat16)
                _, sep_r, ratio_r = arm(
                    vt, li, K_SCAN,
                    'subrnd%d|%d' % (j, li), spreads)
                t2['L%d' % li]['sub_rnd%d' % j] = {
                    'ratio': round(ratio_r, 4)}
                sub_ratios.append(ratio_r)
            full_ratios = []
            for j, v in enumerate(rnd_full):
                vt = torch.tensor(
                    v, device='cuda',
                    dtype=torch.bfloat16)
                _, sep_r, ratio_r = arm(
                    vt, li, K_SCAN,
                    'fullrnd%d|%d' % (j, li),
                    spreads)
                t2['L%d' % li]['full_rnd%d' % j] = {
                    'ratio': round(ratio_r, 4)}
                full_ratios.append(ratio_r)
            t2['L%d' % li]['sub_rnd_median'] = round(
                float(np.median(sub_ratios)), 4)
            t2['L%d' % li]['full_rnd_median'] = round(
                float(np.median(full_ratios)), 4)
            log('T2 L%d xdir ratio=%.4f sub_rnd '
                'med=%.4f full_rnd med=%.4f'
                % (li, ratio_x,
                   float(np.median(sub_ratios)),
                   float(np.median(full_ratios))),
                lines)
        sel17_sub = t2['L%d' % REF_LAYER][
            'sub_rnd_median']
        xdir17 = t2['L%d' % REF_LAYER]['xdir']['ratio']
        selective = bool(sel17_sub >= SEL_GATE
                         and sel17_sub >= SEL_MULT
                         * raw_x[REF_LAYER])
        T2 = dict(t2)
        T2['selective'] = selective
        T2['sel_gate'] = SEL_GATE
        T2['sel_mult'] = SEL_MULT
        log('T2 selective=%s (sub17=%.4f vs xdir17=%.4f)'
            % (selective, sel17_sub,
               raw_x[REF_LAYER]), lines)

        # a9 2945 repro (RAW ratio, not rounded)
        a9_diffs = {
            '15': abs(raw_x[15] - REF_2945['15']),
            '17': abs(raw_x[17] - REF_2945['17'])}
        a9_ok = bool(a9_diffs['15'] < A9_TOL
                     and a9_diffs['17'] < A9_TOL)
        log('a9 2945 repro diffs %s ok=%s'
            % ({k: ('%.2e' % v)
                for k, v in a9_diffs.items()}, a9_ok),
            lines)

        # ---------- T3 tracking (descriptive) ----------
        trk = {}
        base_trk = forward_trk(batch['func'])
        for li_inj in TRK_LAYERS:
            inj_trk = forward_trk(
                batch['func'],
                coef={li_inj: 1.0}, scale=S_SCAN)
            prof = {}
            for li in range(li_inj, NL):
                Dl = inj_trk[li] - base_trk[li]
                nr = float(np.median(
                    np.linalg.norm(Dl, axis=1))) \
                    / S_SCAN
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
                % (li_inj, li_inj, NL - 1, eraser),
                lines)
            save['trk_L%d' % li_inj] = np.array(
                [prof[li]['trk_ratio']
                 for li in range(li_inj, NL)])
        T3 = {'layers': list(TRK_LAYERS), 'trk': trk,
              'trk_drop': TRK_DROP,
              'note': 'descriptive; no verdict branch'}

        a11_diff = max(spreads.values())
        a11_ok = bool(a11_diff < 1e-6)
        log('a11 same-session determinism %.2e ok=%s'
            % (a11_diff, a11_ok), lines)

        if not (a9_ok and a11_ok):
            verdict = 'anchor_fail_all_void'
        elif carry_bool and selective:
            verdict = \
                'word_carry_selective_amplification_qwen'
        elif carry_bool:
            verdict = \
                'word_carry_general_amplification_qwen'
        else:
            verdict = 'context_entangled_qwen'
    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('VERDICT %s' % verdict, lines)

    elapsed = time.monotonic() - t0

    anchors = {
        'a0_words': a0_ok, 'a8_collision': a8_ok,
        'a1_diff': a1_diff, 'a2_rel': a2_rel,
        'a3_diff': a3_diff, 'a4_diff': a4_diff,
        'a5_diff': a5_diff, 'a6_sep_f': sep_f,
        'a7_diff': a7_diff,
        'a9_diffs': {k: v
                     for k, v in a9_diffs.items()}
        if a9_diffs is not None else None,
        'a9_ok': a9_ok if anchor_prelim else None,
        'a10_ok': a10_ok, 'a11_diff': a11_diff,
    }
    res = {
        'phase': 3002,
        'final_verdict': verdict,
        'anchor_all_ok': bool(
            a0_ok and a8_ok and a1_ok and a3_ok
            and a2_ok and a4_ok and a5_ok and a6_ok
            and a7_ok and a10_ok and a9_ok
            and a11_ok),
        'anchors': anchors,
        'scale': {'sep_f': round(sep_f, 2),
                  'sep_null0': round(sep_n, 2),
                  'med_dS': round(med_dS, 4)},
        'T1': T1, 'T2': T2, 'T3': T3,
        'tags': PREREG['tags'],
        'elapsed_s': round(elapsed, 1),
        'correction_note': 'first run',
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    save['dirs_word'] = dirs_word
    save['Vt8'] = Vt8
    save['u35'] = u35
    save['xdir'] = xdir
    save['words'] = np.array(['%s:%s:%s' % w
                              for w in words],
                             dtype=object)
    save['labels_lang'] = lab_lang
    save['null0_tids'] = np.array(null0_tids)
    save['partner_idx'] = np.array(partner)
    save['proj_base'] = proj_f0
    save['proj_null'] = pB
    save['proj_sameword'] = pC
    save['proj_swapword'] = pD
    npz_path = os.path.join(
        OUT, 'omega_g2_robustness_source_qwen.npz')
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
